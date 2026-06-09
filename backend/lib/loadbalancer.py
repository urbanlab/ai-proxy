"""Health-aware load balancer for upstream model endpoints.

When the same `model_name` is declared multiple times in `config.yaml`, each
entry is treated as an interchangeable upstream endpoint. This module health
checks every such endpoint in the background and hands out healthy endpoints in
round-robin order, so requests for a model are spread across its replicas and
unhealthy backends are skipped automatically.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from lib.metric import log_endpoint_health

logger = logging.getLogger("loadbalancer")


class LoadBalancer:
    def __init__(self, config: Dict[str, Any], interval: int = 30, timeout: int = 5):
        self.interval = interval
        self.timeout = timeout
        self._groups: Dict[str, List[Dict[str, Any]]] = {}
        self._health: Dict[str, bool] = {}
        self._rr: Dict[str, int] = {}
        self._task: Optional[asyncio.Task] = None
        self.load_config(config)

    def load_config(self, config: Dict[str, Any]) -> None:
        """(Re)build endpoint groups from a config dict."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for model in config.get("model_list", []):
            groups.setdefault(model["model_name"], []).append(model)
        self._groups = groups
        # Optimistically assume endpoints are healthy until the first probe runs,
        # so the proxy works immediately on startup.
        for endpoints in groups.values():
            for ep in endpoints:
                self._health.setdefault(self._endpoint_id(ep), True)

    @staticmethod
    def _endpoint_id(ep: Dict[str, Any]) -> str:
        p = ep.get("params", {})
        return f"{ep['model_name']}|{p.get('api_base')}|{p.get('model')}"

    def _monitored_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """Unique endpoints that belong to a load-balanced (>1 replica) group."""
        seen: Dict[str, Dict[str, Any]] = {}
        for endpoints in self._groups.values():
            if len(endpoints) < 2:
                continue
            for ep in endpoints:
                seen[self._endpoint_id(ep)] = ep
        return seen

    def has_model(self, model_name: str) -> bool:
        return model_name in self._groups

    def select(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return a model config for `model_name`, balanced across healthy replicas.

        Single-endpoint models are returned as-is. For load-balanced models we
        round-robin over the healthy replicas, falling back to the full set if
        every replica currently looks unhealthy (better to try than to fail).
        """
        endpoints = self._groups.get(model_name)
        if not endpoints:
            return None
        if len(endpoints) == 1:
            return endpoints[0]

        healthy = [ep for ep in endpoints if self._health.get(self._endpoint_id(ep), True)]
        pool = healthy if healthy else endpoints
        idx = self._rr.get(model_name, 0) % len(pool)
        self._rr[model_name] = idx + 1
        return pool[idx]

    def health_snapshot(self) -> Dict[str, Any]:
        """Human-readable view of monitored endpoints and their status."""
        out: Dict[str, Any] = {}
        for name, endpoints in self._groups.items():
            if len(endpoints) < 2:
                continue
            out[name] = [
                {
                    "api_base": ep["params"].get("api_base"),
                    "model": ep["params"].get("model"),
                    "healthy": self._health.get(self._endpoint_id(ep), True),
                }
                for ep in endpoints
            ]
        return out

    async def _check_endpoint(self, session: aiohttp.ClientSession, ep: Dict[str, Any]) -> bool:
        p = ep.get("params", {})
        base = (p.get("api_base") or "").rstrip("/")
        if not base:
            return False
        url = f"{base}/models"

        headers: Dict[str, str] = {}
        api_key = p.get("api_key")
        if "anthropic" in base:
            if api_key and api_key != "no_token":
                headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        elif api_key and api_key != "no_token":
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                # Any non-5xx response means the upstream is reachable and serving.
                return resp.status < 500
        except Exception as e:
            logger.warning("Health check failed for %s: %s", url, e)
            return False

    async def check_all(self) -> None:
        endpoints = self._monitored_endpoints()
        if not endpoints:
            return
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[self._check_endpoint(session, ep) for ep in endpoints.values()]
            )
        for (eid, ep), healthy in zip(endpoints.items(), results):
            if self._health.get(eid) != healthy:
                logger.info("Endpoint %s is now %s", eid, "healthy" if healthy else "unhealthy")
            self._health[eid] = healthy
            log_endpoint_health(ep["model_name"], ep["params"].get("api_base", ""), healthy)

    async def _run(self) -> None:
        while True:
            try:
                await self.check_all()
            except Exception as e:
                logger.warning("Health check cycle error: %s", e)
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        """Launch the background health-check loop (idempotent)."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())
