"""Health-aware, load-aware balancer for upstream model endpoints.

When the same `model_name` is declared multiple times in `config.yaml`, each
entry is treated as an interchangeable upstream endpoint. This module health
checks every such endpoint in the background and hands out healthy endpoints so
requests for a model are spread across its replicas and unhealthy backends are
skipped automatically.

Dispatch is **least-connections**: a request is sent to the replica currently
handling the fewest in-flight requests, so a free GPU is always preferred over a
busy one (round-robin only balances totals over time, not current occupancy).

An endpoint may declare `max_concurrency` in its params. The balancer then never
sends more than that many simultaneous requests to it; a GPU that serves one
request at a time should set `max_concurrency: 1`. When every eligible replica is
at its limit, callers using `acquire()` **queue** until a slot frees instead of
piling another request onto a busy backend. Endpoints without `max_concurrency`
are unlimited (a `default_max_concurrency` at the top level sets a fallback).
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
        # In-flight request count per endpoint id (drives least-connections
        # dispatch) and the per-endpoint concurrency cap (None = unlimited).
        self._inflight: Dict[str, int] = {}
        self._limit: Dict[str, Optional[int]] = {}
        self._default_limit: Optional[int] = None
        # Queue of futures for callers waiting on a free slot. A waiter appends a
        # future and awaits it; `release()` (which is synchronous, so a cancelled
        # request can never skip it) wakes every waiter to re-check. We avoid
        # asyncio.Condition on purpose: its release path must `await` the lock, and
        # a request cancelled mid-await would then never decrement its in-flight
        # count — leaking the slot until the endpoint looks permanently full.
        self._waiters: List["asyncio.Future"] = []
        self._task: Optional[asyncio.Task] = None
        self.load_config(config)

    def load_config(self, config: Dict[str, Any]) -> None:
        """(Re)build endpoint groups from a config dict."""
        self._default_limit = config.get("default_max_concurrency")
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for model in config.get("model_list", []):
            groups.setdefault(model["model_name"], []).append(model)
            eid = self._endpoint_id(model)
            self._limit[eid] = model.get("params", {}).get(
                "max_concurrency", self._default_limit
            )
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

    def endpoint_id(self, ep: Dict[str, Any]) -> str:
        """Public alias for callers that need to identify an endpoint."""
        return self._endpoint_id(ep)

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

    def candidates(self, model_name: str) -> List[Dict[str, Any]]:
        """Ordered endpoints to attempt for `model_name`, best first.

        Healthy replicas come first in round-robin order (so concurrent requests
        still spread across GPUs), then currently-unhealthy ones as a last
        resort. The caller walks this list, failing over on connection errors,
        so a downed endpoint never reaches the client as an error while another
        replica can serve the request.
        """
        endpoints = self._groups.get(model_name)
        if not endpoints:
            return []
        if len(endpoints) == 1:
            return list(endpoints)

        healthy = [ep for ep in endpoints if self._health.get(self._endpoint_id(ep), True)]
        dead = [ep for ep in endpoints if not self._health.get(self._endpoint_id(ep), True)]
        if healthy:
            idx = self._rr.get(model_name, 0) % len(healthy)
            self._rr[model_name] = idx + 1
            healthy = healthy[idx:] + healthy[:idx]
        return healthy + dead

    def endpoints(self, model_name: str) -> List[Dict[str, Any]]:
        """Raw endpoint group for `model_name`, in declared order — NO round-robin
        rotation and no health ordering. For callers that only need the shared,
        replica-identical metadata (cost, vision, limits); actual dispatch goes
        through `acquire()`. Rotating here would corrupt the `_rr` counter that
        `_pick_least_loaded` relies on, pinning every request to one replica."""
        return list(self._groups.get(model_name) or [])

    def _eligible_pool(self, model_name: str) -> List[Dict[str, Any]]:
        """Healthy replicas for `model_name`, or the full set if none look healthy
        (better to try a possibly-down endpoint than to fail outright)."""
        endpoints = self._groups.get(model_name) or []
        healthy = [ep for ep in endpoints if self._health.get(self._endpoint_id(ep), True)]
        return healthy if healthy else endpoints

    def _has_capacity(self, ep: Dict[str, Any]) -> bool:
        eid = self._endpoint_id(ep)
        limit = self._limit.get(eid)
        return limit is None or self._inflight.get(eid, 0) < limit

    def _pick_least_loaded(self, model_name: str, eps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Least-connections pick among `eps`, round-robin breaking ties so equally
        idle replicas are used evenly."""
        min_load = min(self._inflight.get(self._endpoint_id(e), 0) for e in eps)
        least = [e for e in eps if self._inflight.get(self._endpoint_id(e), 0) == min_load]
        if len(least) == 1:
            return least[0]
        idx = self._rr.get(model_name, 0) % len(least)
        self._rr[model_name] = idx + 1
        return least[idx]

    def _try_reserve(self, model_name: str, tried: set, waited: bool):
        """Synchronously (no await ⇒ atomic under asyncio) try to reserve a slot.

        Returns ("ok", ep) having bumped its in-flight count, ("done", None) when
        every replica has already been tried (failover exhausted), or
        ("wait", None) when replicas remain but all are at capacity."""
        servable = [
            ep for ep in self._eligible_pool(model_name)
            if self._endpoint_id(ep) not in tried
        ]
        if not servable:
            return "done", None
        free = [ep for ep in servable if self._has_capacity(ep)]
        if not free:
            return "wait", None
        ep = self._pick_least_loaded(model_name, free)
        eid = self._endpoint_id(ep)
        self._inflight[eid] = self._inflight.get(eid, 0) + 1
        tried.add(eid)
        logger.info(
            "dispatch %s -> %s (inflight now %s)%s",
            model_name, eid, self._inflight, " [after queue]" if waited else "",
        )
        return "ok", ep

    async def acquire(self, model_name: str, tried: set) -> Optional[Dict[str, Any]]:
        """Reserve a slot on the best available replica for `model_name` and return
        its endpoint config, or None if every replica has already been tried.

        Selection is least-connections over the replicas not in `tried`; the chosen
        endpoint's id is added to `tried` and its in-flight count is bumped. If all
        eligible replicas are at their `max_concurrency`, this waits (queues) until a
        slot frees rather than overloading a busy backend. The caller MUST pair every
        non-None return with exactly one `release(ep)`."""
        waited = False
        while True:
            status, ep = self._try_reserve(model_name, tried, waited)
            if status == "ok":
                return ep
            if status == "done":
                return None
            # Every eligible replica is at capacity: queue until a slot frees.
            logger.info(
                "queue %s: all replicas at capacity, waiting (inflight %s, limits %s)",
                model_name, self._inflight, self._limit,
            )
            waited = True
            fut: "asyncio.Future" = asyncio.get_running_loop().create_future()
            self._waiters.append(fut)
            try:
                await fut
            finally:
                # Drop our waiter whether we were woken or the request was
                # cancelled while queued, so it can't linger and be double-set.
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass

    def release(self, ep: Dict[str, Any]) -> None:
        """Return a slot reserved by `acquire` and wake queued waiters.

        SYNCHRONOUS on purpose: callers invoke it from a `finally`, and a request
        cancelled mid-stream must never skip the in-flight decrement (that is what
        pinned endpoints at capacity and wedged the queue). No `await` here means
        cancellation cannot interrupt it."""
        eid = self._endpoint_id(ep)
        if self._inflight.get(eid, 0) > 0:
            self._inflight[eid] -= 1
        logger.info("release %s (inflight now %s)", eid, self._inflight)
        # Wake every waiter to re-check; the one that grabs the freed slot wins,
        # the rest re-queue. set_result is synchronous, so this can't be cancelled.
        waiters, self._waiters = self._waiters, []
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)

    def mark_unhealthy(self, ep: Dict[str, Any]) -> None:
        """Flag an endpoint dead immediately (e.g. on a connection failure),
        without waiting for the next background probe cycle."""
        eid = self._endpoint_id(ep)
        if self._health.get(eid) is not False:
            logger.info("Endpoint %s marked unhealthy (connection failure)", eid)
        self._health[eid] = False
        log_endpoint_health(ep["model_name"], ep["params"].get("api_base", ""), False)

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
                    "inflight": self._inflight.get(self._endpoint_id(ep), 0),
                    "max_concurrency": self._limit.get(self._endpoint_id(ep)),
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
        # A GPU that serves one request at a time (max_concurrency=1) usually can't
        # answer a /models probe while it is mid-generation, so probing a busy
        # endpoint would time out and wrongly flag an actively-serving GPU as dead —
        # which then routes ALL traffic to the other replica until the next cycle.
        # In-flight traffic already proves the endpoint is alive, so skip the probe
        # for busy endpoints and keep them healthy; a truly hung request is caught
        # by the per-request timeouts (which call mark_unhealthy) instead.
        busy = {eid for eid in endpoints if self._inflight.get(eid, 0) > 0}
        to_probe = {eid: ep for eid, ep in endpoints.items() if eid not in busy}
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[self._check_endpoint(session, ep) for ep in to_probe.values()]
            )
        health: Dict[str, bool] = dict(zip(to_probe.keys(), results))
        for eid in busy:
            health[eid] = True  # serving traffic ⇒ demonstrably alive
        for eid, healthy in health.items():
            ep = endpoints[eid]
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
