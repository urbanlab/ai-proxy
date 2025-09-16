from prometheus_client import start_http_server, REGISTRY
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from prometheus_client.registry import Collector
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Tuple

class SQLiteCollector(Collector):
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_db_connection(self):
        return sqlite3.connect(self.db_path)
    
    def collect(self):
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Export raw time-series data
            yield from self._get_timeseries_metrics(cursor)
            
            # Export current totals (for single-value panels)
            yield from self._get_current_totals(cursor)
            
            # Export distribution metrics (for pie charts)
            yield from self._get_distribution_metrics(cursor)
            
        finally:
            conn.close()
    
    def _get_timeseries_metrics(self, cursor):
        """Export time-series data - let Grafana handle time filtering"""
        
        # Requests per hour by model and user (time series)
        cursor.execute("""
            SELECT 
                strftime('%s', datetime(strftime('%Y-%m-%d %H:00:00', created_at))) as timestamp,
                model_name,
                user_name,
                COUNT(*) as requests,
                COALESCE(SUM(tokens_used), 0) as tokens,
                AVG(response_latency) as avg_latency,
                COALESCE(SUM(co2_emission), 0) as co2
            FROM requests 
            WHERE created_at >= datetime('now', '-30 days')  -- Last 30 days
            GROUP BY 
                strftime('%Y-%m-%d %H:00:00', created_at),
                model_name,
                user_name
            ORDER BY timestamp
        """)
        results = cursor.fetchall()
        
        # Create separate metrics for each measurement
        requests_metric = GaugeMetricFamily(
            'llm_requests_hourly', 
            'Requests per hour by model and user', 
            labels=['model', 'user', 'timestamp']
        )
        tokens_metric = GaugeMetricFamily(
            'llm_tokens_hourly', 
            'Tokens per hour by model and user', 
            labels=['model', 'user', 'timestamp']
        )
        latency_metric = GaugeMetricFamily(
            'llm_latency_hourly', 
            'Average latency per hour by model and user', 
            labels=['model', 'user', 'timestamp']
        )
        co2_metric = GaugeMetricFamily(
            'llm_co2_hourly', 
            'CO2 emissions per hour by model and user', 
            labels=['model', 'user', 'timestamp']
        )
        
        for timestamp, model, user, requests, tokens, avg_latency, co2 in results:
            labels = [model, user, str(timestamp)]
            requests_metric.add_metric(labels, requests)
            tokens_metric.add_metric(labels, tokens)
            latency_metric.add_metric(labels, avg_latency or 0.0)
            co2_metric.add_metric(labels, co2)
        
        yield requests_metric
        yield tokens_metric
        yield latency_metric
        yield co2_metric
    
    def _get_current_totals(self, cursor):
        """Export current totals for single-value panels"""
        
        # Total requests by model
        cursor.execute("SELECT model_name, COUNT(*) FROM requests GROUP BY model_name")
        results = cursor.fetchall()
        metric = GaugeMetricFamily('llm_total_requests_by_model', 'Total requests by model', labels=['model'])
        for model, count in results:
            metric.add_metric([model], count)
        yield metric
        
        # Total tokens by model
        cursor.execute("SELECT model_name, COALESCE(SUM(tokens_used), 0) FROM requests GROUP BY model_name")
        results = cursor.fetchall()
        metric = GaugeMetricFamily('llm_total_tokens_by_model', 'Total tokens by model', labels=['model'])
        for model, tokens in results:
            metric.add_metric([model], tokens)
        yield metric
        
        # Total requests by user
        cursor.execute("SELECT user_name, COUNT(*) FROM requests GROUP BY user_name")
        results = cursor.fetchall()
        metric = GaugeMetricFamily('llm_total_requests_by_user', 'Total requests by user', labels=['user'])
        for user, count in results:
            metric.add_metric([user], count)
        yield metric
        
        # Total tokens by user
        cursor.execute("SELECT user_name, COALESCE(SUM(tokens_used), 0) FROM requests GROUP BY user_name")
        results = cursor.fetchall()
        metric = GaugeMetricFamily('llm_total_tokens_by_user', 'Total tokens by user', labels=['user'])
        for user, tokens in results:
            metric.add_metric([user], tokens)
        yield metric
        
        # Average latency by model
        cursor.execute("""
            SELECT model_name, AVG(response_latency) 
            FROM requests 
            WHERE response_latency IS NOT NULL 
            GROUP BY model_name
        """)
        results = cursor.fetchall()
        metric = GaugeMetricFamily('llm_avg_latency_by_model', 'Average latency by model', labels=['model'])
        for model, avg_latency in results:
            metric.add_metric([model], avg_latency or 0.0)
        yield metric
        
        # Average latency by user
        cursor.execute("""
            SELECT user_name, AVG(response_latency) 
            FROM requests 
            WHERE response_latency IS NOT NULL 
            GROUP BY user_name
        """)
        results = cursor.fetchall()
        metric = GaugeMetricFamily('llm_avg_latency_by_user', 'Average latency by user', labels=['user'])
        for user, avg_latency in results:
            metric.add_metric([user], avg_latency or 0.0)
        yield metric
        
        # Min/Max tokens per request by user
        cursor.execute("""
            SELECT user_name, MIN(tokens_used), MAX(tokens_used)
            FROM requests 
            WHERE tokens_used IS NOT NULL AND tokens_used > 0
            GROUP BY user_name
        """)
        results = cursor.fetchall()
        min_metric = GaugeMetricFamily('llm_min_tokens_by_user', 'Min tokens per request by user', labels=['user'])
        max_metric = GaugeMetricFamily('llm_max_tokens_by_user', 'Max tokens per request by user', labels=['user'])
        for user, min_tokens, max_tokens in results:
            min_metric.add_metric([user], min_tokens or 0)
            max_metric.add_metric([user], max_tokens or 0)
        yield min_metric
        yield max_metric
    
    def _get_distribution_metrics(self, cursor):
        """Export distribution data for pie charts"""
        
        # User distribution by requests
        cursor.execute("SELECT user_name, COUNT(*) FROM requests GROUP BY user_name")
        results = cursor.fetchall()
        metric = GaugeMetricFamily('llm_user_distribution_requests', 'User distribution by requests', labels=['user'])
        for user, count in results:
            metric.add_metric([user], count)
        yield metric
        
        # User distribution by tokens
        cursor.execute("SELECT user_name, COALESCE(SUM(tokens_used), 0) FROM requests GROUP BY user_name")
        results = cursor.fetchall()
        metric = GaugeMetricFamily('llm_user_distribution_tokens', 'User distribution by tokens', labels=['user'])
        for user, tokens in results:
            metric.add_metric([user], tokens)
        yield metric
        
        # Model distribution by requests
        cursor.execute("SELECT model_name, COUNT(*) FROM requests GROUP BY model_name")
        results = cursor.fetchall()
        metric = GaugeMetricFamily('llm_model_distribution_requests', 'Model distribution by requests', labels=['model'])
        for model, count in results:
            metric.add_metric([model], count)
        yield metric
        
        # Model distribution by tokens
        cursor.execute("SELECT model_name, COALESCE(SUM(tokens_used), 0) FROM requests GROUP BY model_name")
        results = cursor.fetchall()
        metric = GaugeMetricFamily('llm_model_distribution_tokens', 'Model distribution by tokens', labels=['model'])
        for model, tokens in results:
            metric.add_metric([model], tokens)
        yield metric

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SQLite Prometheus Exporter for LLM Analytics')
    parser.add_argument('--db-path', default='/data/requests.db', help='Path to SQLite database')
    parser.add_argument('--port', type=int, default=8001, help='Port to serve metrics on')
    
    args = parser.parse_args()
    
    # Register the collector
    REGISTRY.register(SQLiteCollector(args.db_path))
    
    # Start the HTTP server
    start_http_server(args.port)
    print(f"Serving metrics on port {args.port}")
    print(f"Database path: {args.db_path}")
    print("Metrics available at http://localhost:8001/metrics")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Exporter stopped")

if __name__ == '__main__':
    main()