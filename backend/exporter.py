from prometheus_client import start_http_server, REGISTRY
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from prometheus_client.registry import Collector
import sqlite3
import time
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
            # Export cumulative counters (these will be scraped regularly by Prometheus)
            yield from self._get_counter_metrics(cursor)
            
            # Export current state gauges
            yield from self._get_gauge_metrics(cursor)
            
        finally:
            conn.close()
    
    def _get_counter_metrics(self, cursor):
        """Export cumulative counters - Prometheus will calculate rates from these"""
        
        # Total requests by model and user (Counter)
        cursor.execute("""
            SELECT model_name, user_name, COUNT(*) as total
            FROM requests 
            GROUP BY model_name, user_name
        """)
        results = cursor.fetchall()
        
        metric = CounterMetricFamily(
            'llm_requests_total', 
            'Total number of requests by model and user', 
            labels=['model', 'user']
        )
        for model, user, total in results:
            metric.add_metric([model, user], total)
        yield metric
        
        # Total tokens by model and user (Counter)
        cursor.execute("""
            SELECT model_name, user_name, COALESCE(SUM(tokens_used), 0) as total_tokens
            FROM requests 
            GROUP BY model_name, user_name
        """)
        results = cursor.fetchall()
        
        metric = CounterMetricFamily(
            'llm_tokens_total', 
            'Total tokens processed by model and user', 
            labels=['model', 'user']
        )
        for model, user, tokens in results:
            metric.add_metric([model, user], tokens)
        yield metric
        
        # Total CO2 emissions by model and user (Counter)
        cursor.execute("""
            SELECT model_name, user_name, COALESCE(SUM(co2_emission), 0) as total_co2
            FROM requests 
            GROUP BY model_name, user_name
        """)
        results = cursor.fetchall()
        
        metric = CounterMetricFamily(
            'llm_co2_grams_total', 
            'Total CO2 emissions in grams by model and user', 
            labels=['model', 'user']
        )
        for model, user, co2 in results:
            metric.add_metric([model, user], co2)
        yield metric
        
        # Sum of all latencies (Counter) - for calculating averages
        cursor.execute("""
            SELECT model_name, user_name, 
                   COALESCE(SUM(response_latency), 0) as total_latency
            FROM requests 
            WHERE response_latency IS NOT NULL
            GROUP BY model_name, user_name
        """)
        results = cursor.fetchall()
        
        metric = CounterMetricFamily(
            'llm_latency_seconds_total', 
            'Sum of all response latencies by model and user', 
            labels=['model', 'user']
        )
        for model, user, latency_sum in results:
            metric.add_metric([model, user], latency_sum)
        yield metric
    
    def _get_gauge_metrics(self, cursor):
        """Export current state metrics"""
        
        # Average latency by model (Gauge)
        cursor.execute("""
            SELECT model_name, AVG(response_latency) as avg_latency
            FROM requests 
            WHERE response_latency IS NOT NULL 
            GROUP BY model_name
        """)
        results = cursor.fetchall()
        
        metric = GaugeMetricFamily(
            'llm_latency_seconds_avg', 
            'Average response latency by model', 
            labels=['model']
        )
        for model, avg_latency in results:
            metric.add_metric([model], avg_latency or 0.0)
        yield metric
        
        # Average latency by user (Gauge)
        cursor.execute("""
            SELECT user_name, AVG(response_latency) as avg_latency
            FROM requests 
            WHERE response_latency IS NOT NULL 
            GROUP BY user_name
        """)
        results = cursor.fetchall()
        
        metric = GaugeMetricFamily(
            'llm_latency_seconds_avg_by_user', 
            'Average response latency by user', 
            labels=['user']
        )
        for user, avg_latency in results:
            metric.add_metric([user], avg_latency or 0.0)
        yield metric
        
        # Min/Max tokens per request by user (Gauge)
        cursor.execute("""
            SELECT user_name, MIN(tokens_used), MAX(tokens_used)
            FROM requests 
            WHERE tokens_used IS NOT NULL AND tokens_used > 0
            GROUP BY user_name
        """)
        results = cursor.fetchall()
        
        min_metric = GaugeMetricFamily(
            'llm_tokens_min', 
            'Minimum tokens per request by user', 
            labels=['user']
        )
        max_metric = GaugeMetricFamily(
            'llm_tokens_max', 
            'Maximum tokens per request by user', 
            labels=['user']
        )
        
        for user, min_tokens, max_tokens in results:
            min_metric.add_metric([user], min_tokens or 0)
            max_metric.add_metric([user], max_tokens or 0)
        
        yield min_metric
        yield max_metric
        
        # Recent activity (last hour) - Gauge
        cursor.execute("""
            SELECT model_name, user_name, COUNT(*) as recent_requests
            FROM requests 
            WHERE created_at >= datetime('now', '-1 hour')
            GROUP BY model_name, user_name
        """)
        results = cursor.fetchall()
        
        metric = GaugeMetricFamily(
            'llm_requests_last_hour', 
            'Number of requests in the last hour', 
            labels=['model', 'user']
        )
        for model, user, count in results:
            metric.add_metric([model, user], count)
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
    print("Metrics available at http://localhost:{args.port}/metrics")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Exporter stopped")

if __name__ == '__main__':
    main()