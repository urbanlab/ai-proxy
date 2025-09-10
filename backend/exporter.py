from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY, CollectorRegistry
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from prometheus_client.registry import Collector
import sqlite3
import time
from datetime import datetime, timedelta
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
            # Total CO2 consumption
            cursor.execute("SELECT SUM(co2_emission) FROM Requests")
            total_co2 = cursor.fetchone()[0] or 0
            yield GaugeMetricFamily('total_co2_emissions', 'Total CO2 emissions in grams', value=total_co2)
            
            # CO2 consumption by time periods
            co2_periods = self._get_co2_by_periods(cursor)
            yield GaugeMetricFamily('co2_emissions_daily', 'CO2 emissions today', value=co2_periods['day'])
            yield GaugeMetricFamily('co2_emissions_weekly', 'CO2 emissions this week', value=co2_periods['week'])
            yield GaugeMetricFamily('co2_emissions_monthly', 'CO2 emissions this month', value=co2_periods['month'])
            yield GaugeMetricFamily('co2_emissions_yearly', 'CO2 emissions this year', value=co2_periods['year'])
            
            # Total requests
            cursor.execute("SELECT COUNT(*) FROM Requests")
            total_requests = cursor.fetchone()[0] or 0
            yield GaugeMetricFamily('total_requests', 'Total number of requests', value=total_requests)
            
            # Requests by time periods
            request_periods = self._get_requests_by_periods(cursor)
            yield GaugeMetricFamily('requests_hourly', 'Requests in last hour', value=request_periods['hour'])
            yield GaugeMetricFamily('requests_daily', 'Requests today', value=request_periods['day'])
            yield GaugeMetricFamily('requests_monthly', 'Requests this month', value=request_periods['month'])
            
            # Requests per user
            cursor.execute("SELECT user_name, COUNT(*) FROM Requests GROUP BY user_name")
            user_requests = cursor.fetchall()
            user_request_metric = CounterMetricFamily('requests_per_user_total', 'Total requests per user', labels=['user'])
            for user, count in user_requests:
                user_request_metric.add_metric([user], count)
            yield user_request_metric
            
            # Requests per user by time periods
            yield from self._get_user_requests_by_periods(cursor)
            
            # Requests per model by time periods
            yield from self._get_model_requests_by_periods(cursor)
            
            # CO2 emissions per model
            cursor.execute("SELECT model_name, SUM(co2_emission) FROM Requests GROUP BY model_name")
            model_co2 = cursor.fetchall()
            model_co2_metric = CounterMetricFamily('co2_emissions_per_model_total', 'Total CO2 emissions per model', labels=['model'])
            for model, co2 in model_co2:
                model_co2_metric.add_metric([model], co2 or 0)
            yield model_co2_metric
            
        finally:
            conn.close()
    
    def _get_co2_by_periods(self, cursor) -> Dict[str, float]:
        periods = {}
        
        # Day
        cursor.execute("""
            SELECT SUM(co2_emission) FROM Requests 
            WHERE created_at >= datetime('now', 'start of day')
        """)
        periods['day'] = cursor.fetchone()[0] or 0
        
        # Week
        cursor.execute("""
            SELECT SUM(co2_emission) FROM Requests 
            WHERE created_at >= datetime('now', '-7 days')
        """)
        periods['week'] = cursor.fetchone()[0] or 0
        
        # Month
        cursor.execute("""
            SELECT SUM(co2_emission) FROM Requests 
            WHERE created_at >= datetime('now', 'start of month')
        """)
        periods['month'] = cursor.fetchone()[0] or 0
        
        # Year
        cursor.execute("""
            SELECT SUM(co2_emission) FROM Requests 
            WHERE created_at >= datetime('now', 'start of year')
        """)
        periods['year'] = cursor.fetchone()[0] or 0
        
        return periods
    
    def _get_requests_by_periods(self, cursor) -> Dict[str, int]:
        periods = {}
        
        # Hour
        cursor.execute("""
            SELECT COUNT(*) FROM Requests 
            WHERE created_at >= datetime('now', '-1 hour')
        """)
        periods['hour'] = cursor.fetchone()[0] or 0
        
        # Day
        cursor.execute("""
            SELECT COUNT(*) FROM Requests 
            WHERE created_at >= datetime('now', 'start of day')
        """)
        periods['day'] = cursor.fetchone()[0] or 0
        
        # Month
        cursor.execute("""
            SELECT COUNT(*) FROM Requests 
            WHERE created_at >= datetime('now', 'start of month')
        """)
        periods['month'] = cursor.fetchone()[0] or 0
        
        return periods
    
    def _get_user_requests_by_periods(self, cursor):
        # Requests per user per hour
        cursor.execute("""
            SELECT user_name, COUNT(*) FROM Requests 
            WHERE created_at >= datetime('now', '-1 hour')
            GROUP BY user_name
        """)
        hourly_user_requests = cursor.fetchall()
        hourly_metric = GaugeMetricFamily('requests_per_user_hourly', 'Requests per user in last hour', labels=['user'])
        for user, count in hourly_user_requests:
            hourly_metric.add_metric([user], count)
        yield hourly_metric
        
        # Requests per user per day
        cursor.execute("""
            SELECT user_name, COUNT(*) FROM Requests 
            WHERE created_at >= datetime('now', 'start of day')
            GROUP BY user_name
        """)
        daily_user_requests = cursor.fetchall()
        daily_metric = GaugeMetricFamily('requests_per_user_daily', 'Requests per user today', labels=['user'])
        for user, count in daily_user_requests:
            daily_metric.add_metric([user], count)
        yield daily_metric
        
        # Requests per user per month
        cursor.execute("""
            SELECT user_name, COUNT(*) FROM Requests 
            WHERE created_at >= datetime('now', 'start of month')
            GROUP BY user_name
        """)
        monthly_user_requests = cursor.fetchall()
        monthly_metric = GaugeMetricFamily('requests_per_user_monthly', 'Requests per user this month', labels=['user'])
        for user, count in monthly_user_requests:
            monthly_metric.add_metric([user], count)
        yield monthly_metric
    
    def _get_model_requests_by_periods(self, cursor):
        # Requests per model per hour
        cursor.execute("""
            SELECT model_name, COUNT(*) FROM Requests 
            WHERE created_at >= datetime('now', '-1 hour')
            GROUP BY model_name
        """)
        hourly_model_requests = cursor.fetchall()
        hourly_metric = GaugeMetricFamily('requests_per_model_hourly', 'Requests per model in last hour', labels=['model'])
        for model, count in hourly_model_requests:
            hourly_metric.add_metric([model], count)
        yield hourly_metric
        
        # Requests per model per day
        cursor.execute("""
            SELECT model_name, COUNT(*) FROM Requests 
            WHERE created_at >= datetime('now', 'start of day')
            GROUP BY model_name
        """)
        daily_model_requests = cursor.fetchall()
        daily_metric = GaugeMetricFamily('requests_per_model_daily', 'Requests per model today', labels=['model'])
        for model, count in daily_model_requests:
            daily_metric.add_metric([model], count)
        yield daily_metric
        
        # Requests per model per month
        cursor.execute("""
            SELECT model_name, COUNT(*) FROM Requests 
            WHERE created_at >= datetime('now', 'start of month')
            GROUP BY model_name
        """)
        monthly_model_requests = cursor.fetchall()
        monthly_metric = GaugeMetricFamily('requests_per_model_monthly', 'Requests per model this month', labels=['model'])
        for model, count in monthly_model_requests:
            monthly_metric.add_metric([model], count)
        yield monthly_metric

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SQLite Prometheus Exporter')
    parser.add_argument('--db-path', default='/data/requests.db', help='Path to SQLite database')
    parser.add_argument('--port', type=int, default=8001, help='Port to serve metrics on')
    parser.add_argument('--interval', type=int, default=30, help='Scrape interval in seconds')
    
    args = parser.parse_args()
    
    # Register the collector
    REGISTRY.register(SQLiteCollector(args.db_path))
    
    # Start the HTTP server
    start_http_server(args.port)
    print(f"Serving metrics on port {args.port}")
    print(f"Database path: {args.db_path}")
    print("Metrics available at http://localhost:8001/metrics")
    
    # Keep the server running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Exporter stopped")

if __name__ == '__main__':
    main()