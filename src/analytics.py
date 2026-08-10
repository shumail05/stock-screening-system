from datetime import datetime, timedelta
import pandas as pd

class AnalyticsEngine:
    def calculate_exchange_quantity(self, historical_df, window_minutes):
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = historical_df[historical_df['timestamp'] >= cutoff]
        return int(recent['volume'].sum()) if not recent.empty else 0

    def calculate_avg_ltp(self, historical_df, window_minutes):
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = historical_df[historical_df['timestamp'] >= cutoff]
        return round(float(recent['ltp'].mean()), 2) if not recent.empty else 0.0
