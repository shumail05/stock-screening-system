import pandas as pd
import numpy as np

class SMMACalculator:
    def calculate_smma(self, df, period=20):
        return df['ltp'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    def detect_crossover(self, df, fast_period=20, slow_period=120):
        fast_smma = self.calculate_smma(df, fast_period)
        slow_smma = self.calculate_smma(df, slow_period)
        prev_fast = fast_smma.shift(1)
        prev_slow = slow_smma.shift(1)
        bullish = (prev_fast <= prev_slow) & (fast_smma > slow_smma)
        bearish = (prev_fast >= prev_slow) & (fast_smma < slow_smma)
        return {
            'fast_smma': fast_smma,
            'slow_smma': slow_smma,
            'bullish_signal': bullish,
            'bearish_signal': bearish
        }
