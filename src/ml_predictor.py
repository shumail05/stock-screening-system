import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.training_date = None
        self.accuracy = 0.0
        self.metrics = {
            'total_signals': 0,
            'avoided_signals': 0,
            'profitable_signals': 0,
            'loss_signals': 0,
            'avoidance_rate': 0.0,
            'profit_rate': 0.0,
            'loss_rate': 0.0
        }

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_features(self, df):
        features = pd.DataFrame()
        features['sma_fast'] = df['ltp'].ewm(alpha=1/20, adjust=False).mean()
        features['sma_slow'] = df['ltp'].ewm(alpha=1/120, adjust=False).mean()
        features['rsi'] = self.calculate_rsi(df['ltp'])
        features['volume_ma'] = df['volume'].rolling(20).mean()
        features['price_change'] = df['ltp'].pct_change()
        features['volatility'] = df['ltp'].rolling(20).std()
        if 'ltq' in df.columns:
            features['ltq_ma'] = df['ltq'].rolling(20).mean()
            features['ltq_ratio'] = df['ltq'] / (df['volume'].rolling(20).mean() + 1e-9)
        return features.dropna()

    def create_profitability_labels(self, df, holding_period=20):
        labels = []
        for i in range(len(df) - holding_period):
            entry = df.iloc[i]['ltp']
            future = df.iloc[i+1:i+1+holding_period]['ltp']
            labels.append(1 if future.iloc[-1] > entry else 0)
        return pd.Series(labels)

    def train(self, historical_data):
        X = self.prepare_features(historical_data)
        y = self.create_profitability_labels(historical_data)
        min_len = min(len(X), len(y))
        X, y = X.iloc[:min_len], y.iloc[:min_len]
        if len(X) < 10:
            return 0.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.training_date = datetime.now().isoformat()
        self.accuracy = float(accuracy_score(y_test, self.model.predict(X_test)))
        self._save_model()
        self._update_metrics_history('train', self.accuracy)
        return self.accuracy

    def predict_signal_quality(self, current_data):
        features = self.prepare_features(current_data)
        if len(features) == 0:
            return {'probability': 50.0, 'recommendation': 'AVOID', 'reason': 'Insufficient data'}
        latest = features.iloc[[-1]]
        if not self.is_trained:
            return self._rule_based_prediction(latest.iloc[0], current_data)
        probability = float(self.model.predict_proba(latest)[0][1]) * 100
        if probability > 70:
            return {'probability': round(probability, 2), 'recommendation': 'ACCEPT', 'reason': 'High probability based on conditions'}
        if probability > 50:
            return {'probability': round(probability, 2), 'recommendation': 'CAUTION', 'reason': 'Moderate probability'}
        return {'probability': round(probability, 2), 'recommendation': 'AVOID', 'reason': 'Low probability / adverse conditions'}

    def _rule_based_prediction(self, feat, df):
        reasons = []
        score = 50.0
        if feat['sma_fast'] > feat['sma_slow']:
            score += 10
            reasons.append('Fast SMMA above Slow SMMA')
        else:
            score -= 10
            reasons.append('Fast SMMA below Slow SMMA')
        if feat['rsi'] > 70:
            score -= 15
            reasons.append('RSI overbought')
        elif feat['rsi'] < 30:
            score += 10
            reasons.append('RSI oversold')
        if 'ltq_ma' in feat and feat['ltq_ma'] > 0:
            if feat['ltq_ratio'] > 1.5:
                score += 10
                reasons.append('High LTQ relative to volume')
            elif feat['ltq_ratio'] < 0.5:
                score -= 10
                reasons.append('Low LTQ relative to volume')
        if feat['volume_ma'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5:
            score += 10
            reasons.append('Volume spike detected')
        if feat['volatility'] > df['ltp'].rolling(20).std().mean() * 2:
            score -= 10
            reasons.append('High volatility')
        score = max(0, min(100, score))
        if score > 70:
            rec = 'ACCEPT'
        elif score > 50:
            rec = 'CAUTION'
        else:
            rec = 'AVOID'
        reason = '; '.join(reasons) if reasons else 'Neutral market conditions'
        return {'probability': round(score, 2), 'recommendation': rec, 'reason': reason}

    def backtest_crossover(self, df, fast_period=20, slow_period=120, holding_period=20):
        if len(df) < slow_period + holding_period + 1:
            return {'buy_profitable': None, 'sell_profitable': None, 'reason': 'Insufficient data for backtest'}
        fast_smma = df['ltp'].ewm(alpha=1/fast_period, adjust=False, min_periods=fast_period).mean()
        slow_smma = df['ltp'].ewm(alpha=1/slow_period, adjust=False, min_periods=slow_period).mean()
        crossover_points = []
        for i in range(slow_period, len(df) - holding_period):
            prev_fast = fast_smma.iloc[i-1]
            prev_slow = slow_smma.iloc[i-1]
            curr_fast = fast_smma.iloc[i]
            curr_slow = slow_smma.iloc[i]
            if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
                continue
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                entry = df.iloc[i]['ltp']
                future = df.iloc[i+1:i+1+holding_period]['ltp']
                profit = ((future.iloc[-1] - entry) / entry) * 100 if entry != 0 else 0
                crossover_points.append({'type': 'bullish', 'entry': entry, 'profit_pct': profit})
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                entry = df.iloc[i]['ltp']
                future = df.iloc[i+1:i+1+holding_period]['ltp']
                profit = ((entry - future.iloc[-1]) / entry) * 100 if entry != 0 else 0
                crossover_points.append({'type': 'bearish', 'entry': entry, 'profit_pct': profit})
        if not crossover_points:
            return {'buy_profitable': None, 'sell_profitable': None, 'reason': 'No crossovers in history'}
        buy_signals = [c for c in crossover_points if c['type'] == 'bullish']
        sell_signals = [c for c in crossover_points if c['type'] == 'bearish']
        buy_profitable = sum(1 for c in buy_signals if c['profit_pct'] > 0) / len(buy_signals) if buy_signals else None
        sell_profitable = sum(1 for c in sell_signals if c['profit_pct'] > 0) / len(sell_signals) if sell_signals else None
        reasons = []
        if buy_signals:
            avg_buy_profit = np.mean([c['profit_pct'] for c in buy_signals])
            if buy_profitable < 0.4:
                reasons.append(f'Buy signals historically weak ({buy_profitable*100:.0f}% success, avg {avg_buy_profit:.1f}%)')
        if sell_signals:
            avg_sell_profit = np.mean([c['profit_pct'] for c in sell_signals])
            if sell_profitable < 0.4:
                reasons.append(f'Sell signals historically weak ({sell_profitable*100:.0f}% success, avg {avg_sell_profit:.1f}%)')
        if not reasons:
            reasons.append('Historical crossover performance acceptable')
        return {
            'buy_profitable': round(buy_profitable, 2) if buy_profitable is not None else None,
            'sell_profitable': round(sell_profitable, 2) if sell_profitable is not None else None,
            'reason': '; '.join(reasons)
        }

    def evaluate_crossover_signals(self, df, fast_period=20, slow_period=120, holding_period=20):
        backtest = self.backtest_crossover(df, fast_period, slow_period, holding_period)
        total = 0
        avoided = 0
        profitable = 0
        losses = 0
        if backtest.get('buy_profitable') is not None:
            total += 1
            if backtest['buy_profitable'] < 0.5:
                avoided += 1
            else:
                profitable += 1
        if backtest.get('sell_profitable') is not None:
            total += 1
            if backtest['sell_profitable'] < 0.5:
                avoided += 1
            else:
                profitable += 1
        if total == 0:
            return {
                'total_signals': 0,
                'avoided_signals': 0,
                'profitable_signals': 0,
                'loss_signals': 0,
                'avoidance_rate': 0.0,
                'profit_rate': 0.0,
                'loss_rate': 0.0,
                'reason': 'No crossovers detected'
            }
        loss_signals = total - avoided - profitable
        self.metrics = {
            'total_signals': total,
            'avoided_signals': avoided,
            'profitable_signals': profitable,
            'loss_signals': loss_signals,
            'avoidance_rate': round(avoided / total * 100, 2),
            'profit_rate': round(profitable / total * 100, 2),
            'loss_rate': round(loss_signals / total * 100, 2)
        }
        self._save_metrics()
        return self.metrics

    def _save_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self, f)

    def _save_metrics(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(METRICS_PATH, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def _update_metrics_history(self, event_type, value):
        history = []
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as f:
                history = json.load(f)
        history.append({
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'value': value,
            'metrics': self.metrics
        })
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

    @classmethod
    def load(cls):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        return cls()

    def get_training_history(self):
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as f:
                return json.load(f)
        return []

    def get_current_metrics(self):
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                return json.load(f)
        return self.metrics
