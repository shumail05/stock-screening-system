import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return accuracy_score(y_test, self.model.predict(X_test))

    def predict_signal_quality(self, current_data):
        if not self.is_trained:
            return {'probability': 50.0, 'recommendation': 'AVOID', 'reason': 'Model not trained'}
        features = self.prepare_features(current_data)
        latest = features.iloc[[-1]]
        probability = float(self.model.predict_proba(latest)[0][1]) * 100
        if probability > 70:
            return {'probability': round(probability, 2), 'recommendation': 'ACCEPT', 'reason': 'High probability based on conditions'}
        if probability > 50:
            return {'probability': round(probability, 2), 'recommendation': 'CAUTION', 'reason': 'Moderate probability'}
        return {'probability': round(probability, 2), 'recommendation': 'AVOID', 'reason': 'Low probability / adverse conditions'}
