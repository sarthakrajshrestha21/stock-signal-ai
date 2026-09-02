import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os
import json
from datetime import datetime

class RecursiveSignalEngine:
    """
    Iterative learning engine for stock signals.
    Retrains on new data = "recursive self-improvement" in practice.
    """
    def __init__(self, model_path="model.pkl", history_path="history.json"):
        self.model_path = model_path
        self.history_path = history_path
        self.model = None
        self.is_fitted = False
        self.training_history = []
        self.feature_columns = None

        # Load existing model if available (recursive learning)
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.is_fitted = True
            print("✅ Loaded existing model. Continuing recursive learning...")

        # Load training history
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)

    def parse_excel(self, file_path_or_buffer):
        """Parse multi-sheet Excel where sheet names are dates (YYYY_MM_DD)"""
        xl = pd.ExcelFile(file_path_or_buffer)
        all_data = []

        for sheet_name in xl.sheet_names:
            try:
                date = datetime.strptime(sheet_name, '%Y_%m_%d')
                df = xl.parse(sheet_name)
                df['Date'] = date
                all_data.append(df)
            except ValueError:
                continue  # Skip sheets that aren't date-formatted

        if not all_data:
            raise ValueError("No valid date sheets found. Sheets must be named YYYY_MM_DD")

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        return combined

    def engineer_features(self, df):
        """Create technical features from raw stock data"""
        df = df.copy()
        df = df.sort_values(['Symbol', 'Date'])

        # Group by symbol for calculations
        grouped = df.groupby('Symbol')

        # Returns
        df['returns_1d'] = grouped['Close'].pct_change()
        df['returns_5d'] = grouped['Close'].pct_change(5)
        df['returns_10d'] = grouped['Close'].pct_change(10)

        # Volatility
        df['volatility_5d'] = grouped['returns_1d'].transform(lambda x: x.rolling(5).std())
        df['volatility_20d'] = grouped['returns_1d'].transform(lambda x: x.rolling(20).std())

        # Moving average signals (you already have 120/180 day in data)
        df['ma_ratio'] = df['120 Days'] / df['180 Days']
        df['close_vs_120ma'] = (df['Close'] - df['120 Days']) / df['120 Days']
        df['close_vs_180ma'] = (df['Close'] - df['180 Days']) / df['180 Days']

        # 52-week range position
        df['range_position'] = (df['Close'] - df['52 Weeks Low']) / (df['52 Weeks High'] - df['52 Weeks Low'] + 1e-9)
        df['dist_from_52w_high'] = (df['Close'] - df['52 Weeks High']) / df['52 Weeks High']
        df['dist_from_52w_low'] = (df['Close'] - df['52 Weeks Low']) / df['52 Weeks Low']

        # VWAP signals
        df['vwap_deviation'] = df['VWAP %']
        df['close_vs_vwap'] = (df['Close'] - df['VWAP']) / df['VWAP']

        # Volume features
        df['volume_ma_5'] = grouped['Vol'].transform(lambda x: x.rolling(5).mean())
        df['volume_ma_20'] = grouped['Vol'].transform(lambda x: x.rolling(20).mean())
        df['volume_ratio_5'] = df['Vol'] / (df['volume_ma_5'] + 1e-9)
        df['volume_ratio_20'] = df['Vol'] / (df['volume_ma_20'] + 1e-9)

        # Turnover/Transaction efficiency
        df['avg_trade_size'] = df['Turnover'] / (df['Trans'] + 1e-9)
        df['turnover_ratio'] = df['Turnover'] / (df['volume_ma_20'] * df['VWAP'] + 1e-9)

        # Range analysis
        df['body_size'] = abs(df['Close'] - df['Open']) / df['Open']
        df['upper_shadow'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / df['Open']
        df['lower_shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / df['Open']

        # Target: Will stock go up > 1.5% in next 5 days? (swing signal)
        df['future_return_5d'] = grouped['Close'].shift(-5) / df['Close'] - 1
        df['signal'] = (df['future_return_5d'] > 0.015).astype(int)

        return df.dropna()

    def get_feature_columns(self):
        """Define which columns are features"""
        return [
            'returns_1d', 'returns_5d', 'returns_10d',
            'volatility_5d', 'volatility_20d',
            'ma_ratio', 'close_vs_120ma', 'close_vs_180ma',
            'range_position', 'dist_from_52w_high', 'dist_from_52w_low',
            'vwap_deviation', 'close_vs_vwap',
            'volume_ratio_5', 'volume_ratio_20',
            'avg_trade_size', 'turnover_ratio',
            'body_size', 'upper_shadow', 'lower_shadow',
            'Range %', 'Trans'
        ]

    def train(self, df, validate=True):
        """Train or retrain model = recursive learning step"""
        df = self.engineer_features(df)
        self.feature_columns = self.get_feature_columns()

        # Ensure all features exist
        available_features = [f for f in self.feature_columns if f in df.columns]

        X = df[available_features]
        y = df['signal']

        # Time-series aware split (CRITICAL for stocks)
        tscv = TimeSeriesSplit(n_splits=5)

        validation_results = {}
        if validate and len(df) > 100:
            scores = []
            precisions = []
            recalls = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=6,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)

                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
                precisions.append(precision_score(y_test, pred, zero_division=0))
                recalls.append(recall_score(y_test, pred, zero_division=0))

            validation_results = {
                'accuracy': float(np.mean(scores)),
                'precision': float(np.mean(precisions)),
                'recall': float(np.mean(recalls)),
                'fold_accuracies': [float(s) for s in scores]
            }

        # Final fit on ALL data (for production signals)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=15,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_columns = available_features

        # Save model for recursive learning
        joblib.dump(self.model, self.model_path)

        # Log training history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'samples': int(len(df)),
            'symbols': int(df['Symbol'].nunique()),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'validation': validation_results,
            'features_used': available_features
        }
        self.training_history.append(history_entry)

        with open(self.history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        return validation_results, available_features

    def predict_signals(self, latest_df):
        """Generate trade signals for latest data"""
        if not self.is_fitted:
            raise ValueError("❌ Model not trained yet! Upload data and train first.")

        df = self.engineer_features(latest_df)

        # Use latest row per symbol
        latest = df.groupby('Symbol').last().reset_index()

        # Only use features the model was trained on
        available_features = [f for f in self.feature_columns if f in latest.columns]
        X = latest[available_features]

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        latest['prediction'] = predictions
        latest['confidence'] = probabilities
        latest['recommendation'] = latest['confidence'].apply(
            lambda c: '🔥 STRONG_BUY' if c > 0.75 
            else '🟢 BUY' if c > 0.6 
            else '🟡 WATCH' if c > 0.5 
            else '🟠 WEAK' if c > 0.4 
            else '🔴 AVOID'
        )

        # Add some context
        latest['current_price'] = latest['Close']
        latest['vwap'] = latest['VWAP']
        latest['day_range'] = latest['Range %']

        return latest[['Symbol', 'Date', 'current_price', 'vwap', 'confidence', 
                       'recommendation', 'prediction', 'day_range', 'Vol']].sort_values('confidence', ascending=False)

    def get_feature_importance(self):
        """Show what the AI learned"""
        if not self.is_fitted:
            return None

        importances = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importances

    def get_training_history(self):
        """Show recursive learning progress"""
        return self.training_history
