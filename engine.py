import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os
import json
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RecursiveSignalEngine:
    """
    Iterative learning engine for stock signals.
    FLEXIBLE PARSER: Adapts to your file format automatically.
    """
    def __init__(self, model_path="model.pkl", history_path="history.json"):
        self.model_path = model_path
        self.history_path = history_path
        self.model = None
        self.is_fitted = False
        self.training_history = []
        self.feature_columns = None
        self.parse_log = []

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.is_fitted = True
            self.parse_log.append("Loaded existing model from previous session")

        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)

    COLUMN_ALIASES = {
        'Symbol': ['symbol', 'ticker', 'sym', 'stock', 'scrip', 'name', 'instrument', 'trading symbol'],
        'Open': ['open', 'open price', 'opening', 'opening price', 'open_p', 'op'],
        'High': ['high', 'high price', 'day high', 'high_p', 'hi'],
        'Low': ['low', 'low price', 'day low', 'low_p', 'lo'],
        'Close': ['close', 'close price', 'closing', 'closing price', 'close_p', 'cl', 'last'],
        'LTP': ['ltp', 'last traded price', 'last price', 'last traded', 'ltp price'],
        'VWAP': ['vwap', 'volume weighted average price', 'vwap price', 'avg price'],
        'Vol': ['vol', 'volume', 'volume traded', 'qty', 'quantity', 'shares', 'vol.', 'volume.'],
        'Turnover': ['turnover', 'turnover value', 'total turnover', 'turnover_rs', 'value', 'turnover (rs)'],
        'Trans': ['trans', 'trans.', 'transactions', 'trades', 'no of trades', 'trade count', 'txn', 'transaction count', 'no. of trades'],
        'Range %': ['range %', 'range%', 'range', 'intraday range', 'day range %', 'range (%)', 'range_percentage'],
        'VWAP %': ['vwap %', 'vwap%', 'vwap deviation', 'close vs vwap %', 'vwap deviation %', 'vwap_perc'],
        '120 Days': ['120 days', '120d', '120 day ma', '120_ma', 'ma120', 'sma120', '120 days ma', '120-day ma'],
        '180 Days': ['180 days', '180d', '180 day ma', '180_ma', 'ma180', 'sma180', '180 days ma', '180-day ma'],
        '52 Weeks High': ['52 weeks high', '52 week high', '52w high', '52_week_high', 'year high', '52wh', 'high_52w'],
        '52 Weeks Low': ['52 weeks low', '52 week low', '52w low', '52_week_low', 'year low', '52wl', 'low_52w'],
        'Date': ['date', 'trading date', 'trade date', 'dt', 'timestamp', 'date_time'],
        'Prev. Close': ['prev. close', 'prev close', 'previous close', 'prev_close', 'previous_close', 'prevclose'],
        'Diff': ['diff', 'difference', 'change', 'diff.', 'difference.'],
        'Range': ['range', 'day range', 'price range', 'daily range', 'intraday range']
    }

    def _normalize_column_name(self, col):
        return str(col).strip().lower().replace('_', ' ').replace('-', ' ')

    def _find_column(self, df, target):
        df_cols_lower = {self._normalize_column_name(c): c for c in df.columns}

        if target in df.columns:
            return target

        aliases = self.COLUMN_ALIASES.get(target, [target.lower()])
        for alias in aliases:
            alias_norm = self._normalize_column_name(alias)
            if alias_norm in df_cols_lower:
                return df_cols_lower[alias_norm]

        target_norm = self._normalize_column_name(target)
        for norm_col, orig_col in df_cols_lower.items():
            if target_norm in norm_col or norm_col in target_norm:
                if len(target_norm) >= 3 and len(norm_col) >= 3:
                    return orig_col

        return None

    def _map_columns(self, df, sheet_name=""):
        mapped = {}
        missing = []
        for standard_name in list(self.COLUMN_ALIASES.keys()) + ['Date']:
            found = self._find_column(df, standard_name)
            if found:
                mapped[standard_name] = found
            else:
                missing.append(standard_name)
        return mapped, missing

    DATE_PATTERNS = [
        (r'^(\d{4})_(\d{2})_(\d{2})$', '%Y_%m_%d'),
        (r'^(\d{4})-(\d{2})-(\d{2})$', '%Y-%m-%d'),
        (r'^(\d{4})(\d{2})(\d{2})$', '%Y%m%d'),
        (r'^(\d{2})_(\d{2})_(\d{4})$', '%d_%m_%Y'),
        (r'^(\d{2})-(\d{2})-(\d{4})$', '%d-%m-%Y'),
        (r'^(\d{1,2})[\-/](\d{1,2})[\-/](\d{4})$', 'auto'),
        (r'^(\d{4})[\-/](\d{1,2})[\-/](\d{1,2})$', 'auto'),
    ]

    def _parse_sheet_name(self, sheet_name):
        s = str(sheet_name).strip()

        for pattern, fmt in self.DATE_PATTERNS:
            match = re.match(pattern, s)
            if match:
                if fmt == 'auto':
                    for try_fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y']:
                        try:
                            return datetime.strptime(s, try_fmt)
                        except:
                            continue
                else:
                    try:
                        return datetime.strptime(s, fmt)
                    except:
                        continue

        try:
            excel_epoch = datetime(1899, 12, 30)
            return excel_epoch + timedelta(days=int(s))
        except:
            pass

        return None

    def parse_excel(self, file_path_or_buffer):
        self.parse_log = []
        xl = pd.ExcelFile(file_path_or_buffer)
        all_data = []

        self.parse_log.append(f"Found {len(xl.sheet_names)} sheet(s): {str(xl.sheet_names[:5])}{'...' if len(xl.sheet_names) > 5 else ''}")

        # STRATEGY 1: Multi-sheet with date-named sheets
        date_sheets = []
        for sheet_name in xl.sheet_names:
            parsed_date = self._parse_sheet_name(sheet_name)
            if parsed_date:
                date_sheets.append((sheet_name, parsed_date))

        if date_sheets:
            self.parse_log.append(f"Detected multi-sheet format with {len(date_sheets)} date-named sheets")

            for sheet_name, date in date_sheets:
                try:
                    df = xl.parse(sheet_name)
                    mapped, missing = self._map_columns(df, sheet_name)

                    if not mapped:
                        self.parse_log.append(f"Sheet '{sheet_name}': No recognizable columns, skipping")
                        continue

                    df = df.rename(columns={v: k for k, v in mapped.items() if k != 'Date'})
                    df['Date'] = date

                    keep_cols = [k for k in mapped.keys() if k in df.columns] + ['Date']
                    df = df[[c for c in keep_cols if c in df.columns]]

                    all_data.append(df)

                except Exception as e:
                    self.parse_log.append(f"Sheet '{sheet_name}': Error - {str(e)[:50]}")

        # STRATEGY 2: Single sheet with Date column
        if not all_data:
            self.parse_log.append("Trying single-sheet format (looking for Date column)...")

            for sheet_name in xl.sheet_names:
                try:
                    df = xl.parse(sheet_name)
                    mapped, missing = self._map_columns(df)

                    if 'Date' in mapped and mapped['Date'] in df.columns:
                        df = df.rename(columns={v: k for k, v in mapped.items()})
                        date_col = mapped['Date']
                        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.dropna(subset=['Date'])

                        if len(df) > 0:
                            all_data.append(df)
                            self.parse_log.append(f"Sheet '{sheet_name}': Found {len(df)} rows with Date column")
                            break
                except Exception as e:
                    self.parse_log.append(f"Sheet '{sheet_name}': {str(e)[:50]}")

        # STRATEGY 3: Single sheet, no Date column
        if not all_data:
            self.parse_log.append("Trying single-sheet without Date column...")

            for sheet_name in xl.sheet_names:
                try:
                    df = xl.parse(sheet_name)
                    mapped, missing = self._map_columns(df)

                    if mapped:
                        df = df.rename(columns={v: k for k, v in mapped.items()})
                        df['Date'] = datetime.now()
                        all_data.append(df)
                        self.parse_log.append(f"Sheet '{sheet_name}': Using current date (no date found in data)")
                        break
                except:
                    pass

        if not all_data:
            raise ValueError(
                "Could not parse any sheets. "
                "Tried: date-named sheets, Date column, single-sheet. "
                "Please check your column names match: Symbol, Open, High, Low, Close, LTP, VWAP, Vol, Turnover, Trans, Range %, VWAP %, 120 Days, 180 Days, 52 Weeks High, 52 Weeks Low"
            )

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        # Force numeric types on critical columns (handles text/empty cells in 800+ sheets)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'LTP', 'VWAP', 'Vol', 
                        'Turnover', 'Trans', 'Range %', 'VWAP %', 
                        '120 Days', '180 Days', '52 Weeks High', '52 Weeks Low']
        for col in numeric_cols:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')
                # Replace inf values that break plotly
                if combined[col].dtype in ['float64', 'int64', 'float32']:
                    combined[col] = combined[col].replace([np.inf, -np.inf], np.nan)

        # Drop rows where critical columns are NaN
        critical_for_display = ['Symbol', 'Date', 'Close']
        before_drop = len(combined)
        combined = combined.dropna(subset=[c for c in critical_for_display if c in combined.columns])
        dropped = before_drop - len(combined)
        if dropped > 0:
            self.parse_log.append(f"Dropped {dropped} rows with missing critical data")

        self.parse_log.append(f"Total: {len(combined):,} rows | {combined['Symbol'].nunique()} symbols | {combined['Date'].nunique()} dates")

        critical = ['Symbol', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Vol']
        missing_critical = [c for c in critical if c not in combined.columns]
        if missing_critical:
            self.parse_log.append(f"Missing critical columns: {missing_critical}. Some features may not work.")

        return combined

    def get_parse_log(self):
        return self.parse_log

    def engineer_features(self, df):
        df = df.copy()
        df = df.sort_values(['Symbol', 'Date'])
        grouped = df.groupby('Symbol')

        df['returns_1d'] = grouped['Close'].pct_change()
        df['returns_5d'] = grouped['Close'].pct_change(5)
        df['returns_10d'] = grouped['Close'].pct_change(10)

        df['volatility_5d'] = grouped['returns_1d'].transform(lambda x: x.rolling(5).std())
        df['volatility_20d'] = grouped['returns_1d'].transform(lambda x: x.rolling(20).std())

        if '120 Days' in df.columns and '180 Days' in df.columns:
            df['ma_ratio'] = df['120 Days'] / df['180 Days']
            df['close_vs_120ma'] = (df['Close'] - df['120 Days']) / df['120 Days']
            df['close_vs_180ma'] = (df['Close'] - df['180 Days']) / df['180 Days']
        else:
            df['ma_ratio'] = np.nan
            df['close_vs_120ma'] = np.nan
            df['close_vs_180ma'] = np.nan

        if '52 Weeks High' in df.columns and '52 Weeks Low' in df.columns:
            df['range_position'] = (df['Close'] - df['52 Weeks Low']) / (df['52 Weeks High'] - df['52 Weeks Low'] + 1e-9)
            df['dist_from_52w_high'] = (df['Close'] - df['52 Weeks High']) / df['52 Weeks High']
            df['dist_from_52w_low'] = (df['Close'] - df['52 Weeks Low']) / df['52 Weeks Low']
        else:
            df['range_position'] = np.nan
            df['dist_from_52w_high'] = np.nan
            df['dist_from_52w_low'] = np.nan

        if 'VWAP %' in df.columns:
            df['vwap_deviation'] = df['VWAP %']
        else:
            df['vwap_deviation'] = np.nan

        if 'VWAP' in df.columns:
            df['close_vs_vwap'] = (df['Close'] - df['VWAP']) / df['VWAP']
        else:
            df['close_vs_vwap'] = np.nan

        df['volume_ma_5'] = grouped['Vol'].transform(lambda x: x.rolling(5).mean())
        df['volume_ma_20'] = grouped['Vol'].transform(lambda x: x.rolling(20).mean())
        df['volume_ratio_5'] = df['Vol'] / (df['volume_ma_5'] + 1e-9)
        df['volume_ratio_20'] = df['Vol'] / (df['volume_ma_20'] + 1e-9)

        if 'Turnover' in df.columns and 'Trans' in df.columns:
            df['avg_trade_size'] = df['Turnover'] / (df['Trans'] + 1e-9)
            df['turnover_ratio'] = df['Turnover'] / (df['volume_ma_20'] * df['VWAP'] + 1e-9)
        else:
            df['avg_trade_size'] = np.nan
            df['turnover_ratio'] = np.nan

        df['body_size'] = abs(df['Close'] - df['Open']) / df['Open']
        df['upper_shadow'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / df['Open']
        df['lower_shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / df['Open']

        df['future_return_5d'] = grouped['Close'].shift(-5) / df['Close'] - 1
        df['signal'] = (df['future_return_5d'] > 0.015).astype(int)

        return df.dropna()

    def get_feature_columns(self):
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
        df = self.engineer_features(df)
        self.feature_columns = self.get_feature_columns()
        available_features = [f for f in self.feature_columns if f in df.columns]
        available_features = [f for f in available_features if df[f].notna().sum() > 10]

        if len(available_features) < 5:
            raise ValueError(f"Not enough valid features found. Only have: {available_features}. Check your column names.")

        X = df[available_features]
        y = df['signal']

        validation_results = {}
        if validate and len(df) > 100:
            tscv = TimeSeriesSplit(n_splits=5)
            scores, precisions, recalls = [], [], []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = RandomForestClassifier(
                    n_estimators=100, max_depth=6, min_samples_split=20,
                    min_samples_leaf=10, random_state=42, n_jobs=-1
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

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=15,
            min_samples_leaf=8, random_state=42, n_jobs=-1
        )
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_columns = available_features

        joblib.dump(self.model, self.model_path)

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
        if not self.is_fitted:
            raise ValueError("Model not trained yet! Upload data and train first.")

        df = self.engineer_features(latest_df)
        latest = df.groupby('Symbol').last().reset_index()

        available_features = [f for f in self.feature_columns if f in latest.columns]
        X = latest[available_features]

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        latest['prediction'] = predictions
        latest['confidence'] = probabilities
        latest['recommendation'] = latest['confidence'].apply(
            lambda c: 'STRONG_BUY' if c > 0.75 
            else 'BUY' if c > 0.6 
            else 'WATCH' if c > 0.5 
            else 'WEAK' if c > 0.4 
            else 'AVOID'
        )

        latest['current_price'] = latest['Close']
        latest['vwap'] = latest['VWAP'] if 'VWAP' in latest.columns else np.nan
        latest['day_range'] = latest['Range %'] if 'Range %' in latest.columns else np.nan

        return latest[['Symbol', 'Date', 'current_price', 'vwap', 'confidence', 
                       'recommendation', 'prediction', 'day_range', 'Vol']].sort_values('confidence', ascending=False)

    def get_feature_importance(self):
        if not self.is_fitted:
            return None
        importances = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importances

    def get_training_history(self):
        return self.training_history
