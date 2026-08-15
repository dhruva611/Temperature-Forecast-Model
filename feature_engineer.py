import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # --- DateTime extraction ---
        if 'date_time' in X.columns:
            dt = pd.to_datetime(X['date_time'])
        elif isinstance(X.index, pd.DatetimeIndex):
            dt = X.index
        else:
            raise ValueError("No 'date_time' column and index is not a DatetimeIndex")

        X['month'] = dt.month
        X['day'] = dt.day
        X['hour'] = dt.hour

        # --- Cyclic encoding ---
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        X['day_sin'] = np.sin(2 * np.pi * X['day'] / 31)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / 31)
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)

        # --- Season ---
        def get_season(month):
            if month in [3, 4, 5]:
                return 'summer'
            elif month in [6, 7, 8, 9]:
                return 'monsoon'
            elif month in [10, 11]:
                return 'post-monsoon'
            else:
                return 'winter'
        X['season'] = X['month'].apply(get_season)

        # --- Precipitation ---
        X['precip_flag'] = (X['precipMM'] > 0).astype(int)
        X['precip_amount'] = X['precipMM']

        # --- Sunrise/Sunset (robust) ---
        X['day_length_hours'] = np.nan
        X['day_progress'] = np.nan

        if 'sunrise' in X.columns and 'sunset' in X.columns:
            try:
                # Ensure we have clean string representations
                sunrise_str = X['sunrise'].fillna('').astype(str).str.strip()
                sunset_str = X['sunset'].fillna('').astype(str).str.strip()

                # Parse time strings in 12-hour format with AM/PM
                sunrise_time = pd.to_datetime(sunrise_str, format='%I:%M %p', errors='coerce')
                sunset_time = pd.to_datetime(sunset_str, format='%I:%M %p', errors='coerce')

                # Extract just the time component
                sunrise_times = sunrise_time.dt.time
                sunset_times = sunset_time.dt.time

                # Create full datetime by combining date from dt with parsed times
                sunrise_dt = pd.Series([
                    pd.Timestamp.combine(d.date(), t) if pd.notnull(d) and t is not None else pd.NaT
                    for d, t in zip(dt, sunrise_times)
                ], index=X.index)

                sunset_dt = pd.Series([
                    pd.Timestamp.combine(d.date(), t) if pd.notnull(d) and t is not None else pd.NaT
                    for d, t in zip(dt, sunset_times)
                ], index=X.index)

                # Calculate day length in hours
                day_length_seconds = (sunset_dt - sunrise_dt).dt.total_seconds()
                X['day_length_hours'] = day_length_seconds / 3600

                # Calculate progress through the day (0 at sunrise, 1 at sunset)
                time_since_sunrise_seconds = (dt - sunrise_dt).dt.total_seconds()
                X['day_progress'] = (time_since_sunrise_seconds / 3600) / X['day_length_hours']
                X['day_progress'] = X['day_progress'].clip(0, 1)

            except Exception as e:
                print(f"Warning: sunrise/sunset parsing failed → {e}")
                # Keep NaN values for these features

        # Drop raw columns if not needed
        X.drop(columns=['sunrise', 'sunset'], inplace=True, errors='ignore')

        return X
