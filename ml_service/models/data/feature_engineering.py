import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TFTFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def add_time_idx(self, df):
        df = df.sort_values(["symbol", "timestamp"])
        df["time_idx"] = df.groupby("symbol").cumcount()
        return df

    def encode_static_categoricals(self, df, columns=["symbol"]):
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def scale_features(self, df, columns):
        df[columns] = self.scaler.fit_transform(df[columns])
        return df
