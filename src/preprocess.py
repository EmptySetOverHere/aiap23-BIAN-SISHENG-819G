from collections import OrderedDict
import pandas as pd
import warnings

class Preprocessor:
    def __init__(self):
        self.cleaning_methods = {
            "BounceRate": lambda df: df["BounceRate"].clip(lower=0),
            "ExitRate": lambda df: df["ExitRate"].clip(lower=0),
            "ProductPageTime": lambda df: df["ProductPageTime"].clip(lower=0),
            "CustomerType": lambda df: (
                df["CustomerType"]
                .replace(["Unknown", "None", "", "nan", pd.NA], "Unknown")
                .replace("returning_Visitor", "Returning_Visitor")
            ),
            "GeographicRegion": lambda df: df["GeographicRegion"].clip(lower=0),
        }
        self.feature_transforms = OrderedDict({
            "BounceExitRatio": lambda df: df["BounceRate"] / (df["ExitRate"] + 1e-6),
            "IsSpecialDay": lambda df: (df["SpecialDayProximity"] > 0).astype(int),
            "ExitRate^2": lambda df: df["ExitRate"].pow(2),
            "TrafficSource^2": lambda df: df["TrafficSource"].pow(2),
            "PageValue^3": lambda df: df["PageValue"].pow(3),
            "ProductPageTime^3": lambda df: df["ProductPageTime"].pow(3),
            "BounceRate^3": lambda df: df["BounceRate"].pow(3),
            "EngagementScore": lambda df: df["PageValue^3"] * df["ProductPageTime^3"],
        })

    def clean_data(self, df):
        for col, func in self.cleaning_methods.items():
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found, skipping cleaning.")
                continue
            df[col] = func(df)
        df = df.dropna()
        return df

    def add_features(self, df):
        for name, func in self.feature_transforms.items():
            try:
                df[name] = func(df)
            except KeyError as e:
                warnings.warn(f"Skipping feature '{name}': missing source column {e}")
        return df

    def run(self, df):
        df = self.clean_data(df)
        df = self.add_features(df)
        return df
        

