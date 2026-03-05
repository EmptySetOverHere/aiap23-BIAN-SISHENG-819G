from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import warnings


def clean_data(df):
    """
        Clean the data by handling missing values and correcting erroneous values.
        Columns to be cleaned as discussed in EDA:
        - BounceRate: Replace negative values with NA
        - ExitRate: Replace negative values with NA 
        - CustomerType: Replace "Unknown", "None", empty string, and NA with "Unknown"
        - GeographicRegion: Replace negative values with 0 (assuming negative values are errors)
    """
    try:
        df["BounceRate"] = df["BounceRate"].clip(lower=0)
    except KeyError:
        warnings.warn("BounceRate column not found in the dataframe. Skip BounceRate clean-up.")
        
    try:
        df["ExitRate"] = df["ExitRate"].clip(lower=0)
    except KeyError:
        warnings.warn("ExitRate column not found in the dataframe. Skip ExitRate clean-up.")

    try:
        df["CustomerType"] = df["CustomerType"].replace(["Unknown", "None", "", pd.NA], "Unknown")
    except KeyError:
        warnings.warn("CustomerType column not found in the dataframe. Skip CustomerType clean-up.")

    try:
        df["GeographicRegion"] = df["GeographicRegion"].clip(lower=0)
    except KeyError:
        warnings.warn("GeographicRegion column not found in the dataframe. Skip GeographicRegion clean-up.")
    
    df = df.dropna()
    return df


def transform_columns(df, numeric_cols, cat_cols, gt_col = None):
    
    scaler = StandardScaler()
    labeller = LabelEncoder()

    col_trans = ColumnTransformer(
        transformers=[
            ('num', scaler, numeric_cols),
            ('cat', labeller, cat_cols)
        ]
    )
    
    X_trans = col_trans.fit_transform(df)
    columns = numeric_cols + cat_cols + ([gt_col] if gt_col else [])
    df = pd.DataFrame(X_trans, columns=columns)
    return df 


def engineer_features(df):
    try:
        df["BounceExitRatio"] = df["BounceRate"] / (df["ExitRate"] + 1e-6)
    except KeyError:
        warnings.warn("BounceRate or ExitRate column not found in the dataframe. Skip adding BounceExitRatio feature.")
    
    try:
        df["IsSpecialDay"] = (df["SpecialDayProximity"] > 0).astype(int)
    except KeyError:
        warnings.warn("SpecialDayProximity column not found in the dataframe. Skip adding IsSpecialDay feature.")
    
    try:
        df["ExitRate^2"] = df["ExitRate"].pow(2)
    except KeyError:
        warnings.warn("ExitRate column not found in the dataframe. Skip adding ExitRate^2 feature.")
        
    try:
        df["TrafficSource^2"] = df["TrafficSource"].pow(2)
    except KeyError:
        warnings.warn("TrafficSource column not found in the dataframe. Skip adding TrafficSource^2 feature.")
        
    try:
        df["PageValue^3"] = df["PageValue"].pow(3)
    except KeyError:
        warnings.warn("PageValue column not found in the dataframe. Skip adding LogPageValue feature.")

    try:
        df["ProductPageTime^3"] = df["ProductPageTime"].pow(3)
    except KeyError:
        warnings.warn("ProductPageTime column not found in the dataframe. Skip adding LogProductTime feature.")

    try:
        df["BounceRate^3"] = df["BounceRate"].pow(3)
    except KeyError:
        warnings.warn("BounceRate column not found in the dataframe. Skip adding BounceRate^3 feature.")
    
    try:
        df["EngagementScore"] = df["PageValue^3"] * df["ProductPageTime^3"]
    except KeyError:
        warnings.warn("PageValue or ProductPageTime column not found in the dataframe. Skip adding EngagementScore feature.")
    
    return df
    
def run_preprocessing(df, numeric_cols, cat_cols):
    df = clean_data(df)
    X_processed = transform_columns(df, numeric_cols, cat_cols)
    df = engineer_features(df)
    return X_processed