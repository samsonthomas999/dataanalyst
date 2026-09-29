import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
import os

load_dotenv()

def validate_file(file):
    if file.name.endswith(".csv"):
        return True
    else:
        return False


def clean_data(file):
    df = pd.read_csv(file)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna("Unknown", inplace=True)
    return df

def explore_data(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "stats": df.describe().to_dict(),
        "preview": df.head()
    }
