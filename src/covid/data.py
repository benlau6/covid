from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
HK_COVID_INDIVIDUAL_FILENAME = "enhanced_sur_covid_19_eng.csv"
HK_COVID_SUMMARY_FILENAME = "latest_situation_of_reported_cases_covid_19_eng.csv"


def read_individual_csv():
    """
    Index(['Case no.', 'Report date', 'Date of onset', 'Gender', 'Age',
       'Name of hospital admitted', 'Hospitalised/Discharged/Deceased',
       'HK/Non-HK resident', 'Classification*', 'Case status*'],
      dtype='object')
    """
    path = DATA_DIR / HK_COVID_INDIVIDUAL_FILENAME
    df = pd.read_csv(path)
    return df


def read_summary_csv():
    """
    Index(['As of date', 'As of time', 'Number of confirmed cases',
       'Number of ruled out cases',
       'Number of cases still hospitalised for investigation',
       'Number of cases fulfilling the reporting criteria',
       'Number of death cases', 'Number of discharge cases',
       'Number of probable cases',
       'Number of hospitalised cases in critical condition',
       'Number of cases tested positive for SARS-CoV-2 virus by nucleic acid tests',
       'Number of cases tested positive for SARS-CoV-2 virus by rapid antigen tests',
       'Number of positive nucleic acid test laboratory detections',
       'Number of death cases related to COVID-19'],
      dtype='object')
    """
    path = DATA_DIR / HK_COVID_SUMMARY_FILENAME
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["As of date"], format="%d/%m/%Y")
    df.index.name = "date"

    # It was checked that the government changed the way they report the number of confirmed cases
    df["Number of confirmed cases"] = np.where(
        df["Number of confirmed cases"].isnull(),
        df[
            "Number of cases tested positive for SARS-CoV-2 virus by nucleic acid tests"
        ],
        df["Number of confirmed cases"],
    )

    # It was checked by isnull().sum() that the following columns are mostly null
    remove_cols = [
        "As of time",
        "Number of hospitalised cases in critical condition",
        "Number of discharge cases",
        "Number of cases still hospitalised for investigation",
        "Number of cases fulfilling the reporting criteria",
        "Number of probable cases",
        "Number of cases tested positive for SARS-CoV-2 virus by nucleic acid tests",
        "Number of cases tested positive for SARS-CoV-2 virus by rapid antigen tests",
        "Number of positive nucleic acid test laboratory detections",
        "Number of ruled out cases",
        "Number of death cases related to COVID-19",
    ]
    df = df.drop(columns=remove_cols)
    df = df.dropna()
    column_rename = {
        "As of date": "date",
        "Number of death cases": "deaths",
        "Number of confirmed cases": "confirmed",
        "Number of hospitalised cases in critical condition": "critical",
    }
    df = df.rename(columns=column_rename)
    df = df.sort_index()

    confirmed = df["confirmed"].diff().values
    # there will be adjustedment, which lead to possible decreasing cumulative confirmed cases
    # so we need to adjust it to be non-negative
    confirmed = np.where(confirmed < 0, 0, confirmed)
    confirmed[0] = confirmed[1]
    df["confirmed"] = confirmed

    return df


def assign_days_since(df, col: str = "confirmed", days: int = 100):
    date = get_start_idx(df, col, threshold=days)
    days_since = (df.index - date).days
    df[f"days_since_{days}"] = days_since
    return df


def get_start_idx(df, col: str, threshold: int):
    return df[col].gt(threshold).idxmax()


def get_data_model(num_days: int = 100, since_val: int = 100):
    df = read_summary_csv()
    days_since_col = f"days_since_{since_val}"
    df = assign_days_since(df, days=since_val)
    df = df.loc[df[days_since_col] >= 0]
    df = df.head(num_days)
    confirmed = df["confirmed"].values
    days = df[days_since_col].values
    dates = df.index.values
    return confirmed, days, dates
