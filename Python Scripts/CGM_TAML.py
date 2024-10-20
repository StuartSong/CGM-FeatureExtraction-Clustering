## Import cgm library
import cgmquantify_stuart as cgm
import pandas as pd
import numpy as np
import random

def quantify_glycemic_features(df):
    df = df.reset_index(drop=True)
    features = {}
    
    # cgm.summary(df) returns 6 elements: 'mean', 'median', 'min', 'max', 'fq', 'tq'
    summary_stats = cgm.summary(df)
    feature_names = ['mean', 'median', 'min', 'max', 'fq', 'tq']
    features.update(dict(zip(feature_names, summary_stats)))
    
    # Add other features
    features['interdaysd'] = cgm.interdaysd(df)
    features['interdaycv'] = cgm.interdaycv(df)
    features['TOR'] = cgm.TOR(df)
    features['TIR'] = cgm.TIR(df)
    features['MGE'] = cgm.MGE(df)
    features['MGN'] = cgm.MGN(df)
    features['J_index'] = cgm.J_index(df)
    features['LBGI'] = cgm.LBGI(df)
    features['HBGI'] = cgm.HBGI(df)
    features['ADRR'] = cgm.DRR(df)
    features['TA140'] = cgm.TAT(df, thres=140)
    features['TA200'] = cgm.TAT(df, thres=200)
    features['TIR_70_180'] = cgm.TIR_lo_hi(df)
    features['TA180'] = cgm.TAT(df, thres=180)
    features['TA250'] = cgm.TAT(df, thres=250)
    features['TB70'] = cgm.TBT(df, thres=70)
    features['TB54'] = cgm.TBT(df, thres=54)
    features['TITR'] = cgm.TIR_lo_hi(df, up=140, dw=70)
    features['GRI'] = cgm.GRI(df)
    features['PA140'] = cgm.count_peaks(df, 140)
    features['PA180'] = cgm.count_peaks(df, 180)
    features['PA200'] = cgm.count_peaks(df, 200)
    
    return features

def feature_extraction_fixed_hour_window_0oclock(df, id, hour=24):
    df = df.sort_values(by='Time').reset_index(drop=True)
    window_size = pd.Timedelta(hours=hour)
    
    # Convert time to seconds
    converted_time = [time.hour * 3600 + time.minute * 60 + time.second for time in df.Time.dt.time]
    
    # Start between 23:57:30 to 00:02:30
    converted_time_index = [
        index for index, time in enumerate(converted_time)
        if time >= 23 * 3600 + 57 * 60 + 30 or time < 0 * 3600 + 2 * 60 + 30
    ]
    
    window_number = 0
    # Number of data points for 70% coverage with 5-minute intervals
    window_data_threshold = pd.Timedelta(hours=hour) / pd.Timedelta(minutes=5) * 0.70
    
    feature_list = []
    
    for start_time_index in converted_time_index:
        window_number += 1
        start_time = df.Time[start_time_index]
        window_condition = (df.Time >= start_time) & (df.Time < start_time + window_size)
        df_window = df[window_condition]
        
        if len(df_window) > window_data_threshold:
            df_window = df_window.copy()
            df_window.set_index('Time', inplace=True)
            df_window.drop(columns=['ID'], inplace=True, errors='ignore')
            df_window = df_window.resample('5min').median()
            df_window.reset_index(inplace=True)
            
            features = quantify_glycemic_features(df_window.drop(columns=["Time"]))
            features['id'] = f"{id}_win{window_number}"
            feature_list.append(features)
    
    # Convert list of feature dictionaries to DataFrame
    feature_table = pd.DataFrame(feature_list)
    
    # Reorder columns to match desired feature names
    feature_names = [
        "id", "mean", "median", "min", "max", "fq", "tq",
        "interdaysd", "interdaycv", "TOR", "TIR", "MGE", "MGN", "J_index",
        "LBGI", "HBGI", "ADRR", "TA140", "TA200", "TIR_70_180", "TA180",
        "TA250", "TB70", "TB54", "TITR", "GRI", "PA140", "PA180", "PA200"
    ]
    feature_table = feature_table.reindex(columns=feature_names)
    
    return feature_table


def fixed_time_sliding_window_0oclock_1random_day(df, id, hour=24):
    df = df.sort_values(by='Time').reset_index(drop=True)
    df['Day'] = df['Time'].dt.date
    
    window_size = pd.Timedelta(hours=hour)

    # Convert time to seconds since midnight
    converted_time = df['Time'].dt.hour * 3600 + df['Time'].dt.minute * 60 + df['Time'].dt.second

    # Find indices where time is between 23:57:30 and 00:02:30
    condition = (converted_time >= 23 * 3600 + 57 * 60 + 30) | (converted_time < 2 * 60 + 30)
    converted_time_index = df.index[condition].tolist()

    # Number of data points for 95% coverage with 5-minute intervals
    window_data_threshold = (pd.Timedelta(hours=hour) / pd.Timedelta(minutes=5)) * 0.95

    # Shuffle the indices to pick a random day
    random.shuffle(converted_time_index)

    feature_list = []
    for start_time_index in converted_time_index:
        start_time = df.loc[start_time_index, 'Time']
        end_time = start_time + window_size
        window_condition = (df['Time'] >= start_time) & (df['Time'] < end_time)
        df_window = df[window_condition].copy()

        if len(df_window) > window_data_threshold:
            # Interpolate the 'Glucose' column
            df_window['Glucose'] = df_window['Glucose'].interpolate(method='spline', order=2)

            # Quantify glycemic features
            features = quantify_glycemic_features(df_window)

            # Add 'id' to features
            features['id'] = id
            feature_list.append(features)
            break  # Only process one window

    if feature_list:
        feature_table = pd.DataFrame(feature_list)
    else:
        # If no window meets the threshold, return an empty DataFrame with the feature columns
        feature_table = pd.DataFrame(columns=[
            "id", "mean", "median", "min", "max", "fq", "tq",
            "interdaysd", "interdaycv", "TOR", "TIR", "MGE", "MGN",
            "J_index", "LBGI", "HBGI", "ADRR", "TA140", "TA200"
        ])
        feature_table['id'] = id

    # Reorder columns to match desired feature names
    feature_names = [
        "id", "mean", "median", "min", "max", "fq", "tq",
        "interdaysd", "interdaycv", "TOR", "TIR", "MGE", "MGN",
        "J_index", "LBGI", "HBGI", "ADRR", "TA140", "TA200"
    ]
    feature_table = feature_table.reindex(columns=feature_names)

    return feature_table
