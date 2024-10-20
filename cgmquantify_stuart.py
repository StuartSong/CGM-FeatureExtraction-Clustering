import pandas as pd
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

#     Functions:
#     intradaycv(): Computes and returns the intraday coefficient of variation of glucose 
#     intradaysd(): Computes and returns the intraday standard deviation of glucose 
#     TIR(): Computes and returns the time in range
#     TOR(): Computes and returns the time outside range
#     PIR(): Computes and returns the percent time in range
#     POR(): Computes and returns the percent time outside range
#     MGE(): Computes and returns the mean of glucose outside the specified range
#     MGN(): Computes and returns the mean of glucose inside the specified range
#     MAGE(): Computes and returns the mean amplitude of glucose excursions
#     J_index(): Computes and returns the J-index
#     LBGI(): Computes and returns the low blood glucose index
#     HBGI(): Computes and returns the high blood glucose index
#     ADRR(): Computes and returns the average daily risk range, an assessment of total daily glucose variations within risk space
#     MODD(): Computes and returns the mean of daily differences. Examines mean of value + value 24 hours before
#     CONGA24(): Computes and returns the continuous overall net glycemic action over 24 hours
#     GMI(): Computes and returns the glucose management index
#     eA1c(): Computes and returns the American Diabetes Association estimated HbA1c
#     summary(): Computes and returns glucose summary metrics, including interday mean glucose, interday median glucose, interday minimum glucose, interday maximum glucose, interday first quartile glucose, and interday third quartile glucose
#     TAT(): Computes and returns time above a threshold
#     TBT(): Computes and returns time below a threshold
#     GRI(): Computes the glucose risk index, a composite score for hypo- and hyperglycemia
#     count_peaks(): Counts the number of glucose peaks above a given threshold
#     TAT_revised(): Computes and returns a more precise time above threshold by considering episode durations

def interdaycv(df):
    """
        Computes and returns the interday coefficient of variation of glucose
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            cvx (float): interday coefficient of variation averaged over all days
            
    """
    cvx = (np.std(df['Glucose']) / (np.mean(df['Glucose'])))*100
    return cvx

def interdaysd(df):
    """
        Computes and returns the interday standard deviation of glucose
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            interdaysd (float): interday standard deviation averaged over all days
            
    """
    interdaysd = np.std(df['Glucose'])
    return interdaysd
    

def TIR(df, sd=1, sr=5):
    """
        Computes and returns the time in range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            TIR (float): time in range, units=minutes
            
    """
    up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
    time = len(df[(df['Glucose']<= up) & (df['Glucose']>= dw)])*sr 
    return time

def TIR_lo_hi(df, up = 180, dw = 70, sr=5):
    """
        Computes and returns the time in range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            TIR (float): time in range, units=minutes
            
    """
    time = len(df[(df['Glucose']<= up) & (df['Glucose']>= dw)])*sr 
    return time

def TOR(df, sd=1, sr=5):
    """
        Computes and returns the time outside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing  range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            TOR (float): time outside range, units=minutes
            
    """
    up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
    TOR = len(df[(df['Glucose']>= up) | (df['Glucose']<= dw)])*sr
    return TOR

def POR(df, sd=1, sr=5):
    """
        Computes and returns the percent time outside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            POR (float): percent time outside range, units=%
            
    """
    up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
    TOR = len(df[(df['Glucose']>= up) | (df['Glucose']<= dw)])*sr
    POR = (TOR/(len(df)*sr))*100
    return POR

def PIR(df, sd=1, sr=5):
    """
        Computes and returns the percent time inside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            PIR (float): percent time inside range, units=%
            
    """
    up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
    TIR = len(df[(df['Glucose']<= up) & (df['Glucose']>= dw)])*sr
    PIR = (TIR/(len(df)*sr))*100
    return PIR

def MGE(df, sd=1):
    """
        Computes and returns the mean of glucose outside specified range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
        Returns:
            MGE (float): the mean of glucose excursions (outside specified range)
            
    """
    up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
    MGE = np.mean(df[(df['Glucose']>= up) | (df['Glucose']<= dw)])
    return float(MGE)

def MGN(df, sd=1):
    """
        Computes and returns the mean of glucose inside specified range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
        Returns:
            MGN (float): the mean of glucose excursions (inside specified range)
            
    """
    up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
    MGN = np.mean(df[(df['Glucose']<= up) & (df['Glucose']>= dw)])
    return float(MGN)


def J_index(df):
    """
        Computes and returns the J-index
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            J (float): J-index of glucose
            
    """
    J = 0.001*((np.mean(df['Glucose'])+np.std(df['Glucose']))**2)
    return J

def LBGI_HBGI(df):
    """
        Connecter function to calculate rh and rl, used for ADRR function
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            LBGI (float): Low blood glucose index
            HBGI (float): High blood glucose index
            rl (float): See calculation of LBGI
            rh (float): See calculation of HBGI
            
    """
    f = ((np.log(df['Glucose'])**1.084) - 5.381)
    rl = []
    for i in f: 
        if (i <= 0):
            rl.append(22.77*(i**2))
        else:
            rl.append(0)

    LBGI = np.mean(rl)

    rh = []
    for i in f: 
        if (i > 0):
            rh.append(22.77*(i**2))
        else:
            rh.append(0)

    HBGI = np.mean(rh)
    
    return LBGI, HBGI, rh, rl

def LBGI(df):
    """
        Computes and returns the low blood glucose index
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            LBGI (float): Low blood glucose index
            
    """
    f = ((np.log(df['Glucose'])**1.084) - 5.381)
    rl = []
    for i in f: 
        if (i <= 0):
            rl.append(22.77*(i**2))
        else:
            rl.append(0)

    LBGI = np.mean(rl)
    return LBGI

def HBGI(df):
    """
        Computes and returns the high blood glucose index
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            HBGI (float): High blood glucose index
            
    """
    f = ((np.log(df['Glucose'])**1.084) - 5.381)
    rh = []
    for i in f: 
        if (i > 0):
            rh.append(22.77*(i**2))
        else:
            rh.append(0)

    HBGI = np.mean(rh)
    return HBGI

def DRR(df):
    """
        Computes and returns the daily risk range, an assessment of total daily glucose variations within risk space
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            DRR (float): daily risk range
            
    """
    LBGI, HBGI, rh, rl = LBGI_HBGI(df)
    LR = np.max(rl)
    HR = np.max(rh)
    DRR = LR+HR
    return DRR


def CONGA24(df):
    """
        Computes and returns the continuous overall net glycemic action over 24 hours
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Requires:
            uniquevalfilter (function)
        Returns:
            CONGA24 (float): continuous overall net glycemic action over 24 hours
            
    """
    df['Timefrommidnight'] =  df['Time'].dt.time
    lists=[]
    for i in range(0, len(df['Timefrommidnight'])):
        lists.append(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[0:2])*60 + int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[3:5]) + round(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[6:9])/60))
    df['Minfrommid'] = lists
    df = df.drop(columns=['Timefrommidnight'])
    
    #Calculation of MODD and CONGA:
    MODD_n = []
    uniquetimes = df['Minfrommid'].unique()

    for i in uniquetimes:
        MODD_n.append(uniquevalfilter(df, i))
    
    #Remove zeros from dataframe for calculation (in case there are random unique values that result in a mean of 0)
    MODD_n[MODD_n == 0] = np.nan
    
    CONGA24 = np.nanstd(MODD_n)
    return CONGA24

def summary(df): 
    """
        Computes and returns glucose summary metrics
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            meanG (float): interday mean of glucose
            medianG (float): interday median of glucose
            minG (float): interday minimum of glucose
            maxG (float): interday maximum of glucose
            Q1G (float): interday first quartile of glucose
            Q3G (float): interday third quartile of glucose
            
    """
    meanG = np.nanmean(df['Glucose'])
    medianG = np.nanmedian(df['Glucose'])
    minG = np.nanmin(df['Glucose'])
    maxG = np.nanmax(df['Glucose'])
    Q1G = np.nanpercentile(df['Glucose'], 25)
    Q3G = np.nanpercentile(df['Glucose'], 75)
    
    return meanG, medianG, minG, maxG, Q1G, Q3G

def TAT(df, thres, sr=5):
    """
        Computes and returns the time outside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing  range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            TOR (float): time outside range, units=minutes
            
    """
    thres = float(thres)
#     print(len(df[(df['Glucose']>= thres)]))
    TOR = len(df[(df['Glucose']>= thres)])*sr
    return TOR

def TBT(df, thres, sr=5):
    """
        Computes and returns the time outside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing  range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            TOR (float): time outside range, units=minutes
            
    """
    thres = float(thres)
#     print(len(df[(df['Glucose']>= thres)]))
    TOR = len(df[(df['Glucose']<= thres)])*sr
    return TOR

def GRI(df):
    """
        Computes and returns the glucose risk index (GRI)
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time, and Glucose columns
        Returns:
            GRI (float): glucose risk index, a composite score of hypo- and hyperglycemia risk
    """
    TAR1 = TAT(df, thres = 180)
    TAR2 = TAT(df, thres = 250)
    TBR1 = TBT(df, thres = 70)
    TBR2 = TBT(df, thres = 54)
    GRI = (3.0*TBR2)+(2.4*TBR1)+(1.6*TAR2)+(0.8*TAR1)
    return GRI

def count_peaks(df, threshold):
    """
        Counts and returns the number of glucose peaks above a given threshold
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time, and Glucose columns
            threshold (float): the glucose value threshold to define a peak
        Returns:
            peaks (int): the number of peaks (instances where glucose exceeds the threshold)
    """
    peaks = 0
    above_threshold = False
 
    for glucose_level in df['Glucose']:
        if above_threshold:
            if glucose_level < threshold:
                peaks += 1
                above_threshold = False
        else:
            if glucose_level > threshold:
                above_threshold = True
    if above_threshold:
        peaks += 1
 
    return peaks

def TAT_revised(df, thres):
    """
        Computes and returns the total time above a specified glucose threshold, considering episode durations
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time, and Glucose columns
            thres (float): the glucose value threshold
        Returns:
            TAT_thres (float): total time above the threshold (in minutes), taking into account the duration of each episode
    """
    
    # Add above_thres column
    df['above_thres'] = df['Glucose'] > thres

    # Identify episodes of above-140 levels
    df['episode_id'] = (df['above_thres'] != df['above_thres'].shift()).cumsum()

    # Calculate episode durations
    df['duration'] = df.groupby('episode_id')['Time'].diff().fillna(pd.Timedelta(seconds=0))

    # Filter and sum durations where glucose levels were above 140
    total_duration_above_thres = df[df['above_thres']].groupby('episode_id')['duration'].sum()

    TAT_thres = float(total_duration_above_thres.dt.total_seconds().sum()/60)

    return TAT_thres
