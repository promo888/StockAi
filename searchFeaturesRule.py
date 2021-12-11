#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import numpy as np
import pandas as pd
import random
import preproc
from parameters import *

# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
#tf.random.set_seed(314)
random.seed(314)


def isUp(df_data, field):
    df_data['diff_' + '%s' % field] = df_data[field].diff(1) > 0
    df_data['isUp_' + '%s' % field] = [1 if x > 0 else 0 for x in df_data['diff_' + '%s' % field]]


def load_data(ticker, n_steps=3, scale=True, shuffle=True, lookup_step=1,
              test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
              test_days_ago=0, validation_days=0):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    if test_days_ago > 0:
        df = df[-test_days_ago:]


    # todo check indicators znachimost relevance/divergence on each train cycle
    # my add indicators
    ##isUp(df, 'adjclose')
    preproc.isUp(df, "adjclose")
    preproc.pctChange1p(df, "adjclose")
    preproc.pctChange1p(df, "high")
    preproc.pctChange1p(df, "low")
    preproc.sma(df, "adjclose", 2)
    preproc.sma(df, "high", 2)
    preproc.sma(df, "low", 2)
    preproc.sma(df, "high", 3)
    preproc.sma(df, "low", 3)
    preproc.ema(df, "adjclose", 2)
    preproc.volatility(df, "adjclose", 10)

     #O'neil test
    preproc.sma(df, "adjclose", 5)
    preproc.sma(df, "adjclose", 10)
    preproc.sma(df, "adjclose", 20)
    preproc.sma(df, "adjclose", 50)
    preproc.sma(df, "adjclose", 100)
    preproc.sma(df, "adjclose", 200)

    preproc.sma(df, "adjclose", 10)
    preproc.sma(df, "high", 10)
    preproc.sma(df, "low", 10)
    preproc.sma(df, "volume", 10)

    #

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()

    # isUp(df, 'adjclose')
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df[PREDICT_FIELD].shift(-lookup_step)
    ##df['future'] = df['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    #last_sequence = np.array(df[feature_columns].tail(lookup_step))

    print("Before drop NaNs len: ", len(df))
    # drop NaNs
    df.dropna(inplace=True)
    print("After drop NaNs len: ", len(df))

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])


    # split the dataset
    # #result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
    #                                                                                             test_size=test_size,
    #                                                                                             shuffle=shuffle)
    # return the result
    return result


res = load_data("riot")
print("debug")
lows5d = np.where(res["df"]["adjclose"] < res["df"]["sma_200_adjclose"])