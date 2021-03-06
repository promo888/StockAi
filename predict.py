import timeit

from sklearn.svm import LinearSVC as svm
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.tree import DecisionTreeRegressor as dtr

import matplotlib.pyplot as plt
import pylab
import preproc
from helper import *
#import helper
from tabulate import tabulate
import datetime
import ast
from itertools import combinations
from timeit import default_timer as timer
from datetime import timedelta
import sqlite3
from sqlalchemy import create_engine
from yahoo_fin import stock_info as si
import yfinance as yf
import datetime
import numpy as np
import pandas as pd
import hashlib
from horology import timed

class PeriodResults:
    def __init__(self, df_data, prediction_period=1, prediction_field="close"):
        self.df_data = df_data
        #self.df_data_copy = df_data.copy()
        self.prediction_period = prediction_period
        self.prediction_field = prediction_field
        self.prediction_accuracy_pct = None
        self.total_predictions = None
        self.good_predictions = None
        self.bad_predictions = None
        self.total_profit_pct = None #OHLC? if field is close -> calc Upwards/Downwards from HighLow within a period
        self.total_profit_pct_avg = None #when prediction_period > 1
        self.total_profit_pct_med = None
        self.max_profit_pct = None
        self.min_profit_pct = None
        self.total_loss_pct = None
        self.total_loss_pct_avg = None
        self.total_loss_pct_med = None
        self.max_loss_pct = None
        self.min_loss_pct = None
        self.max_up_profit_pct = None #high for long is prediction for CloseField
        self.max_down_profit_pct = None #max drawdown in period for Buy/Sell Entry/Exit


class DataReader:
    @timed
    def __init__(self, ticker, fromDate="1990-01-01", toDate="2021-12-31", interval="1d", index_as_date=True, fromWeb=True):
        self.ticker = ticker
        self.fromDate = fromDate
        self.toDate = toDate
        self.interval = interval
        self.index_as_date = index_as_date
        self.fromWeb = fromWeb
        self.inputs = None
        if fromWeb and self.inputs is None:
            #self.inputs = si.get_data(ticker, fromDate, toDate, index_as_date, interval)
            self.inputs = yf.download(ticker, start=fromDate, end=toDate) #"SPY AAPL" multiple tickers Todo for correlation features
            self.inputs.columns = [x.lower() for x in self.inputs.columns]
        else: #todo read from file/db
            pass

    def getData(self):
        return self.inputs


class PreprocData:
    def __init__(self):
        self.rulesId = {}

    @staticmethod
    def run(funcsList):
        for func in funcsList:
            for funcName in func:
                eval(funcName)(*func[funcName]) #todo byModule/byClass like in preprocFile


#todo strat and rules logic
class PredictionResults:
    def __init__(self):
        self.results = []

class Rules: #todo
    def __init__(self, ruleType, rulesList):
        self.ruleType = ruleType
        self.rulesList = rulesList #trainRules=trainType+params: IndicatorsList or NN type

class Strategies:
    def __init__(self, stratType, stratsList):
        self.stratType = stratType
        self.stratsList = stratsList




class Predictor:

    def __init__(self, ticker, df_data, predictField, predictFieldShift, startTrain, trainPeriods, predictPeriods, trainRules=None, predictRules=None): #todo start_end_train fromToDate <-> fromToIndex
        self.df_data = df_data
        self.predictField = predictField
        self.predictFieldShift = predictFieldShift # -1 is (+1d) used to predict tmrw value
        self.df_data_copy = None #df_data.copy()
        self.ticker = ticker
        self.web_data = None
        self.startTrain = startTrain
        self.startTrainDate = self.df_data.index[startTrain]
        self.endTrainDate = self.df_data.index[startTrain+trainPeriods]
        self.toPredictDate = self.df_data.index[startTrain+trainPeriods+predictPeriods]
        self.trainPeriods = trainPeriods
        self.predictPeriods = predictPeriods #sort/learn/apply new rules/strats -> retrain after each prediction

        # todo
        self.periodResults = {"startTrainDate": self.startTrainDate, "endTrainDate": self.endTrainDate, "trainPeriods": trainPeriods,
                              "predictPeriods": predictPeriods, "periodRules": [], "periodsResults": [], "period_up_rules": [],
                              "period_down_rules": [], "predicted_up_ratio": None, "predicted_down_ratio": None,
                              "predicted_up_total_profit": None, "predicted_up_max_profit": None, "predicted_up_max_loss": None,
                              "predicted_down_total_profit": None, "predicted_down_max_profit": None, "predicted_down_max_loss": None,

                             }
        self.periodsResultsList = [] #store rules/predictions/stats by period
        self.trainRules = trainRules
        self.predictRules = predictRules



    def setNewDataCopy(self):
        self.df_data_copy = df_data.copy(deep=True)
        return self.df_data_copy


    def setPredictField(self):
        self.df_data_copy[self.predictField] = self.df_data_copy[self.predictField].shift(self.predictFieldShift)
        return self.df_data_copy.dropna(inplace=True)



    # def preprocData(self, startFrom, endTo, rules=[]): #todo to continue
    #     self.df_data_copy = self.df_data.iloc(startFrom, endTo).copy(deep=True)
    #     preproc.isUp(self.df_data_copy, "open")
    #     preproc.isUp(self.df_data_copy, "high")
    #     preproc.isUp(self.df_data_copy, "low")
    #     preproc.isUp(self.df_data_copy, "close")
    #
    #
    #     if len(rules) == 0: #default
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=5, period=3, fromField="open")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=5, period=3, fromField="high")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=5, period=3, fromField="low")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=5, period=3, fromField="close")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=10, period=5, fromField="open")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=10, period=5, fromField="high")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=10, period=5, fromField="low")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=10, period=5, fromField="close")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=20, period=10, fromField="open")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=20, period=10, fromField="high")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=20, period=10, fromField="low")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=20, period=10, fromField="close")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=50, period=30, fromField="open")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=50, period=30, fromField="high")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=50, period=30, fromField="low")
    #         preproc.isHighLowFromPeriodBack(self.df_data_copy, pct=50, period=30, fromField="close")
    #
    #     else:
    #         pass #todo
    #
    #     #self.df_data_copy.dropna(inplace=True)
    #     self.setPredictField() # shift PredictField and dropna - removes today/last element, since there is no values for tomorrow


    def trainAndPredictPeriod(self, startIdx, endIdx, predictField, predictPeriods):
        #period_data = preprocData(startIdx, endIdx, loadFromFile=False) #todo copy data/append?
        #dropna
        # rules = findBestWorstRules(byRulesRule, from, to) #-> storeRules
        # predictions = applyStrat(byStratsRule, from, to) #-> predict and store results
        # self.predictionResults.append(rules)
        # self.predictionResults.append(predictions)


        pass


    def cleanData(self):
        self.df_data_copy.dropna(inplace=True)
        self.df_data_copy.columns.drop(list(self.df_data_copy.filter(regex='diff_')))  # remove tmp cols
        #self.df_data_copy = self.df_data_copy.drop('ticker', 1)

    @timed
    def preprocBinaryFeatures(self, df_data):
        start = timeit.default_timer()
        # split continuous and binary features
        binary_data_cols = [col for col in df_data if
                            np.isin(df_data[col].dropna().unique(),
                                    [0, 1]).all()]
        continuous_data_cols = [col for col in df_data if
                                not np.isin(df_data[col].dropna().unique(),
                                            [0, 1]).all()]
        self.inputs_data_binary = df_data[binary_data_cols]  # get rid of continuous features/values
        self.inputs_data_continuous = df_data[continuous_data_cols]  # get rid of continuous features/values

        # range binary features if exist
        if not self.inputs_data_binary is None and self.inputs_data_binary.shape[0] > 0:
            # self.inputs_data["rules"] = {"binary_features_sum", self.inputs_data_binary.sum(axis=1)}
            df_data["rules_binary_features_sum"] = self.inputs_data_binary.sum(axis=1)

            #temp - feature correlation - reduce correlating features
            corr_data = self.inputs_data_binary.copy(deep=True)
            corr = corr_data.corr()
            #corr = df_data.corr() #copy needed to display removed features
            # # corr.head()
            print(f"Total binary features {self.inputs_data_binary.shape[1]}")
            print("removing correlating binary features")
            columns = np.full((corr.shape[0],), True, dtype=bool)
            #
            removed_cols = []
            try:
                for i in range(corr.shape[0]):
                    for j in range(i + 1, corr.shape[0]): #todo optimize duplicate loop
                        #print(f"i:{i}, j:{j}")
                        if corr.iloc[i, j] >= 0.9:
                            if columns[j]:
                                columns[j] = False
                                removed_cols.append(j)
                selected_columns = self.inputs_data_binary.columns[columns]
                print(f"Uncorrelated binary features: {selected_columns.shape}") #selected_columns.values
                self.inputs_data_binary = self.inputs_data_binary[selected_columns]
            except:
                raise Exception(f"in removing correlating binary features")
            print(f"Removed correlating binary features:{self.inputs_data_binary.columns[removed_cols].values}")

        print(f"preprocBinaryFeatures {self.df_data_copy.shape} took {timeit.default_timer() - start}s")
        print()
        return self.inputs_data_binary

    @timed
    def preprocBinaryFeatures2(self, df_data, corr_field="close"):
        start = timeit.default_timer()
        # split continuous and binary features
        binary_data_cols = [col for col in df_data if
                            np.isin(df_data[col].dropna().unique(),
                                    [0, 1]).all()]
        continuous_data_cols = [col for col in df_data if
                                not np.isin(df_data[col].dropna().unique(),
                                            [0, 1]).all()]
        self.inputs_data_binary = df_data[binary_data_cols]  # get rid of continuous features/values
        self.inputs_data_continuous = df_data[continuous_data_cols]  # get rid of continuous features/values

        # range binary features if exist
        if not self.inputs_data_binary is None and self.inputs_data_binary.shape[0] > 0:

            corr_data = self.inputs_data_binary.copy(deep=True)
            corr_data['close'] = df_data['close']
            corr_data['high'] = df_data['high']
            corr_data['low'] = df_data['low']
            corr_matrix = corr_data.corr()
            print(f"Total binary features {self.inputs_data_binary.shape[1]}")
            print("removing correlating binary features..")
            binary_best_down_keys = corr_matrix[corr_field].where(corr_matrix[corr_field] <= -0.1).filter(
                regex='is.*').dropna().keys()
            binary_best_up_keys = corr_matrix[corr_field].where(corr_matrix[corr_field] >= 0.1).filter(
                regex='is.*').dropna().keys()
            binary_best_keys = list(binary_best_down_keys) + list(binary_best_up_keys)


            #removed_cols = list(set(corr_data.filter(regex='is.*'))-set(binary_best_keys))
            ##corr_data.drop(removed_cols, axis=1, inplace=True) #todo to check bad perf on highly vorrelted features
            self.inputs_data_binary = corr_data.filter(regex='is.*')
            corr_data["rules_binary_features_sum"] = self.inputs_data_binary.sum(axis=1)
            #print(f"Removed weak correlating binary features:{removed_cols}")

        print(f"preprocBinaryFeatures {self.df_data_copy.shape} took {timeit.default_timer() - start}s")
        print()
        self.chartBinarySumStrat(corr_data)
        return self.inputs_data_binary


    def chartBinarySumStrat(self, df_data, corr_field="close"): #todo binary_sum for high/low
        profitPct = 0
        maxprofit = 0
        maxloss = 0
        closedTradesCount = 0
        good_trades_count = 0
        bad_trades_count = 0
        inpos = False
        inposValue = 0
        take_profit_pct = 7 #15% 10
        stop_loss_pct = 10f #8% 5
        total_trades = []
        long_trades = []
        short_trades = []
        good_trades = []
        bad_trades = []
        good_long_trades = []
        bad_long_trades = []
        good_short_trades = []
        bad_short_trades = []
        entries = []
        exits = []
        trades = []
        df_data['direction'] = 0 # 0 - OutOfMarket, 1 - short, 2 - long, closePos = 3
        for i, row in df_data.iterrows():
            if row['rules_binary_features_sum'] <= 2 and not inpos:  # openLong
                inposValue = row[corr_field]
                inpos = True
                trades.append({'Date': {i}, 'TradeType': 'Open', 'PositionType': 'Long', 'Price': row['close']})
                row['direction'] = 2
                df_data.at[i, 'direction'] = 2
                next
            elif inpos and trades[-1]['PositionType'] == 'Long':  # takeProfitLong
                profit = (row["high"]/inposValue - 1) * 100
                if profit >= take_profit_pct:
                    good_trades_count += 1
                    if take_profit_pct > maxprofit:
                        maxprofit = take_profit_pct
                    profitPct += take_profit_pct
                    closedTradesCount += 1
                    trades.append({'Date': {i}, 'TradeType': 'CloseTakeProfit', 'PositionType': 'Long', 'Price': inposValue*(1+take_profit_pct/100),
                               'profitPct': take_profit_pct})
                    row['direction'] = 3
                    df_data.at[i, 'direction'] = 4
                    inposValue = 0
                    inpos = False
                    next

            elif inpos and trades[-1]['PositionType'] == 'Long':  # stopLossLong
                profit = (inposValue / row["low"] - 1) * 100
                if profit >= stop_loss_pct:
                    bad_trades_count += 1
                    if profit > maxloss:
                        maxloss = profit
                    profitPct -= stop_loss_pct
                    closedTradesCount += 1
                    trades.append({'Date': {i}, 'TradeType': 'CloseStopLoss', 'PositionType': 'Long', 'Price': inpos*(1-stop_loss_pct/100),
                               'profitPct': stop_loss_pct*-1})
                    row['direction'] = 3
                    df_data.at[i, 'direction'] = 5
                    inposValue = 0
                    inpos = False
                    next

            elif row['rules_binary_features_sum'] >= 28 and inpos and trades[-1]['PositionType'] == 'Long':  # closeLong
                profit = (row[corr_field] / inposValue - 1) * 100
                profitPct += profit
                inposValue = 0
                inpos = False
                if profit >= 0:
                    good_trades_count += 1
                    if profit > maxprofit:
                        maxprofit = profit
                else:
                    bad_trades_count += 1
                    if profit < maxloss:
                        maxloss = profit
                closedTradesCount += 1
                trades.append({'Date': {i}, 'TradeType': 'Close', 'PositionType': 'Long', 'Price': row['close'], 'profitPct':{(row['close']/trades[-1]['Price']-1)*100}})
                row['direction'] = 3
                df_data.at[i, 'direction'] = 3
                next

            elif row['rules_binary_features_sum'] >= 29 and not inpos:  # openShort
                inposValue = row['close']
                inpos = True
                trades.append({'Date': {i}, 'TradeType': 'Open', 'PositionType': 'Short', 'Price': row['close']})
                row['direction'] = 1
                df_data.at[i, 'direction'] = 1
                next

            elif inpos and trades[-1]['PositionType'] == 'Short':  # closeShortStopLoss
                profit = (inposValue/row['high'] - 1) * 100
                if profit <= -stop_loss_pct:
                    profitPct -= stop_loss_pct
                    bad_trades_count += 1
                    if stop_loss_pct > maxloss:
                        maxloss = stop_loss_pct
                    closedTradesCount += 1
                    trades.append({'Date': {i}, 'TradeType': 'CloseStopLoss', 'PositionType': 'Short', 'Price': inposValue*(1+stop_loss_pct/100),
                               'profitPct': stop_loss_pct*-1})

                    row['direction'] = 3
                    df_data.at[i, 'direction'] = 5
                    inposValue = 0
                    inpos = False
                    next

            elif inpos and trades[-1]['PositionType'] == 'Short':  # closeShortTakeProfit
                profit = (inposValue / row['low'] - 1) * 100
                if profit >= take_profit_pct:
                    profitPct += take_profit_pct
                    good_trades_count += 1
                    if profit > maxprofit:
                        maxprofit = profit
                    closedTradesCount += 1
                    trades.append({'Date': {i}, 'TradeType': 'CloseTakeProfit', 'PositionType': 'Short', 'Price': inposValue*(1-take_profit_pct/100),
                               'profitPct': take_profit_pct})
                    row['direction'] = 3
                    df_data.at[i, 'direction'] = 4
                    inposValue = 0
                    inpos = False
                    next

            elif row['rules_binary_features_sum'] <= 2 and inpos and trades[-1]['PositionType'] == 'Short':  # closeShort
                profit = (inposValue / row['close'] - 1) * 100
                profitPct += profit
                if profit >= 0:
                    good_trades_count += 1
                    if profit > maxprofit:
                        maxprofit = profit
                else:
                    bad_trades_count += 1
                    if profit < maxloss:
                        maxloss = profit
                closedTradesCount += 1
                trades.append({'Date': {i}, 'TradeType': 'Close', 'PositionType': 'Short', 'Price': row['close'],
                               'profitPct': {(trades[-1]['Price']/row['close'] - 1) * 100}})
                row['direction'] = 3
                df_data.at[i, 'direction'] = 3
                inposValue = 0
                inpos = False
                next

        # print(f"Binary_Features Strat Results: \nTotalProfit:{}\nLongProfit:{}ShortProfit"
        #       f"\nMaxProfit:{}\nMaxLoss:{}\nTotalCloseTrades:{}\nTotalLongTrades{}\nTotalShortTrades"
        #       f"\nLongTradesGoodBadTradesRatio:{}\nShortTradesGoodBadTradesRatio:{}\nGoodBadTradesRatio{}"
        #       f"\nLongMedianProfitPct:{}\nShortMedianProfitPct:{}\nLongMedianLossPct:{}\nShortMedianLossPct:{}")

        for t in trades:
            print(t, "\n")

        df = pd.DataFrame(
            {'Date': df_data.index,
             'Close': df_data["close"],
             'Direction': df_data['direction']

             #,
             # 'BuyLong': [t['Price'] for t in trades if t['PositionType']=="Long" and t['TradeType']=="Open"],
             # 'CloseLong': [t['Price'] for t in trades if t['PositionType'] == "Long" and t['TradeType'] == "Close"],
             # 'SellShort': [t['Price'] for t in trades if t['PositionType'] == "Short" and t['TradeType'] == "Open"],
             # 'CloseShort': [t['Price'] for t in trades if t['PositionType'] == "Short" and t['TradeType'] == "Close"],
             }
        )
        df.dropna(inplace=True)
        colormap = np.array(['blue', 'red', 'green', 'orange', 'magenta', 'pink'])
        categories = np.array(df["Direction"])
        # plt.legend(["wait", "sell", "buy"], loc="upper right")
        # df.plot.legend(["wait", "sell", "buy"], loc="upper right")
        #plt.plot('Date', 'Close', data=df, marker='o', markerfacecolor='blue', markersize=10, color='skyblue', linewidth=2)

        df.plot.scatter("Date", 'Close', color=colormap[categories], title=self.ticker)
        #plt.savefig(f"charts/{self.ticker}_binarySumStrat.png")
        #plt.show()
        pylab.show()



    def shiftDfData(self, df_data, shift_field, shift_period=0, inplace=True):
        df = df_data
        if not inplace:
            df = df.copy()
        df[shift_field] = df[shift_field].shift(shift_period)
        return df.dropna()

    #@timed
    def queryBinRuleSqlDb(self, table, fields, values, predict_field_kv, start_idx, end_idx): #todo minPctChange>=3 + multipleShiftTargetData
        query = f"select * from {table} where "
        cols = " AND ".join([f"{fields[x]} = {values[x]}" for x in range(len(fields))])
        start_idx = f" AND rowid>={start_idx}" #todo indexDates
        end_idx = f" AND rowid<{end_idx}"
        where = f" AND {predict_field_kv.split('=')[0]}={predict_field_kv.split('=')[1]}" if len(predict_field_kv) > 0 else ""
        AND = "" #f" AND pctChange_close > 10" if 1 in fields else " AND pctChange_close < -10"
        query = f"{query} {cols} {start_idx} {end_idx} {where} {AND}"
        #print(query)
        return query


    def exploreBinaryRules(self, dfBinaryFeatures, dbTable, predictField, shiftPeriods, startFromPeriod, explorePeriod, rule_pct=0, rule_limit=1): #rule_pct=0.7
        start = timer() #timeit.default_timer()
        dfTarget = self.inputs_data_binary.copy(deep=True) #t = timeit.Timer(lambda: self.inputs_data_binary.copy(deep=True)).timeit(100)
        last_date = dfTarget[-1:].index[0]
        new_predict_field = f"predict_{predictField}" #todo +{shiftPeriods}
        dfTarget[new_predict_field] = self.df_data_copy[predictField]
        dfTarget = dfTarget[startFromPeriod: (startFromPeriod+explorePeriod)]

        # print(f"{startFromPeriod}, {dfTarget.index[0]}, {startFromPeriod+explorePeriod}, {dfTarget.index[-1]}")
        # return

        new_table = f"{dbTable}_{startFromPeriod}" #for different threads
        if shiftPeriods != 0:
            dfTarget = self.shiftDfData(dfTarget, new_predict_field, shiftPeriods, inplace=True)
        #dfTarget['id'] = dfTarget.index
        Db.persistDataFrame(dfTarget, new_table) # dfBinaryFeatures.loc[dfTarget.index, :]
        dfFeatures = dfTarget.filter(regex='^is.*').columns #self.inputs_data_binary.copy()
        ##dfTarget.index = pd.to_datetime(dfTarget.index)
        #dfTarget.index = dfTarget.index.map(lambda x: x.strftime('%Y-%m-%d'))

        #calc/runtime 1cycle-1period started - start
        #start = timer()
        #dfFeatures.drop(predictField, axis='columns', inplace=True)
        total_bin_features = len(dfFeatures) #todo chunks if rules_amount > 1000
        max_indicators = 3 #2 #3 #5 #7 #10 ##total_bin_features - 1 if total_bin_features <= 11 else 11 #todo upto 30 - bigCombsArray?
        ##periods = exploreStart  #500 #50 #200
        #rule_pct = 0.55 #0.6 #0.8 #0.5 0.75
        total_records = dfTarget.shape[0]
        #assert periods < total_records
        totalUp = dfTarget.query(f"{new_predict_field}==1").shape[0]
        totalDown = dfTarget.query(f"{new_predict_field}==0").shape[0]
        count = 0
        #print(f"Exploring {max_indicators} pairs from {dfFeatures.shape[0]} indicators with {len([x for x in combinations(dfFeatures, max_indicators+1)])} combinations")
        for i in range(1, max_indicators+1): #todo param range from 1 to 10 indicators
            combs = combinations(dfFeatures, i)
            for comb in combs: #todo some x1=1,x2=0.. permutations
                queryUp = " and ".join([f"{x}==1" for x in comb]) + f" and {new_predict_field}==1"
                queryDown = " and ".join([f"{x}==0" for x in comb]) + f" and {new_predict_field}==0"
                #queryStart = timer()
                #print(f"comb:{comb}")
                #print(f"queryUp:{queryUp}")
                #print(f"queryDown:{queryDown}")

                #UP - todo func #todo fix bug -  rowid is starting from 1 not 0
                rule_UP_all_periods = Db.execute(
                    self.queryBinRuleSqlDb(new_table, list(comb), [1] * len(comb), "", startFromPeriod, startFromPeriod+explorePeriod))

                predicted_UP_true = None
                periodsPredictedUp = 0
                if len(rule_UP_all_periods) > 0:
                    predicted_UP_true = Db.execute(
                        self.queryBinRuleSqlDb(new_table, list(comb), [1] * len(comb), f"{new_predict_field}=1", startFromPeriod, startFromPeriod+explorePeriod))
                    periodsPredictedUp = len(predicted_UP_true)
                rulePctUp = False
                if not predicted_UP_true is None and periodsPredictedUp > 0:
                    rulePctUp = periodsPredictedUp / len(rule_UP_all_periods) >= rule_pct  and periodsPredictedUp / len(rule_UP_all_periods) <= rule_limit\
                        if len(rule_UP_all_periods) > 0 and periodsPredictedUp <= len(rule_UP_all_periods) else \
                            periodsPredictedUp - periodsPredictedUp - len(rule_UP_all_periods) >= rule_pct


                # DOWN - todo func
                rule_DOWN_all_periods = Db.execute(
                   self.queryBinRuleSqlDb(new_table, list(comb), [0] * len(comb), "", startFromPeriod, startFromPeriod+explorePeriod))
                predicted_DOWN_true = None
                periodsPredictedDown = 0
                if len(rule_DOWN_all_periods) > 0:
                    predicted_DOWN_true = Db.execute(
                      self.queryBinRuleSqlDb(new_table, list(comb), [0] * len(comb), f"{new_predict_field}=0", startFromPeriod, startFromPeriod+explorePeriod))
                    periodsPredictedDown = len(predicted_DOWN_true)

                rulePctDown = False
                if not predicted_DOWN_true is None and periodsPredictedDown > 0:
                    rulePctDown = periodsPredictedDown / len(rule_DOWN_all_periods) >= rule_pct  and  periodsPredictedDown / len(rule_DOWN_all_periods) <= rule_limit\
                if len(rule_DOWN_all_periods) > 0 and periodsPredictedDown <= len(rule_DOWN_all_periods) else \
                    periodsPredictedDown - periodsPredictedDown - len(rule_DOWN_all_periods) >= rule_pct

           #
           #      #test/debug some rule
           #      if (queryDown == "isUp_ema_100_close==0 and isUp_ema_3_close==0 and predict_isUp_close==0"):
           #          print("BestRule: " + queryDown)
           #          print(tabulate(ruleTradesDown[["isUp_close", "predict_isUp_close"]]))
           #          print(f"Took {timer()-start} secs")
           #          exit(1)
           #
                ruleIdUp = hashlib.md5((str(comb) + 'up').encode('utf-8')).hexdigest()
                ruleIdDown = hashlib.md5((str(comb) + 'down').encode('utf-8')).hexdigest()
                if (rulePctUp):
                    #continue
                    binary_comb_results = {"ticker": self.ticker,
                                                 "last_date": last_date,
                                                 "start_date": dfTarget.index[0],
                                                 "end_date": dfTarget.index[-1],
                                                 "target_field": predictField,
                                                 "ruleId": ruleIdUp,
                                                 "ruleDesc": queryUp,
                                                 "predictField": predictField,
                                                 #"periods_back_test": periods,
                                                 #"periods_predict": shiftPeriods * -1,
                                                 "direction": "UP",
                                                 "total": totalUp,
                                                 "total_rules": len(rule_UP_all_periods),
                                                 "predicted_true": periodsPredictedUp,
                                                 "predicted_true_pct": periodsPredictedUp/len(rule_UP_all_periods),
                                                 "predicted_trade_dates": [x[0] for x in rule_UP_all_periods],
                                                 # f"{predictField}_20dBackPctIsUp": rule20dTrue / 20,
                                                 # f"{predictField}_10dBackPctIsUp": rule10dTrue / 10,
                                                 # f"{predictField}_5dBackPctIsUp": rule5dTrue / 5,
                                                 # f"{predictField}_50dBackPctIsDown": rule50dFalse / 50,
                                                 # f"{predictField}_20dBackPctIsDown": rule20dFalse  / 20,
                                                 # f"{predictField}_10dBackPctIsDown": rule10dFalse  / 10,
                                                 # f"{predictField}_5dBackPctIsDown": rule5dFalse  / 5
                                                 }
                    ##self.binary_combs_results.append(binary_comb_results)
                    self.periodResults["period_up_rules"].append(binary_comb_results)

                    #self.binary_combs_trades_up.append({'rule_id': ruleId, 'rule_desc': queryUp, 'direction': 'UP', 'trades': rule_UP_found_all})
                    #self.binary_combs_trades.append({'rule_id': ruleIdUp, 'rule_desc': queryUp, 'direction': 'UP', 'trades': rule_UP_found_all})
                    #pd.DataFrame.from_records(binary_comb_results, index='ruleId').to_sql('bin_rules_trades', con=self.db, if_exists='append', index=False)
                if (rulePctDown):
                    #continue
                    binary_comb_results = {"ticker": self.ticker,
                                                 "last_date": last_date,
                                                 "start_date": dfTarget.index[0],
                                                 "end_date": dfTarget.index[-1],
                                                 "target_field": predictField,
                                                 "ruleId": ruleIdDown,
                                                 "ruleDesc": queryDown,
                                                 "predictField": predictField,
                                                 #"periods_back_test": periods,
                                                 #"periods_predict": shiftPeriods * -1,
                                                 "direction": "DOWN",
                                                 "total": totalDown,
                                                 "total_rules": len(rule_DOWN_all_periods),
                                                 "predicted_true": periodsPredictedDown,
                                                 "predicted_true_pct": periodsPredictedDown/len(rule_DOWN_all_periods),
                                                 "predicted_trade_dates": [x[0] for x in rule_DOWN_all_periods],
                                                 # f"{predictField}_20dBackPctIsUp": rule20dTrue / 20,
                                                 # f"{predictField}_10dBackPctIsUp": rule10dTrue / 10,
                                                 # f"{predictField}_5dBackPctIsUp": rule5dTrue / 5,
                                                 # f"{predictField}_50dBackPctIsDown": rule50dFalse / 50,
                                                 # f"{predictField}_20dBackPctIsDown": rule20dFalse  / 20,
                                                 # f"{predictField}_10dBackPctIsDown": rule10dFalse  / 10,
                                                 # f"{predictField}_5dBackPctIsDown": rule5dFalse  / 5
                                                 }
                    ##self.binary_combs_results.append(binary_comb_results)
                    self.periodResults["period_down_rules"].append(binary_comb_results)

                    #self.binary_combs_trades_down.append({'rule_id': ruleId, 'rule_desc': queryDown, 'direction': 'DOWN', 'trades': rule_UP_found_all})
                    #self.binary_combs_trades.append({'rule_id': ruleIdUp, 'rule_desc': queryDown, 'direction': 'DOWN', 'trades': rule_DOWN_found_all})
                    #pd.DataFrame.from_records(binary_comb_results, index='ruleId').to_sql('bin_rules_trades', con=self.db, if_exists='append', index=False)
           #      #good self.inputs_data.query(query).shape[0]
           #      #bad total-good vs query bad = total query = bad query
           #      #last date self.inputs_data.query(query)[-1:].index[0]
           #      #todo where query 100d-50d-20d-5d >75% target + >75% voters + >75% assets (or 1-3 highly correlated assets)

        #if Db.con.has_table(new_table):
        #Db.execute(f"DROP TABLE {new_table} IF EXISTS")
        #print(f"exploreBinaryRules took {timer()-start} secs")



    def validateBinaryRules(self, df, tableName, predictField, startExplore, startPredict, predictPeriods, minGoodPeriodsPredicted=1):
        start = timer()
        # worst_up_rules = [item for item in self.periodResults["period_up_rules"] if item["predicted_true_pct"] < 0.2]
        # best_up_rules = [item for item in self.periodResults["period_up_rules"] if item["predicted_true_pct"] > 0.8]
        # # worst_up_rules_dates = set([item for item in worst_up_rules for item in item["predicted_trade_dates"]])
        # # best_up_rules_dates = set([item for item in best_up_rules for item in item["predicted_trade_dates"]])
        #
        # worst_down_rules = [item for item in self.periodResults["period_down_rules"] if
        #                     item["predicted_true_pct"] < 0.3]
        # best_down_rules = [item for item in self.periodResults["period_down_rules"] if
        #                    item["predicted_true_pct"] > 0.7]
        # worst_down_rules_dates = set([item for item in worst_down_rules for item in item["predicted_trade_dates"]])
        # best_down_rules_dates = set([item for item in best_down_rules for item in item["predicted_trade_dates"]])

        worst_up_rules_q = [item["ruleDesc"] for item in self.periodResults["period_up_rules"] if
                            item["start_date"] == df.index[startExplore] and item["predicted_true_pct"] <= 0.2 and item["predicted_true"] >=minGoodPeriodsPredicted]
        best_up_rules_q = [item["ruleDesc"] for item in self.periodResults["period_up_rules"] if
                           item["start_date"] == df.index[startExplore] and item["predicted_true_pct"] >= 0.8 and item["predicted_true"] >=minGoodPeriodsPredicted ]
        worst_down_rules_q = [item["ruleDesc"] for item in self.periodResults["period_down_rules"] if
                              item["start_date"] == df.index[startExplore] and item["predicted_true_pct"] <= 0.2 and item["predicted_true"] >=minGoodPeriodsPredicted]
        best_down_rules_q = [item["ruleDesc"] for item in self.periodResults["period_down_rules"] if
                             item["start_date"] == df.index[startExplore] and item["predicted_true_pct"] >= 0.8 and item["predicted_true"] >=minGoodPeriodsPredicted]

        up_rules = best_up_rules_q + worst_down_rules_q
        up_rules_queries = [f"select * from {tableName} where {' and '.join(item.split(' and ')[:-1])} and rowid>={startPredict} and rowid<{startPredict + predictPeriods}"
                                for item in up_rules]
        down_rules = best_down_rules_q + worst_up_rules_q
        down_rules_queries = [f"select * from {tableName} where {' and '.join(item.split(' and ')[:-1])} and rowid>={startPredict} and rowid<{startPredict + predictPeriods}"
                                for item in down_rules]
        predict_dates_up = [{'date': x[0], 'matched': False, 'id': x[1]} for x in Db.execute(
            f"select id, rowid from {tableName} where  rowid>={startPredict} and rowid<{startPredict + predictPeriods}")]
        predict_days_up = [x['date'].replace(" 00:00:00.000000", "") for x in predict_dates_up]
        #up
        for q in up_rules_queries:
            rs = Db.execute(q)
            if len(rs) == predictPeriods or len([i for i in predict_dates_up if i['matched'] == True]) == predictPeriods:
                for d in predict_dates_up:
                    d['matched'] = True
                break
            for d in rs:
                for dd in predict_dates_up:
                    if d['id'] == dd['date']:
                        dd['matched'] = True
        # up_entries = [d["id"] for d in predict_dates_up if d['matched'] == True]
        # up_exits = [d + abs(self.predictFieldShift) for d in up_entries]
        # rsPredictionsUp = Db.execute(f"select {predictField} from {tableName} where rowid in ({','.join([str(i) for i in up_exits])})")
        # print(f"Up {len([x for x in rsPredictions if x[0] == 1])} good trades from total {len(up_entries)} trades in period between {predict_days_up[0]}-{predict_days_up[-1]}")
        # #print(up_rules)
        # profit = None
        # if len(up_exits) > 0:
        #     profit = Db.execute(
        #         f"select sum(pctChange_Close) from {tableName.replace('_binary_features', '')} where id in (select id from {tableName} where rowid in ({','.join([str(i) for i in up_exits])}))")
        # if profit is not None:
        #     print(up_entries)
        #     print(f"Up Period Profit: {profit}% in period between {predict_days_up[0]}-{predict_days_up[-1]}")

        # down
        predict_dates_down = [{'date': x[0], 'matched': False, 'id': x[1]} for x in Db.execute(
            f"select id, rowid from {tableName} where  rowid>={startPredict} and rowid<{startPredict + predictPeriods}")]
        predict_days_down = [x['date'].replace(" 00:00:00.000000", "") for x in predict_dates_down]
        for q in down_rules_queries:
            rs = Db.execute(q)
            if len(rs) == predictPeriods or len([i for i in predict_dates_down if i['matched'] == True]) == predictPeriods:
                for d in predict_dates_down:
                    d['matched'] = True
                break
            for d in rs:
                for dd in predict_dates_down:
                    if d['id'] == dd['date']:
                        dd['matched'] = True

        up_entries = [d["id"] for d in predict_dates_up if d['matched'] == True]
        down_entries = [d["id"] for d in predict_dates_down if d['matched'] == True]
        up_entries = list(set(up_entries).difference(set(down_entries)))
        down_entries = list(set(down_entries).difference(set(up_entries)))

        up_exits = [d + abs(self.predictFieldShift) for d in up_entries]
        rsPredictionsUp = Db.execute(f"select {predictField} from {tableName} where rowid in ({','.join([str(i) for i in up_exits])})")
        print(len(up_rules), 'up_rules', len(set(up_rules)))
        profit_up = None
        if len(up_exits) > 0:
            print(
                f"Up {len([x for x in rsPredictionsUp if x[0] == 1])} good trades from total {len(up_entries)} trades in period between {predict_days_up[0]}-{predict_days_up[-1]}")
            profit_up = Db.execute(
                f"select sum(pctChange_Close) from {tableName.replace('_binary_features', '')} where id in (select id from {tableName} where rowid in ({','.join([str(i) for i in up_exits])}))")
        if profit_up is not None:
            profit_up = profit_up[0][0]
            print(up_entries)
            print(f"Up Period Profit: {profit_up}% in period between {predict_days_up[0]}-{predict_days_up[-1]}")

        down_exits = [d + abs(self.predictFieldShift) for d in down_entries]
        rsPredictionsDown = Db.execute(f"select {predictField} from {tableName} where rowid in ({','.join([str(i) for i in down_exits])})")
        print(len(down_rules), 'down_rules', len(set(down_rules)))
        profit_down = None
        if len(down_exits)>0:
            print(f"Down {len([x for x in rsPredictionsDown if x[0] == 0])} good trades from total {len(down_entries)} trades in period between {predict_days_down[0]}-{predict_days_down[-1]}")
            profit_down = Db.execute(f"select sum(pctChange_Close) from {tableName.replace('_binary_features', '')} where id in (select id from {tableName} where rowid in ({','.join([str(i) for i in down_exits])}))")
        if profit_down is not None:
            print(down_entries)
            profit_down = profit_down[0][0] * -1 #profit -> change signs
            print(f"Down Period Profit: {profit_down}% in period between {predict_days_down[0]}-{predict_days_down[-1]}")

        if profit_up is not None and profit_down is not None:
            total_period_profit = profit_up+profit_down
        elif profit_up is None and profit_down is not None:
            total_period_profit = profit_down
        elif profit_up is not None and profit_down is None:
            total_period_profit = profit_up
        else:
            total_period_profit = 0
        print(f"Period PL {total_period_profit}%")
        self.periodResults['periodsResults'].append({
            'from': startPredict,
            'to': startPredict + predictPeriods,
            'total_period_profit': total_period_profit,
            'period_profit_up': profit_up,
            'period_profit_down': profit_down
        })


        #todo good-bad + minTrainedTradesAmount
        #?? set(down_entries).intersection(set(up_entries))
        #x = list(set(up_entries).symmetric_difference(set(down_entries)))
        #r = Db.execute(f"select {predictField} from {tableName} where rowid in ({','.join([str(i) for i in x])})")

        trades = []
        # for p in range(startPredict, startPredict+predictPeriods):
        #     #print(f"Validating {df.index[p]}")
        #     predict_date = startPredict
        #     predict_interval = predictPeriods
        #     trainPeriod_up_best = [
        #         f"select * from {tableName} where {' and '.join(item.split(' and ')[:-1])} and rowid>={predict_date} and rowid<{predict_date + predict_interval}"
        #         for item in best_up_rules_q]
        #     trainPeriod_up_worst = [
        #         f"select * from {tableName} where {' and '.join(item.split(' and ')[:-1])} and rowid>={predict_date} and rowid<{predict_date + predict_interval}"
        #         for item in worst_up_rules_q]
        #     trainPeriod_down_best = [
        #         f"select * from {tableName} where {' and '.join(item.split(' and ')[:-1])]} and rowid>={predict_date} and rowid<{predict_date + predict_interval}"
        #         for item in best_down_rules_q]
        #     trainPeriod_down_worst = [
        #         f"select * from {tableName} where {' and '.join(item.split(' and ')[:-1])} and rowid>={predict_date} and rowid<{predict_date + predict_interval}"
        #         for item in worst_down_rules_q]
        #
        #     # rs = Db.execute(trainPeriod0[0])
        #
        #     #shorts trainPeriod_down_best + trainPeriod_up_worst ToDo
        #     # longs trainPeriod_up_best   + trainPeriod_down_worst
        #     signal = False
        #     for q in trainPeriod_up_best:
        #         # print(q)
        #         rs = Db.execute(q)
        #         if len(rs) > 0:
        #             signal = True
        #             #print(f"Rule_up_best Found: {q}")
        #             #print(rs)
        #             break
        #
        #     for q in trainPeriod_up_worst:
        #         #print(q)
        #         rs = Db.execute(q)
        #         if len(rs)>0:
        #             signal = False
        #             #print(f"Rule_up_worst Found: {q}")
        #             #print(rs)
        #             signal = False
        #             break
        #
        #     for q in trainPeriod_down_best:
        #         # print(q)
        #         rs = Db.execute(q)
        #         if len(rs) > 0:
        #             signal = False
        #             #print(f"Rule_down_best Found: {q}")
        #             break
        #
        #     if signal:  # check long Entry #todo check for Short + todo min 2indicators + remove predictField after shifting?
        #         is_good =  (df.iloc[predict_date][f"{predictField}"] == 1)
        #         #print(f"Prediction for {df.index[predict_date]}", is_good)  # change for check PredictField - Shift
        #         trades.append(is_good)


        #print(f"validateBinaryRules took {timer() - start} secs , from {df.index[predict_date]} {len([x for x in trades if x])} longs are good from {self.predictPeriods} total")


    def run(self): #cross validation by periods? threadPool/gpu grid
        #assert abs(self.startTrain) < len(self.df_data_copy)
        #assert len(self.df_data_copy) > self.trainPeriods
        start = timeit.default_timer()
        self.df_data_copy = preprocData(self.df_data, self.startTrain, len(self.df_data), loadFromFile=False)
        tableName = self.ticker
        tableName = tableName.replace("^", "")
        self.ticker = tableName
        self.cleanData()
        binary_features = self.preprocBinaryFeatures2(self.df_data_copy)
        Db.persistDataFrame(self.df_data_copy, tableName) #with corelated (all) features
        Db.persistDataFrame(binary_features, f"{self.ticker}_binary_features") #uncorrelated binary features
        total_periods = len(binary_features)
        for i in range(self.trainPeriods, total_periods, self.predictPeriods): #TOdo LESS THEN 3/5/10D remainder
            startExplore = i - self.trainPeriods
            endExplore = startExplore + self.trainPeriods
            #print(f"startExplore: {startExplore}, endExplore: {endExplore}")
            self.exploreBinaryRules(binary_features, f"{self.ticker}_binary_features", self.predictField, self.predictFieldShift, startExplore, self.trainPeriods)
            self.validateBinaryRules(binary_features, f"{self.ticker}_binary_features", self.predictField, startExplore, endExplore, self.predictPeriods)
            #print('d')

        # self.exploreBinaryRules(binary_features, f"{self.ticker}_binary_features", self.predictField,
        #                         self.predictFieldShift, 0, self.trainPeriods)

        print()
        print("Total Run profit", sum([x['total_period_profit'] for x in self.periodResults['periodsResults']]))
        print(f"Run took {timeit.default_timer()-start}s")
        exit(0) #todo to exclude predict and "predict_field" from the rule set

        while self.startTrain + self.trainPeriods + self.predictPeriods < len(self.df_data_copy):
            trainFrom = self.df_data.index[self.startTrain]
            trainTo = self.df_data.index[self.startTrain+self.trainPeriods]
            self.trainAndPredictPeriod(trainFrom, trainTo, self.predictField, self.predictPeriods)
            self.startTrain += self.predictPeriods #look backwards period then predict / moving window/period
        if len(self.df_data_copy) - (self.startTrain + self.trainPeriods) > 0: #< self.predictPeriods: #predict last days if < predictPeriods remained
            trainFrom = self.df_data.index[self.startTrain]
            trainTo = self.df_data.index[self.startTrain + self.trainPeriods]
            self.trainAndPredictPeriod(trainFrom, trainTo, self.predictField, self.predictPeriods)

        #self.df_data.index[startTrain+trainPeriods+predictPeriods] < self.self.df_data_copy.index[-1]
        while self.toPredictDate < self.df_data_copy.index[-1]:
            self.trainAndPredictPeriod(self.startTrainDate, self.endTrainDate, self.predictField, self.predictPeriods)
            self.startTrainDate = self.toPredictDate
            #self.endTrainDate =

        return self

if __name__ == "__main__":
    #PreprocData.run([{"print": ["a", "b"]}])
    start = timeit.default_timer() #^GSPC AI
    ticker = '^GSPC' #'PLTR' 1995 -5555
    data = DataReader(ticker=ticker, fromDate="2005-01-01", toDate="2022-12-31", interval="1d", index_as_date=True).getData() #todo global -> distrubute to Predictors Grid/Threads/Gpus
    predictor = Predictor(ticker, data, predictField="isUp_close", predictFieldShift=-1, startTrain=-3000, trainPeriods=100, predictPeriods=5) #todo tochange? startTrain to last
    #-333
    predictor.run()
    print(f"Run took {timeit.default_timer() - start}s")