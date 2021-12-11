import pandas as pd
# import ray
# ray.init(num_cpus=6)
#import modin.pandas as pd
from pony.orm import *
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

import preproc
from technical_indicators_lib import *
#rom talib_test import bb #
import timeit
import pickle
from yahoo_fin import stock_info as si

# instantiate the class
# obv = OBV()
# rsi = RSI()
# bb = BollingerBands()
# cci = CCI()
# roc = ROC()
# macd = MACD()
# stoch2 = StochasticKAndD()

db = Database()

class DictToObject(object):
    def __init__(self, myDict):
        for key, value in myDict.items():
            if type(value) == dict:
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)
        return self

#
# class DbEntityFromDict(db.Entity):
#     https://docs.ponyorm.org/api_reference.html#sqlite


class Db:
    con = None

    def __init__(self, provider="sqlite", isMemory=True):
        if isMemory:
            self.db = Database() #pony? todo?
            #self.db = create_engine('sqlite://', echo=False)

            con = self.db
            self.db.bind(provider=provider, filename=':memory:')

    @staticmethod
    def persistDataFrame(dfData, toTable, pkCol="id", provider="sqlite", isMemory=True):
        if Db.con is None:
            Db.con = create_engine('sqlite:///:memory:', echo=False) #('sqlite://', echo=False)
        dfData.to_sql(toTable, con=Db.con, if_exists='append', index=True, index_label=pkCol)
        #dfData.to_sql(toTable, Db.con) #, if_exists="append") #todo create table/index ->Pony
        #rs = con.execute(f"select * from {toTable}").fetchall()
        #return

    @staticmethod
    def execute(query, provider="sqlite", isMemory=True, echo=False):
        if Db.con is None:
            Db.con = create_engine('sqlite:///:memory:', echo=echo)
        return Db.con.execute(query).fetchall()

def saveDataCsv(df_data, filename, data_foler="data"):
    df_data.to_csv(data_foler + "/" + filename)

def loadDataCsv(filename, data_foler="data"):
    return pd.read_csv(data_foler + "/" + filename)

def getDataFromYahoo(symbol, fromDate="2000-01-01", toDate="2021-31-12"):
    startYahoo = timeit.default_timer()
    web_data = si.get_data(symbol, fromDate, toDate)
    web_data.to_csv("data/%s_%s_%s.csv" % (symbol.replace("^", ""), fromDate, toDate))
    print("Get Yahoo ticker %s took %.2f secs" % (symbol, timeit.default_timer() - startYahoo))
    return web_data


def saveModel(model, filename, models_dir="models"):
    try:
        pickle.dump(model, open(models_dir+"/"+filename, 'wb'))
    except:
        pass


def loadModel(filename, models_dir="models"):
    try:
        return pickle.load(open(models_dir + "/" + filename, 'rb'))
    except:
        pass


def pctChange(fromAmount, toAmount, direction="long"):
    if fromAmount == toAmount:
        return 0
    else:
        if direction == "long" and toAmount > fromAmount:
            return (toAmount/fromAmount - 1) * 100           #advance
        if direction == "long" and toAmount < fromAmount:
            return (fromAmount / toAmount - 1) * -1 * 100    #decline
        if direction == "short" and toAmount < fromAmount:
            return (fromAmount / toAmount - 1) * 100         #advance
        if direction == "short" and fromAmount < toAmount:
            return (toAmount / fromAmount - 1) * -1 * 100    #decline


#todo days declinse/advance, percentAdvanceDecline
def preprocData(ohlcv, startIdx, endIdx, loadFromFile=True):
    start = timeit.default_timer()
    if loadFromFile:
        try:
            ohlcv = pd.read_csv('preproc_data.csv', index_col=0)
            end = timeit.default_timer()
            print("Preproc data from file load took %.2f seconds" % (end - start))
            return ohlcv[startIdx:endIdx].copy(deep=True)
        except Exception as ex:
            raise ex

    ohlcv = ohlcv[startIdx:endIdx].copy(deep=True)

    ohlcv = preproc.ema(ohlcv, 'close', 2)
    ohlcv = preproc.ema(ohlcv, 'close', 3)
    ohlcv = preproc.ema(ohlcv, 'close', 5)
    ohlcv = preproc.ema(ohlcv, 'close', 10)
    ohlcv = preproc.ema(ohlcv, 'close', 20)
    ohlcv = preproc.ema(ohlcv, 'close', 50)
    ohlcv = preproc.ema(ohlcv, 'close', 100)

#    return ohlcv

    ohlcv = preproc.ema(ohlcv, 'high', 2)
    ohlcv = preproc.ema(ohlcv, 'high', 3)
    ohlcv = preproc.ema(ohlcv, 'high', 5)
    ohlcv = preproc.ema(ohlcv, 'high', 10)
    ohlcv = preproc.ema(ohlcv, 'high', 20)
    ohlcv = preproc.ema(ohlcv, 'high', 50)
    ohlcv = preproc.ema(ohlcv, 'high', 100)

    ohlcv = preproc.ema(ohlcv, 'low', 2)
    ohlcv = preproc.ema(ohlcv, 'low', 3)
    ohlcv = preproc.ema(ohlcv, 'low', 5)
    ohlcv = preproc.ema(ohlcv, 'low', 10)
    ohlcv = preproc.ema(ohlcv, 'low', 20)
    ohlcv = preproc.ema(ohlcv, 'low', 50)
    ohlcv = preproc.ema(ohlcv, 'low', 100)

    ohlcv = preproc.isUp(ohlcv, 'close')
    ohlcv = preproc.isUp(ohlcv, 'ema_2_close')
    ohlcv = preproc.isUp(ohlcv, 'ema_3_close')
    ohlcv = preproc.isUp(ohlcv, 'ema_5_close')
    ohlcv = preproc.isUp(ohlcv, 'ema_10_close')
    ohlcv = preproc.isUp(ohlcv, 'ema_20_close')
    ohlcv = preproc.isUp(ohlcv, 'ema_50_close')
    ohlcv = preproc.isUp(ohlcv, 'ema_100_close')

    ohlcv = preproc.isUp(ohlcv, 'ema_2_high')
    ohlcv = preproc.isUp(ohlcv, 'ema_3_high')
    ohlcv = preproc.isUp(ohlcv, 'ema_5_high')
    ohlcv = preproc.isUp(ohlcv, 'ema_10_high')
    ohlcv = preproc.isUp(ohlcv, 'ema_20_high')
    ohlcv = preproc.isUp(ohlcv, 'ema_50_high')
    ohlcv = preproc.isUp(ohlcv, 'ema_100_high')

    ohlcv = preproc.isUp(ohlcv, 'ema_2_low')
    ohlcv = preproc.isUp(ohlcv, 'ema_3_low')
    ohlcv = preproc.isUp(ohlcv, 'ema_5_low')
    ohlcv = preproc.isUp(ohlcv, 'ema_10_low')
    ohlcv = preproc.isUp(ohlcv, 'ema_20_low')
    ohlcv = preproc.isUp(ohlcv, 'ema_50_low')
    ohlcv = preproc.isUp(ohlcv, 'ema_100_low')


    ohlcv = preproc.pctChange1p(ohlcv, 'open')
    ohlcv = preproc.pctChange1p(ohlcv, 'close')
    ohlcv = preproc.pctChange1p(ohlcv, 'low')
    ohlcv = preproc.pctChange1p(ohlcv, 'high')

    ohlcv = preproc.pctChange1p(ohlcv, 'ema_2_high')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_3_high')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_5_high')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_10_high')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_20_high')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_50_high')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_100_high')

    ohlcv = preproc.pctChange1p(ohlcv, 'ema_2_close')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_3_close')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_5_close')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_10_close')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_20_close')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_50_close')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_100_close')

    ohlcv = preproc.pctChange1p(ohlcv, 'ema_2_low')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_3_low')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_5_low')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_10_low')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_20_low')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_50_low')
    ohlcv = preproc.pctChange1p(ohlcv, 'ema_100_low')

    # ohlcv = preproc.volatility(ohlcv, 'close', 2)
    # ohlcv = preproc.volatility(ohlcv, 'close', 3)
    # ohlcv = preproc.volatility(ohlcv, 'close', 5)
    # ohlcv = preproc.volatility(ohlcv, 'close', 10)
    # ohlcv = preproc.volatility(ohlcv, 'close', 20)
    # ohlcv = preproc.volatility(ohlcv, 'close', 50)
    # ohlcv = preproc.volatility(ohlcv, 'close', 100)
    #
    # ohlcv = preproc.volatility(ohlcv, 'high', 2)
    # ohlcv = preproc.volatility(ohlcv, 'high', 3)
    # ohlcv = preproc.volatility(ohlcv, 'high', 5)
    # ohlcv = preproc.volatility(ohlcv, 'high', 10)
    # ohlcv = preproc.volatility(ohlcv, 'high', 20)
    # ohlcv = preproc.volatility(ohlcv, 'high', 50)
    # ohlcv = preproc.volatility(ohlcv, 'high', 100)
    #
    # ohlcv = preproc.volatility(ohlcv, 'low', 2)
    # ohlcv = preproc.volatility(ohlcv, 'low', 3)
    # ohlcv = preproc.volatility(ohlcv, 'low', 5)
    # ohlcv = preproc.volatility(ohlcv, 'low', 10)
    # ohlcv = preproc.volatility(ohlcv, 'low', 20)
    # ohlcv = preproc.volatility(ohlcv, 'low', 50)
    # ohlcv = preproc.volatility(ohlcv, 'low', 100)
    # ohlcv = preproc.volatility(ohlcv, 'low', 200)
    # ohlcv = preproc.volatility(ohlcv, 'ema_2_low', 2)
    # ohlcv = preproc.volatility(ohlcv, 'ema_3_low', 3)
    # ohlcv = preproc.volatility(ohlcv, 'ema_5_low', 5)
    # ohlcv = preproc.volatility(ohlcv, 'ema_10_low', 10)
    # ohlcv = preproc.volatility(ohlcv, 'ema_20_low', 20)
    # ohlcv = preproc.volatility(ohlcv, 'ema_50_low', 50)
    # ohlcv = preproc.volatility(ohlcv, 'ema_100_low', 100)


    # ohlcv = preproc.rsi(ohlcv, "close", 14)
    # ohlcv = preproc.rsi(ohlcv, "high", 14)
    # ohlcv = preproc.rsi(ohlcv, "low", 14)
    # ohlcv = preproc.rsi(ohlcv, "open", 14)

    # ohlcv = preproc.rsi(ohlcv, "close", 10)
    # ohlcv = preproc.rsi(ohlcv, "high", 10)
    # ohlcv = preproc.rsi(ohlcv, "low", 10)
    # ohlcv = preproc.rsi(ohlcv, "close", 20)
    # ohlcv = preproc.rsi(ohlcv, "high", 20)
    # ohlcv = preproc.rsi(ohlcv, "low", 20)
    #
    # ohlcv = preproc.rsi(ohlcv, 'ema_2_close')
    # ohlcv = preproc.rsi(ohlcv, 'ema_3_close')
    # ohlcv = preproc.rsi(ohlcv, 'ema_5_close')
    # ohlcv = preproc.rsi(ohlcv, 'ema_10_close')
    # ohlcv = preproc.rsi(ohlcv, 'ema_20_close')
    # ohlcv = preproc.rsi(ohlcv, 'ema_50_close')
    # ohlcv = preproc.rsi(ohlcv, 'ema_100_close')
    #
    # ohlcv = preproc.rsi(ohlcv, 'ema_2_high')
    # ohlcv = preproc.rsi(ohlcv, 'ema_3_high')
    # ohlcv = preproc.rsi(ohlcv, 'ema_5_high')
    # ohlcv = preproc.rsi(ohlcv, 'ema_10_high')
    # ohlcv = preproc.rsi(ohlcv, 'ema_20_high')
    # ohlcv = preproc.rsi(ohlcv, 'ema_50_high')
    # ohlcv = preproc.rsi(ohlcv, 'ema_100_high')
    #
    ohlcv = preproc.rsi(ohlcv, 'ema_2_low')
    ohlcv = preproc.rsi(ohlcv, 'ema_3_low')
    ohlcv = preproc.rsi(ohlcv, 'ema_5_low')
    ohlcv = preproc.rsi(ohlcv, 'ema_10_low')
    ohlcv = preproc.rsi(ohlcv, 'ema_20_low')
    ohlcv = preproc.rsi(ohlcv, 'ema_50_low')
    ohlcv = preproc.rsi(ohlcv, 'ema_100_low')
    #
    # ohlcv = preproc.rsi(ohlcv, 'ema_2_low', 20)
    #ohlcv = preproc.rsi(ohlcv, 'ema_3_low', 10)
    #ohlcv = preproc.rsi(ohlcv, 'ema_5_low', 10)
    # ohlcv = preproc.rsi(ohlcv, 'ema_10_low', 10)
    # ohlcv = preproc.rsi(ohlcv, 'ema_20_low', 10)
    #ohlcv = preproc.rsi(ohlcv, 'ema_50_low', 5)
    #ohlcv = preproc.rsi(ohlcv, 'ema_100_low', 5)
    # #
    # ohlcv = preproc.rsi(ohlcv, 'ema_2_high', 20)
    # ohlcv = preproc.rsi(ohlcv, 'ema_3_high', 20)
    # ohlcv = preproc.rsi(ohlcv, 'ema_5_high', 20)
    # ohlcv = preproc.rsi(ohlcv, 'ema_10_high', 10)
    # ohlcv = preproc.rsi(ohlcv, 'ema_20_high', 10)
    # ohlcv = preproc.rsi(ohlcv, 'ema_50_high', 5)
    # ohlcv = preproc.rsi(ohlcv, 'ema_100_high', 5)

    #ohlcv = preproc.stoch(ohlcv, 'close', 5) #5+14 SAME ACCURACY,but together less
    ohlcv = preproc.stoch(ohlcv, 'close', 14) #TODO REMOVE CORRELATING FEATURES FOR ACCURATE PREDICTIONS
    ohlcv = preproc.stoch(ohlcv, 'close', 25) #todo Optimize best? + bad together other periods
    ohlcv = preproc.stoch(ohlcv, 'open', 25)
    ohlcv = preproc.stoch(ohlcv, 'high', 25)
    ohlcv = preproc.stoch(ohlcv, 'low', 25)

    # ohlcv = preproc.stoch(ohlcv, 'ema_2_low', 14)
    # ohlcv = preproc.stoch(ohlcv, 'ema_3_low', 14)
    # ohlcv = preproc.stoch(ohlcv, 'ema_5_low', 14)
    # ohlcv = preproc.stoch(ohlcv, 'ema_10_low', 10)
    # ohlcv = preproc.stoch(ohlcv, 'ema_20_low', 10)
    # ohlcv = preproc.stoch(ohlcv, 'ema_50_low', 5)
    # ohlcv = preproc.stoch(ohlcv, 'ema_100_low', 5)

    # ohlcv = preproc.isPeriodHighBack(ohlcv, 3)
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 5)
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 9)
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 14)
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 18)
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 50)
    #
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 3, "low")
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 5, "low")
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 9, "low")
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 14, "low")
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 18, "low")
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 50, "low")
    #
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 3, "close")
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 5, "close")
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 9, "close")
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 14, "close")
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 18, "close")
    # ohlcv = preproc.isPeriodHighBack(ohlcv, 50, "close")

    # ohlcv = preproc.isPeriodLowBack(ohlcv, 3)
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 5)
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 9)
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 14)
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 18)
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 50)
    #
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 3, "close")
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 5, "close")
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 9, "close")
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 14, "close")
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 18, "close")
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 50, "close")
    #
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 3, "open")
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 5, "open")
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 9, "open")
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 14, "open")
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 18, "open")
    # ohlcv = preproc.isPeriodLowBack(ohlcv, 50, "open")

    #ohlcv = preproc.pctChangePriceByPeriod(ohlcv, 2)
    #ohlcv = preproc.pctChangePriceByPeriod(ohlcv, 3)
    #ohlcv = preproc.pctChangePriceByPeriod(ohlcv, 5)

    ohlcv = preproc.pctChangePriceByPeriod(ohlcv, 3)
    ohlcv = preproc.pctChangePriceByPeriod(ohlcv, 5)
    ohlcv = preproc.pctChangePriceByPeriod(ohlcv, 9)
    ohlcv = preproc.pctChangePriceByPeriod(ohlcv, 18)
    ohlcv = preproc.pctChangePriceByPeriod(ohlcv, 48)
    ohlcv = preproc.pctChangePriceByPeriod(ohlcv, 98)

    # ohlcv = preproc.isPctChangePriceByPeriod(ohlcv, 5, 3) #todo rectified cross tuning for same class sequences
    # ohlcv = preproc.isPctChangePriceByPeriod(ohlcv, 5, 5)
    # ohlcv = preproc.isPctChangePriceByPeriod(ohlcv, 5, 10)
    # ohlcv = preproc.isPctChangePriceByPeriod(ohlcv, 5, 20)
    #ohlcv = preproc.isPctChangePriceByPeriod(ohlcv, 5, 50)
   # ohlcv = preproc.isPctChangePriceByPeriod(ohlcv, 5, 100)
   # ohlcv = preproc.isPctChangePriceByPeriod(ohlcv, 10, 200)

    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 0.3, 2)
    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 0.5, 2)
    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 1, 2)
    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 2, 2)
    ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 3, 2)
    ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 5, 2)

    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 0.3, 3)
    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 0.5, 3)
    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 1, 3)
    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 3, 3)
    ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 10, 3)
    ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 20, 3)

    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 0.3, 5)
    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 0.5, 5)
    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 1, 5)
    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 2, 5)
    # ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 3, 5)
    ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 5, 5)
    ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 10, 5)
    ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 20, 10)
    ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 30, 20)
    ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 50, 20)
    ohlcv = preproc.isHighLowFromPeriodBack(ohlcv, 100, 50)




    #no impact ? todo crossover with accuracy
    #ohlcv = preproc.stochrsi(ohlcv, 'close', 25)
    #ohlcv = preproc.stochrsi(ohlcv, 'low', 10)
    #ohlcv = preproc.stochrsi(ohlcv, 'high', 10)
    #ohlcv = preproc.stochrsi(ohlcv, 'close', 10)

    #ohlcv = preproc.williams(ohlcv, 'close', 25)
    #ohlcv = preproc.williams(ohlcv, 'close', 50)

    #ohlcv = preproc.ppo(ohlcv, 'close', 20, 10, 5)
    #ohlcv = preproc.ppo(ohlcv, 'close', 10, 5, 3)
    #ohlcv = preproc.ppo(ohlcv, 'close', 50, 20, 10)
    #ohlcv = preproc.ppo(ohlcv, 'low', 10, 5, 3)

    #ohlcv = preproc.pvo(ohlcv, 'close')
    #ohlcv = preproc.pvo(ohlcv, 'low')
    #ohlcv = preproc.pvo(ohlcv, 'close', 10, 5, 3)

    # ohlcv = preproc.tsi(ohlcv, 'close')
    # ohlcv = preproc.tsi(ohlcv, 'low')
    # ohlcv = preproc.tsi(ohlcv, 'high')



    #bad impact
    #ohlcv = preproc.roc(ohlcv, 'close', 10)
    # ohlcv = preproc.roc(ohlcv, 'close', 25)
    #ohlcv = preproc.roc(ohlcv, 'low', 10)
    #ohlcv = preproc.roc(ohlcv, 'high', 5)

    #ohlcv = preproc.kama(ohlcv, 'close', 25)



    #ohlcv = preproc.bbands(ohlcv, 'close', 20)
    #ohlcv = preproc.bb(ohlcv, 'close', 20)



    # Method 1: get the data by sending a dataframe
    #df = obv.get_value_df(ohlcv)
    #df = rsi.get_value_df(ohlcv, 3)
    #df = rsi.get_value_df(ohlcv, 5)
    #df = rsi.get_value_df(ohlcv, 9)
    #df = rsi.get_value_df(ohlcv, 14)
    #df = rsi.get_value_df(ohlcv, 21)


    # Method 2: get the data by sending series values
    # obv_values = obv.get_value_list(df["close"], df["volume"])

    # ohlcv = preproc.bbp(ohlcv, 'close')
    # ohlcv = preproc.bbp(ohlcv, 'high') #todo periodNumHigh
    # ohlcv = preproc.bbp(ohlcv, 'low')

    end = timeit.default_timer()
    print(f"preprocData {ohlcv.shape}) took %.2f seconds {end-start}")
    # if loadFromFile:
    #     ohlcv.to_csv('preproc_data.csv')
    #saveDataCsv(ohlcv, )

    return ohlcv