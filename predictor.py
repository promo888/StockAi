import hashlib
import time

import pandas as pd
# import ray
# ray.init(num_cpus=6)
# import modin.pandas as pd
import numpy as np
#from yahoo_fin import stock_info as si


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE as rfe
from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as dtc

#semi optimized for inverse predictions
from sklearn.svm import LinearSVC as svm
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.tree import DecisionTreeRegressor as dtr

import matplotlib.pyplot as plt
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

class StockPredictor:

    __doc__ = """
    """

    def __init__(self, ticker="^GSPC", predict_field="close", shift_n_periods=0,
                 from_index="2018-01-01", to_index="2021-12-31'",
                 periods_back_test=500, rulePct=0.7,
                 data_dir="data", models_dir="models", charts_dir="charts",
                 explore_score_field="GOOD Ratio",
                 explore_chart_axs=["pctChange_close", "rules_binary_features_sum"],
                 displayChart=True,
                 debug=True):
        self.config = self.loadConfigFromFile()
        self.db = create_engine('sqlite://', echo=False) #self.initSqlDb()
        self.ticker = ticker
        self.web_data = None
        self.inputs_data = None
        self.bkp_data = None
        self.inputs_data_binary = None
        self.inputs_data_continuous = None
        self.targets_data = None #for supervised learning
        self.start_from_index = from_index
        self.end_to_index = to_index
        self.periods_back_test = periods_back_test
        self.rulePct = rulePct
        self.look_back = shift_n_periods if shift_n_periods < 0 else 0 #todo forward
        self.look_forward = shift_n_periods if shift_n_periods > 0 else 0
        self.predict_field = predict_field #todo to change along with binary/DL rules
        self.predict_field_shift_periods = self.look_back if shift_n_periods < 0 else self.look_forward
        self.predictors = []
        self.predicted_data = []
        self.ensemble_predictions = []
        self.predictions_accuracy = []
        self.best_binary_rule = None  #long
        self.worst_binary_rule = None # short

        self.binary_combs_results = [] # id, desc, total_trades, profit_trades, loss_trades, profit_median, loss_median, profit_max, profit_median, loss_max, loss_median
        self.binary_combs_trades = []
        self.binary_combs_trades_up = []
        self.binary_combs_trades_down = []

        self.data_folder = data_dir #"data"
        self.models_folder = models_dir #"models"
        self.charts_folder = charts_dir #"charts"
        self.predictions_folder = None #"charts"
        self.explore_score_field = explore_score_field
        self.explore_chart_axs = explore_chart_axs
        self.displayChart = displayChart
        self.debug = debug

        #range current price to check switch for opposite direction/strat remained
        self.last_period_binary_range = {}
            # {"close5plus": None,  #simulate current price + 5%
            #                       "close5minus": None,
            #                       "close10plus": None,
            #                       "close10minus": None,
            #                       "close": None,  #current price
            #                       "date": None,
            #                       "ticker": None
            #                              }




        self.cprint(f"Predictor started {datetime.datetime.now()} \nParams: \n %s%s%s" %
              (
                  "_predict field: " + self.predict_field + ",",
                  "_predict_field_shift:" + str(self.predict_field_shift_periods) + ",",
                  "_RuleModel,StratModel,ActorModel,ExploratoryResults, SratResults, RuleResults byAssetsClassify, byModelClassify -> feedPredictor.. - TODO"
              ))

        # self.web_data = None
        # self.input_data = loadModel("input_data", "data")
        # self.preproc_data = loadModel("preproc_data", "data")
        # self._models_dir = models_dir
        # self.lr_predictor = loadModel("lr")
        # self.gbc_predictor = loadModel("gbs_clf")
        # self.svm_predictor = loadModel("svms_clf")
        # self.dtc_predictor = loadModel("dtclf")
        # self.dtr_predictor = loadModel("dtr")
        # self.dtr_predictor2 = loadModel("dtr_loop")

        #start explore
        #self.inputs_data = \
        #self.loadDataCsv() #(last_periods=400) #load default file when no network
        if self.inputs_data is None:
            self.getTickerFromYahoo(fromDate=self.start_from_index, toDate=self.end_to_index)
        self.startExplore()
        if self.displayChart:
            self.visBinaryRuleData(self.inputs_data, self.explore_chart_axs)

    def initSqlDb(self):
        return sqlite3.connect(":memory:")


    def queryBinSqlDb(self, table, fields, values, predict_field_kv, start_from_idx=0):
        query = f"select * from {table} where "
        cols = " AND ".join([f"{fields[x]} = {values[x]}" for x in range(len(fields))])
        start_from = f" AND id>={start_from_idx}"
        where = f" AND {predict_field_kv.split('=')[0]}={predict_field_kv.split('=')[1]}" if len(predict_field_kv) > 0 else ""
        AND = "" #f" AND pctChange_close > 10" if 1 in fields else " AND pctChange_close < -10"
        return f"{query} {cols} {start_from} {where} {AND}"



    def querySqlDb(self, sql, start_from_idx=0):
        query = sql
        start_from = f" AND id>={start_from_idx}"
        return f"{query} {start_from}"


    def startExplore(self):
        self.runPreprocData()
        self.cleanupData()
        self.preprocBinaryFeatures()
        #self.copyPreprocData()
        self.exploreBinaryFeatures(self.predict_field, self.predict_field_shift_periods, self.periods_back_test, self.rulePct)
        ##self.exploreBinaryFeatures(self.predict_field, self.predict_field_shift_periods, self.periods_back_test, 0.75)
        ##self.exploreBinaryFeatures(self.predict_field, self.predict_field_shift_periods, self.periods_back_test, 0, 0.25)
        exit(0)
        # self.savePreprocData()
        self.simulateLastPeriodBinaryFeatures()
        # self.revertPreprocData()
        # self.preprocData() #todo last element prediction+update preproc/run configs
        self.getBinaryStratDetails(score_field=self.explore_score_field)


    def cprint(self, msg):
        if self.debug:
            print(msg)

    def loadConfigFromFile(self, filename="test_config.dict"):
        with open(filename, 'r') as myfile:
            data = myfile.read()
            config = eval(data) #ast.literal_eval(data.replace("\n", ""))
            config["run"] = sorted(config["run"], key=lambda x: x["exec_id"], reverse=False) #ASC execId
            return config


    def saveDataCsv(self, data, filename, data_foler="data"):
        with open(data_foler+"/"+filename, 'w') as f:
            f.write(data.to_csv())


    def loadDataCsv(self, filename="GSPC_1990-01-01_2021-12-31.csv", data_foler="data", index_as_date=True, last_periods=0, first_periods=0): #todo getByIdDateLastN
        filepath = data_foler+"/"+filename #default
        # if index_as_date: #todo id,date,custom
        self.inputs_data = pd.read_csv(filepath, index_col=0)
        #         #self.inputs_data["DateTime"] = self.inputs_data.index
        #     #self.inputs_data.set_index("DateTime")
        #     #todo change index to id int/long incremented
        # else:
        #     self.inputs_data = pd.read_csv(filepath)  # [self.start_from_index:self.end_to_index]
        #     self.inputs_data["ID"] = self.inputs_data.index
        #     self.inputs_data.set_index("ID")
        #     # index_col = self.inputs_data.columns[0]
        #     # self.inputs_data["DateTime"] = self.inputs_data[index_col]
        #     # self.inputs_data.columns.drop(index_col))
        #     #todo remove redundant index col after renaming
        if last_periods > 0:
            self.inputs_data = self.inputs_data[-last_periods:]

        return self.inputs_data

    def getTickerFromYahoo(self, fromDate="1990-01-01", toDate="2021-12-31", interval="1d", index_as_date=True):
        startYahoo = timeit.default_timer()
        filename = "%s_%s_%s.csv" % (self.ticker.replace("^", ""), fromDate, toDate)
        self.web_data = si.get_data(self.ticker, fromDate, toDate, index_as_date, interval)
        # if index_as_date:
        #     self.inputs_data["DateTime"] = self.inputs_data.index
        #     self.inputs_data.set_index("DateTime")
        self.saveDataCsv(self.web_data, filename)
        self.inputs_data = self.web_data.copy(deep=True)
        self.cprint("Get Yahoo ticker %s took %.2f secs" % (self.ticker, timeit.default_timer() - startYahoo))


    def pctChange(self, field):
        self.inputs_data['pctChange_%s' % field] = self.inputs_data[field].pct_change() * 100
        self.inputs_data.dropna(inplace=True)


    def runPreprocData(self):
        start = time.time()
        for r in self.config["run"]:
            obj, func, params = r["object"], r["func"], r["params"]
            params = [self.__dict__[p.split(".")[1]] if 'str' in str(type(p)) and len(p.split(".")) > 1 else p for p in r["params"]]
            self.inputs_data = getattr(globals()[obj], func)(*params) #call to config's preproc functions
        self.cprint(f"Preproc data took {time.time()-start:.2f} secs")


    def copyPreprocData(self):
        self.bkp_data = self.inputs_data.copy(deep=True)


    def simulateLastPeriodBinaryFeatures(self):
        self.last_period_binary_range["ticker"] = self.ticker
        self.last_period_binary_range["last_date"] = str(self.inputs_data[-1:].index[0].date())
        self.last_period_binary_range["last_date_close"] = self.inputs_data[-1:]["close"][0]
        self.last_period_binary_range["last_date_binary_sum"] = self.inputs_data[-1:]["rules_binary_features_sum"][0]

        self.copyPreprocData()
        for p in [0.9, 0.95, 1.05, 1.1]: #simulate advance/decline close/high close/low from -10 to +10% change
            self.revertWebData()  # to revert last changes
            if p < 1: #simulate decline
                self.inputs_data[-1:]["close"] = self.inputs_data[-1:]["close"][0] * p
                self.inputs_data[-1:]["adjclose"] = self.inputs_data[-1:]["close"]
                self.inputs_data[-1:]["low"] = self.inputs_data[-1:]["close"]
            if p > 1: #simulate advance
                self.inputs_data[-1:]["close"] = self.inputs_data[-1:]["close"][0] * p
                self.inputs_data[-1:]["adjclose"] = self.inputs_data[-1:]["close"]
                self.inputs_data[-1:]["high"] = self.inputs_data[-1:]["close"]

            self.runPreprocData()
            self.preprocBinaryFeatures()
            self.last_period_binary_range[f"last_date_binary_sum_{p}"] = \
                self.inputs_data[-1:]["rules_binary_features_sum"][0]
            assert self.last_period_binary_range["last_date"] == str(self.inputs_data[-1:].index[0].date())

        self.revertPreprocData()
        self.cprint(f"\nLast Period:\n{self.last_period_binary_range}")

    def revertWebData(self):
        self.inputs_data = self.web_data.copy(deep=True)


    def revertPreprocData(self):
        self.inputs_data = self.bkp_data.copy(deep=True)

    def cleanupData(self):
        self.inputs_data.dropna(inplace=True)
        self.inputs_data = self.inputs_data[self.inputs_data.columns.drop(["ticker", "volume"])]  # todo cutParams
        self.inputs_data = self.inputs_data[
            self.inputs_data.columns.drop(list(self.inputs_data.filter(regex='diff_')))]  # remove tmp cols


    def preprocBinaryFeatures(self):
        # split continuous and binary features
        binary_data_cols = [col for col in self.inputs_data if
                            np.isin(self.inputs_data[col].dropna().unique(),
                                    [0, 1]).all()]
        continuous_data_cols = [col for col in self.inputs_data if
                                not np.isin(self.inputs_data[col].dropna().unique(),
                                            [0, 1]).all()]
        self.inputs_data_binary = self.inputs_data[binary_data_cols]  # get rid of continuous features/values
        self.inputs_data_continuous = self.inputs_data[continuous_data_cols]  # get rid of continuous features/values #todo setUniq 4 better compliance

        # range binary features if exist
        if not self.inputs_data_binary is None and self.inputs_data_binary.shape[0] > 0:
            # self.inputs_data["rules"] = {"binary_features_sum", self.inputs_data_binary.sum(axis=1)}
            self.inputs_data["rules_binary_features_sum"] = self.inputs_data_binary.sum(axis=1)

            #temp - feature correlation - reduce correlating features
            corr_data = self.inputs_data_binary.copy(deep=True)
            corr = corr_data.corr()
            # # corr.head()
            print(f"Total binary rules {self.inputs_data_binary.shape}")
            #print("removing correlating binary features")
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
                #print(f"Discorrelated binary features: {selected_columns.shape}") #selected_columns.values
                self.inputs_data_binary = self.inputs_data_binary[selected_columns]
            except:
                raise Exception(f"in removing correlating binary features")
            print(f"Removed correlating binary features:{removed_cols}")


    def chartBestUpDownRules(self,min_outputs=2):
        self.inputs_data['date'] = self.inputs_data.index
        rulesUp = sorted(
            [x for x in self.binary_combs_results if x['direction'] == "UP" and x['predicted_true'] >= min_outputs],
            key=lambda i: (i['predicted_true_pct']), reverse=True)
        if len(rulesUp) > 0:
            ruleUpDates = [pd.to_datetime(x) for x in rulesUp[0]['predicted_trade_dates']]
                # [pd.to_datetime(x, format='%d-%m-%Y').strftime('%Y-%m-%d') for x in
                #            ruleUpDates[0]['predicted_trade_dates']]
        ruleUpCloses = self.inputs_data.query(f"date in {ruleUpDates}")['close'] if len(rulesUp) > 0 else []
        if len(rulesUp) > 0:
            print(f"Best Up rule: {rulesUp[0]['ruleDesc']} predicted {rulesUp[0]['predicted_true']}/{rulesUp[0]['total_comb']} {'{:.2f}'.format(rulesUp[0]['predicted_true_pct'])}%")

        rulesDown = sorted(
            [x for x in self.binary_combs_results if x['direction'] == "DOWN" and x['predicted_true'] >= 5],
            key=lambda i: (i['predicted_true_pct']), reverse=True)
        if len(rulesDown) > 0:
            # ruleDownDates = [pd.to_datetime(x, format='%d-%m-%Y').strftime('%Y-%m-%d') for x in
            #                  ruleDownDates[0]['predicted_trade_dates']]
            ruleDownDates =  [pd.to_datetime(x) for x in rulesDown[0]['predicted_trade_dates']]
        ruleDownCloses = self.inputs_data.query(f"date in {ruleDownDates}")['close'] if len(rulesDown) > 0 else []
        if len(rulesDown) > 0:
            print(f"Best Down rule: {rulesDown[0]['ruleDesc']} predicted {rulesDown[0]['predicted_true']}/{rulesDown[0]['total_comb']} {'{:.2f}'.format(rulesDown[0]['predicted_true_pct'])}%")


#temp all dates from all rules
        # l = []
        # ll = [x['predicted_trade_dates'] for x in self.binary_combs_results if x['direction'] == 'UP']
        # for i in ll:
        #     l.extend(i)
        # ruleUpDates = list(set(l))
        # ruleUpCloses = self.inputs_data.query(f"date in {ruleUpDates}")['close']
        # l = []
        # ll = [x['predicted_trade_dates'] for x in self.binary_combs_results if x['direction'] == 'DOWN']
        # for i in ll:
        #     l.extend(i)
        # ruleDownDates = list(set(l))
        # ruleDownCloses = self.inputs_data.query(f"date in {ruleDownDates}")['close']
# ###


        plt.plot([pd.to_datetime(x, infer_datetime_format=True) for x in self.inputs_data['date']],
                 self.inputs_data['close'], marker='o', markerfacecolor='blue')

        if len(ruleUpCloses) > 0:
            plt.plot(pd.to_datetime(ruleUpDates, infer_datetime_format=True), ruleUpCloses, marker='^',
                     markerfacecolor='green')
        if len(ruleDownCloses) > 0:
            plt.plot(pd.to_datetime(ruleDownDates, infer_datetime_format=True), ruleDownCloses, marker='v',
                     markerfacecolor='red')

        plt.show()

    def exploreBinaryFeatures(self, predictField, shiftPeriods, explore_periods_back=1, rule_pct=0.7, rule_limit=1):
        #preprocBinaryFeatures / init
        #start = timer()
        #self.shiftDfData(predictField, shiftPeriods)
        last_date = self.inputs_data[-1:].index[0]
        dfTarget = self.inputs_data #self.inputs_data_binary   #.copy()
        new_predict_field = f"predict_{predictField}"
        dfTarget[new_predict_field] = dfTarget[predictField]
        dfTarget = self.shiftDfData(new_predict_field, shiftPeriods, dfTarget, inplace=True)
        dfFeatures = dfTarget.filter(regex='^is.*').columns #self.inputs_data_binary.copy()
        dfTarget.index = pd.to_datetime(dfTarget.index)
        #dfTarget.index = dfTarget.index.map(lambda x: x.strftime('%Y-%m-%d'))
        dfTarget['id'] = [i for i in range(len(dfTarget.index))]
        dfTarget['date'] = dfTarget.index
        dfTarget.to_sql('bin_rules_stats', con=self.db, if_exists='append', index=True)
        #res = self.db.execute(self.queryBinSqlDb('bin_rules_stats', list(comb), [1] * len(comb))).fetchall() #("SELECT * FROM bin_rules_stats").fetchall()
        #preproc/mine end

        #calc/runtime 1cycle-1period started - start
        start = timer()
        #dfFeatures.drop(predictField, axis='columns', inplace=True)
        total_bin_features = len(dfFeatures) #todo chunks if rules_amount > 1000
        max_indicators = 2 #3 #5 #7 #10 ##total_bin_features - 1 if total_bin_features <= 11 else 11 #todo upto 30 - bigCombsArray?
        periods = explore_periods_back  #500 #50 #200
        #rule_pct = 0.55 #0.6 #0.8 #0.5 0.75
        total_records = dfTarget.shape[0]
        assert periods < total_records
        totalUp = dfTarget.iloc[-periods:].query(f"{new_predict_field}==1").shape[0]
        totalDown = dfTarget.iloc[-periods:].query(f"{new_predict_field}==0").shape[0]
        count = 0
        print(f"Exploring {max_indicators} pairs from {dfFeatures.shape[0]} indicators with {len([x for x in combinations(dfFeatures, max_indicators+1)])} combinations")
        for i in range(1, max_indicators+1):#todo Exclude targetField assert amount=true+false -> 0/1 full coverage
            combs = combinations(dfFeatures, i)
            for comb in combs:
                queryUp = " and ".join([f"{x}==1" for x in comb]) + f" and {new_predict_field}==1"
                queryDown = " and ".join([f"{x}==0" for x in comb]) + f" and {new_predict_field}==0"
                queryStart = timer()

                #UP - todo func
                rule_UP_found_all = self.db.execute(
                    self.queryBinSqlDb('bin_rules_stats', list(comb), [1] * len(comb), "",
                                       total_records - periods)).fetchall()

                predicted_UP_true = None
                rulesPredictedUp = 0
                if len(rule_UP_found_all) > 0:
                    predicted_UP_true = self.db.execute(
                        self.queryBinSqlDb('bin_rules_stats', list(comb), [1] * len(comb), f"{new_predict_field}=1",
                                       total_records - periods)).fetchall()
                    rulesPredictedUp = len(predicted_UP_true)
                rulePctUp = False
                if not predicted_UP_true is None and rulesPredictedUp > 0:
                    rulePctUp = rulesPredictedUp / len(rule_UP_found_all) >= rule_pct  and rulesPredictedUp / len(rule_UP_found_all) <= rule_limit\
                        if len(rule_UP_found_all) > 0 and rulesPredictedUp <= len(rule_UP_found_all) else \
                            rulesPredictedUp - rulesPredictedUp - len(rule_UP_found_all) >= rule_pct


                # DOWN - todo func
                rule_DOWN_found_all = self.db.execute(
                   self.queryBinSqlDb('bin_rules_stats', list(comb), [0] * len(comb), "",
                                        total_records - periods)).fetchall()
                predicted_DOWN_true = None
                rulesPredictedDown = 0
                if len(rule_DOWN_found_all) > 0:
                    predicted_DOWN_true = self.db.execute(
                      self.queryBinSqlDb('bin_rules_stats', list(comb), [0] * len(comb), f"{new_predict_field}=0",
                                          total_records - periods)).fetchall()
                    rulesPredictedDown = len(predicted_DOWN_true)

                rulePctDown = False
                if not predicted_DOWN_true is None and rulesPredictedDown > 0:
                    rulePctDown = rulesPredictedDown / len(rule_DOWN_found_all) >= rule_pct  and  rulesPredictedDown / len(rule_DOWN_found_all) <= rule_limit\
                if len(rule_DOWN_found_all) > 0 and rulesPredictedDown <= len(rule_DOWN_found_all) else \
                    rulesPredictedDown - rulesPredictedDown - len(rule_DOWN_found_all) >= rule_pct

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
                                                 "date": last_date,
                                                 "target_field": predictField,
                                                 "ruleId": ruleIdUp,
                                                 "ruleDesc": queryUp,
                                                 "predictField": predictField,
                                                 "periods_back_test": periods,
                                                 "periods_predict": shiftPeriods * -1,
                                                 "direction": "UP",
                                                 "total": totalUp,
                                                 "total_comb": len(rule_UP_found_all),
                                                 "predicted_true": rulesPredictedUp,
                                                 "predicted_true_pct": rulesPredictedUp/len(rule_UP_found_all),
                                                 "predicted_trade_dates": [x[0] for x in rule_UP_found_all],
                                                 # f"{predictField}_20dBackPctIsUp": rule20dTrue / 20,
                                                 # f"{predictField}_10dBackPctIsUp": rule10dTrue / 10,
                                                 # f"{predictField}_5dBackPctIsUp": rule5dTrue / 5,
                                                 # f"{predictField}_50dBackPctIsDown": rule50dFalse / 50,
                                                 # f"{predictField}_20dBackPctIsDown": rule20dFalse  / 20,
                                                 # f"{predictField}_10dBackPctIsDown": rule10dFalse  / 10,
                                                 # f"{predictField}_5dBackPctIsDown": rule5dFalse  / 5
                                                 }
                    self.binary_combs_results.append(binary_comb_results)
                    #self.binary_combs_trades_up.append({'rule_id': ruleId, 'rule_desc': queryUp, 'direction': 'UP', 'trades': rule_UP_found_all})
                    #self.binary_combs_trades.append({'rule_id': ruleIdUp, 'rule_desc': queryUp, 'direction': 'UP', 'trades': rule_UP_found_all})
                    #pd.DataFrame.from_records(binary_comb_results, index='ruleId').to_sql('bin_rules_trades', con=self.db, if_exists='append', index=False)
                if (rulePctDown):
                    #continue
                    binary_comb_results = {"ticker": self.ticker,
                                                 "date": last_date,
                                                 "target_field": predictField,
                                                 "ruleId": ruleIdDown,
                                                 "ruleDesc": queryDown,
                                                 "predictField": predictField,
                                                 "periods_back_test": periods,
                                                 "periods_predict": shiftPeriods * -1,
                                                 "direction": "DOWN",
                                                 "total": totalDown,
                                                 "total_comb": len(rule_DOWN_found_all),
                                                 "predicted_true": rulesPredictedDown,
                                                 "predicted_true_pct": rulesPredictedDown/len(rule_DOWN_found_all),
                                                 "predicted_trade_dates": [x[0] for x in rule_DOWN_found_all],
                                                 # f"{predictField}_20dBackPctIsUp": rule20dTrue / 20,
                                                 # f"{predictField}_10dBackPctIsUp": rule10dTrue / 10,
                                                 # f"{predictField}_5dBackPctIsUp": rule5dTrue / 5,
                                                 # f"{predictField}_50dBackPctIsDown": rule50dFalse / 50,
                                                 # f"{predictField}_20dBackPctIsDown": rule20dFalse  / 20,
                                                 # f"{predictField}_10dBackPctIsDown": rule10dFalse  / 10,
                                                 # f"{predictField}_5dBackPctIsDown": rule5dFalse  / 5
                                                 }
                    self.binary_combs_results.append(binary_comb_results)
                    #self.binary_combs_trades_down.append({'rule_id': ruleId, 'rule_desc': queryDown, 'direction': 'DOWN', 'trades': rule_UP_found_all})
                    #self.binary_combs_trades.append({'rule_id': ruleIdUp, 'rule_desc': queryDown, 'direction': 'DOWN', 'trades': rule_DOWN_found_all})
                    #pd.DataFrame.from_records(binary_comb_results, index='ruleId').to_sql('bin_rules_trades', con=self.db, if_exists='append', index=False)
           #      #good self.inputs_data.query(query).shape[0]
           #      #bad total-good vs query bad = total query = bad query
           #      #last date self.inputs_data.query(query)[-1:].index[0]
           #      #todo where query 100d-50d-20d-5d >75% target + >75% voters + >75% assets (or 1-3 highly correlated assets)

        #print(f"calc took {timer()-start} secs")
        #exit(1) #testPerf

        self.binary_combs_results = sorted(self.binary_combs_results, key=lambda x: x["predicted_true_pct"], reverse=False)  # Sort best rules DESC
        print(f"\n\nBIN_DF ticker: {self.ticker}, RulePct predict {rule_pct*100}%, for: {periods} periods back, took: {timer()-start} secs,\
                {len(self.binary_combs_results)} valid rules found,\n{tabulate(self.binary_combs_results)}")
        print(f"calc took {timer() - start} secs")


        # self.inputs_data['date'] = self.inputs_data.index
        # # longsDfDates = [y for y in self.inputs_data['date'] if
        # #          y in [pd.to_datetime(x[0]) for x in self.binary_combs_trades_up[0]]]
        # # longsDfCloses = self.inputs_data.query(f"date in {longs}")["close"]
        # ruleDfDates = [pd.to_datetime(x[0], format='%d-%m-%Y').strftime('%Y-%m-%d') for x in self.binary_combs_trades[0]]
        #     #[y for y in self.inputs_data['date'] if y in [pd.to_datetime(x[0]) for x in self.binary_combs_trades_down[0]]]
        # ruleDfCloses = self.inputs_data.query(f"date in {ruleDfDates}")['close']
        # ##dfChart = self.inputs_data[["close"]]
        # #plt.plot('Date', 'Close', data=dfChart)
        # fig, ax = plt.subplots(figsize=(20, 15))
        # #plt.scatter(shortDfDates, shortsDfCloses, marker='v', markerfacecolor='red')
        # #dfChart.plot(ax=ax)
        # #plt.plot(longsDfDates, longsDfCloses * 1.01, marker='^', markerfacecolor='green')
        # plt.plot([pd.to_datetime(x, infer_datetime_format=True) for x in self.inputs_data['date']],
        #          self.inputs_data['close'], marker='o', markerfacecolor='blue')
        # plt.plot(pd.to_datetime(ruleDfDates, infer_datetime_format=True), ruleDfCloses, marker='v', markerfacecolor='red')
        # plt.show()
        self.chartBestUpDownRules(min_outputs=5)
        #exit(1) #testPerf


    def preprocData(self, use_only_binary_features=True):

        #self.inputs_data = helper.preprocData(self.inputs_data)
        preproc.pctChange1p(self.inputs_data, 'open')
        preproc.pctChange1p(self.inputs_data, 'close')
        preproc.pctChange1p(self.inputs_data, 'low')
        preproc.pctChange1p(self.inputs_data, 'high')

        # preproc.ema(self.inputs_data, 'close', 2)
        # preproc.ema(self.inputs_data, 'close', 3)
        # preproc.ema(self.inputs_data, 'close', 5)
        # preproc.ema(self.inputs_data, 'close', 7)
        # preproc.ema(self.inputs_data, 'close', 10)
        # preproc.ema(self.inputs_data, 'close', 20)
        preproc.ema(self.inputs_data, 'close', 50)
        preproc.ema(self.inputs_data, 'close', 100)

#
        # preproc.ema(self.inputs_data, 'high', 2)
        # preproc.ema(self.inputs_data, 'high', 3)
        # preproc.ema(self.inputs_data, 'high', 5)
        # preproc.ema(self.inputs_data, 'high', 7)
        # preproc.ema(self.inputs_data, 'high', 10)
        # preproc.ema(self.inputs_data, 'high', 20)
        preproc.ema(self.inputs_data, 'high', 50)
        preproc.ema(self.inputs_data, 'high', 100)

        # preproc.ema(self.inputs_data, 'low', 2)
        # preproc.ema(self.inputs_data, 'low', 3)
        # preproc.ema(self.inputs_data, 'low', 5)
        # preproc.ema(self.inputs_data, 'low', 7)
        # preproc.ema(self.inputs_data, 'low', 10)
        # preproc.ema(self.inputs_data, 'low', 20)
        preproc.ema(self.inputs_data, 'low', 50)
        preproc.ema(self.inputs_data, 'low', 100)

        preproc.isUp(self.inputs_data, 'high')
        # preproc.isUp(self.inputs_data, 'ema_2_high')
        # preproc.isUp(self.inputs_data, 'ema_3_high')
        # preproc.isUp(self.inputs_data, 'ema_5_high')
        # preproc.isUp(self.inputs_data, 'ema_7_high')
        # preproc.isUp(self.inputs_data, 'ema_10_high')
        # preproc.isUp(self.inputs_data, 'ema_20_high')
        preproc.isUp(self.inputs_data, 'ema_50_high')
        preproc.isUp(self.inputs_data, 'ema_100_high')

        preproc.isUp(self.inputs_data, 'low')
        # preproc.isUp(self.inputs_data, 'ema_2_low')
        # preproc.isUp(self.inputs_data, 'ema_3_low')
        # preproc.isUp(self.inputs_data, 'ema_5_low')
        # preproc.isUp(self.inputs_data, 'ema_7_low')
        # preproc.isUp(self.inputs_data, 'ema_10_low')
        # preproc.isUp(self.inputs_data, 'ema_20_low')
        preproc.isUp(self.inputs_data, 'ema_50_low')
        preproc.isUp(self.inputs_data, 'ema_100_low')
#

        preproc.isUp(self.inputs_data, 'close')
        # preproc.isUp(self.inputs_data, 'ema_2_close')
        # preproc.isUp(self.inputs_data, 'ema_3_close')
        # preproc.isUp(self.inputs_data, 'ema_5_close')
        # preproc.isUp(self.inputs_data, 'ema_7_close')
        # preproc.isUp(self.inputs_data, 'ema_10_close')
        # preproc.isUp(self.inputs_data, 'ema_20_close')
        preproc.isUp(self.inputs_data, 'ema_50_close')
        preproc.isUp(self.inputs_data, 'ema_100_close')





        ###Rule2
        # preproc.ema(self.inputs_data, 'high', 2)
        # preproc.ema(self.inputs_data, 'high', 3)
        # preproc.ema(self.inputs_data, 'high', 5)
        # preproc.ema(self.inputs_data, 'high', 10)
        # preproc.ema(self.inputs_data, 'high', 20)
        # preproc.ema(self.inputs_data, 'high', 50)
        # preproc.ema(self.inputs_data, 'high', 100)
        #
        # preproc.ema(self.inputs_data, 'low', 2)
        # preproc.ema(self.inputs_data, 'low', 3)
        # preproc.ema(self.inputs_data, 'low', 5)
        # preproc.ema(self.inputs_data, 'low', 10)
        # preproc.ema(self.inputs_data, 'low', 20)
        # preproc.ema(self.inputs_data, 'low', 50)
        # preproc.ema(self.inputs_data, 'low', 100)
        #
        # preproc.isUp(self.inputs_data, 'ema_2_high')
        # preproc.isUp(self.inputs_data, 'ema_3_high')
        # preproc.isUp(self.inputs_data, 'ema_5_high')
        # preproc.isUp(self.inputs_data, 'ema_10_high')
        # preproc.isUp(self.inputs_data, 'ema_20_high')
        # preproc.isUp(self.inputs_data, 'ema_50_high')
        # preproc.isUp(self.inputs_data, 'ema_100_high')
        #
        # preproc.isUp(self.inputs_data, 'ema_2_low')
        # preproc.isUp(self.inputs_data, 'ema_3_low')
        # preproc.isUp(self.inputs_data, 'ema_5_low')
        # preproc.isUp(self.inputs_data, 'ema_10_low')
        # preproc.isUp(self.inputs_data, 'ema_20_low')
        # preproc.isUp(self.inputs_data, 'ema_50_low')
        # preproc.isUp(self.inputs_data, 'ema_100_low')
        #
        # #Rule3
        # preproc.isPeriodHighFromBack(self.inputs_data, 2, "high")
        # preproc.isPeriodHighFromBack(self.inputs_data, 3, "high")
        # preproc.isPeriodHighFromBack(self.inputs_data, 5, "high")
        # preproc.isPeriodHighFromBack(self.inputs_data, 10, "high")
        # preproc.isPeriodHighFromBack(self.inputs_data, 20, "high")
        # preproc.isPeriodHighFromBack(self.inputs_data, 50, "high")
        # preproc.isPeriodHighFromBack(self.inputs_data, 100, "high")
        # preproc.isPeriodHighFromBack(self.inputs_data, 2, "low")
        # preproc.isPeriodHighFromBack(self.inputs_data, 3, "low")
        # preproc.isPeriodHighFromBack(self.inputs_data, 5, "low")
        # preproc.isPeriodHighFromBack(self.inputs_data, 10, "low")
        # preproc.isPeriodHighFromBack(self.inputs_data, 20, "low")
        # preproc.isPeriodHighFromBack(self.inputs_data, 50, "low")
        # preproc.isPeriodHighFromBack(self.inputs_data, 100, "low")
        #
        # preproc.isPeriodLowBack(self.inputs_data, 2, "low")
        # preproc.isPeriodLowBack(self.inputs_data, 3, "low")
        # preproc.isPeriodLowBack(self.inputs_data, 5, "low")
        # preproc.isPeriodLowBack(self.inputs_data, 10, "low")
        # preproc.isPeriodLowBack(self.inputs_data, 20, "low")
        # preproc.isPeriodLowBack(self.inputs_data, 50, "low")
        # preproc.isPeriodLowBack(self.inputs_data, 100, "low")
        # preproc.isPeriodLowBack(self.inputs_data, 2, "high")
        # preproc.isPeriodLowBack(self.inputs_data, 3, "high")
        # preproc.isPeriodLowBack(self.inputs_data, 5, "high")
        # preproc.isPeriodLowBack(self.inputs_data, 10, "high")
        # preproc.isPeriodLowBack(self.inputs_data, 20, "high")
        # preproc.isPeriodLowBack(self.inputs_data, 50, "high")
        # preproc.isPeriodLowBack(self.inputs_data, 100, "high")
        #
        # preproc.isPeriodHighFromBack(self.inputs_data, 2, "close")
        # preproc.isPeriodHighFromBack(self.inputs_data, 3, "close")
        # preproc.isPeriodHighFromBack(self.inputs_data, 5, "close")
        # preproc.isPeriodHighFromBack(self.inputs_data, 10, "close")
        # preproc.isPeriodHighFromBack(self.inputs_data, 20, "close")
        # preproc.isPeriodHighFromBack(self.inputs_data, 50, "close")
        # preproc.isPeriodHighFromBack(self.inputs_data, 100, "close")
        # preproc.isPeriodLowBack(self.inputs_data, 2, "close")
        # preproc.isPeriodLowBack(self.inputs_data, 3, "close")
        # preproc.isPeriodLowBack(self.inputs_data, 5, "close")
        # preproc.isPeriodLowBack(self.inputs_data, 10, "close")
        # preproc.isPeriodLowBack(self.inputs_data, 20, "close")
        # preproc.isPeriodLowBack(self.inputs_data, 50, "close")
        # preproc.isPeriodLowBack(self.inputs_data, 100, "close")

        # preproc.isHighLowFromPeriodBack(self.inputs_data, 0.5, 2)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 1, 2)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 2, 2)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 3, 2)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 5, 2)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 10, 2)
        #
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 0.3, 3)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 0.5, 3)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 1, 3)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 3, 3)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 3, 3)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 5, 3)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 10, 3)
        #
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 0.3, 5)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 0.5, 5)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 1, 5)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 2, 5)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 3, 5)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 5, 5)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 10, 5)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 20, 5)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 30, 5)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 50, 10)
        # preproc.isHighLowFromPeriodBack(self.inputs_data, 50, 20)

        #assign more weights for HighRulesSum - low: 1, high: 2 - to ignore same sum ambiguity
        # high_cols = self.inputs_data.filter(regex='is.*high').columns
        # for col in high_cols:
        #     self.inputs_data.loc[self.inputs_data[col] == 1, col] = 2
        #self.inputs_data.drop(columns=high_cols, inplace=True)

        #todo differentRules: sum, regression, separate..
        # low_cols = self.inputs_data.filter(regex='is.*low').columns
        # for col in low_cols:
        #     self.inputs_data.loc[self.inputs_data[col] == 1, col] = -1
        #self.inputs_data.drop(columns=low_cols, axis=1, inplace=True)


        #cleanup
        self.inputs_data.dropna(inplace=True)
        self.inputs_data = self.inputs_data[self.inputs_data.columns.drop(["ticker", "volume"])]  # todo cutParams
        self.inputs_data = self.inputs_data[self.inputs_data.columns.drop(list(self.inputs_data.filter(regex='diff_')))]  # remove tmp cols

        #split continuous and binary features
        binary_data_cols = [col for col in self.inputs_data if
                            np.isin(self.inputs_data[col].dropna().unique(),
                                    [0, 1]).all()]
        continuous_data_cols = [col for col in self.inputs_data if
                                not np.isin(self.inputs_data[col].dropna().unique(),
                                            [0, 1]).all()]
        self.inputs_data_binary = self.inputs_data[binary_data_cols]  # get rid of continuous features/values
        self.inputs_data_continuous = self.inputs_data[continuous_data_cols]  # get rid of continuous features/values #todo setUniq 4 better compliance


        # range binary features if exist
        if not self.inputs_data_binary is None and self.inputs_data_binary.shape[0] > 0:
            # self.inputs_data["rules"] = {"binary_features_sum", self.inputs_data_binary.sum(axis=1)}
            self.inputs_data["rules_binary_features_sum"] = self.inputs_data_binary.sum(axis=1)


        # assign targets for regression/learning/explore/charts
        #self.shiftData(self.predict_field, self.predict_field_shift_periods)

        # #range binary features if exist
        # if not self.inputs_data_binary is None and self.inputs_data_binary.shape[0] > 0:
        #     #self.inputs_data["rules"] = {"binary_features_sum", self.inputs_data_binary.sum(axis=1)}
        #     self.inputs_data["rules_binary_features_sum"] = self.inputs_data_binary.sum(axis=1)


    # def getLastDateBinaryRule(self):
    #     # simulate current/today/last price close
    #     # if self.inputs_data[-1:].index[0].date() == datetime.datetime.today().date():
    #     last_element = {"date": self.inputs_data[-1:].index[0].date(),
    #                     "close": self.inputs_data.iloc[-1]["close"],
    #                     "rules_binary_features_sum": self.inputs_data.iloc[-1]["rules_binary_features_sum"]
    #                     }
    #     self.bkp_data = self.inputs_data.copy(deep=True)


    def find_loc_by_date(self, df, dates):
        marks = []
        for date in dates:
            marks.append(df.index.get_loc(date))
        return marks

    def appendResults(self, content, filename="explore.txt"):
        with open(f"{filename}", "a") as f:
            f.write("\n")
            f.write(content)
            f.write("\n")

    def visBinaryRuleData(self, df_data, cols_ordered_list, last_n_records=-100):
        assert len(cols_ordered_list) >= 1
        # Data
        dates_index = df_data.index
        closes = df_data["close"]
        pctChanges = df_data[cols_ordered_list[0]] if len(cols_ordered_list) >= 1 else None
        predictions = self.inputs_data["rules_binary_features_sum"]
        df = pd.DataFrame(
            {'Date': dates_index,
             '%_Close': pctChanges, #.shift(-1)#todo shifted earlier, #like predictField #nextDay pctChange for RuleValidation visualization, spliy
             'Close': closes,
             'Prediction': predictions
            }
        )##[-last_n_records:]
        df.dropna(inplace=True)

        for i in range(1, len(cols_ordered_list)):
            df["y_values_%s" % i] = df_data[cols_ordered_list[i]]

        # multiple line plots #todo 'y_values_1', 'y_values_2', 'y_values_3',
        plt.plot('Date', '%_Close', data=df, marker='o', markerfacecolor='blue', markersize=10, color='skyblue', linewidth=2)
        df.plot.scatter('Prediction', '%_Close') #todo subplots
        df.plot.scatter('%_Close', 'Prediction')

        ##longs = df_data[["close", "rules_binary_features_sum"]].where(
        ##    df_data["rules_binary_features_sum"] > 20).dropna()
        ##df = df_data[["close"]]
        ##longs = df_data[["close"]].where(df_data["rules_binary_features_sum"] <= 2).dropna().index
        ##shorts = df_data[["close"]].where(df_data["rules_binary_features_sum"] >= 10).dropna().index
        # longs_and_shorts = pd.merge(longs, shorts, how='inner', left_index=True, right_index=True)
        #df.plot(linestyle='-', markevery=longs, marker='+', markerfacecolor='green')
        ##df.plot(linestyle='-', markevery=self.find_loc_by_date(df, longs), marker='^', markerfacecolor='green')
        ##df.plot(linestyle='-', markevery=self.find_loc_by_date(df, shorts), marker='o', markerfacecolor='red')

        # df_data["direction"] = 0
        # df_data = df_data[-100:]
        # df_data.loc[(df_data["rules_binary_features_sum"] == 0), "direction"] = 1
        # df_data.loc[(df_data["rules_binary_features_sum"] == 6), "direction"] = 2
        # colormap = np.array(['b', 'r', 'g'])
        # categories = np.array(df_data["direction"])
        # plt.scatter(df_data.index, df_data['close'], color=colormap[categories])

        #df = df[-100:]
        df["direction"] = 0
        df.loc[(df["Prediction"] == int(self.worst_binary_rule)), "direction"] = 1 #short
        df.loc[(df["Prediction"] == int(self.best_binary_rule)), "direction"] = 2 #long
        colormap = np.array(['b', 'r', 'g'])
        categories = np.array(df["direction"])
        #plt.legend(["wait", "sell", "buy"], loc="upper right")
        #df.plot.legend(["wait", "sell", "buy"], loc="upper right")
        df.plot.scatter("Date", 'Close', color=colormap[categories], title=self.ticker)
        plt.savefig(f"charts/{self.ticker}.png")

        if self.displayChart:
            plt.show()



    def shiftAndSplitData(self, shift_field="close", shift_period=0):
        if shift_period != 0:
            self.inputs_data[shift_field] = self.inputs_data[shift_field].shift(shift_period).dropna()
            self.inputs_data_binary = self.inputs_data_binary[:self.inputs_data.shape[0] - shift_period]
            if shift_field in self.inputs_data_binary.columns:
                self.inputs_data_binary[shift_field] = self.inputs_data[shift_field]
            self.inputs_data_continuous = self.inputs_data_continuous[:self.inputs_data.shape[0] - shift_period]
            if shift_field in self.inputs_data_continuous.columns:
                self.inputs_data_continuous[shift_field] = self.inputs_data[shift_field]

            self.inputs_data.dropna(inplace=True)  # cut shifted
            self.inputs_data_binary.dropna(inplace=True)
            self.inputs_data_continuous.dropna(inplace=True)

            #todo fix multiple fields/series update
            #self.targets_data = self.inputs_data[shift_field]

    def shiftDfData(self, shift_field="close", shift_period=0, dfData="default", inplace=False):
        df = self.inputs_data if dfData is "default" else dfData
        if not inplace:
            df = df.copy()
        df[shift_field] = df[shift_field].shift(shift_period)
        return df.dropna()

    def predictByAllModels(self):
        pass

    def predictByModel(self, model, predict_field, df_new_instance):
        return model.predict(df_new_instance)


    def getBinaryStratDetails(self, profit_field="isUp_close",  #"pctChange_close",
                              strat_indicator="rules_binary_features_sum",
                              score_field="MedianProfit",
                              shift_fields=["pctChange_close", "pctChange_low", "pctChange_high"],
                              shift_period=-1):
        for f in shift_fields:
            self.shiftAndSplitData(f, shift_period)
        strat_results = {}
        for n in set(self.inputs_data[strat_indicator]):
            nn = str(n)
            profits = self.inputs_data.where(self.inputs_data[strat_indicator] == n).dropna()[[profit_field]]
            strat_results[nn] = profits
            #todo Drawdown on ema
            strat_results[nn + "_MaxProfit"] = profits[profit_field].max()
            strat_results[nn + "_MaxLoss"] = profits[profit_field].min()
            strat_results[nn + "_MedianProfit"] = profits[profit_field].median()
            strat_results[nn + "_AverageProfit"] = profits[profit_field].mean()
            strat_results[nn + "_TotalProfit"] = profits[profit_field].sum()

            strat_results[nn + "_TradesCount"] = profits[profit_field].count()
            best_profits = profits.where(profits[profit_field] > 0).dropna()
            strat_results[nn + "_BestTradesCount"] = best_profits.count()[0]
            strat_results[nn + "_BestTradesTotalProfit"] = best_profits.sum()[0]
            strat_results[nn + "_BestTradesMaxProfit"] = best_profits.max()[0]
            strat_results[nn + "_BestTradesMinProfit"] = best_profits.min()[0]
            strat_results[nn + "_BestTradesAvgProfit"] = best_profits.mean()[0]
            strat_results[nn + "_BestTradesMedianProfit"] = best_profits.median()[0]
            worst_profits = profits.where(profits[profit_field] < 0).dropna()
            strat_results[nn + "_WorstTradesCount"] = worst_profits.count()[0]
            strat_results[nn + "_WorstTradesTotalProfit"] = worst_profits.sum()[0]
            strat_results[nn + "_WorstTradesMaxProfit"] = worst_profits.max()[0]
            strat_results[nn + "_WorstTradesMinProfit"] = worst_profits.min()[0]
            strat_results[nn + "_WorstTradesAvgProfit"] = worst_profits.mean()[0]
            strat_results[nn + "_WorstTradesMedianProfit"] = worst_profits.median()[0]
            strat_results[nn + "_BAD Ratio"] = worst_profits.count()[0] / profits[profit_field].count() * 100  if worst_profits.count()[0] > 0 else 0
            strat_results[nn + "_GOOD Ratio"] = best_profits.count()[0] / profits[profit_field].count() * 100 if best_profits.count()[0] > 0 else 0
            #self.cprint(f"{nn} StratDetails:\n {strat_results}\n")
            #todo: GooDRatio combined = good_ratio/total_good_trades + good_ratio/total_bad_trades - bad_ratio/good_trades - bad_ratio/bad_trades
            #todo add bad_ratio<=0.2 rule to GoodRatio combined

        score_fields = [i for i in strat_results.keys() if score_field in i]
        maxMedianProfit = -1000000
        maxMedianProfitIndicator = "None"
        minMedianProfit = 1000000
        minMedianProfitIndicator = "None"
        for k in score_fields:
            if strat_results[k] > maxMedianProfit:
                maxMedianProfit = strat_results[k]
                maxMedianProfitIndicator = k
                self.best_binary_rule = k.split("_")[0]
            if strat_results[k] < minMedianProfit or strat_results[k] == 0:
                minMedianProfit = strat_results[k]
                minMedianProfitIndicator = k
                self.worst_binary_rule = k.split("_")[0]
        best_advance = strat_results[maxMedianProfitIndicator.split("_")[0]]
        best_advance_total = best_advance.count()[0]
        best_advance_min = best_advance.min()[0]
        best_advance_max = best_advance.max()[0]
        best_advance_avg = best_advance.mean()[0]
        best_advance_median = best_advance.median()[0]
        worst_advance = strat_results[minMedianProfitIndicator.split("_")[0]]
        worst_advance_total = worst_advance.count()[0]
        worst_advance_min = worst_advance.min()[0]
        worst_advance_max = worst_advance.max()[0]
        worst_advance_avg = worst_advance.mean()[0]
        worst_advance_median = worst_advance.median()[0]
        start_date = self.inputs_data[0:1].index[0]
        end_date = self.inputs_data[-1:].index[0]
        self.cprint(f"Ticker: {self.ticker}, StartDate:{start_date}, EndDate: {end_date}\n predict_field: {self.predict_field}, predict_field_shift:{self.predict_field_shift_periods} \
              Strategy - strat_indicator: {strat_indicator}, profit_field:{profit_field}, score_field: {score_field}\n \
              maxProfitIndicator:{maxMedianProfitIndicator}:\n \
              maxProfitGoodLongRatio:{'%.2f' % maxMedianProfit}%\n \
              Total Long Trades: {best_advance_total}\n \
              minProfitIndicator:{minMedianProfitIndicator}:\n \
              minProfitBadRatio:{'%.2f' % minMedianProfit}%\n \
              Total Short Trades: {worst_advance_total}\n")
        #self.cprint("StratDetails", strat_results)

        df_strat_results = pd.DataFrame()
        for k in strat_results:
            df_strat_results[k] = [strat_results[k]]

        best_strat_filter = maxMedianProfitIndicator.split('_')[0]
        worst_strat_filter = minMedianProfitIndicator.split('_')[0]
        #trades ommited in filter by + '_'
        best_trades =tabulate(df_strat_results.filter(regex=best_strat_filter+'_'), headers='keys')
        worst_trades = tabulate(df_strat_results.filter(regex=worst_strat_filter+'_'), headers='keys')

        best_strat = f"\nBest Strat Long %:\n{best_trades}\n"
        worst_strat = f"\nWorst Strat Short %:\n{worst_trades}\n"
        self.cprint(best_strat)
        self.cprint(worst_strat)
        self.appendResults(f"{self.ticker}_{start_date}_{end_date} predict:{self.predict_field} shift:{self.predict_field_shift_periods}")
        self.appendResults(best_strat)
        self.appendResults(worst_strat)

    # ticker="^GSPC" #TDOC #MBRX #WISH #JPM
    #\ #isUp_close pctChange_close #isUp_close #todo: isUp_ema_3_high in preproc
voter1 = StockPredictor(ticker="^GSPC", predict_field="isUp_close",
                        shift_n_periods=-1,  periods_back_test=100, rulePct=0, displayChart=True, debug=True)#0.75 bc 3/4 or or 80 4/5 ?0.66 2/3
print("d")

#tickers = ["AI", "FSLY", "TDOC", "JPM", "KGC", "GDX", "NET", "^GSPC", "^VIX", "^DJI", "GLD"]
# for t in tickers:
#     StockPredictor(ticker=t, predict_field="isUp_close", shift_n_periods=-1, displayChart=False)

##voter1.getTickerFromYahoo()
#voter1.loadDataCsv(last_periods=400)
#voter1.preprocData()
#voter1.getStratExploreResults(score_field="GOOD Ratio")
#voter1.visData(voter1.inputs_data, ["pctChange_close", "rules_binary_features_sum"])
#exit(0)