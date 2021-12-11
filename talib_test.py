import ta  # as talib
# import pandas_datareader.data as web
from sklearn.svm import LinearSVC
from yahoo_fin import stock_info as si
import pandas as pd
import numpy as np
# from talib import RSI, BBANDS
import preproc
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from ta.volatility import BollingerBands as BBANDS #deault 20
from ta.momentum import rsi as RSI #70/30 defaut; 75/25 in trend, 20/80-85 entry
import matplotlib.pyplot as plt
from helper import *

from predictor import StockPredictor
pred = StockPredictor("isUp_adjclose", -1)
#exit()

from sklearn.tree import DecisionTreeClassifier as dtc # Import Decision Tree Classifier
#from sklearn.model_selection import train_test_split  # Import train_test_split function
#from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

startTest = timeit.default_timer()

DATA_FROM_DATE = '2018-01-01' #'2015-01-01'#'2007-01-01'
DATA_TO_DATE = '2021-12-31'
TAKE_PROFIT_PCT = 3 #5
STOP_LOSS_PCT = -3 #-2 #todo checkOpen - adjust close or open-1% tuning?
PREDICT_FROM_FIELD = "5pctChange-high_5periods_back" #"isUp_adjclose"  #"5pctChange-high_5periods_back" #"5pctChange-low_5periods_back" # "isUp_adjclose" # "0.3pctChange-low_2periods_back" # "isUp_adjclose" #'5pctChange_low_2periods_back'##"isUp_adjclose" #
PREDICT_FROM_FIELD_SHIFT = -5 #-5 #-2 #-1
FORWARD_TO = 5 #5
BACK_TO = -5 #-5
PREDICT_FIELD = PREDICT_FROM_FIELD #"isUp_adjclose_tmrw"
SYMBOL = "^GSPC"#"gsx" #"^GSPC"#"mdb"#"gsx" #"kgc"#"MDB" #"GSX" #"GDX" #KGC "MDB" #"NET"#'TLSA' #'FSLY'#'PLTR' #'FSLY'#
NUM_FEATURES_TRAINING = 33 #50 #33#33 #50   #33b   #33 # 20-30-50
NUM_PERIODS_TRAINING = 200 #15 #200#13#13#200 #11b #11 20-30-50-100
NUM_PERIODS_PREDICT = 1 #2 #3 #30#20
NUM_LAST_PERIODS_VALIDATION = 1
print('Ticker: ' + SYMBOL)
#todo train(startFrom-Period% (-100) - endTo-Period% (0))),
# ,classify/predict(fromToPeriod)
# ,rotate/loop accuract for predictions for a recordset with a running window
# ,Shuffle predictions for unseed data (forward window), montecarlo
# ,Train patterns + optimize fields/features (loopLearn) in RunTime
# , Optimize/Fit Params (classification > 75-80, prediction(8/10) MIN for 100 assets * 100 last periods
# , Dummy reocurring correlations Research



def bbp(price):
    up, mid, low = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    bbp = (price['adjclose'] - low) / (up - low)
    return bbp


startYahoo = timeit.default_timer()
max_holding = 100
data = getDataFromYahoo(SYMBOL, DATA_FROM_DATE, DATA_TO_DATE) #si.get_data(SYMBOL, DATA_FROM_DATE, DATA_TO_DATE)
print("Get Yahoo ticker took %.2f secs" % (timeit.default_timer() - startYahoo))
# web.DataReader(name=symbol, data_source='yahoo', start=start, end=end)
# print(price.shape, price.keys)
#price = price.iloc[::-1] # reverse dates - sort last=0 idx
data.dropna(inplace=True)




data = preprocData(data)
#data.select_dtypes(include="bool").astype(int, copy=False) #Convert booleans to int
data = data[data.columns.drop(list(data.filter(regex='diff_')))] #remove tmp cols
binary_data_cols = [col for col in data if np.isin(data[col].dropna().unique(), [0, 1]).all()]
data_binary = data[binary_data_cols] #get rid of continuos features
# close = data[PREDICT_FROM_FIELD].values
# print(help(BBANDS))
# bb = BBANDS(price['adjclose'], 14) #(close)  # , timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
# up, mid, low = bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
# rsi = RSI(price['adjclose']) #(close) #, timeperiod=14)
# print("RSI (first 10 elements)\n", rsi[14:24])
#
# bbpp = bb.bollinger_pband()
# price['BB_up'] = up
# price['BB_low'] = low
# price['BB_mid'] = mid
# price["BBP"] = bbpp
# price['RSI'] = rsi

data.dropna(inplace=True)
def visData(df_data, *df_features):
    f_len = len(df_features)
    #fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, sharex=True, figsize=(12, 8))
    # ax0.plot(data.index, data['adjclose'], label='adjclose')
    # ax0.set_xlabel('Date')
    # ax0.set_ylabel('adjclose')
    # ax0.grid()
    # for day, holding in holdings.iterrows():
    #     order = holding['Order']
    #     if order > 0:
    #         ax0.scatter(x=day, y=data.loc[day, 'adjclose'], color='green')
    #     elif order < 0:
    #         ax0.scatter(x=day, y=data.loc[day, 'adjclose'], color='red')

    # ax1.plot(price.index, price['RSI'], label='RSI')
    #ax1.fill_between(data.index, y1=30, y2=70, color='#adccff')
    #ax1.set_xlabel('Date')
    # ax1.set_ylabel('RSI')
    #ax1.grid()

    # ax2.plot(price.index, price['BB_up'], label='BB_up')
    # ax2.plot(data.index, data['adjclose'], label='AdjClose')
    # ax0.plot(data.index, data['high'], label='High', color='green')
    # ax0.plot(data.index, data['low'], label='Low', color='red')
    # ax2.plot(price.index, price['BB_low'], label='BB_low')
    # ax2.fill_between(price.index, y1=price['BB_low'], y2=price['BB_up'], color='#adccff')
    #ax2.set_xlabel('Date')
    # ax2.set_ylabel('Bollinger Bands')
    #ax2.grid()

    # Data
    df = pd.DataFrame(
        {'x_values': df_data.index,
         'y1_values': df_data[PREDICT_FROM_FIELD],
         #'y2_values': np.random.randn(10) + range(1, 11),
         #'y3_values': np.random.randn(10) + range(11, 21)
         })[-100:]

    # multiple line plots
    plt.plot('x_values', 'y1_values', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

    for x in df_features:
        pass

    #fig.tight_layout()
    plt.legend()
    plt.show()

#print([x for x in data.columns.values if "pctChange-" in x])
vis_features = [x for x in data.columns.values if "pctChange-" in x]
#visData(data, vis_features)
#exit(0)
########preproc end


def shift(df_data, field, period):
    df_data[field] = df_data[field].shift(period)
    df_data.dropna(inplace=True) #cut shifted

    return df_data[field]

#preproc with features set
data.drop('ticker', axis='columns', inplace=True) #get rid of the labels strings and dates if required
target = pd.DataFrame()
target[PREDICT_FIELD] = shift(data_binary, PREDICT_FROM_FIELD, PREDICT_FROM_FIELD_SHIFT) #data["isUp_adjclose"].shift(-1)
target = target[-data.shape[0]:] #len of preproc data Last N
#todo -shift cut df or dropna - test dates #target = target[-data.shape[0]:] #len of preproc data Last N
# data_binary = data[:PREDICT_FROM_FIELD_SHIFT]
# target = target[:PREDICT_FROM_FIELD_SHIFT] #cut shifted back/forward
data_binary = data_binary[-data.shape[0]:]
last_element = data_binary[-1:]
#data = data[:-1] # get rid of last row target variable #todo cutLast(data, targets...)
data.dropna(inplace=True)
target.dropna(inplace=True)
data_binary.dropna(inplace=True)
#target = pd.DataFrame(data)

#data.drop(predict_field, axis='columns', inplace=True)
#data_copy = data.copy()

predict_last_periods = NUM_PERIODS_PREDICT #20
learning_set = data_binary[-NUM_PERIODS_TRAINING-predict_last_periods:-predict_last_periods]
learning_set_target = target[PREDICT_FIELD][-NUM_PERIODS_TRAINING - predict_last_periods:-predict_last_periods]
validation_set = data_binary[-predict_last_periods:]
validation_set_target = target[PREDICT_FIELD][-predict_last_periods:]

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import timeit

#All periods data - last period prediction as overall? last 10d as a validation set 1-10% of total data?
XX = data_binary[:-NUM_LAST_PERIODS_VALIDATION]  #train set inputs price
YY = target[PREDICT_FIELD][:-NUM_LAST_PERIODS_VALIDATION] #train set output/target target
XX_test = data_binary[-NUM_LAST_PERIODS_VALIDATION:]  #validation set inputs price
YY_test = target[PREDICT_FIELD][-NUM_LAST_PERIODS_VALIDATION:] #validation set outputs to compare with predictions target


#last period/batch
X = data_binary[-NUM_PERIODS_TRAINING-predict_last_periods:-predict_last_periods]  #train set inputs price
Y = target[PREDICT_FIELD][-NUM_PERIODS_TRAINING - predict_last_periods:-predict_last_periods] #train set output/target target
X_test = data_binary[-predict_last_periods:]  #validation set inputs price
Y_test = target[PREDICT_FIELD][-predict_last_periods:] #validation set outputs to compare with predictions target

#scale inputs - targets are 0,1 bool values
#scaler = preprocessing.StandardScaler()
# X.apply(lambda x: StandardScaler().fit_transform(x))
# X_test(lambda x: StandardScaler().fit_transform(x))
# X = preprocessing.StandardScaler().fit(X)
# X_test = preprocessing.StandardScaler().fit(X_test)

#target.drop(predict_field, axis='columns', inplace=True) #tune without trainer? later stage
#target = data #get rid of expected result feature
#confusionMatrix, learn sequence, compare each period sequence with past + learn Y_test sequences

#Classifiers
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# # Feature extraction
# model = LogisticRegression()
# rfe = RFE(model, 3) #3 #10
# lr_rfe_model = rfe.fit(learning_set, learning_set_target)#learning_set[predict_field]) #target[predict_field][:-predict_last_periods]) #learn data
# saveModel(rfe, "rfe") # lr_rfe_model = loadModel("rfe")
# lr_model = model.fit(learning_set, learning_set_target)# loadModel("lr") # learning_set[predict_field]) #target[predict_field][:-predict_last_periods]) #learn data
# saveModel(lr_model, "lr")
# # exit()
#
#
#
#
# #lr_model = loadModel("lr")
# print("Num Periods Training: %s, Num Period Validations: %s" % (len(learning_set), len(validation_set)))
# print("Num Features: %s" % (lr_rfe_model.n_features_))
# #print("Selected Features: %s" % (fit.support_))
# print("Feature Ranking: %s" % (lr_rfe_model.ranking_))
# print("Selected Features Ranking: %s" % ([data.columns.array[idx] for idx in lr_rfe_model.ranking_]))
# print("Total features: ", max(lr_rfe_model.ranking_)) #len(price.columns)
# print(data.head())
# print(data.tail())
# best_features_rfe = set([data.columns.array[idx] for idx in lr_rfe_model.ranking_][:NUM_FEATURES_TRAINING]) #20-30-50
# data = data[set(best_features_rfe)] #select best N features
# print("RFE Selected Features:", best_features_rfe)


#
# import xgboost
# from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_val_score
# X = XX#data#learning_set
# Y = YY#target#learning_set_target
# # CV model
# model = xgboost.XGBClassifier()
# kfold = KFold(n_splits=100, random_state=7)#StratifiedKFold(n_splits=100, random_state=7) #KFold(n_splits=100, random_state=7)
# results = cross_val_score(model, X, Y, cv=kfold)
# print("XGBOOST Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#https://scikit-learn.org/stable/modules/preprocessing.html
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#todo to continue
#X, y = make_classification(random_state=42)
#X = preprocessing.StandardScaler().fit(data[:-1])
#y = preprocessing.StandardScaler().fit(target[predict_field])
# #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# X_train = X
# y_train = Y
#pipe = make_pipeline(StandardScaler(), LogisticRegression())
#pipe.fit(X_train, y_train)  # apply scaling on training data
# Pipeline(steps=[('standardscaler', StandardScaler()),
#                 ('logisticregression', LogisticRegression())])
inps = data_binary[:PREDICT_FROM_FIELD_SHIFT] #[:-1] #avoid last row without a target
trgts = target[PREDICT_FIELD][:PREDICT_FROM_FIELD_SHIFT]    #-1 saved for nextDay predictions     #[:-1]
svms_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])
gbs_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("standardscaler", LogisticRegression()),
])
##svms_clf.fit(inps[:-20], trgts[:-20].values.ravel())
svms_clf.fit(inps, trgts.values.ravel())
saveModel(svms_clf, "svms_clf") #todo kmeans
svms_clf = loadModel("svms_clf")
##gbs_clf.fit(inps[:-20], trgts[:-20].values.ravel())
gbs_clf.fit(inps, trgts.values.ravel())
saveModel(gbs_clf, "gbs_clf")
gbs_clf = loadModel("gbs_clf")
print(type(svms_clf))
print("svms_clf.score(%s inps,%s trgts) " % (len(inps), len(trgts)), svms_clf.score(inps, trgts))
print("gbs_clf.score(%s inps,%s trgts)"  % (len(inps), len(trgts)), gbs_clf.score(inps, trgts))
#print('pipe.score: ', pipe.score(X_test, y_test))  # apply scaling on testing data, without leaking training data.
#gbs_clf.predict(trgts[:-20])

#exit(0)

# from sklearn.svm import SVR
# svm_poly_reg = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
# svm_poly_reg.fit(XX[-120:-20], YY[-120:-20])
# prediction = svm_poly_reg.predict(XX[-20:])

# from sklearn.svm import LinearSVR
# svm_reg = LinearSVR(epsilon=1.5)
# svm_reg.fit(X, y)


# from sklearn.datasets import make_moons
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import GradientBoostingClassifier as gbc
#
# #fit as a hidden param? - LONG computation
# start = timeit.default_timer()
# polynomial_svm_clf = Pipeline([
#  ("poly_features", PolynomialFeatures(degree=3)), #3
#  ("scaler", StandardScaler()),
#  #("svm_clf", LinearSVC(C=10, loss="hinge"))
#     ("gbc_clf", gbc())
# ])
# polynomial_svm_clf.fit(XX[-120:-20], YY[-120:-20])
# prediction = polynomial_svm_clf.predict(XX[-20:])
# end = timeit.default_timer()
#
#print("PolynomialF accuracy: ", accuracy_score(YY[-20:], prediction), " time = ", end - start)
# #print(confusion_matrix(Y_test, prediction))
# print("\n")


# algorithm 2 ------------------------------------------------------------------
#print(" Random Forest ... ")

DATA_FROM_DATE = timeit.default_timer()
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
# rf_model = classifier.fit(learning_set, learning_set_target.values.ravel()) #, target[predict_field][:-predict_last_periods]) #learn (X, Y)
# prediction = rf_model.predict(validation_set) #(validation_set) #Y_test
# DATA_TO_DATE = timeit.default_timer()
#
# print("%s events accuracy = " % len(prediction), accuracy_score(Y_test, prediction), " time = ", DATA_TO_DATE - DATA_FROM_DATE) #Y_test
# print(confusion_matrix(Y_test, prediction)) #Y_test
# print(Y_test.tail()) #Y_test.tail()
# print("\n")

# # algorithm 3 ------------------------------------------------------------------
print(" Gradient Boosting ... ")

DATA_FROM_DATE = timeit.default_timer()
from sklearn.ensemble import GradientBoostingClassifier as gbc
classifier = gbc()
gbc_model = classifier.fit(XX, YY.values.ravel())
saveModel(classifier, "gbs_clf") #
gbc_model = loadModel("gbs_clf")
#XX_test = X_test.drop(columns=predict_field, inplace=True) #test
#######XX_test = XX_test[best_features_rfe]
prediction = gbc_model.predict(XX_test) #gbc_model.predict(XX_test)
##gbc_model.predict(last_element) #tmrw
DATA_TO_DATE = timeit.default_timer()
#classifier.predict(X_test)

# gbc_model2 = classifier.fit(target[:-50], Y[:-50])
# prediction2= gbc_model.predict(target[-50:])

#######XX = XX[best_features_rfe]
#XX_test = XX_test[best_features_rfe] #???
print(" Train accuracy = ", accuracy_score(YY_test, prediction), " time = ", DATA_TO_DATE - DATA_FROM_DATE)
# print(' classifier.score(X, Y) - TrainSet %s: ' % len(XX), classifier.score(XX, YY))
# print(' classifier.score(X_test, Y_test) - ValidationSet: ', classifier.score(XX_test, YY_test))
print(' classifier.score(X, Y) - TrainSet %s: ' % len(XX), gbc_model.score(XX, YY))
print(' classifier.score(X_test, Y_test) - ValidationSet: ', gbc_model.score(XX_test, YY_test))

#print(confusion_matrix(Y_test, prediction))
print("YY_test\n", YY_test.index, "\n", YY_test, '\nprediction\n', prediction)
##classifier.predict(last_element)
print("\n")


# print(" Decision tree classifier ... ")
# tree_classifier = dtc(max_depth=3)
#dtc_model = tree_classifier.fit(XX, YY.values.ravel())
#saveModel(tree_classifier, "dt_clf")
# dtc_model = loadModel("dt_clf")
# prediction = dtc_model.predict(XX_test)
# ##dtc_model.predict(last_element) #tmrw
# print(" ValidationSet accuracy = ", accuracy_score(YY_test, prediction), " time = ", DATA_TO_DATE - DATA_FROM_DATE)
# print(' DecisionTree classifier.score(X, Y) - TrainSet %s: ' % len(XX), dtc_model.score(XX, YY))
# print(' DecisionTree classifier.score(X_test, Y_test) - ValidationSet: ', dtc_model.score(XX_test, YY_test))
# print("\n")

from sklearn.tree import DecisionTreeRegressor
print(" Decision treeRegressor classifier ... ")
tree_reg_clf = DecisionTreeRegressor(max_depth=2)
dtr_model = tree_reg_clf.fit(XX, YY.values.ravel())
saveModel(tree_reg_clf, "dtr")
dtr_model = loadModel("dtr")
#print("DecisionTreeRegressor Features importances: " + tree_reg_clf.feature_importances_) #todo
prediction = dtr_model.predict(XX_test)
print(" ValidationSet accuracy = ", "\nYY_test:\n", YY_test, "\nprediction:\n", prediction)
print(' TreeReg classifier.score(X, Y) - TrainSet %s: ' % len(XX), dtr_model.score(XX, YY))
print(' TreeReg classifier.score(X_test, Y_test) - ValidationSet: ', dtr_model.score(XX_test, YY_test))
print("\n")
#
# from sklearn.tree import export_graphviz
# export_graphviz(
# tree_classifier,
# out_file="ticker_tree.dot",
# feature_names=best_features_rfe,
# class_names=YY,
# rounded=True,
# filled=True
# )




#exit(0)
#
# # algorithm 4 ------------------------------------------------------------------
print(" SVM ... ")

DATA_FROM_DATE = timeit.default_timer()
from sklearn import svm
classifier = svm.SVC()
svc_model = classifier.fit(XX, YY.values.ravel())
saveModel(classifier, "svc_clf")
prediction = svc_model.predict(XX_test)
DATA_TO_DATE = timeit.default_timer()

print(" ValidationSet accuracy = ", accuracy_score(YY_test, prediction), " time = ", DATA_TO_DATE - DATA_FROM_DATE)
print(' classifier.score(X, Y) - TrainSet: ', classifier.score(XX, YY))
print(' classifier.score(X_test, Y_test) - ValidationSet: ', classifier.score(XX_test, YY_test))
#print(confusion_matrix(Y_test, prediction))
print("Y_test\n", YY_test, '\nprediction\n', prediction)
print("\n")


history_results = {}
test_results = {}
signals = []
def getHistoricalClassAccuracy(history_data, history_targets, num_learn_periods, num_predict_periods=1, algo_type="undefined_test_type"):
    classifierSvm = svm.SVC() #loadModel("svms_clf")#
    classifierGbc = gbc() #loadModel("gbs_clf")#
    classifierDtr = dtc() #loadModel("dtr") #
    assert len(history_data) == len(history_targets)
    total_predictions = len(history_data) - 1
    test_results["Classifier"] = {"algo_type": "CompoundClassifier",
                                        "good_predictions_percent": 0,
                                        "learn_periods": num_learn_periods,
                                        "predict_next_periods": num_predict_periods,
                                        "total_predictions": total_predictions - num_learn_periods,
                                        "svm_predictions": [],
                                        "gbc_predictions": [],
                                        "gbc_scaler_predictions": [],
                                        "dtr_predictions": [],
                                        "svm_good_amount": 0,
                                        "gbc_good_amount": 0,
                                        "gbc_scaler_good_amount": 0,
                                        "compound_good_amount": 0,
                                        "compound_bad_amount": 0,
                                        "trades": [],
                                        "dates": [] #prediction for day with -shift as +n forward prediction
                                        }


    good_ratio = 0
    i = num_learn_periods #check from the end (latest data dates)
    test_results["Classifier"]["dates"] = history_data[num_learn_periods:].index.values
    binary_data_cols = [col for col in history_data if np.isin(history_data[col].dropna().unique(), [0, 1]).all()]
    for i in range(total_predictions-num_learn_periods+PREDICT_FROM_FIELD_SHIFT): #total predictions up to -2d from thr last record
        data_binary = history_data[binary_data_cols]  # get rid of continuos features
        learning_data = data_binary[i:i+num_learn_periods]
        learning_targets_data = history_targets[i:i+num_learn_periods]
        predict_data = data_binary[i+num_learn_periods:i+num_learn_periods+num_predict_periods]
        predict_target = history_targets[i+num_learn_periods:i+num_learn_periods+num_predict_periods]

        svm_model = classifierSvm.fit(learning_data, learning_targets_data.values.ravel())
        gbc_model = classifierGbc.fit(learning_data, learning_targets_data.values.ravel())
        predictionSvc = svm_model.predict(predict_data)
        predictionGbc = gbc_model.predict(predict_data)
        real_outputs = pd.array(predict_target[PREDICT_FIELD])

       # dtc_model = tree_classifier.fit(learning_data, learning_targets_data.values.ravel())
       # predictionDtc = dtc_model.predict(predict_data)
       # dtc_prediction_accuracy = accuracy_score(real_outputs, predictionDtc)

        dtr_model = classifierDtr.fit(learning_data, learning_targets_data.values.ravel())
       ##saveModel(classifierDtr, "dtr_loop")
       #dtr_model = loadModel("dtr")
        predictionDtr = dtr_model.predict(predict_data)
        dtc_prediction_accuracy = accuracy_score(real_outputs, predictionDtr)
        test_results["Classifier"]["dtr_predictions"].append(predictionDtr[0])

        svc_prediction_accuracy = accuracy_score(real_outputs, predictionSvc)
        test_results["Classifier"]["svm_predictions"].append(predictionSvc[0])
        if svc_prediction_accuracy > 0.8:
            test_results["Classifier"]["svm_good_amount"] += 1
        gbc_prediction_accuracy = accuracy_score(real_outputs, predictionGbc)
        test_results["Classifier"]["gbc_predictions"].append(predictionGbc[0])

       # gbc_scaler_model = gbs_clf.fit(learning_data, learning_targets_data.values.ravel())
       # predictionGbc_scaler = gbc_model.predict(predict_data)
       # gbc_scaler_prediction_accuracy = accuracy_score(real_outputs, predictionGbc_scaler)
       # if gbc_scaler_prediction_accuracy > 0.8:
       #     test_results["Classifier"]["gbc_scaler_good_amount"] += 1


        if gbc_prediction_accuracy > 0.8: #1 for logit
            test_results["Classifier"]["gbc_good_amount"] += 1

       ##predictionDtr[0] < predictionGbc and predictionGbc == predictionSvc:
        if predictionGbc == predictionSvc:# == predictionDtc: #== predictionGbc_scaler:
            test_results["Classifier"]["compound_good_amount"] += 1 #compound_bad_amount
           # dtr reversed results
           # test_results["Classifier"]["trades"].append(
           #     1 if (predictionDtr[0] < predictionGbc and predictionGbc == predictionSvc) else 0)

           #gbc results
            test_results["Classifier"]["trades"].append(predictionGbc)
               #1 if predictionGbc == predictionSvc else 0)

           #todo for/by high/low
           #inverse_prediction_high = "buy" if predictionDtr[0] == 0 else "sell"

    #compound best 4now predictionGbc == predictionSvc:
    #compound worst4now predictionDtr[0] != predictionGbc and predictionGbc == predictionSvc:

#todo All classifiers/networks 80% confirm +bktest by 100 periods last 5y aat least 50assets+correlations
    test_results["Classifier"]["good_predictions_percent"] = "%.2f%%" % (test_results["Classifier"]["compound_good_amount"] / test_results["Classifier"]["total_predictions"] * 100)
    print("Test Results:\n", test_results)

    predicted_data = history_data[-len(test_results["Classifier"]["dtr_predictions"]):]
    ##predicted_data["prediction"] = test_results["Classifier"]["dtr_predictions"]
    predicted_data["prediction"] = test_results["Classifier"]["gbc_predictions"]

    for i in range(len(predicted_data) + BACK_TO ): #-6 #todo shif_back BUG
        trade = {}
        max_period_profit_long = pctChange(predicted_data["adjclose"][i:i+1][0], max(predicted_data["high"][i: i + FORWARD_TO]), "long")
        max_period_loss_long = pctChange(predicted_data["adjclose"][i:i+1][0], min(predicted_data["low"][i:i + FORWARD_TO]), "long")
        max_period_profit_short = pctChange(predicted_data["adjclose"][i:i+1][0], min(predicted_data["low"][i:i + FORWARD_TO]), "short")
        max_period_loss_short = pctChange(predicted_data["adjclose"][i:i+1][0], max(predicted_data["high"][i: i + FORWARD_TO]), "short")
        last_period_indirection_close_long = pctChange(predicted_data["adjclose"][i:i+1][0], predicted_data["adjclose"][i+FORWARD_TO-1: i+FORWARD_TO][0], "long")
        last_period_indirection_close_short = pctChange(predicted_data["adjclose"][i:i+1][0], predicted_data["adjclose"][i+FORWARD_TO-1: i+FORWARD_TO][0], "short")
        if predicted_data["prediction"][i] == 1: #todo 1 - bug for not inveted#inverted? #0: #long (inverted for DTR)
            trade["date"] = predicted_data.index[i:i + 1][0]
            trade["direction"] = "long"
            trade["trade_pl"] = TAKE_PROFIT_PCT if max_period_profit_long >= TAKE_PROFIT_PCT and max_period_loss_long > STOP_LOSS_PCT else (STOP_LOSS_PCT if max_period_loss_long <= STOP_LOSS_PCT else 0) ##last_period_indirection_close_long) #last_period_indirection_close_long  # ? 5->percent var
            trade["trade_max_profit"] = max_period_profit_long
            trade["trade_max_loss"] = max_period_loss_long
        else: #short
            trade["date"] = predicted_data.index[i:i+1][0]
            trade["direction"] = "short"
            trade["trade_pl"] = TAKE_PROFIT_PCT if max_period_profit_short >= TAKE_PROFIT_PCT and max_period_loss_short > STOP_LOSS_PCT else (STOP_LOSS_PCT if max_period_loss_short <= STOP_LOSS_PCT else 0) #last_period_indirection_close_short)  #last_period_indirection_close_short  # ? 5->percent var
            trade["trade_max_profit"] = max_period_profit_short
            trade["trade_max_loss"] = max_period_loss_short

        signals.append(trade)
#todo pl by first date profit or loss
#todo to train for ensemble data[data["pctChange_open"]<-0.02]["pctChange_open"]
#todo shuffle/select/fit best indicators/features per batch/period
#todo NN+clf+r high/low 1-10 period ratio
shift_back = PREDICT_FROM_FIELD_SHIFT #PREDICT_FROM_FIELD_SHIFT * 1 - 1 if PREDICT_FROM_FIELD_SHIFT <= -2 else 1
#getHistoricalClassAccuracy(data[:-1], target, NUM_PERIODS_TRAINING, NUM_PERIODS_PREDICT, algo_type="undefined_test_type")
getHistoricalClassAccuracy(data, target, NUM_PERIODS_TRAINING, NUM_PERIODS_PREDICT, algo_type="undefined_test_type")
#getHistoricalClassAccuracy(X[-200:], Y[-200:], 11, 1, algo_type="undefined_test_type")

#print(signals)
signals_results = pd.DataFrame.from_dict(signals)
print("TotalPL: %.2f%%, TotalIndirection: %.2f, TotalOutdirection: %.2f" % (
                                              sum(signals_results["trade_pl"]),
                                              sum(signals_results["trade_max_profit"]),
                                              sum(signals_results["trade_max_loss"])))
hl_ratio = 3 #2/1 #todo target, param
print("HL good trade ratio:'*%s': %s from total: %s trades " % (hl_ratio, (signals_results["trade_max_profit"]/signals_results["trade_max_loss"] * -1 >= hl_ratio).sum(), signals_results.shape[0]))
print("Close good trade ratio - BAD:%s, GOOD:%s, Total:%s trades" % (signals_results[(signals_results["trade_pl"] < 0)].shape[0], signals_results[(signals_results["trade_pl"] > 0)].shape[0], signals_results.shape[0]))
print("HoldProfit: %s%%, ShortEnabled: %s" % (sum(signals_results[signals_results["trade_pl"] != 0]["trade_pl"]), len(signals_results["direction"] == "short") > 0 ))
#todo mediam takeprofit/stoploss accuracy+adjustment per batch?
#signals_results[signals_results["trade_max_loss"] <= -1]
# + signals_results[signals_results["trade_max_profit"] >= 1]
#signals_results[signals_results["trade_pl"] == STOP_LOSS_PCT]
#data[(data['pctChange_adjclose'] < -0.03)]
#signals_results.loc[signals_results["trade_pl"] > 0]
#signals_rsignals_results[signals_results["trade_max_loss"] <= -1]esults.loc[signals_results["trade_max_profit"] / signals_results["trade_max_loss"] * -1 > 0.75]
#print("%s percent where indirection ratio > *3" % np.where()# ratio (range profit/range loss>=66%")
#print("Total good predictions %s " % np.bincount(where)) #hl indirection ratio
print("Test took %.2f secs" % (timeit.default_timer() - startTest))

exit(0)

holdings = pd.DataFrame(index=data.index, data={'Holdings': np.array([np.nan] * data.shape[0])})
# holdings["Order"] = holdings.loc[((price['RSI'] < 25) & (price['BBP'] < 0)), 'Holdings'] = max_holding
# holdings["Order"] = holdings.loc[((price['RSI'] > 75) & (price['BBP'] > 1)), 'Holdings'] = 0
#holdings.ffill(inplace=True)
#holdings.fillna(0, inplace=True)
holdings = holdings.dropna()

#holdings['Order'] = holdings.diff()
#holdings.dropna(inplace=True)

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
ax0.plot(data.index, data['adjclose'], label='adjclose')
ax0.set_xlabel('Date')
ax0.set_ylabel('adjclose')
ax0.grid()
for day, holding in holdings.iterrows():
    order = holding['Order']
    if order > 0:
        ax0.scatter(x=day, y=data.loc[day, 'adjclose'], color='green')
    elif order < 0:
        ax0.scatter(x=day, y=data.loc[day, 'adjclose'], color='red')

#ax1.plot(price.index, price['RSI'], label='RSI')
ax1.fill_between(data.index, y1=30, y2=70, color='#adccff')
ax1.set_xlabel('Date')
#ax1.set_ylabel('RSI')
ax1.grid()

#ax2.plot(price.index, price['BB_up'], label='BB_up')
ax2.plot(data.index, data['adjclose'], label='AdjClose')
ax0.plot(data.index, data['high'], label='High', color='green')
ax0.plot(data.index, data['low'], label='Low', color='red')
#ax2.plot(price.index, price['BB_low'], label='BB_low')
#ax2.fill_between(price.index, y1=price['BB_low'], y2=price['BB_up'], color='#adccff')
ax2.set_xlabel('Date')
#ax2.set_ylabel('Bollinger Bands')
ax2.grid()

fig.tight_layout()
##plt.show()
print("debug")