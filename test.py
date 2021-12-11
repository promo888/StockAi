from stock_prediction import create_model, load_data, np
from parameters import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd

TEST_TICKER = "FB"#"TSLA" #"FSLY" #"SQ" #"QS" #"FSLY" #"FB" "NET" "XPEV" "NIO" "SPWR" "SOLO"
TEST_FROM_DAYS_AG0 = 200
TEST_VALIDATION_DAYS = 5

def plot_graph(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    ##y_test = np.squeeze(data["column_scaler"][PREDICT_FIELD].inverse_transform(np.expand_dims(y_test, axis=0)))
    ##y_pred = np.squeeze(data["column_scaler"][PREDICT_FIELD].inverse_transform(y_pred))
    plt.plot(y_test[-200:], c='g', linestyle="", marker="o")
    plt.plot(y_pred[-200:], c='r', linestyle="", marker="o")
    plt.xlabel("Days")
    plt.ylabel("Value")
    plt.legend(["Actual ", "Predicted "])

    #my#
    # 0 1 advance/decline logRegression
    prediction_matches = []
    #realv = data["df"][-len(data["y_test"]):]
    #realv = list(realv[PREDICT_FIELD])
    #predictedv = y_pred #y_test#data["y_test"]
    # p = [x for x in zip(predictedv, realv)]
    # for pr in p: #todo rule? if prev2,3*0 ->then 0=1 (swap direction for each subsequent 0 in series (until 1 observed->resetAdvisor))
    #     if pr[0] > 0.5 and pr[1] == 1: #todo remove? 0.5
    #         prediction_matches.append(1)
    #     elif pr[0] > 0.5 and pr[1] == 0:
    #         prediction_matches.append(0)
    #     elif pr[0] < 0.5 and pr[1] == 0:
    #         prediction_matches.append(1)
    #     elif pr[0] < 0.5 and pr[1] == 1:
    #         prediction_matches.append(0)
    #good = len([x for x in prediction_matches if x==1 ])
    #print("%s matches from total %s %s" % (good, len(prediction_matches), good/len(prediction_matches)))
    #print(prediction_matches) #to do np.where>0.5
    #print(y_test)
    #print(y_pred)

    #my#

    plt.show()


def get_accuracy(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test_before_inverse = y_test
    y_pred_before_inverse = y_pred
    y_test = np.squeeze(data["column_scaler"][PREDICT_FIELD].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"][PREDICT_FIELD].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))

    #my
    df_train_len = len(data["X_train"])
    df_test_len = len(y_test)
    print('Before inverse data["y_test"]', data["y_test"])
    print(len(y_test), " tested___ ", y_test)
    print(len(y_pred), " predicted ", y_pred)
    df_test_dates = data["df"].index[df_train_len+LOOKUP_STEP: df_train_len + LOOKUP_STEP + df_test_len]  # axes[0].values() # DatetimeIndex
    print("Predict for next {} date: ".format(LOOKUP_STEP) + pd.Series(df_test_dates.format()))
    periods = 15
    print("Last {}d\nActual   :{}\nPredicted:{}".format(periods, y_test[-periods:], y_pred[-periods:]))

    #remove added dummy dates todo param
    # remove_days = -92
    # print("Predict for next {} date: ".format(LOOKUP_STEP) + pd.Series(df_test_dates.format())[:remove_days])
    # y_pred = y_pred[:remove_days]
    # y_test = y_test[:remove_days]
    #
    return accuracy_score(y_test, y_pred)


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler[PREDICT_FIELD].inverse_transform(prediction)[0][0]
    return predicted_price


# load the data
data = load_data(TEST_TICKER, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS, shuffle=False)

# construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# evaluate the model
mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
mean_absolute_error = data["column_scaler"][PREDICT_FIELD].inverse_transform([[mae]])[0][0]
print("### Ticker: " + TEST_TICKER)
print("Mean Absolute Error:", mean_absolute_error)
# predict the future price
future_price = predict(model, data)
##print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
print("Future price after {} days is {}$".format(LOOKUP_STEP, future_price))
print("Accuracy Score (%s data points) :" % len(data), get_accuracy(model, data))
plot_graph(model, data)


TEST_TICKERS = ["TSLA", "SQ", "FB", "KODK"] #crypto, fx
TEST_MODEL = "TODO"
TEST_MODEL_BATCH_PERIOD = 3 #20 #3-5 20-60 # sequence/batch learning model
TEST_START_PERIOD_DAYS_BACK = 60
TEST_VALIDATION_SIZE_FORWARD = 7
TEST_PREDICT_PERIOD_DAYS_FORWARD = 1 #2-5-7

#for tkr in TEST_TICKERS:
