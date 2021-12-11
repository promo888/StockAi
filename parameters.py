import os
import time
from tensorflow.keras.layers import LSTM

#Define PREDICT_FIELD
PREDICT_FIELD = 'isUp_adjclose'#'adjclose'#

# Window size or the sequence length
N_STEPS = 3 #20 #3 #50 #3 #20 #70 #used in models 20tsla, 3ctic #batch number in model training
# Lookup step, 1 is the next day
LOOKUP_STEP = 1 #1#2 #1 #3 #Days to predict

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.055 #0.045 #0.2 #0.005 #0.9#0.05 #0.1 #0.2
# features to use
FEATURE_COLUMNS = ['open', 'adjclose', 'isUp_adjclose', 'sma_2_adjclose', 'sma_2_high', 'sma_3_high', 'sma_2_low', 'sma_3_low', 'ema_2_adjclose']#
    #["adjclose", "open", "high", "low", "isUp_adjclose"] #"volume",
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.5 #0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 3 #3 #30 #9#30 #32 #64
EPOCHS = 200 #200 #400

# Tesla stock market
ticker = "FSLY" #"TSLA" #"SQ" #"FSLY" #"CTIC" #"TSLA"#
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
##model_name = f"my_{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
##model_name = f"2020-10-20_CTIC-huber_loss-adam-LSTM-seq-3-step-1-layers-3-units-256"
##model_name = f"2020-09-03_TSLA-huber_loss-adam-LSTM-seq-70-step-1-layers-3-units-256"
model_name = f"2020-10-20_TSLA-huber_loss-adam-LSTM-seq-3-step-1-layers-3-units-256"
#model_name = f"2020-10-23_CTIC-huber_loss-adam-LSTM-seq-3-step-1-layers-3-units-256"
#model_name = f"my_2020-12-17_SQ-huber_loss-adam-LSTM-seq-3-step-1-layers-3-units-256"

if BIDIRECTIONAL:
    model_name += "-b"