import plotly
from fastquant import get_stock_data
# df = get_stock_data("AMZN", "2018-01-01", "2021-01-01")
# print(df.head(10))
# print(df.tail(10))

from fastquant import backtest
# backtest('smac', df, fast_period=15, slow_period=40)
#backtest('rsi', df, rsi_period=14, rsi_upper=70, rsi_lower=30)
#backtest("smac", df, fast_period=range(15, 30, 3), slow_period=range(40, 55, 3), verbose=False)

# res = backtest("smac", df, fast_period=range(15, 30, 3), slow_period=range(40, 55, 3), verbose=False)
# # Optimal parameters: {'init_cash': 100000, 'buy_prop': 1, 'sell_prop': 1, 'execution_type': 'close', 'fast_period': 15, 'slow_period': 40}
# # Optimal metrics: {'rtot': 0.022, 'ravg': 9.25e-05, 'rnorm': 0.024, 'rnorm100': 2.36, 'sharperatio': None, 'pnl': 2272.9, 'final_value': 102272.90}
# print(res[['fast_period', 'slow_period', 'final_value']].head())

# ticker_sentiment = "fnv"
# from_date = "2015-01-01"
# to_date = "2020-12-31"
from fastquant import get_yahoo_data, get_bt_news_sentiment
ticker = "aapl"
data = get_yahoo_data(ticker, "2020-01-01", "2021-12-31") #tsla #spwr
sentiments = get_bt_news_sentiment(keyword=ticker, page_nums=3) #value tesla, 3 #undervalue overvalue value drop negative overvalue downgrade revenue upside downside up down [array words]
##backtest("sentiment", data, sentiments=sentiments, senti=0.2) #0.2 0.1-sensitive

exit(0)

from fastquant import get_crypto_data, backtest
from fbprophet import Prophet
from matplotlib import pyplot as plt

# Pull crypto data
##df = get_crypto_data("BTC/USDT", "2019-01-01", "2020-08-31")
df = get_stock_data("fsly", "2010-01-01", "2021-12-31")
##df = get_stock_data("fsly", "2010-01-01", "2020-12-31")
#df = get_stock_data("gdx", "2010-01-01", "2020-12-31")

# Fit model on closing prices
ts = df.reset_index()[["dt", "close"]]
ts.columns = ['ds', 'y']
m = Prophet(daily_seasonality=True, yearly_seasonality=True).fit(ts)
forecast = m.make_future_dataframe(periods=1, freq='D')

# Predict and plot
pred = m.predict(forecast)
fig1 = m.plot(pred)
plt.title('BTC/USDT: Forecasted Daily Closing Price', fontsize=25)
#plt.show()
# Convert predictions to expected 1 day returns
expected_1day_return = pred.set_index("ds").yhat.pct_change().shift(-1).multiply(100)

# Backtest the predictions, given that we buy bitcoin when the predicted next day return is > +1.5%, and sell when it's < -1.5%.
#df["custom"] = expected_1day_return.multiply(-1)
#backtest("custom", df.dropna(),upper_limit=1.5, lower_limit=-1.5)
#backtest('rsi', df, rsi_period=14, rsi_upper=70, rsi_lower=30)
backtest('bbands', df)
#backtest('macd', df)
#backtest('buynhold', df)
#backtest('sentiment', df)
#backtest('emac', df)
#backtest('smac', df)
#backtest('ternary', df)
#backtest('base', df)
#backtest('custom', df)
