import yfinance as yf

#define the ticker symbol
tickerSymbol = "AI"#'AYX'#'PLTR'#'FSLY' #RIOT

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-1-1', end='2021-12-31')

#see your data
tickerDf

#info on the company
print(tickerData.info, "\n")
#open
#sharesPercentSharesOut
#shortPercentOfFloat
#shortRatio
#volume
#averageVolume
#averageDailyVolume10Day
#fiftyTwoWeekLow
#fiftyTwoWeekHigh
#priceToBook
#forwardPE
#priceToSalesTrailing12Months


#get event data for ticker
print(tickerData.calendar, "\n")

#get recommendation data for ticker
print(tickerData.recommendations, "\n")


# /home/igor/anaconda3/envs/stock-prediction/bin/python /home/igor/PycharmProjects/stock-prediction/yfinance_test.py
# {'zip': '94107', 'sector': 'Technology', 'fullTimeEmployees': 752, 'longBusinessSummary': "Fastly, Inc. operates an edge cloud platform for processing, serving, and securing its customer's applications. The edge cloud is a category of Infrastructure as a Service that enables developers to build, secure, and deliver digital experiences at the edge of the Internet. It is a programmable platform designed for Web and application delivery. As of December 31, 2019, the company's edge network spans 68 points-of-presence worldwide. It serves customers operating in digital publishing, media and entertainment, technology, online retail, travel and hospitality, and financial technology services industries. The company was formerly known as SkyCache, Inc. and changed its name to Fastly, Inc. in May 2012. Fastly, Inc. was founded in 2011 and is headquartered in San Francisco, California.", 'city': 'San Francisco', 'phone': '844 432 7859', 'state': 'CA', 'country': 'United States', 'companyOfficers': [], 'website': 'http://www.fastly.com', 'maxAge': 1, 'address1': '475 Brannan Street', 'industry': 'Software—Application', 'address2': 'Suite 300', 'previousClose': 104.13, 'regularMarketOpen': 101.97, 'twoHundredDayAverage': 87.824524, 'trailingAnnualDividendYield': None, 'payoutRatio': 0, 'volume24Hr': None, 'regularMarketDayHigh': 104.46, 'navPrice': None, 'averageDailyVolume10Day': 6686280, 'totalAssets': None, 'regularMarketPreviousClose': 104.13, 'fiftyDayAverage': 93.407814, 'trailingAnnualDividendRate': None, 'open': 101.97, 'toCurrency': None, 'averageVolume10days': 6686280, 'expireDate': None, 'yield': None, 'algorithm': None, 'dividendRate': None, 'exDividendDate': None, 'beta': None, 'circulatingSupply': None, 'startDate': None, 'regularMarketDayLow': 99.63, 'priceHint': 2, 'currency': 'USD', 'regularMarketVolume': 6656149, 'lastMarket': None, 'maxSupply': None, 'openInterest': None, 'marketCap': 11722279936, 'volumeAllCurrencies': None, 'strikePrice': None, 'averageVolume': 7260583, 'priceToSalesTrailing12Months': 43.877213, 'dayLow': 99.63, 'ask': 105, 'ytdReturn': None, 'askSize': 900, 'volume': 6656149, 'fiftyTwoWeekHigh': 136.5, 'forwardPE': -491.80954, 'fromCurrency': None, 'fiveYearAvgDividendYield': None, 'fiftyTwoWeekLow': 10.63, 'bid': 0, 'tradeable': False, 'dividendYield': None, 'bidSize': 900, 'dayHigh': 104.46, 'exchange': 'NYQ', 'shortName': 'Fastly, Inc.', 'longName': 'Fastly, Inc.', 'exchangeTimezoneName': 'America/New_York', 'exchangeTimezoneShortName': 'EST', 'isEsgPopulated': False, 'gmtOffSetMilliseconds': '-18000000', 'quoteType': 'EQUITY', 'symbol': 'FSLY', 'messageBoardId': 'finmb_144816347', 'market': 'us_market', 'annualHoldingsTurnover': None, 'enterpriseToRevenue': 42.492, 'beta3Year': None, 'profitMargins': -0.24068001, 'enterpriseToEbitda': -289.387, '52WeekChange': 3.5179353, 'morningStarRiskRating': None, 'forwardEps': -0.21, 'revenueQuarterlyGrowth': None, 'sharesOutstanding': 102400000, 'fundInceptionDate': None, 'annualReportExpenseRatio': None, 'bookValue': 5.006, 'sharesShort': 15188052, 'sharesPercentSharesOut': 0.1338, 'fundFamily': None, 'lastFiscalYearEnd': 1577750400, 'heldPercentInstitutions': 0.55684, 'netIncomeToCommon': -64301000, 'trailingEps': -0.65, 'lastDividendValue': None, 'SandP52WeekChange': 0.1843121, 'priceToBook': 20.631243, 'heldPercentInsiders': 0.07644, 'nextFiscalYearEnd': 1640908800, 'mostRecentQuarter': 1601424000, 'shortRatio': 1.89, 'sharesShortPreviousMonthDate': 1606694400, 'floatShares': 91291455, 'enterpriseValue': 11352077312, 'threeYearAverageReturn': None, 'lastSplitDate': None, 'lastSplitFactor': None, 'legalType': None, 'lastDividendDate': None, 'morningStarOverallRating': None, 'earningsQuarterlyGrowth': None, 'dateShortInterest': 1609372800, 'pegRatio': -16.87, 'lastCapGain': None, 'shortPercentOfFloat': 0.16229999, 'sharesShortPriorMonth': 17521995, 'impliedSharesOutstanding': 113500000, 'category': None, 'fiveYearAverageReturn': None, 'regularMarketPrice': 101.97, 'logo_url': 'https://logo.clearbit.com/fastly.com'}
#
#                                 Value
# Earnings Date     2021-02-17 00:00:00
# Earnings Average                 -0.1
# Earnings Low                    -0.12
# Earnings High                   -0.03
# Revenue Average              82030000
# Revenue Low                  80220000
# Revenue High                 84300000
#
#                                   Firm        To Grade      From Grade Action
# Date
# 2019-06-11 11:05:53    Stifel Nicolaus             Buy                   init
# 2019-06-11 12:15:26        DA Davidson             Buy                   init
# 2019-06-11 13:30:50              Baird      Outperform                   init
# 2019-06-12 11:26:44      Raymond James  Market Perform                   init
# 2019-06-12 12:36:30          Citigroup         Neutral                   init
# 2019-06-21 12:14:27       Craig-Hallum             Buy                   init
# 2019-08-14 10:48:21      Piper Jaffray      Overweight                   init
# 2019-10-18 13:31:50          Citigroup         Neutral                   main
# 2020-01-02 11:50:28       PiperJaffray      Overweight         Neutral     up
# 2020-05-07 14:19:16        Oppenheimer      Outperform                   main
# 2020-05-07 15:38:26  B of A Securities             Buy                   reit
# 2020-05-07 15:57:48              Baird      Outperform                   main
# 2020-05-08 11:57:34          Citigroup         Neutral                   main
# 2020-07-06 09:54:52      Piper Sandler         Neutral      Overweight   down
# 2020-07-09 15:30:27          Citigroup            Sell         Neutral   down
# 2020-07-10 10:41:07  B of A Securities    Underperform             Buy   down
# 2020-07-10 13:59:16       Craig-Hallum            Hold             Buy   down
# 2020-08-06 13:03:17      Credit Suisse      Outperform                   main
# 2020-08-06 14:05:59             Stifel             Buy                   main
# 2020-08-06 17:27:05      Piper Sandler         Neutral                   main
# 2020-08-24 11:45:24      Raymond James      Outperform  Market Perform     up
# 2020-08-28 12:46:39      Credit Suisse      Outperform                   main
# 2020-10-15 10:10:14             Stifel            Hold             Buy   down
# 2020-10-15 10:11:32              Baird         Neutral      Outperform   down
# 2020-10-15 12:14:18      Credit Suisse      Outperform                   main
# 2020-10-23 10:11:46      Piper Sandler     Underweight         Neutral   down
# 2020-10-29 11:37:01  B of A Securities    Underperform                   main
# 2020-11-25 11:01:36      Credit Suisse         Neutral      Outperform   down
# 2021-01-21 11:38:38        Oppenheimer      Outperform         Perform     up


