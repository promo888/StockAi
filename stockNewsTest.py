from stocknews import StockNews

tickers = ['FSLY', 'WISH', 'CLNE', 'BE', 'TSLA'] #, 'XPEV', 'CLNE', 'RIOT']
assets = {'FSLY': "Fastly"}
assets = {'FSLY': "Fastly",
          'WISH': "WISH",
          'CLNE': "Clean Energy",
          'BE': "Bloom Energy",
          'TSLA': "Tesla"}
sn = StockNews(tickers, wt_key='MY_WORLD_TRADING_DATA_KEY', save_news=True)
df = sn.summarize()

# sorting dataframe
#data.sort_values("STOCK", inplace=True)

# making boolean series for a team name
# filter1 = data["Team"] == "Atlanta Hawks"
#
# # making boolean series for age
# filter2 = data["Age"] > 24
#
# # filtering data on basis of both filters
# data.where(filter1 & filter2, inplace=True)
#
# # display
# data

#todo OWN classificator on words conjunctions/emotions/.. article's score positive/negative
#NN learn positive negative (exclude/delete predicat's words)
#filter/cluster up/down from the nextDay close + HL (H/L>2,3) - by news / words
#todo inside sn.summarize() or before
df2 = StockNews(tickers, wt_key='MY_WORLD_TRADING_DATA_KEY', save_news=False).read_rss()
for index, row in df2.iterrows():
    #print(row)
    print("\n")
    news_ticker = row["stock"].lower()
    if news_ticker in assets:
        #assets[row["stock"]] #todo split, Search and separate search count
        if not assets[news_ticker] in assets[row["summary"]].lower():
            continue

    print(row["p_date"])
    print("Header:\n\t", row["title"])
    print("Header sentiment:\t", row["sentiment_title"])
    print("Text:\n\t", row["summary"])
    print("Text sentiment:\t", row["sentiment_summary"])
    print("\n\n")
print(df)