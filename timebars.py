import numpy                                        #uv add numpy
import pandas                                       #uv add pandas
from datetime import datetime, timedelta
import plotly                                       #uv add plotly
import plotly.graph_objects as graphObjectLibrary

#Collect historical candlestick data
#Inspired from https://davidzhao12.medium.com/advances-in-financial-machine-learning-for-dummies-part-1-7913aa7226f5

#We fetch our data from Alpaca
#https://alpaca.markets/sdks/python/getting_started.html#introduction

from alpaca.data.historical import CryptoHistoricalDataClient    #uv add alpaca-py
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

#no keys required for crypto data
client = CryptoHistoricalDataClient()

requestParameters = CryptoBarsRequest(
                        #symbol_or_symbols=["BTC/USD", "ETH/USD"],
                        #timeframe=TimeFrame.Day, #Day gives you a faster loop/completes faster
                        symbol_or_symbols=["BTC/USD"],
                        timeframe=TimeFrame.Minute,
                        #start="2026-04-01",
                        start=datetime(2026, 4, 1), #Start date year,month, day --> of minute data
                        end=datetime(2026, 4, 2) #End date year, month, day --> of minute data
                 )

candleBars = client.get_crypto_bars(requestParameters)

#print(candleBars) #print the data so we can see how it is stuctured. Often it is a Json object. Look for list `[...]` and objects `{...}` keys and values.
#print(candleBars.data["BTC/USD"][0].open)

time = []
open = []
high = []
low = []
close = []
volume = []

#Formatting the data to match the plotting library
for item in candleBars.data["BTC/USD"]: 
    #print(item)                #Print the data, inspect how is it stuctured?
    #print(item.timestamp)      #test print the data to get a value from a key (test and see if you get what you want)
    time.append(item.timestamp)
    #time.append(item["timestamp"])
    open.append(item.open)
    high.append(item.high)
    low.append(item.low)
    close.append(item.close)
    volume.append(item.volume)

#Plotting the candlesticks
figure = graphObjectLibrary.Figure(data=[graphObjectLibrary.Candlestick(x=time,         #x: x-axis holds our time
                                                                        open=open,      #Then opening value for a stick
                                                                        high=high,      #The highest value for a stick
                                                                        low=low,        #The lowest value for a stick
                                                                        close=close     #The closing value for a stick
                                                                        )])

figure.show()
