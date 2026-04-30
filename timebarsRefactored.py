"""
Time bars - Financial Data Pipeline 

Fetches and visualizes cryptocurrency time-based candlestick data 
from Alpaca Market API. 
https://alpaca.markets/sdks/python/getting_started.html#introduction 

Author: Leo
Date: 2026-04-27
"""

import numpy                                                        #uv add numpy
import pandas                                                       #uv add pandas
from datetime import datetime, timedelta
import plotly                                                       #uv add plotly
import plotly.graph_objects as graphObjectLibrary

from alpaca.data.historical import CryptoHistoricalDataClient       #uv add alpaca-py
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

#----------------------------------
#CONFIGURATION
#----------------------------------
CONFIG = {
    "SYMBOL": "BTC/USD",
    "TIMEFRAME": TimeFrame.Minute, 
    "START_DATE": datetime(2026, 4, 1),     #Start date year,month, day --> of minute data
    "END_DATE": datetime(2026, 4, 2)        #End date year, month, day --> of minute data
}

#----------------------------------
#Data Fetching 
#----------------------------------
def fetchCryptoData(_symbol, _timeframe, _startDate, _endDate):
    """
    Fetches historical crypto data from Alpaca Market API. 
    (historical OHLCV bar data for a cryptocurrency symbol from the Alpaca API)

    Args:
        _symbol (str): Crypto symbol (The trading pair symbol (e.g., "BTC/USD", "ETH/USD")).
        _timeframe (TimeFrame): Bar timeframe - The bar interval (e.g., TimeFrame.Minute, TimeFrame.Hour, TimeFrame.Day)
        _startDate (datetime): Start date - Start of the data range.
        _endDate (datetime): End date - End of the data range.
    
    Returns: 
        Alpaca bar object containing OHLCV data   
        
    Example:
        >>> bars = fetchCryptoData(
        ...     "BTC/USD",
        ...     TimeFrame.Minute,
        ...     datetime(2026, 4, 1),
        ...     datetime(2026, 4, 2)
        ... )
        Fetched data for BTC/USD from 2026-04-01 to 2026-04-02
    """
    #no keys required for crypto data
    client = CryptoHistoricalDataClient()

    requestParameters = CryptoBarsRequest(
                            symbol_or_symbols=[_symbol],
                            timeframe=_timeframe,
                            start=_startDate, 
                            end=_endDate 
                    )

    candleBars = client.get_crypto_bars(requestParameters)

    #print(candleBars) #print the data so we can see how it is stuctured. Often it is a Json object. Look for list `[...]` and objects `{...}` keys and values.
    #print(candleBars.data["BTC/USD"][0].open)
    
    print(f"Fetched data for {_symbol} from {_startDate.date()} to {_endDate.date()}")
    
    return candleBars 

#Formatting the data into a format/shape we can work with
def formatDataToLists(_candleBars, _symbol):
    """
    Convert Alpaca bars to separate OHLCV lists for plotting.
    (Extracts OHLCV bar data for a symbol into separate lists.)

    Args:
        _candleBars (BarSet): Alpaca bars object - The raw bar data returned from the Alpaca API.
        _symbol (str): Symbol to extract (e.g., "BTC/USD", "ETH/USD") - The trading pair symbol used to index into the bar data (e.g., "BTC/USD", "ETH/USD").

    Returns:
        tuple: (time, openPrice, highPrice, lowPrice, closePrice, volumeUnits) as parallel lists.
        
    Example:
        >>> bars = fetchCryptoData("BTC/USD", TimeFrame.Minute, datetime(2026, 4, 1), datetime(2026, 4, 2))
        >>> time, openPrice, highPrice, lowPrice, closePrice, volumeUnits = formatDataToLists(bars, "BTC/USD")
        Formatted 1440 bars
    """
    time = []
    openPrice = []
    highPrice = []
    lowPrice = []
    closePrice = []
    volumeUnits = []

    #Formatting the data into a format/shape we can work with
    for item in _candleBars.data[_symbol]: 
        #print(item)                #Print the data, inspect how is it stuctured?
        #print(item.timestamp)      #test print the data to get a value from a key (test and see if you get what you want)
        time.append(item.timestamp)
        openPrice.append(item.open)
        highPrice.append(item.high)
        lowPrice.append(item.low)
        closePrice.append(item.close)
        volumeUnits.append(item.volume)
        
    print(f"Formatted {len(time)} bars")
    
    return time, openPrice, highPrice, lowPrice, closePrice, volumeUnits


#1. Fetch data
bars = fetchCryptoData(
                        CONFIG["SYMBOL"],           #"BTC/USD"
                        CONFIG["TIMEFRAME"],        #TimeFrame.Minute
                        CONFIG["START_DATE"],       #datetime(2026, 4, 1)
                        CONFIG["END_DATE"]          #datetime(2026, 4, 2)
                       )

#2. Format data 
time, openPrice, highPrice, lowPrice, closePrice, volumeUnits = formatDataToLists(bars, CONFIG["SYMBOL"])

#Plotting the candlesticks
figure = graphObjectLibrary.Figure(data=[graphObjectLibrary.Candlestick(x=time,                 #x: x-axis holds our time
                                                                        open=openPrice,         #Then opening price for a stick
                                                                        high=highPrice,         #The highest price for a stick
                                                                        low=lowPrice,           #The lowest price for a stick
                                                                        close=closePrice        #The closing price for a stick
                                                                        )])

figure.show()

    
