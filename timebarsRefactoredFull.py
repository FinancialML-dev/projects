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

def barsToDataframe(_time, _open, _high, _low, _close, _volume): 
    """
    Converts OHLCV lists into a pandas DataFrame indexed by time.

    Args:
        _time (list): List of timestamps used as the DataFrame index.
        _open (list): List of open prices.
        _high (list): List of high prices.
        _low (list): List of low prices.
        _close (list): List of close prices.
        _volume (list): List of volume values.

    Returns:
        pandas.DataFrame: OHLCV DataFrame indexed by timestamp.

    Example:
        >>> bars = fetchCryptoData("BTC/USD", TimeFrame.Minute, datetime(2026, 4, 1), datetime(2026, 4, 2))
        >>> time, openPrice, highPrice, lowPrice, closePrice, volumeUnits = formatDataToLists(bars, "BTC/USD")
        >>> dataframe = barsToDataframe(time, openPrice, highPrice, lowPrice, closePrice, volumeUnits)
        >>> print(dataframe.head())
                                open      high       low     close  volume
        2026-04-01 04:00:00  82500.0  82550.0  82480.0  82510.0    1.23
    """
    dataframe = pandas.DataFrame({
        "open": _open,
        "high": _high,
        "low": _low,
        "close": _close,
        "volume": _volume
    }, index=_time)
    
    return dataframe   

#----------------------------------
#Visualization 
#----------------------------------
def plotCandleSticks(_time, _open, _high, _low, _close, _title="Candlestick Chart"):
    """
    Plot/Builds an interactive Plotly candlestick chart from OHLC data.

    Args:
        _time (list): List of timestamps for the x-axis.
        _open (list): List of open prices.
        _high (list): List of high prices.
        _low (list): List of low prices.
        _close (list): List of close prices.
        _title (str, optional): Chart title. Defaults to "Candlestick Chart".

    Returns:
        plotly.graph_objects.Figure: Interactive candlestick figure (call .show() to display).

    Example:
        >>> bars = fetchCryptoData("BTC/USD", TimeFrame.Minute, datetime(2026, 4, 1), datetime(2026, 4, 2))
        >>> time, openPrice, highPrice, lowPrice, closePrice, volumeUnits = formatDataToLists(bars, "BTC/USD")
        >>> figure = plotCandleSticks(time, openPrice, highPrice, lowPrice, closePrice, _title="BTC/USD - April 2026")
        >>> figure.show()
    """
    figure = graphObjectLibrary.Figure(data=[graphObjectLibrary.Candlestick(
        x=_time,            #x: x-axis holds our time
        open=_open,         #Then opening value for a stick
        high=_high,         #The highest value for a stick
        low=_low,           #The lowest value for a stick
        close=_close,       #The closing value for a stick
        name="OHLC"
    )])
    
    figure.update_layout(
        title=_title,
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_light",
        xaxis_rangeslider_visible=True
    )
    
    return figure

def plotCandlesticksWithVolume(_time, _open, _high, _low, _close, _volume, _title="OHLC with Volume"): 
    """
    Plot/Builds an interactive Plotly chart with a candlestick panel and a volume bar panel.

    The chart is split into two stacked subplots sharing the same x-axis:
    the top panel (70%) shows OHLC candlesticks, and the bottom panel (30%)
    shows volume bars colored green for bullish bars and red for bearish bars.

    Args:
        _time (list): List of timestamps for the x-axis.
        _open (list): List of open prices.
        _high (list): List of high prices.
        _low (list): List of low prices.
        _close (list): List of close prices.
        _volume (list): List of volume values.
        _title (str, optional): Chart title. Defaults to "OHLC with Volume".

    Returns:
        plotly.graph_objects.Figure: Interactive two-panel figure (call .show() to display).

    Example:
        >>> bars = fetchCryptoData("BTC/USD", TimeFrame.Minute, datetime(2026, 4, 1), datetime(2026, 4, 2))
        >>> time, openPrice, highPrice, lowPrice, closePrice, volumeUnits = formatDataToLists(bars, "BTC/USD")
        >>> figure = plotCandlesticksWithVolume(time, openPrice, highPrice, lowPrice, closePrice, volumeUnits,
        ...                                  _title="BTC/USD - April 2026")
        >>> figure.show()
    """
    from plotly.subplots import make_subplots
    
    #Create subplots: candlestick + volume 
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price", "Volume")
    )  
    
    #Candlestick chart
    figure.add_trace(
        graphObjectLibrary.Candlestick(
            x=_time,            #x: x-axis holds our time
            open=_open,         #Then opening value for a stick
            high=_high,         #The highest value for a stick
            low=_low,           #The lowest value for a stick
            close=_close,       #The closing value for a stick
            name="OHLC"
        ),
        row=1,
        col=1
    )
    
    #Volume bar chart 
    colors = ["red" if _close[i] < _open[i] else "green" for i in range(len(_close))]
    figure.add_trace(
        graphObjectLibrary.Bar(x=_time, y=_volume, marker_color=colors, name="Volume"),
        row=2,
        col=1
    )
    
    figure.update_layout(
        title=_title,
        xaxis_rangeslider_visible=True,
        template="plotly_white"                 #https://plotly.com/python/templates/
    )
    
    figure.update_yaxes(title_text="Price (USD)", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    
    return figure

#----------------------------------
#Data Export  
#----------------------------------
def exportToCSV(_time, _open, _high, _low, _close, _volume, _filename="timebars-data.csv"):
    """
    Exports OHLCV data to a CSV file using a pandas DataFrame.

    Args:
        _time (list): List of timestamps used as the DataFrame index.
        _open (list): List of open prices.
        _high (list): List of high prices.
        _low (list): List of low prices.
        _close (list): List of close prices.
        _volume (list): List of volume values.
        _filename (str, optional): Output file name. Defaults to "timebars-data.csv".

    Returns:
        None

    Example:
        >>> bars = fetchCryptoData("BTC/USD", TimeFrame.Minute, datetime(2026, 4, 1), datetime(2026, 4, 2))
        >>> time, openPrice, highPrice, lowPrice, closePrice, volumeUnits = formatDataToLists(bars, "BTC/USD")
        >>> exportToCSV(time, openPrice, highPrice, lowPrice, closePrice, volumeUnits, "btc-april-2026.csv")
        --> Data exported to btc-april-2026.csv
    """

    dataframe = barsToDataframe(_time, _open, _high, _low, _close, _volume)
    dataframe.to_csv(_filename)
    
    print(f"--> Data exported to {_filename}")
    
#----------------------------------
#Statistics
#----------------------------------
def printStatistics(_time, _open, _high, _low, _close, _volume):
    """
    Prints a summary of key statistics for the given OHLCV data.

    Displays total bars, date range, price range, opening/closing prices,
    price change ($ and %), and total volume.

    Args:
        _time (list): List of timestamps.
        _open (list): List of open prices.
        _high (list): List of high prices.
        _low (list): List of low prices.
        _close (list): List of close prices.
        _volume (list): List of volume values.

    Returns:
        None

    Example:
        >>> bars = fetchCryptoData("BTC/USD", TimeFrame.Minute, datetime(2026, 4, 1), datetime(2026, 4, 2))
        >>> time, openPrice, highPrice, lowPrice, closePrice, volumeUnits = formatDataToLists(bars, "BTC/USD")
        >>> printStatistics(time, openPrice, highPrice, lowPrice, closePrice, volumeUnits)
        xxxxxxxxxxxxxxxxxxxx
        Data Statistics
        --------------------
        Total bars:         1440
        Date range:         2026-04-01 to 2026-04-02
        Price range:        $81,200.00 - $83,750.00
        Opening price:      $82,500.00
        Closing price:      $83,100.00
        Price change:       $600.00 (0.73%)
        Total volume:       12,483.45
        xxxxxxxxxxxxxxxxxxxx
    """

    print(f"\n" + "x"*20)
    print("Data Statistics")
    print("-"*20)
    print(f"Total bars:         {len(_time)}")
    print(f"Date range:         {_time[0].date()} to {_time[-1].date()}")
    print(f"Price range:        ${min(_low):.2f} - ${max(_high):.2f}")
    print(f"Opening price:      ${_open[0]:.2f}")
    print(f"Closing price:      ${_close[-1]:.2f}")
    print(f"Price change:       ${_close[-1] - _open[0]:.2f} ({ ( (_close[-1] - _open[0]) / _open[0]*100 ):.2f}%)")
    print(f"Total volume:       {sum(_volume):,.2f}")    #the `,` just makes large numbers easier to read by grouping digits in threes.
    print("x"*20 + "\n")


#bars = fetchCryptoData("BTC/USD", TimeFrame.Minute, datetime(2026, 4, 1), datetime(2026, 4, 2))
#time, openPrice, highPrice, lowPrice, closePrice, volumeUnits = formatDataToLists(bars, "BTC/USD")
#figure = plotCandlesticksWithVolume(time, openPrice, highPrice, lowPrice, closePrice, volumeUnits, _title="BTC/USD - April 2026")
#figure.show()


"""
## Best Practices

1. **Encapsulate Code in Functions:** Place most of your code inside functions or classes to avoid unintended execution during imports.

2. **Use `__name__` to Control Execution:** Utilize the `if __name__ == "__main__":` idiom to control when your code runs.

3. **Define a Main Function:** Create a `main()` function to contain the primary logic of your script.

> By following these practices, you can write more organized and reusable Python code.
"""

#----------------------------------
#Main Execution 
#----------------------------------
def main():
    """Main execution function"""
    print("\n" + "-"*20)
    print("Time bars - financial data pipeline")
    print("-"*20 + "\n")
    
    #1. Fetch data
    bars = fetchCryptoData(
        CONFIG["SYMBOL"],
        CONFIG["TIMEFRAME"],
        CONFIG["START_DATE"],
        CONFIG["END_DATE"]
    )
    
    #2. Format data 
    time, open, high, low, close, volume = formatDataToLists(bars, CONFIG["SYMBOL"])
    
    #3. Print statistics 
    printStatistics(time, open, high, low, close, volume)
    
    #4. Visualize 
    print("Generating candlestick chart...")
    figure = plotCandlesticksWithVolume(
        time, open, high, low, close, volume, 
        _title=f"{CONFIG["SYMBOL"]} Time Bars - {CONFIG["START_DATE"].date()}"
    )
    figure.show()
    
    #5. Export csv file (optional)
    #exportToCSV(time, open, high, low, close, volume, _filename="btc-timebars.csv")
    
    print("--> Pipeline complete!...\n")
    
   
if __name__ == "__main__":      #The `if __name__ == "__main__"`: block ensures that main() is called only when the script is executed directly, not when it is imported as a module.
    main() 

