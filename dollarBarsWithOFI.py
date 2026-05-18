"""
Dollar Bars - Alternative Financial Data Sampling 

Implements Lopez de Prado's dollar bars concept from 
"Advances in Financial Machine Learning" Chapter 2.

Instead of sampling by fixed time intervals, dollar bars sample 
based on cumulative dollar volume, creating bars with more uniform 
information content.

Author: Leo 
Date: 2026-04-27 - 2026-05-04
Reference: Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.  
"""

import pandas 
import math 
from datetime import datetime, timedelta
import plotly.graph_objects as plotlyLibrary 
from plotly.subplots import make_subplots 

from alpaca.data.historical import CryptoHistoricalDataClient    #uv add alpaca-py
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

#-------------------------------
#Configuration 
#-------------------------------
CONFIG = {
    "SYMBOL": "BTC/USD",                    #"BTC/USD", "ETH/USD"
    "TIMEFRAME": TimeFrame.Day,             #Day gives better volume data (Minute or Hour = 0 volume)  
    "START_DATE": datetime(2026, 4, 1),     #Start date year,month, day --> of days data
    "END_DATE": datetime(2026, 4, 27),      #End date year, month, day --> of days data 
    "DOLLAR_THRESHOLD": 500_000,            #$5M threshold for dollar bars (arbritrarily selected threshold)
}

#-------------------------------
#Data Fetching 
#-------------------------------
def fetchCryptoData(_symbol, _timeframe, _startDate, _endDate): 
    """
    Fetch historical OHLCV bar data for a cryptocurrency from Alpaca Market API.

    Args:
        _symbol (str): The crypto trading pair symbol (e.g., "BTC/USD", "ETH/USD").
        _timeframe (TimeFrame): Alpaca TimeFrame object defining bar resolution
                                (e.g., TimeFrame.Hour, TimeFrame.Day).
        _startDate (datetime): Start of the data range (timezone-aware).
        _endDate (datetime): End of the data range (timezone-aware).

    Returns:
        Alpaca bars object containing OHLCV data keyed by symbol.

    Raises:
        APIError: If the Alpaca API returns an error (e.g., invalid symbol,
                  authentication failure, or rate limit exceeded).
        ValueError: If _startDate is after _endDate.

    Example:
        >>> from datetime import datetime, timezone
        >>> from alpaca.data.timeframe import TimeFrame
        >>> start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> end   = datetime(2024, 1, 31, tzinfo=timezone.utc)
        >>> bars = fetchCryptoData("BTC/USD", TimeFrame.Hour, start, end)
        >>> df = bars.df  # convert to pandas DataFrame
    """
    client = CryptoHistoricalDataClient()

    request_params = CryptoBarsRequest(
        symbol_or_symbols=[_symbol],
        timeframe=_timeframe,       
        start=_startDate,     
        end=_endDate       
    )

    candleBars = client.get_crypto_bars(request_params)
    print(f"--> Fetched {_symbol} data from {_startDate.date()} to {_endDate.date()}")
    
    return candleBars 

def formatBarsToDictionaryList(_bars, _symbol):
    """
    Convert Alpaca Bars into a list of OHLCV dictionaries for a given symbol.

    Each bar is extracted into a plain dictionary (dict) with keys: "time", "open", 
    "high", "low", "close", "volume". Also prints a warning if all volumes are zero,
    which typically happens when using sub-day timeframes on Alpaca crypto data.

    Args:
        _bars (BarSet): Alpaca BarSet/bars object returned by get_crypto_bars(),
                        must contain data for _symbol.
        _symbol (str): The trading pair symbol used to index into _bars.data
                       (e.g., "BTC/USD"). Must match the symbol used when
                       fetching the data.

    Returns:
        list[dict]: A list of dicts, one per bar, each with keys:
                    - "time"   (datetime): Bar timestamp.
                    - "open"   (float):    Opening price.
                    - "high"   (float):    Highest price.
                    - "low"    (float):    Lowest price.
                    - "close"  (float):    Closing price.
                    - "volume" (float):    Traded volume.

    Raises:
        KeyError: If _symbol is not found in _bars.data (symbol mismatch or
                  empty response from the API).

    Example:
        >>> bars = fetchCryptoData("BTC/USD", TimeFrame.Day, start, end)
        >>> barsList = formatBarsToDictionaryList(bars, "BTC/USD")
        >>> print(barsList[0])
        {"time": datetime(...), "open": 42000.0, "high": 43500.0,
         "low": 41800.0, "close": 43100.0, "volume": 1250.5}
    """
    barsList = []

    for item in _bars.data[_symbol]: 
        #print(item)                #Print the data, inspect how is it stuctured?
        barsList.append({
            "time": item.timestamp,
            "open": item.open,
            "high": item.high,
            "low": item.low,
            "close": item.close,
            "volume": item.volume
        })
        
    print(f"--> Formated {len(barsList)} time bars")

    #Check if volumes are non-zero
    totalVolume = sum(item["volume"] for item in barsList)
    if totalVolume == 0:
        print(f"-> WARNING: All volumes are 0. Use at least TimeFrame.Day for actual volume data.")
    else:
        print(f"--> Total Volume: {totalVolume:,.2f}")
    
    return barsList 

#-------------------------------
#Create Dollar Bars (Lopez De Prado)
#-------------------------------
def createDollarBars(_timeBars, _dollarThreshold): 
    """
    Aggregate time bars into dollar bars using the Lopez De Prado method.
    (Create dollar bars from time bars.)

    Iterates through time bars accumulating dollar volume (volume * midpoint price).
    When the running total meets or exceeds _dollarThreshold, a new dollar bar is
    closed and the accumulators reset. The bar timestamp is set to the next calendar
    day after the closing bar.
    (Samples data based on cumulative dollar volume instead of time.
    Each bar contains approximately the same dollar volume traded.)
    
    Algorithm:
    1. Calculate dollar volume for each time bar (volume * midpoint_price)
    2. Accumulate dollar volume
    3. When threshold is exceeded, create new dollar bar
    4. Reset accumulator and repeat

    Args:
        _timeBars (list[dict]): List of OHLCV dicts as returned by
                                formatBarsToDictionaryList(). Each dict must have
                                keys: "time", "open", "high", "low", "close", "volume".
        _dollarThreshold (float): Dollar volume required to close a bar
                                  (e.g., 500_000 for $500K, 500_000_000 for $500M).

    Returns:
        list[dict]: A list of dollar bar dicts, each with keys:
                    - "barStart"     (datetime): Timestamp of the first time bar in the bucket/bar.
                    - "timestamp"    (datetime): Day after the bar closed.
                    - "open"         (float):    Open price of the first bar in the bucket.
                    - "high"         (float):    Highest high across all bars in the bucket.
                    - "low"          (float):    Lowest low across all bars in the bucket.
                    - "close"        (float):    Close price of the last bar in the bucket.
                    - "dollar_volume"(float):    Total dollar volume that triggered the bar.
                    (fewer bars than input, variable time spacing)

    Raises:
        KeyError: If any dict in _timeBars is missing a required OHLCV key.
        ZeroDivisionError: If _dollarBars ends up empty when the caller later
                           computes the compression ratio (all volumes are zero
                           and no bar is ever closed).

    Reference:
        Lopez de Prado, M. (2018). AFML, Chapter 2: Financial Data Structures
        
    Example:
        >>> barsList = formatBarsToDictionaryList(bars, "BTC/USD")
        >>> dollarBars = createDollarBars(barsList, _dollarThreshold=500_000_000)
        >>> print(dollarBars[0])
        {"barStart": datetime(...), "timestamp": datetime(...), "open": 42000.0, "high": 44000.0,
         "low": 41500.0, "close": 43800.0, "dollar_volume": 503_218_400.0}
    """
    dollarBars = []

    #Running accumulators 
    runningVolume = 0
    runningHigh = 0
    runningLow = math.inf
    barStart = None             #track start of each bucket/bar 
    
    print(f"\nCreating dollar bars with ${_dollarThreshold:,.0f} threshold...")

    for i, item in enumerate(_timeBars): 
        #Extract OHLCV  
        time = item["time"]
        openPrice = item["open"]
        high = item["high"]
        low = item["low"]
        closePrice = item["close"]
        volume = item["volume"]
        
        if barStart is None:        #Mark start of new bucket/bar 
            barStart = time 
        
        #Calculate dollar volume using midpoint price (the average of the opening price and the closing price)
        midpointPrice = (openPrice + closePrice)/2 
        dollarVolume = volume*midpointPrice
        
        #Update running high, running low 
        runningHigh = max(runningHigh, high)
        runningLow = min(runningLow, low)

        #Check if threshold exceeded 
        if runningVolume + dollarVolume >= _dollarThreshold:
            #Create new dollar bar 
            barTimestamp = time + timedelta(days=1)      #Next minute after bar close
            
            dollarBars.append({
                "bar_start": barStart,                              #When this bucket/bar opened
                "timestamp": barTimestamp,
                "open": openPrice,
                "high": runningHigh,
                "low": runningLow,
                "close": closePrice,
                "dollar_volume": runningVolume + dollarVolume 
            })
        
            #Reset accumulators 
            runningVolume = 0
            runningHigh = 0
            runningLow = math.inf
            barStart = None                                         #reset for next bucket/bar
        else:
            #Accumulate 
            runningVolume += dollarVolume
            
    print(f"--> Created {len(dollarBars)} dollar bars from {len(_timeBars)} time bars")
    print(f"--> Compression ratio: {len(_timeBars)/len(dollarBars):.2f}x")

    return dollarBars 

def formatDollarBarsToList(_dollarBars):
    """
    Unzip/Convert a list of dollar bar dicts into separate per-field lists.

    Iterates through _dollarBars and splits each field into its own list,
    returning them as a tuple. Useful when a downstream function or plotting
    library expects individual arrays rather than a list of dicts.
    (Transforms the dollar bars output (list of dicts) into 
    separate lists suitable for ML feature engineering and
    stock predictor input.)

    Args:
        _dollarBars (list[dict]): List of dollar bar dicts as returned by
                                  createDollarBars(). Each dict must have keys:
                                  "barStart", "timestamp", "open", "high", "low", 
                                  "close", "dollar_volume".

    Returns:
        tuple: A 7-tuple of lists in this order:
               - barStarts     (list[datatime]): Start timestamp of each bucket/bar.
               - times         (list[datetime]): Bar timestamps.
               - openPrices    (list[float]):    Opening prices.
               - highPrices    (list[float]):    Highest prices.
               - lowPrices     (list[float]):    Lowest prices.
               - closePrices   (list[float]):    Closing prices.
               - dollarVolumes (list[float]):    Dollar volumes.

    Raises:
        KeyError: If any dict in _dollarBars is missing an expected key
                  (e.g., "timestamp" or "dollar_volume").
        TypeError: If _dollarBars is not iterable, or if any element
                   is not a dict.

    Example:
        >>> dollarBars = createDollarBars(timeBars, CONFIG["DOLLAR_THRESHOLD"])
        >>> barStarts, times, opens, highs, lows, closes, volumes = formatDollarBarsToList(dollarBars)
        >>> print(len(times))       #Same as len(dollarBars)
        >>> print(opens[0])         #First bar's open price
        42000.0
    """
    barStarts = []          
    times = []              #dates
    openPrices = []
    highPrices = []
    lowPrices = []
    closePrices = []
    dollarVolumes = []

    #Formatting the data into a format/shape we can work with
    for item in _dollarBars: 
        barStarts.append(item["bar_start"])
        times.append(item["timestamp"])
        openPrices.append(item["open"])
        highPrices.append(item["high"])
        lowPrices.append(item["low"])
        closePrices.append(item["close"])
        dollarVolumes.append(item["dollar_volume"])
        
    print(f"Formatted {len(dollarVolumes)} dollar bars")
    
    return barStarts, times, openPrices, highPrices, lowPrices, closePrices, dollarVolumes


#-------------------------------
#Visualization 
#-------------------------------
def plotTimeBarsVsDollarBars(_timeBars, _dollarBars, _symbol):
    """
    Build a side-by-side candlestick comparison of time bars vs dollar bars.

    Creates a two-row Plotly figure: the top subplot shows daily time bars and
    the bottom shows dollar bars aggregated at the threshold defined in CONFIG.
    The figure is returned (not shown) so the caller can display or export it.

    Args:
        _timeBars (list[dict]): List of time bar dicts as returned by
                                formatBarsToDictionaryList(). Each dict must have
                                keys: "time", "open", "high", "low", "close".
        _dollarBars (list[dict]): List of dollar bar dicts as returned by
                                  createDollarBars(). Each dict must have keys:
                                  "timestamp", "open", "high", "low", "close".
        _symbol (str): Trading pair symbol used in subplot titles (e.g., "BTC/USD").

    Returns:
        plotly.graph_objects.Figure: A Plotly figure with two candlestick subplots,
                                     ready to be displayed with figure.show() or
                                     exported with figure.write_html().

    Raises:
        KeyError: If any bar dict is missing a required OHLC key, or if
                  CONFIG["DOLLAR_THRESHOLD"] is not set.
        ValueError: If either _timeBars or _dollarBars is empty, causing
                    pandas.DataFrame() to produce an empty frame and
                    Plotly to render a blank chart.

    Example:
        >>> timeBars = formatBarsToDictionaryList(bars, "BTC/USD")
        >>> dollarBars = createDollarBars(timeBars, CONFIG["DOLLAR_THRESHOLD"])
        >>> figure = plotTimeBarsVsDollarBars(timeBars, dollarBars, "BTC/USD")
        >>> figure.show()                        # display in browser
        >>> figure.write_html("comparison.html") # or save to file
    """
    figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"{_symbol} Time Bars (Daily)",
            f"{_symbol} Dollar Bars (${CONFIG["DOLLAR_THRESHOLD"]:,.0f} threshold)"
        ),
        vertical_spacing=0.1
    )
    
    #Time Bars (top)
    timeBarsDataframe = pandas.DataFrame(_timeBars)
    figure.add_trace(
        plotlyLibrary.Candlestick(
            x=timeBarsDataframe["time"],
            open=timeBarsDataframe["open"],
            high=timeBarsDataframe["high"],
            low=timeBarsDataframe["low"],
            close=timeBarsDataframe["close"],
            name="Time Bars"
        ),
        row=1,
        col=1
    )
    
    #Dollar Bars (bottom)
    dollarBarsDataframe = pandas.DataFrame(_dollarBars)
    figure.add_trace(
        plotlyLibrary.Candlestick(
            x=dollarBarsDataframe["timestamp"],
            open=dollarBarsDataframe["open"],
            high=dollarBarsDataframe["high"],
            low=dollarBarsDataframe["low"],
            close=dollarBarsDataframe["close"],
            name="Dollar Bars"
        ),
        row=2,
        col=1
    )
    
    figure.update_layout(
        title=f"{_symbol} - Time Bars vs Dollar Bars Comparison",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        template="plotly_white",
        height=800
    )
    
    figure.update_yaxes(title_text="Price (USD)", row=1, col=1)
    figure.update_yaxes(title_text="Price (USD)", row=2, col=1)
    
    return figure 


def plotDollarVolumeDistribution(_dollarBars):
    """
    Build a bar chart showing the dollar volume of each dollar bar over time.

    Plots dollar volume (y-axis) against bar timestamp (x-axis) and overlays a
    horizontal dashed line at CONFIG["DOLLAR_THRESHOLD"] so you can visually
    confirm how consistently each bar hits the target. The figure is returned
    (not shown) so the caller can display or export it.

    Args:
        _dollarBars (list[dict]): List of dollar bar dicts as returned by
                                  createDollarBars(). Each dict must have keys:
                                  "timestamp" (datetime) and "dollar_volume" (float).

    Returns:
        plotly.graph_objects.Figure: A Plotly bar chart figure, ready to be
                                     displayed with figure.show() or exported
                                     with figure.write_html().

    Raises:
        KeyError: If any dict in _dollarBars is missing "timestamp" or
                  "dollar_volume", or if CONFIG["DOLLAR_THRESHOLD"] is not set.
        ValueError: If _dollarBars is empty, causing pandas.DataFrame() to
                    produce an empty frame and Plotly to render a blank chart.

    Example:
        >>> dollarBars = createDollarBars(timeBars, CONFIG["DOLLAR_THRESHOLD"])
        >>> figure = plotDollarVolumeDistribution(dollarBars)
        >>> figure.show()                           # display in browser
        >>> figure.write_html("distribution.html")  # or save to file
    """
    dollarBarsDataframe = pandas.DataFrame(_dollarBars)
    
    figure = plotlyLibrary.Figure()
    
    figure.add_trace(plotlyLibrary.Bar(
        x=dollarBarsDataframe["timestamp"],
        y=dollarBarsDataframe["dollar_volume"],
        name="Dollar Volume per Bar",
        marker_color="#0091d9"
    ))
    
    #Add threshold line 
    figure.add_hline(
        y=CONFIG["DOLLAR_THRESHOLD"],
        line_dash="dash",
        line_color="#717171",
        annotation_text=f"Threshold: ${CONFIG["DOLLAR_THRESHOLD"]:,.0f}"
    )
    
    figure.update_layout(
        title="Dollar Volume Distribution Acress Bars",
        xaxis_title="Time (Days)",
        yaxis_title="Dollar Volume",
        template="plotly_white"
    )
    
    return figure 


#-------------------------------
#Statistics 
#-------------------------------
def printComparisonStatics(_timeBars, _dollarBars):
    """
    Print a statistical comparison between time bars and dollar bars to stdout.

    Reports three sections:
    - Bar counts: number of time bars, dollar bars, and compression ratio.
    - Dollar volume stats: mean, std, min, max vs the configured threshold.
    - Time spacing stats: mean, std, min, max days between consecutive dollar bars
      (only shown if there are at least 2 dollar bars).

    Args:
        _timeBars (list[dict]): List of time bar dicts as returned by
                                formatBarsToDictionaryList(). Only its length
                                is used for the compression ratio.
        _dollarBars (list[dict]): List of dollar bar dicts as returned by
                                  createDollarBars(). Each dict must have keys:
                                  "timestamp" (datetime) and "dollar_volume" (float).

    Returns:
        None

    Raises:
        ZeroDivisionError: If _dollarBars is empty, the compression ratio
                           calculation (len(_timeBars) / len(_dollarBars)) will
                           divide by zero.
        KeyError: If any dict in _dollarBars is missing "timestamp" or
                  "dollar_volume", or if CONFIG["DOLLAR_THRESHOLD"] is not set.

    Example:
        >>> timeBars = formatBarsToDictionaryList(bars, "BTC/USD")
        >>> dollarBars = createDollarBars(timeBars, CONFIG["DOLLAR_THRESHOLD"])
        >>> printComparisonStatics(timeBars, dollarBars)

        -o0o--o0o--o0o--o0o--o0o-
        Dollar Bars Statistics
        --------------------
        Bar counts:
          Time Bars:                365
          Dollar Bars:              42
          Compression Ratio:        8.69x

        Dollar Volume stats:
          Target threshold:         $500,000,000
          Mean:                     $512,300,000
          Std:                      $48,200,000
          Min:                      $501,100,000
          Max:                      $638,900,000

        Time Spacing Between Dollar Bars (days):
          Mean:                 $8.69
          Std:                  $3.12
          Min:                  $1.00
          Max:                  $21.00
    """
    print("\n" + "-o0o-"*5)
    print("Dollar Bars Statistics")
    print("-"*20)
    
    #Basic counts
    print(f"\nBar counts:")
    print(f"  Time Bars:                {len(_timeBars)}")
    print(f"  Dollar Bars:              {len(_dollarBars)}")
    print(f"  Compression Ratio:        {len(_timeBars) / len(_dollarBars):.2f}x")
    
    #Dollar volumes 
    dollarBarsDataframe = pandas.DataFrame(_dollarBars)
    print(f"\nDollar Volume stats:")
    print(f"  Target threshold:         ${CONFIG["DOLLAR_THRESHOLD"]:,.0f}")
    print(f"  Mean:                     ${dollarBarsDataframe["dollar_volume"].mean():,.0f}")
    print(f"  Std:                      ${dollarBarsDataframe["dollar_volume"].std():,.0f}")
    print(f"  Min:                      ${dollarBarsDataframe["dollar_volume"].min():,.0f}")
    print(f"  Max:                      ${dollarBarsDataframe["dollar_volume"].max():,.0f}")
       
    #Time spacing 
    if len(_dollarBars) > 1:
        timestamps = pandas.to_datetime(dollarBarsDataframe["timestamp"]) 
        timeDifferences = timestamps.diff().dt.total_seconds() / 86400       #days (60*60*24)
        print(f"\nTime Spacing Between Dollar Bars (days):")
        print(f"  Mean:                 ${timeDifferences.mean():.2f}")
        print(f"  Std:                  ${timeDifferences.std():.2f}")
        print(f"  Min:                  ${timeDifferences.min():.2f}")
        print(f"  Max:                  ${timeDifferences.max():.2f}")

    print("-"*20 + "\n")

def printSampleBars(_timeBars, _dollarBars, _n=3):
    """
    Print the first N time bars and first N dollar bars to stdout for inspection.

    Displays each time bar with its timestamp, OHLC prices, and volume, then
    displays each dollar bar with its timestamp, OHLC prices, and total dollar
    volume. Useful for a quick sanity check after fetching and aggregating data.

    Args:
        _timeBars (list[dict]): List of time bar dicts as returned by
                                formatBarsToDictionaryList(). Each dict must have
                                keys: "time", "open", "high", "low", "close", "volume".
        _dollarBars (list[dict]): List of dollar bar dicts as returned by
                                  createDollarBars(). Each dict must have keys:
                                  "timestamp", "open", "high", "low", "close",
                                  "dollar_volume".
        _n (int, optional): Number of bars to print from each list. Defaults to 3.

    Returns:
        None

    Raises:
        KeyError: If any bar dict is missing an expected key (e.g., "time" or
                  "dollar_volume").
        IndexError: If either list is empty and slicing fails (does not raise,
                    but prints nothing for that section).

    Example:
        >>> timeBars = formatBarsToDictionaryList(bars, "BTC/USD")
        >>> dollarBars = createDollarBars(timeBars, 500_000_000)
        >>> printSampleBars(timeBars, dollarBars, _n=5)

        First 5 Time Bars:
        --------------------
        Bar 1:
            Time: 2024-01-01 00:00:00+00:00
            OHLC:
                open=42000.00,
                high=43500.00,
                low=41800.00,
                close=43100.00
            Volume: 1250.500000
        ...
        First 5 Dollar Bars:
        --------------------
        Bar 1:
            Time: 2024-01-15 00:00:00+00:00
            OHLC:
                open=42000.00, high=44000.00, low=41500.00, close=43800.00
            Dollar Volume: $503,218,400
    """
    print(f"\nFirst {_n} Time Bars:")
    print(f"-"*20)
    for i, item in enumerate(_timeBars[:_n]):
        print(f"Bar {i+1}:")
        print(f"    Time: {item["time"]}")
        print(f"    OHLC: \n"
              f"        open={item["open"]:.2f},\n"
              f"        high={item["high"]:.2f},\n"
              f"        low={item["low"]:.2f},\n"
              f"        close={item["close"]:.2f}")
        print(f"    Volume: {item["volume"]:.6f}")
        
    print(f"\nFirst {_n} Dollar Bars:")
    print(f"-"*20)
    for i, item in enumerate(_dollarBars[:_n]):
        print(f"Bar {i+1}:")
        print(f"    Time: {item["timestamp"]}")
        print(f"    OHLC: \n"
              f"        open={item["open"]:.2f},\n"
              f"        high={item["high"]:.2f},\n"
              f"        low={item["low"]:.2f},\n"
              f"        close={item["close"]:.2f}")
        print(f"    Dollar Volume: ${item["dollar_volume"]:,.0f}")
  

#-------------------------------
#Data Export 
#-------------------------------  
def exportDollarBars(_dollarBars, _filename="dollar-bars.csv"):
    """
    Export a list of dollar bars to a CSV file.

    Converts _dollarBars to a pandas DataFrame and writes it to disk without
    the row index. If the file already exists it will be overwritten silently.

    Args:
        _dollarBars (list[dict]): List of dollar bar dicts as returned by
                                  createDollarBars(). Each dict should have keys:
                                  "timestamp", "open", "high", "low", "close",
                                  "dollar_volume".
        _filename (str, optional): Destination file path. Defaults to
                                   "dollar-bars.csv" in the current working directory.

    Returns:
        None

    Raises:
        PermissionError: If the file is open in another program (e.g. Excel)
                         and cannot be overwritten.
        OSError: If the directory in _filename does not exist or the path
                 is otherwise invalid.
        ValueError: If _dollarBars is empty, an empty CSV with only headers
                    will be written (no error raised, but worth noting).

    Example:
        >>> dollarBars = createDollarBars(timeBars, CONFIG["DOLLAR_THRESHOLD"])
        >>> exportDollarBars(dollarBars)                          # saves to dollar-bars.csv
        >>> exportDollarBars(dollarBars, "output/btc_bars.csv")   # custom path
    """
    dataframe = pandas.DataFrame(_dollarBars)
    dataframe.to_csv(_filename, index=False)
    print(f"--> Dollar Bars Exported/Saved to {_filename}")
    
    
#-------------------------------
#Main Execution 
#-------------------------------
def main():
    """Main Execution pipeline..."""
    print("\n" + "-"*20)
    print("Dollar bars - Lopez De Prado implementation")
    print("-"*20 + "\n")
    
    #1. Fetch time bars  
    print("Step 1: Fetching time bars...")
    bars = fetchCryptoData(
            CONFIG["SYMBOL"],
            CONFIG["TIMEFRAME"],
            CONFIG["START_DATE"],
            CONFIG["END_DATE"]
        ) 

    timeBars = formatBarsToDictionaryList(bars, CONFIG["SYMBOL"])

    #2. Create dollar bars 
    print("Step 2: Creating dollar bars...")
    dollarBars = createDollarBars(timeBars, CONFIG["DOLLAR_THRESHOLD"]) #500000
    
    #3. Statictics 
    print("Step 3: Comparing time bars vs dollar bars...")
    printComparisonStatics(timeBars, dollarBars)
    printSampleBars(timeBars, dollarBars, _n=3)
    
    #4. Visualization 
    print("Step 4: Generation visualizations...")
    
    #Comparings plot 
    figure1 = plotTimeBarsVsDollarBars(timeBars, dollarBars, CONFIG["SYMBOL"])
    figure1.show()
    
    #Dollar volume distribution 
    figure2 = plotDollarVolumeDistribution(dollarBars)
    figure2.show()
    
    #5. Export (optional)
    #exportDollarBars(dollarBars, "btc-dollar-bars.csv")
    
    print("-"*20)
    print("Pipeline completed!")
    print("-"*20 + "\n")
    
    print("Key Insights:")
    print(f"--> Dollar bars provide more uniform information content")
    print(f"--> {len(dollarBars)} dollar bars vs {len(timeBars)} time bars")
    print(f"--> During high activity: more bars (higher sampling rate)")
    print(f"--> During low activity: fewer bars (lowwe sampling rate)")
    print(f"--> This is Lopez de Prodo's key insight: sample by information, not time\n")


if __name__ == "__main__":
    main()




#Insight
""" 
#Try: 
If volume is 0, the Alpaca free-tier crypto endpoint may not return real volume 
for minute bars. You could try switching to TimeFrame.Hour or TimeFrame.Day 
which tend to have populated volume fields, or lower your 
threshold dramatically (e.g. 50000, 5000, 500) just to confirm the function 
works when volume is non-zero.
"""

#Small test runs for examples 
"""from datetime import datetime, timezone
from alpaca.data.timeframe import TimeFrame
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end   = datetime(2024, 1, 31, tzinfo=timezone.utc)
bars = fetchCryptoData("BTC/USD", TimeFrame.Hour, start, end)
df = bars.df  # convert to pandas DataFrame
print(df.head())"""

"""from datetime import datetime, timezone
from alpaca.data.timeframe import TimeFrame
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end   = datetime(2024, 1, 31, tzinfo=timezone.utc)
bars = fetchCryptoData("BTC/USD", TimeFrame.Day, start, end)
barsList = formatBarsToDictionaryList(bars, "BTC/USD")
print(barsList[0])"""

"""barsList = formatBarsToDictionaryList(bars, "BTC/USD")
dollarBars = createDollarBars(barsList, _dollarThreshold=500_000)
print(dollarBars[0])"""

"""timeBars = formatBarsToDictionaryList(bars, "BTC/USD")
dollarBars = createDollarBars(timeBars, 500_000)
printSampleBars(timeBars, dollarBars, _n=5)"""

"""from datetime import datetime, timezone
from alpaca.data.timeframe import TimeFrame
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end   = datetime(2024, 1, 31, tzinfo=timezone.utc)
bars = fetchCryptoData("BTC/USD", TimeFrame.Day, start, end)
timeBars = formatBarsToDictionaryList(bars, "BTC/USD")
dollarBars = createDollarBars(timeBars, CONFIG["DOLLAR_THRESHOLD"])
times, opens, highs, lows, closes, volumes = formatDollarBarsToList(dollarBars)
print(len(times))       #Same as len(dollarBars)
print(opens[0])         #First bar's open price """