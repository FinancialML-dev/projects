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
    "DOLLAR_THRESHOLD": 5_000_000,          #$5M threshold for dollar bars (arbritrarily selected threshold)
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
        {"timestamp": datetime(...), "open": 42000.0, "high": 44000.0,
         "low": 41500.0, "close": 43800.0, "dollar_volume": 503_218_400.0}
    """
    dollarBars = []

    #Running accumulators 
    runningVolume = 0
    runningHigh = 0
    runningLow = math.inf
    
    print(f"\nCreating dollar bars with ${_dollarThreshold:,.0f} threshold...")

    for i, item in enumerate(_timeBars): 
        #Extract OHLCV  
        time = item["time"]
        openPrice = item["open"]
        high = item["high"]
        low = item["low"]
        closePrice = item["close"]
        volume = item["volume"]
        
        #Calculate dollar volume using midpoint price (the average of the opening price and the closing price)
        midpointPrice = (openPrice + closePrice)/2 
        dollarVolume = volume*midpointPrice
        
        #Update running high, running low 
        runningHigh = max(runningHigh, high)
        runningLow = min(runningLow, low)

        #Check if threshold exceeded 
        if runningVolume + dollarVolume >= _dollarThreshold:
            #Create new dollar bar 
            barTimestamp = time + timedelta(days=1)      #Next day after bar close
            
            dollarBars.append({
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
        else:
            #Accumulate 
            runningVolume += dollarVolume
            
    print(f"--> Created {len(dollarBars)} dollar bars from {len(_timeBars)} time bars")
    print(f"--> Compression ratio: {len(_timeBars)/len(dollarBars):.2f}x")

    return dollarBars 


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
  

bars = fetchCryptoData(
        CONFIG["SYMBOL"],
        CONFIG["TIMEFRAME"],
        CONFIG["START_DATE"],
        CONFIG["END_DATE"]
    )    

timeBars = formatBarsToDictionaryList(bars, CONFIG["SYMBOL"])

#Create bars
dollarBars = createDollarBars(timeBars, 500000) #5,000,000 is an arbritrarily selected threshold

printSampleBars(timeBars, dollarBars, 3)

#create dataframe
dataframe = pandas.DataFrame(dollarBars)

#View the first five entries
print(dataframe.head())


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