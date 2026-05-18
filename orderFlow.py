"""
Order Flow — True Liquidity and Order Flow Features

Fetches individual trade records from Alpaca and computes
Order Flow Imbalance (OFI) per dollar bar.

OFI captures who is driving price: buyers or sellers.
taker_side = "B" → buyer initiated (aggressive buy) - "Ask"
taker_side = "S" → seller initiated (aggressive sell) - "Bid"

Author: Leo
Date: 2026-05-11
Reference: Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
"""

import numpy 
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoTradesRequest

#-------------------------------
#Fetch Trades
#-------------------------------
def fetchCryptoTrades(_symbol, _startDate, _endDate):
    """
    Fetch individual trade records for a crypto pair from Alpaca.
    Each trade includes: timestamp, price, size, taker_side ("B" or "S").

    Args:
        _symbol    (str):      Trading pair symbol (e.g., "BTC/USD", "ETH/USD").
        _startDate (datetime): Start of the data range (inclusive).
        _endDate   (datetime): End of the data range (inclusive).

    Returns:
        Alpaca TradeSet object: Raw trade data keyed by symbol.
                                Access individual trades via .data[_symbol].

    Raises:
        APIError:                 If the Alpaca API rejects the request — e.g.,
                                  invalid symbol, rate limit exceeded, or
                                  the date range is too large for a single request.
        ValueError:               If _startDate is after _endDate (Alpaca will
                                  return an empty result or raise depending on
                                  the SDK version).
        requests.ConnectionError: If there is no network connection or the
                                  Alpaca API endpoint is unreachable.

    Example:
        >>> trades = fetchCryptoTrades(
        ...     "BTC/USD",
        ...     datetime(2026, 4, 1),
        ...     datetime(2026, 4, 2)
        ... )
        >>> tradeList = trades.data["BTC/USD"]
        >>> print(tradeList[0].price, tradeList[0].size, tradeList[0].taker_side)
        82500.0  0.003  B
    """
    client = CryptoHistoricalDataClient()
    request = CryptoTradesRequest(
        symbol_or_symbols=[_symbol],
        start=_startDate,
        end=_endDate  
    )
    trades = client.get_crypto_trades(request)
    
    return trades 


def fetchCryptoTradesChunked(_symbol, _startDate, _endDate, _chunkDays=7): 
    """
    Fetch trades in weekly chunks to avoid API timeouts on large date ranges.
    Returns a flat list of all trade records across the full date range.

    Splits the date range into chunks of _chunkDays, calls fetchCryptoTrades()
    for each chunk, and concatenates the results into a single flat list.

    Args:
        _symbol    (str):      Trading pair symbol (e.g., "BTC/USD", "ETH/USD").
        _startDate (datetime): Start of the full data range (inclusive).
        _endDate   (datetime): End of the full data range (inclusive).
        _chunkDays (int, optional): Number of days per chunk. Defaults to 7.

    Returns:
        list: Flat list of all Alpaca trade objects across the full date range.
              Each trade has: timestamp, price, size, taker_side ("B" or "S").

    Raises:
        APIError:                 If any chunk request is rejected by the Alpaca API —
                                  e.g., invalid symbol or rate limit exceeded. The
                                  error will abort the loop mid-fetch.
        KeyError:                 If _symbol is not present in a chunk's .data dict —
                                  e.g., no trades were returned for that window.
        requests.ConnectionError: If the network connection drops mid-fetch, aborting
                                  the loop and losing any partially collected trades.
        ValueError:               If _startDate is after _endDate (the while loop
                                  never executes and an empty list is returned —
                                  no exception raised, but silently wrong).

    Example:
        >>> trades = fetchCryptoTradesChunked(
        ...     "BTC/USD",
        ...     datetime(2026, 3, 1),
        ...     datetime(2026, 4, 1),
        ...     _chunkDays=7
        ... )
        --> Fetching trades 2026-03-01 to 2026-03-08...
        --> Fetching trades 2026-03-08 to 2026-03-15...
        ...
        --> Total trades fetched: 2,847,391
        >>> print(trades[0].price, trades[0].taker_side)
        82500.0  B
    """
    allTrades = []
    current = _startDate
    
    while current < _endDate:
        chunkEnd = min(current + timedelta(days=_chunkDays), _endDate)
        print(f"--> Fetching trades {current.date()} to {chunkEnd.date()}...")
        
        chunk = fetchCryptoTrades(_symbol, current, chunkEnd)
        allTrades.extend(chunk.data[_symbol]) #version 2: allTrades += chunk.data[_symbol]
        current = chunkEnd 
        
    print(f"--> Total trades fetched: {len(allTrades):,}")
    
    return allTrades
    

#-------------------------------
#Compute Order Flow Imbalance (OFI) per Dollar Bar
#-------------------------------
def calculateOrderFlowImbalance(_buyVolume, _sellVolume):
    """
    Order Flow Imbalance: (buy - sell) / (buy + sell)
    +1.0 = all buying pressure
    -1.0 = all selling pressure
     0.0 = balanced

    Computes element-wise Order Flow Imbalance (OFI). When total volume is zero (no trades in the bar),
    returns 0.0 instead of dividing by zero.

    Args:
        _buyVolume  (float or numpy.ndarray): Total buy-side volume per bar.
        _sellVolume (float or numpy.ndarray): Total sell-side volume per bar.

    Returns:
        float or numpy.ndarray: OFI value(s) in the range [-1.0, +1.0].
                                0.0 is returned for any bar where total volume is zero.

    Raises:
        TypeError: If _buyVolume or _sellVolume are not numeric or numpy-compatible
                   types (numpy.where and arithmetic will fail on incompatible types).

    Example:
        >>> ofi = calculateOrderFlowImbalance(800, 200)
        >>> print(ofi)
        0.6     # (800-200)/(800+200) = 600/1000 = 0.6 → strong buying pressure

        >>> ofi = calculateOrderFlowImbalance(
        ...     numpy.array([800, 200, 500]),
        ...     numpy.array([200, 800, 500])
        ... )
        >>> print(ofi)
        [ 0.6  -0.6   0.0]  # buy-heavy, sell-heavy, balanced
    """
    total = _buyVolume + _sellVolume
    
    return numpy.where(total < 0, (_buyVolume - _sellVolume)/total, 0.0)
    

def aggregateOrderFlowToBars(_trades, _barStarts, _barEnds):
    """
    Map individual trades to dollar bars and compute Order Flow Imbalance (OFI) per bar.

    For each dollar bar defined by [bar_start, bar_end], sums buy and
    sell volume from trades within that window and computes Order Flow Imbalance (OFI).

    Uses a single linear pass through _trades with a bar pointer — trades must be
    in chronological order for correct assignment.

    Args:
        _trades    (list):           Trade records from fetchCryptoTradesChunked().
                                     Each record must have: timestamp, size, taker_side.
        _barStarts (list[datetime]): Opening timestamp of each dollar bar.
        _barEnds   (list[datetime]): Closing timestamp of each dollar bar.
                                     Must be the same length as _barStarts.

    Returns:
        dict with numpy arrays, one value per bar:
            - "order_Flow_Imbalance"  (float): OFI = (buyVol - sellVol) / (buyVol + sellVol), range [-1, +1].
            - "trade_count_imbalance" (float): (buyCount - sellCount) / totalCount, range [-1, +1].
            - "average_trade_size"    (float): Mean trade size per bar. 0.0 for empty bars.

    Raises:
        AttributeError: If any trade in _trades is missing .timestamp, .size, or
                        .taker_side attributes (accessed directly with no guard).
        IndexError:     If _barStarts and _barEnds have different lengths — barIndex
                        advances against numberOfBars but accesses both lists by index.
        ValueError:     If _trades are not in chronological order — the single-pass
                        bar pointer only advances forward, so out-of-order trades will
                        be silently dropped and OFI will be incorrect.

    Example:
        >>> trades = fetchCryptoTradesChunked("BTC/USD", datetime(2026, 4, 1), datetime(2026, 4, 8))
        >>> dollarBars = createDollarBars(timeBars, _dollarThreshold=500_000_000)
        >>> barStarts = [bar["bar_start"] for bar in dollarBars]
        >>> barEnds   = [bar["timestamp"] for bar in dollarBars]
        >>> ofi = aggregateOrderFlowToBars(trades, barStarts, barEnds)
        --> Aggregated Order Flow Imbalance (OFI) for 42 bars
        >>> print(ofi["order_Flow_Imbalance"][:3])
        [ 0.42  -0.18   0.07]   # bar 0: buy-heavy, bar 1: sell-heavy, bar 2: balanced
    """
    numberOfBars = len(_barStarts)
    orderFlowImbalance = numpy.zeros(numberOfBars)
    tradeCountImbalance = numpy.zeros(numberOfBars)
    averageTradeSize = numpy.zeros(numberOfBars)
    
    buyVolume = numpy.zeros(numberOfBars)
    sellVolume = numpy.zeros(numberOfBars)
    buyCount = numpy.zeros(numberOfBars)
    sellCount = numpy.zeros(numberOfBars)
    totalSize = numpy.zeros(numberOfBars)
    tradeCount = numpy.zeros(numberOfBars)
    
    #Walk through all trades once - assign each trade to its bar 
    barIndex = 0
    for item in _trades: 
        timestamp = item.timestamp 
        
        #Advance bar pointer until we find the bar this trade belongs to
        while barIndex < numberOfBars and timestamp >= _barEnds[barIndex]: 
            barIndex += 1
            
            if barIndex > numberOfBars:
                break 
            
            if timestamp >= _barStarts[barIndex]:
                size = item.size
                totalSize[barIndex] += size 
                tradeCount[barIndex] += 1 

                if item.taker_side == "B":
                    buyVolume[barIndex] += size 
                    buyCount[barIndex] += 1
                else:
                    sellVolume[barIndex] += size 
                    sellCount[barIndex] += 1
                
    #Compute metrics per bar 
    orderFlowImbalance = calculateOrderFlowImbalance(buyVolume, sellVolume) 
    
    totalCount = buyCount + sellCount
    tradeCountImbalance = numpy.where(
        totalCount > 0,
        (buyCount - sellCount)/totalCount,
        0.0
    )
    
    averageTradeSize = numpy.where(tradeCount > 0, totalSize/tradeCount, 0.0)
    
    print(f"--> Aggregated Order Flow Imbalance (OFI) for {numberOfBars} bars")
    
    return {
        "order_Flow_Imbalance": orderFlowImbalance,
        "trade_count_imbalance": tradeCountImbalance,
        "average_trade_size": averageTradeSize
    }

  