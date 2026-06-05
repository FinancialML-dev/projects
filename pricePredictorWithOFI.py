"""
Stock/Crypto Price Predictor 

Predicts next candle direction (up/down) using historical price data.
--> Uses simple neural network trained on price returns. 

Author: Leo
Date: 2026-04-28
"""

import torch                                #uv add torch 
import torch.nn 
import numpy 
import matplotlib.pyplot as plotLibrary     #uv add matplotlib 
import bisect                                
from datetime import datetime 
from timebarsWithOFI import fetchCryptoData, formatDataToLists 
from dollarBarsWithOFI import fetchCryptoData, formatBarsToDictionaryList, createDollarBars, formatDollarBarsToList
from alpaca.data.timeframe import TimeFrame 

from sklearn.preprocessing import StandardScaler    #uv add scikit-learn
import copy 
import random

#-------------------------------
#Configuration 
#-------------------------------
BASE_CONFIG = {
    "SYMBOL": "ETH/USD", #"BTC/USD",
    "LOOKBACK": 32,                         #Use last <30> candles for prediction 
    "TRAIN_SPLIT": 0.8,                     #80% training data, 20% test data 
    "EPOCHS": 1000,                          #<100> Iterations
    "LEARNING_RATE": 0.001,                 #0.001 
    "BATCH_SIZE": 2048, #32,
    "DOLLAR_THRESHOLD": 500_000,            #$500K per dollar bar
    "USE_PRICE_RELATIVE_TO_MA": True,       #Toggle on/off
    "USE_VOLUME_FEATURES": True,            #Toggle on/off: intra-bar momentum, bar range, volume ratio, VWAP deviation 
    "USE_ORDER_FLOW_IMBALANCE": True,       #Toggel on/off: Order flow imbalance 
    "MODEL": "lstm",                        #"lstm" - Long Short Term Memory (LSTM) or "feedforward"
    "USE_LONGTERM_CONTEXT": True,           #Price relative to 200-bar Moving Average (MA) (long-term trend context)
    "USE_ADX": True,                        #Trend strength (0-100, >25 = trending)
    "USE_CROSS_ASSET": False, #True         #ETH/BTC ratio relative to its 20-bar MA (only when SYMBOL is ETH/USD)
    "PREDICTION_HOLD_BARS": 1 #5           #N-bar cumulative return for labels and hold period
}

TIME_BARS_CONFIG = {
    "TIMEFRAME": TimeFrame.Minute,          #Intervall in Minutes  
    "START_DATE": datetime(2026, 4, 1),     #Start date year, month, day --> of minute data 
    "END_DATE": datetime(2026, 4, 23),      #End date year, month, day --> of minute data
    "BAR_MODE": "time_bars"
}

DOLLAR_BARS_CONFIG = {
    "TIMEFRAME": TimeFrame.Minute,             #Intervall in Days  
    "START_DATE": datetime(2023, 1, 1),      #Start date year, month, day --> of minutes data 
    "END_DATE": datetime(2026, 5, 7),       #End date year, month, day --> of minutes data
    "BAR_MODE": "dollar_bars"
}

#CONFIG = {**BASE_CONFIG, **TIME_BARS_CONFIG}
CONFIG = {**BASE_CONFIG, **DOLLAR_BARS_CONFIG}

#-------------------------------
#Feature Engineering (input features, variables)
#-------------------------------
def calculateReturns(_prices): 
    """
    Calculates the percentage return between each consecutive price.
    (Calculate price returns (percent change)).

    Args:
        _prices (list): List of close prices.

    Returns:
        numpy.ndarray: Array of returns with length N-1 (one less than input).

    Raises:
        ZeroDivisionError: If any price in _prices is 0 (division by zero
                           in the percentage change formula).
        ValueError: If _prices has fewer than 2 elements (result would be
                    an empty array with no consecutive pairs to compare).
                    
    Example:
        >>> returns = calculateReturns([100, 105, 98, 110])
        >>> print(returns)
        [ 0.05  -0.0667  0.1224]   #+5%, -6.67%, +12.24%
    """

    prices = numpy.array(_prices)
    returns = (prices[1:] - prices[:-1]) / prices[:-1]      #This calculates the percentage (% change) return between consecutive prices (close prices).

    
    return returns 


def calculateRollingVolatility(_returns, _window=14):
    """ 
    Calculate rolling standard deviation of returns over a sliding window.

    For each position i in _returns, computes std of the preceding _window
    values. Positions before the window is fully populated are set to NaN.

    Args:
        _returns (array-like): Sequence of price returns (e.g. from calculateReturns()).
        _window  (int):        Rolling window size. Default 14.

    Returns:
        numpy.ndarray: Array of rolling volatility values, length len(_returns) - 1.
                       First (_window - 1) values are NaN.

    Raises:
        ValueError: If _window < 1 (std of an empty slice is undefined).
        IndexError: If _returns is empty (range produces no iterations,
                    returning an empty array — but downstream callers expecting
                    a populated array will fail).

    Example:
        >>> returns = calculateReturns([100, 105, 98, 110, 108, 115])
        >>> vol = calculateRollingVolatility(returns, _window=3)
        >>> print(vol)
        [nan, nan, 0.0356, 0.0612, 0.0489]   #first 2 are NaN, then rolling std
    """
    volatility = numpy.array([
        numpy.std(_returns[max(0, i - _window):i])
        if i >= _window else numpy.nan
        for i in range(1, len(_returns))
    ])
    
    return volatility


def calculateRSI(_prices, _window=14):
    """ 
    Calculate the Relative Strength Index (RSI) for a price series.

    Computes rolling average gains and losses using convolution, then derives
    RSI = 100 - (100 / (1 + RS)) where RS = averageGain / averageLoss.
    When averageLoss is 0 (all gains), RS is set to inf so RSI = 100 (fully overbought).
    The front is padded with _window NaN values to align with the input length.

    Args:
        _prices (list): List of close prices.
        _window (int):  Rolling window size for average gain/loss. Default 14.

    Returns:
        numpy.ndarray: RSI values, length _window + len(_prices) - 1.
                       First _window values are NaN (window not yet populated).
                       All other values are in range [0, 100].

    Raises:
        ValueError: If _prices has fewer than 2 elements (numpy.diff returns
                    an empty array, producing no gains or losses to average).
        ValueError: If _window < 1 (convolution kernel of size < 1 is invalid).

    Example:
        >>> prices = [44, 46, 45, 48, 47, 50, 49, 51, 52, 50, 53, 54, 52, 55, 56]
        >>> rsi = calculateRSI(prices, _window=14)
        >>> print(rsi[:14])     #first _window values are NaN
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
        >>> print(rsi[14])      #first populated RSI value
        ~66.7               #more gains than losses → above 50
    """
    prices = numpy.array(_prices)
    deltas = numpy.diff(prices)
    
    gains = numpy.where(deltas > 0, deltas, 0)
    losses = numpy.where(deltas < 0, -deltas, 0)
    
    averageGain = numpy.convolve(gains, numpy.ones(_window)/_window, mode="full")[:len(gains)]
    averageLoss = numpy.convolve(losses, numpy.ones(_window)/_window, mode="full")[:len(losses)]
    
    rs = numpy.full_like(averageGain, numpy.inf)    #default: inf (RSI=100)
    mask = averageLoss != 0
    rs[mask] = averageGain[mask] / averageLoss[mask]
    rsi = 100 - (100 / (1 + rs))
    #When rs = inf --> rsi = 100 - 0 = 100 (correct, fully overbought)
    
    #Pad front to match original price length 
    return numpy.concatenate([numpy.full(_window, numpy.nan), rsi])


def calculatePriceRelativeToMA(_prices, _window=20):
    """ 
    Calculate how far each price is above or below its rolling moving average (MA).

    Computes a simple moving average (SMA) via convolution, then returns the
    ratio (price - MA) / MA. Positive values mean price is above the MA;
    negative values mean below. The first (_window - 1) values are NaN
    because the window is not yet fully populated.

    Args:
        _prices (list): List of close prices.
        _window (int):  simple moving average (SMA) window size. Default 20.

    Returns:
        numpy.ndarray: Array of ratios, same length as _prices.
                       First (_window - 1) values are NaN.
                       e.g. 0.02 = 2% above MA, -0.03 = 3% below MA.

    Raises:
        ValueError: If _prices has fewer than _window elements (all values
                    will be NaN — no fully populated window exists).
        ZeroDivisionError: If any MA value is 0 (price series contains zeros,
                           causing division by zero in (price - MA) / MA).

    Example:
        >>> prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        >>> rel = calculatePriceRelativeToMA(prices, _window=5)
        >>> print(rel[:4])      #first (_window-1) values are NaN
        [nan, nan, nan, nan]
        >>> print(round(rel[4], 4))     #first populated value
        0.0196      #price ~2% above its 5-bar MA
    """
    prices = numpy.array(_prices)
    ma = numpy.convolve(prices, numpy.ones(_window)/_window, mode="full")[:len(prices)]
    
    #First (_window-1) values are based on incomplete windows - set to NaN 
    ma[:_window - 1] = numpy.nan 
    
    #Ratio: how far price is above/below its MA (0.02 = 2% above, -0.03 = 3% below)
    priceRelativeToMA = (prices - ma)/ma 
    
    return priceRelativeToMA 


def calculateIntraBarMomentum(_opens, _closes):
    """
    short summery: (close - open)/close - direction with each bar.
    
    Calculate intra-bar momentum as the relative move from open to close.

    Returns (close - open) / close for each bar — the fraction of the closing
    price gained or lost within the bar. Positive = bullish bar, negative = bearish.

    Args:
        _opens  (list): List of bar open prices.
        _closes (list): List of bar close prices. Must be the same length as _opens.

    Returns:
        numpy.ndarray: Array of momentum values, same length as _opens/_closes.
                       e.g. 0.02 = closed 2% above open, -0.03 = closed 3% below open.

    Raises:
        ZeroDivisionError: If any value in _closes is 0 (used as the denominator).
        ValueError: If _opens and _closes have different lengths (numpy will raise
                    when subtracting arrays of mismatched shapes).

    Example:
        >>> opens  = [100, 105, 98]
        >>> closes = [104, 102, 101]
        >>> momentum = calculateIntraBarMomentum(opens, closes)
        >>> print(numpy.round(momentum, 4))
        [ 0.0385 -0.0294  0.0297]   #bar 1: +3.85%, bar 2: -2.94%, bar 3: +2.97%
    """
    
    return (numpy.array(_closes) - numpy.array(_opens)) / numpy.array(_closes)


def calculateBarRange(_highs, _lows, _closes):
    """
    short summery: (high - low)/close - intra-bar as volatility/liquidity proxy.
    
    Calculate intra-bar range normalized by close price.

    Returns (high - low) / close for each bar — a volatility/liquidity proxy
    representing how wide the bar's range was relative to its closing price.
    Larger values indicate higher volatility within the bar.

    Args:
        _highs  (list): List of bar high prices.
        _lows   (list): List of bar low prices.
        _closes (list): List of bar close prices. Must be the same length as
                        _highs and _lows.

    Returns:
        numpy.ndarray: Array of range ratios, same length as the inputs.
                       Always >= 0 (high >= low by definition).
                       e.g. 0.03 = range was 3% of the closing price.

    Raises:
        ZeroDivisionError: If any value in _closes is 0 (used as the denominator).
        ValueError: If _highs, _lows, and _closes have different lengths (numpy
                    will raise when operating on mismatched array shapes).

    Example:
        >>> highs  = [106, 110, 103]
        >>> lows   = [100, 104,  99]
        >>> closes = [104, 108, 101]
        >>> barRange = calculateBarRange(highs, lows, closes)
        >>> print(numpy.round(barRange, 4))
        [0.0577, 0.0556, 0.0396]    #bar 1: range = 5.77% of close, etc.
    """
    
    return (numpy.array(_highs) - numpy.array(_lows)) / numpy.array(_closes)


def calculateVolumeRatio(_volumes, _window=14):
    """
    short summery: Dollar volume/rolling mean - above 1.0 = unusally active bar.
    
    Calculate each bar's dollar volume relative to its rolling mean volume.

    Uses a cumulative-sum approach for efficient rolling window computation.
    Ratio > 1.0 means the bar traded more than its recent average (unusually
    active); ratio < 1.0 means quieter than average. If the rolling mean is
    zero or negative (all-zero window), the value is set to NaN instead.

    Args:
        _volumes (list): List of dollar volumes per bar.
        _window  (int):  Rolling window size for the mean. Default 14.

    Returns:
        numpy.ndarray: Array of volume ratios, same length as _volumes.
                       First _window values are NaN (window not yet populated).
                       e.g. 2.0 = twice the average volume, 0.5 = half the average.

    Raises:
        ValueError: If _window < 1 (rolling mean becomes ill-defined —
                    division by _window produces inf, yielding ratios of 0).
        TypeError:  If _volumes is not iterable or contains non-numeric values
                    (numpy.array conversion will fail).

    Example:
        >>> volumes = [100, 200, 150, 300, 250, 400, 180]
        >>> ratios = calculateVolumeRatio(volumes, _window=3)
        >>> print(numpy.round(ratios, 2))
        [nan, nan, nan, 2.0, 1.15, 1.71, 0.57]
        #bar 3: 300 vs mean([100,200,150])=150 → 2.0x (very active)
        #bar 6: 180 vs mean([300,250,400])=317 → 0.57x (quiet)
    """
    volumes = numpy.array(_volumes, dtype=float)
    padded = numpy.concatenate([[0.0], numpy.cumsum(volumes)])
    ratios = numpy.full(len(volumes), numpy.nan)
    rollingSums = padded[_window:len(volumes)] - padded[:len(volumes) - _window]

    rollingMeans = rollingSums / _window
    with numpy.errstate(divide="ignore", invalid="ignore"):
        ratios[_window:] = numpy.where(rollingMeans > 0, volumes[_window:]/rollingMeans, numpy.nan)
        
    return ratios


def calculateVWAPDeviation(_highs, _lows, _closes, _dollarVolumes, _window=20):
    """
    short summery: (close - VWAP) / VWAP - price vs rolling volume-weighted average price.
    
    Calculate how far the close price deviates from its rolling volume-weighted average price (VWAP).

    For each bar, estimates share volume from dollar volume and typical price
    ((high + low + close) / 3), then computes a rolling VWAP over _window bars
    as sum(dollarVolume) / sum(estimatedVolume). Returns (close - VWAP) / VWAP.
    Positive = price above VWAP (bullish), negative = price below (bearish).
    If windowVolume or VWAP is zero, the value is set to NaN.

    Args:
        _highs        (list): List of bar high prices.
        _lows         (list): List of bar low prices.
        _closes       (list): List of bar close prices.
        _dollarVolumes(list): List of dollar volumes per bar.
                              All four lists must be the same length.
        _window       (int):  Rolling window size in bars. Default 20.

    Returns:
        numpy.ndarray: Array of VWAP deviation ratios, same length as _closes.
                       First _window values are NaN (window not yet populated).
                       e.g. 0.02 = close is 2% above rolling VWAP,
                            -0.03 = close is 3% below rolling VWAP.

    Raises:
        ValueError: If _window < 1 (indices become invalid — numpy.arange
                    produces an empty array and no values are filled).
        ValueError: If input arrays have different lengths (numpy will raise
                    when operating on mismatched array shapes).
        TypeError:  If any input is not iterable or contains non-numeric
                    values (numpy.array conversion will fail).

    Example:
        >>> highs  = [105, 108, 110, 115, 112]
        >>> lows   = [99,  102, 104, 108, 106]
        >>> closes = [102, 105, 107, 112, 109]
        >>> dollarVolumes  = [500_000, 600_000, 550_000, 700_000, 480_000]
        >>> deviation1 = calculateVWAPDeviation(highs, lows, closes, dollarVolumes, _window=2)
        >>> print(numpy.round(deviation1, 4))
        [nan, nan, 0.0326, 0.0571, -0.0051]
        #bar 2: close 3.26% above rolling VWAP → bullish
        #bar 4: close 0.51% below rolling VWAP → slightly bearish
    """
    highs = numpy.array(_highs)
    lows = numpy.array(_lows)
    closes = numpy.array(_closes)
    dvs = numpy.array(_dollarVolumes, dtype=float)
    typicalPrice = (highs + lows + closes)/3.0
    estimatedVolume = numpy.where(typicalPrice > 0, dvs / typicalPrice, 0.0)
    paddedDV = numpy.concatenate([[0.0], numpy.cumsum(dvs)])
    paddedVolume = numpy.concatenate([[0.0], numpy.cumsum(estimatedVolume)]) 
    deviation = numpy.full(len(closes), numpy.nan)
    indices = numpy.arange(_window, len(closes))
    windowDV = paddedDV[indices] - paddedDV[indices - _window]
    windowVolume = paddedVolume[indices] - paddedVolume[indices - _window]
    with numpy.errstate(divide="ignore", invalid="ignore"):
        vwap = numpy.where(windowVolume > 0, windowDV / windowVolume, numpy.nan)
        deviation[_window:] = numpy.where(
            (vwap > 0) & ~numpy.isnan(vwap), 
            (closes[_window:] - vwap) / vwap, 
            numpy.nan
        )
        
    return deviation

def calculateADX(_highs, _lows, _closes, _window=14):
    """
    Calculate the Average Directional Index (ADX) — measures trend strength,
    not direction. High ADX = strong trend (up or down). Low ADX = ranging market.

    Algorithm:
        1. True Range (TR)  = max(high-low, |high-prevClose|, |low-prevClose|)
        2. +DM / -DM        = upward / downward directional movement per bar
        3. Smooth TR, +DM, -DM with a rolling mean of _window bars
        4. +DI / -DI        = 100 * smoothed±DM / smoothedTR
        5. DX               = 100 * |+DI - -DI| / (+DI + -DI)
        6. ADX              = rolling mean of DX over _window bars

    Average Directional Index (ADX) interpretation:
        < 20  → ranging / weak trend
        25–50 → trending (strong signal)
        > 50  → very strong trend (rare)
        example
        ADX > 25: trending  |  ADX < 20: ranging  |  range: [0, 100]

    Args:
        _highs   (list): Bar high prices. Length N.
        _lows    (list): Bar low prices. Length N.
        _closes  (list): Bar close prices. Length N.
                        All three must be the same length.
        _window  (int, optional): Smoothing window for TR, DM, and DX. Defaults to 14.

    Returns:
        numpy.ndarray: ADX values of length N -> len(_closes).
                    First (2 * _window - 1) values are NaN — warmup period
                    for two rounds of smoothing. First element is always NaN
                    (True Range requires a previous close).

    Raises:
        ValueError: If _highs, _lows, and _closes are different lengths —
                    the directional movement arrays will be misaligned and
                    numpy operations will raise a shape mismatch.
        TypeError:  If any input contains non-numeric values —
                    numpy.array() conversion will succeed but arithmetic
                    operations will raise.

    Example:
        >>> highs  = [44, 45, 46, 44, 43, 45, 47, 48, 47, 46, 48, 49, 50, 49, 48,
        ...           50, 51, 52, 51, 50, 52, 53, 54, 53, 52, 54, 55, 56, 55, 54]
        >>> lows   = [42, 43, 44, 42, 41, 43, 45, 46, 45, 44, 46, 47, 48, 47, 46,
        ...           48, 49, 50, 49, 48, 50, 51, 52, 51, 50, 52, 53, 54, 53, 52]
        >>> closes = [43, 44, 45, 43, 42, 44, 46, 47, 46, 45, 47, 48, 49, 48, 47,
        ...           49, 50, 51, 50, 49, 51, 52, 53, 52, 51, 53, 54, 55, 54, 53]
        >>> adx = calculateADX(highs, lows, closes, _window=14)
        >>> print(adx[27])      # first valid value (2 * 14 - 1 = 27)
        32.4                    # strong trend at that bar
        >>> print(numpy.isnan(adx[0]))
        True
    """
    highs = numpy.array(_highs)
    lows = numpy.array(_lows)
    closes = numpy.array(_closes)
    
    #True Range: largest of (high-low), |high-previousClose|, |low-previousClose|
    trueRange = numpy.maximum(
        highs[1:] - lows[1:],
        numpy.maximum(
            numpy.abs(highs[1:] - closes[:-1]),
            numpy.abs(lows[1:] - closes[:-1])
        )
    )
    
    #Directional Movement 
    upMove = highs[1:] - highs[:-1]
    downMove = lows[:-1] - lows[1:]
    
    plusDirectionalMovement = numpy.where( (upMove > downMove) & (upMove > 0), upMove, 0.0 )
    minusDirectionalMovement = numpy.where( (downMove > upMove) & (downMove > 0), downMove, 0.0 )
    
    #Smooth True Range (TR), plus directional movement (+DM), minus directional movement (-DM) with rolling mean "moving average"
    kernel = numpy.ones(_window) / _window
    smoothedTrueRange = numpy.convolve(trueRange, kernel, mode="full")[:len(trueRange)]
    smoothedPlusDirectionalMovement = numpy.convolve(plusDirectionalMovement, kernel, mode="full")[:len(plusDirectionalMovement)]
    smoothedMinusDirectionalMovement = numpy.convolve(minusDirectionalMovement, kernel, mode="full")[:len(minusDirectionalMovement)]
    
    #plus Directional Indicator (+DI) and minus Directional Indicator (-DI)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        plusDirectionalIndicator = numpy.where(smoothedTrueRange > 0, 100*smoothedPlusDirectionalMovement/smoothedTrueRange, 0.0)
        minusDirectionalIndicator = numpy.where(smoothedTrueRange > 0, 100*smoothedMinusDirectionalMovement/smoothedTrueRange, 0.0)
        
        directionalSum = plusDirectionalIndicator + minusDirectionalIndicator
        directionalIndex = numpy.where(directionalSum > 0, 100*numpy.abs(plusDirectionalIndicator-minusDirectionalIndicator)/directionalSum, 0.0)
        
    #Average Directional Index (ADX) = smoothed Directional Index (DX)
    averageDirectionalIndex = numpy.convolve(directionalIndex, kernel, mode="full")[:len(directionalIndex)] 
    averageDirectionalIndex[:2*_window-2] = numpy.nan
    
    #Pad front to match input length N
    return numpy.concatenate([ [numpy.nan], averageDirectionalIndex ])
    

def createFeaturesAndLabels(_closePrices, _openPrices=None, _highPrices=None, _lowPrices=None, _volumes=None, _orderFlowImbalance=None, _crossAssetRatio=None, _lookback=30): 
    """
    Builds a sliding-window dataset of return sequences and binary direction labels.
    (Create feature windows and labels for supervised learning.)
    
    For each position i, the feature is a window of the previous `_lookback` returns,
    and the label is 1 if the next return is positive (price up), 0 if negative (price down).
    N is controlled by CONFIG["PREDICTION_HOLD_BARS"] — N=1 is single next-bar direction, N>1 predicts the direction of a multi-bar hold.
    (For each time step t, features are returns from t-lookback to t-1.
    Label is whether price went up at time t --> (1) or down (0).)

    Feature composition (per CONFIG flags):
        Always included : returns, volatility, RSI                      → _lookback * 3 columns
        USE_PRICE_RELATIVE_TO_MA=True                                   → _lookback * 4 columns
        USE_VOLUME_FEATURES=True + volumes provided                     → _lookback * 6 columns
                                                                          (adds intraBarMomentum,
                                                                          barRange, vwapDeviation)
        USE_ORDER_FLOW_IMBALANCE=True + _orderFlowImbalance provided    → _lookback * 7 columns
        USE_LONGTERM_CONTEXT=True                                       → _lookback * 8 columns
                                                                          (adds price relative to
                                                                          200-period MA; sets
                                                                          validStartIndex to 200)
        USE_ADX=True + _highPrices and _lowPrices provided              → _lookback * 9 columns
                                                                          (adds ADX trend strength;
                                                                           warmup ~27 bars, covered
                                                                           by nanWindow=200)
        USE_CROSS_ASSET=True + _crossAssetRatio provided                → _lookback * 10 columns
                                                                          (adds ETH/BTC ratio relative
                                                                           to its 20-bar MA; only
                                                                           meaningful when SYMBOL
                                                                           is ETH/USD)

    Args:
        _closePrices         (list):           List of close prices. Required.
        _openPrices          (list, optional): List of open prices. Required when
                                               USE_VOLUME_FEATURES is True.
        _highPrices          (list, optional): List of high prices. Required when
                                               USE_VOLUME_FEATURES or USE_ADX is True.
        _lowPrices           (list, optional): List of low prices. Required when
                                               USE_VOLUME_FEATURES or USE_ADX is True.
        _volumes             (list, optional): List of dollar volumes. Required when
                                               USE_VOLUME_FEATURES is True.
        _orderFlowImbalance  (list, optional): List of OFI values per bar. Required when
                                               USE_ORDER_FLOW_IMBALANCE is True.
                                               OFI = (close - open) / (high - low).
        _crossAssetRatio     (list, optional): List of ETH/BTC price ratios aligned to
                                               each bar. Required when USE_CROSS_ASSET is True.
                                               Computed as closePrice[i] / btcClose[i] in the
                                               main pipeline via bisect timestamp alignment.
        _lookback            (int, optional):  Number of past returns per feature window.
                                               Defaults to 30.

    Returns:
        tuple:
            - features (numpy.ndarray): Shape (N, _lookback * F) — each row is a return window. Where F is the
                                        number of active feature arrays (3–10,
                                        controlled by CONFIG flags). Each row is
                                        one concatenated multi-feature window.
            - labels   (numpy.ndarray): Shape (N,) — 1 if N-bar cumulative return (next return) > 0 (up),
                                        0 if N-bar cumulative return (next return) <= 0 (down).
                                        Note: final _lookback + (PREDICTION_HOLD_BARS - 1)
                                        bars are dropped — not enough future bars to label.

    Raises:
        KeyError:   If CONFIG is missing "USE_VOLUME_FEATURES", "USE_PRICE_RELATIVE_TO_MA",
                    "USE_ORDER_FLOW_IMBALANCE", "USE_LONGTERM_CONTEXT", "USE_ADX", 
                    "USE_CROSS_ASSET", "PREDICTION_HOLD_BARS", or "LOOKBACK" keys.
        TypeError:  If USE_VOLUME_FEATURES is True but _openPrices, _highPrices,
                    _lowPrices, or _volumes is None (passed to sub-calculators
                    that expect arrays).
        ValueError: If _closePrices has fewer elements than validStartIndex + _lookback + 1
                    (not enough data to produce even one sample — features and labels will
                    both be empty arrays). Most likely when USE_LONGTERM_CONTEXT=True
                    forces validStartIndex=200 on a short dataset.

    Example:
        >>> # With CONFIG["USE_PRICE_RELATIVE_TO_MA"]=False, USE_VOLUME_FEATURES=False, 
        >>> #      USE_ORDER_FLOW_IMBALANCE=False, USE_LONGTERM_CONTEXT=False, 
        >>> #      USE_ADX=False, USE_CROSS_ASSET=False
        >>> # With all CONFIG flags False and PREDICTION_HOLD_BARS=1 (single next-bar label)
        >>> prices = [100, 102, 101, 105, 103, 107, 109, 108, 111, 110,
        ...           112, 115, 113, 116, 118]   # 15 prices, validStartIndex=14
        >>> features, labels = createFeaturesAndLabels(prices, _lookback=30)
        >>> print(features.shape)   # (N, 90)  — N windows of 90 returns each, — _lookback * 3 feature arrays
        >>> print(labels[0])        # 1 or 0   — next move up, then down, did price go up on the next bar?
    """
    returns = calculateReturns(_closePrices)
    volatility = calculateRollingVolatility(returns, _window=14)
    rsi = calculateRSI(_closePrices, _window=14)
    priceRelativeToMA = calculatePriceRelativeToMA(_closePrices, _window=20)
    useVolumeFeatures = CONFIG.get("USE_VOLUME_FEATURES", False) and _volumes is not None
    useOrderFlowImbalance = _orderFlowImbalance is not None and CONFIG.get("USE_ORDER_FLOW_IMBALANCE", False)
    if useVolumeFeatures:
        intraBarMomentum = calculateIntraBarMomentum(_openPrices, _closePrices)
        barRange = calculateBarRange(_highPrices, _lowPrices, _closePrices)
        volumeRatio = calculateVolumeRatio(_volumes, _window=14)
        vwapDeviation = calculateVWAPDeviation(_highPrices, _lowPrices, _closePrices, _volumes, _window=20)
        
    if useOrderFlowImbalance:
        orderFlowImbalance = numpy.array(_orderFlowImbalance)
    
    useLongTermContext = CONFIG.get("USE_LONGTERM_CONTEXT", False)
    if useLongTermContext:
        longTermMovingAverage = calculatePriceRelativeToMA(_closePrices, _window=200)
        
    useAverageDirectionalIndex = CONFIG.get("USE_ADX", False) and _highPrices is not None and _lowPrices is not None
    if useAverageDirectionalIndex:
        averageDirectionalIndex = calculateADX(_highPrices, _lowPrices, _closePrices, _window=14)
    
    useCrossAsset = CONFIG.get("USE_CROSS_ASSET", False) and _crossAssetRatio is not None
    if useCrossAsset:
        crossAssetRelativeToMovingAverage = calculatePriceRelativeToMA(_crossAssetRatio, _window=20)
    
    #Align all arrays - RSI is based on prices (length N),
    #returns/volatility are length N-1. Trim RSI to match.
    #Align to returns/priceRelativeToMA length (N-1)
    rsi = rsi[1:]   #Drop first element to match returns length
    priceRelativeToMA = priceRelativeToMA[1:]
    if useVolumeFeatures: 
        intraBarMomentum = intraBarMomentum[1:]
        barRange = barRange[1:]
        volumeRatio = volumeRatio[1:]
        vwapDeviation = vwapDeviation[1:]
    
    if useOrderFlowImbalance:
        orderFlowImbalance = orderFlowImbalance[1:]
    
    if useLongTermContext:
        longTermMovingAverage = longTermMovingAverage[1:]
        
    if useAverageDirectionalIndex:
        averageDirectionalIndex = averageDirectionalIndex[1:]
    
    if useCrossAsset:
        crossAssetRelativeToMovingAverage = crossAssetRelativeToMovingAverage[1:]
    
    #Trim NaN values from the start (first 14 are invalid) — use 20 since MA window is larger than RSI/volatility window
    #validStartIndex = 20 #20 #If we add the MA window set validStartIndex = 20
    #validStartIndex = 20 if CONFIG["USE_PRICE_RELATIVE_TO_MA"] else 14
    #validStartIndex = 14 if not CONFIG["USE_PRICE_RELATIVE_TO_MA"] else 20
    #nanWindow = 14 if not CONFIG["USE_PRICE_RELATIVE_TO_MA"] else 20 #For the cases when we vary the LOOKBACK to small sizes 
    nanWindow = 14
    if CONFIG["USE_PRICE_RELATIVE_TO_MA"]:
        nanWindow = max(nanWindow, 20)
    if useVolumeFeatures:
        nanWindow = max(nanWindow, 20)      #VWAP window = 20
    if useLongTermContext:
        nanWindow = max(nanWindow, 200)
    if useCrossAsset:
        nanWindow = max(nanWindow, 20) 
    
    validStartIndex = max(nanWindow, CONFIG["LOOKBACK"])
    
    returns = returns[validStartIndex:]
    volatility = volatility[validStartIndex:]
    rsi = rsi[validStartIndex:]
    priceRelativeToMA = priceRelativeToMA[validStartIndex:]
    
    if useVolumeFeatures: 
        intraBarMomentum = intraBarMomentum[validStartIndex:]
        barRange = barRange[validStartIndex:]
        volumeRatio = volumeRatio[validStartIndex:]
        vwapDeviation = vwapDeviation[validStartIndex:]
    
    if useOrderFlowImbalance:
        orderFlowImbalance = orderFlowImbalance[validStartIndex:]
        
    if useLongTermContext:
        longTermMovingAverage = longTermMovingAverage[validStartIndex:]
    
    if useAverageDirectionalIndex:
        averageDirectionalIndex = averageDirectionalIndex[validStartIndex:]
    
    if useCrossAsset:
        crossAssetRelativeToMovingAverage = crossAssetRelativeToMovingAverage[validStartIndex:]
    
    features = []
    labels = []
    
    for i in range(_lookback, len(returns)):
        #Features (inputs): returns from [i - _lookback:i]  
        #window = returns[i - _lookback:i]
        #Return window - shape (_lookback,)
        returnWindow = returns[i - _lookback:i]
        
        #Volatility window - shape(_lookback,)
        volatilityWindow = volatility[i - _lookback:i]
        
        rsiWindow = rsi[i - _lookback:i]
        
        priceRelativeToMAWindow = priceRelativeToMA[i - _lookback:i]
        
        #Concatenate both into one feature vector - shape (_lookback * 3)
        #window = numpy.concatenate([returnWindow, volatilityWindow, rsiWindow])
        #features.append(window)         #Features shape goes from (N, 64) → (N, 96) — 32 returns + 32 volatility + 32 RSI. 
        #window = numpy.concatenate([returnWindow, volatilityWindow, rsiWindow, priceRelativeToMAWindow]) 
        #features.append(window)         #Features shape goes from (N, 96) → (N, 128) — 32 returns + 32 volatility + 32 RSI + 32 PriceRelativeToMA. 
        
        #Build feature list dynamically
        featureParts = [returnWindow, volatilityWindow, rsiWindow]  #Features shape goes from (N, 64) → (N, 96) — 32 returns + 32 volatility + 32 RSI. 
        
        if CONFIG["USE_PRICE_RELATIVE_TO_MA"]:
            featureParts.append(priceRelativeToMAWindow)            #Features shape goes from (N, 96) → (N, 128) — 32 returns + 32 volatility + 32 RSI + 32 PriceRelativeToMA. 
            
        if useVolumeFeatures:
            featureParts.append(intraBarMomentum[i - _lookback:i])
            featureParts.append(barRange[i - _lookback:i])
            #featureParts.append(volumeRatio[i - _lookback:i])
            featureParts.append(vwapDeviation[i - _lookback:i])
        
        if useOrderFlowImbalance: 
            featureParts.append(orderFlowImbalance[i - _lookback:i])
            
        if useLongTermContext:
            featureParts.append(longTermMovingAverage[i - _lookback:i])
            
        if useAverageDirectionalIndex: 
            featureParts.append(averageDirectionalIndex[i - _lookback:i])
            
        if useCrossAsset:
            featureParts.append(crossAssetRelativeToMovingAverage[i - _lookback:i])
        
        window = numpy.concatenate(featureParts)
        
        #features.append(window) 
        
        #Label: 1 if next return positive, 0 if negative 
        #nextReturn = returns[i] 
        #labels.append(1 if nextReturn > 0 else 0) 
        #Predict next N bars cumulative retun 
        N = CONFIG["PREDICTION_HOLD_BARS"] #or 10 
        if i + N <= len(returns):
            numberOfBarReturn = numpy.prod(1 + returns[i:i+N]) - 1
            labels.append(1 if numberOfBarReturn > 0 else 0) 
            features.append(window) 
    
    return numpy.array(features), numpy.array(labels), returns 

def trainTestSplit(_X, _y, _trainRatio=0.8): 
    """
    --> Split data into training and testing sets.
    
    Splits features (_X -> inputs) and labels (_y -> outputs) into chronological train and test sets.

    Splits at a fixed index (no shuffling) to preserve time order —
    training on past data, testing on future data.

    Args:
        _X (numpy.ndarray): Feature array of shape (N, lookback).
        _y (numpy.ndarray): Label array of shape (N,).
        _trainRatio (float, optional): Fraction of data used for training. Defaults to 0.8.

    Returns:
        tuple: (XtrainData, XtestData, yTrainData, yTestData)
            - XtrainData: First 80% of features (training).
            - XtestData:  Last  20% of features (testing).
            - yTrainData: First 80% of labels   (training).
            - yTestData:  Last  20% of labels   (testing).

    Raises:
        ValueError: If _trainRatio is not in (0, 1) — a ratio of 0 or 1 produces
                    an empty train or test set; negative values or >1 are nonsensical.
        ValueError: If len(_X) != len(_y) — splitIndex is derived from _X but applied
                    to both arrays; mismatched lengths will silently misalign labels.

    Example:
        >>> features, labels = createFeaturesAndLabels(closePrices, _lookback=30)
        >>> XtrainData, XtestData, yTrainData, yTestData = trainTestSplit(features, labels, _trainRatio=0.8)
        >>> print(XtrainData.shape)   # (N*0.8, 30)
        >>> print(XtestData.shape)    # (N*0.2, 30)
    """
    splitIndex = int(_trainRatio * len(_X))     #--> 0.8 * len(features) 
    
    XtrainData = _X[:splitIndex]                #all the first values to the splitIndex (80% of the first values)
    XtestData = _X[splitIndex:]                 #all the last values starting from the splitIndex (20% of the last values)
    yTrainData = _y[:splitIndex]                #... (80% of the first values)
    yTestData = _y[splitIndex:]                 #... (20% of the last values)
    
    return XtrainData, XtestData, yTrainData, yTestData 

#-------------------------------
#Model definition (Neural Network)
#-------------------------------
class PricePredictor(torch.nn.Module): 
    """
    Binary classifier neural network that predicts price direction (up/down).

    Architecture:
        Input(_inputSize) → Linear(64) → ReLU → Dropout(0.2)
                          → Linear(32) → ReLU → Dropout(0.2)
                          → Linear(1)  → Sigmoid → output (0.0 - 1.0)

    The output is a probability — values above 0.5 indicate price going up (1),
    values below 0.5 indicate price going down (0).

    Args:
        _inputSize (int): Number of input features (equal to the lookback window size).

    Raises:
        RuntimeError: If the input tensor passed to forward() has a feature dimension
                      that does not match _inputSize (PyTorch Linear layer will reject
                      mismatched shapes at runtime).
        ValueError:   If _inputSize < 1 (torch.nn.Linear requires a positive in_features).

    Example:
        >>> model = PricePredictor(_inputSize=30)       # 30 = lookback window
        >>> features, labels = createFeaturesAndLabels(closePrices, _lookback=30)
        >>> inputTensor = torch.tensor(features[0], dtype=torch.float32)
        >>> prediction = model(inputTensor)
        >>> print(prediction)
        tensor([0.73])                                  # 73% chance price goes up
    """
    def __init__(self, _inputSize):
        super().__init__()
        self.neuralNetwork = torch.nn.Sequential(   
            torch.nn.Linear(_inputSize, 64),            #input Layer (layer 1): _inputSize (features) --> Hidden layer: 64 neurons/nodes
            torch.nn.ReLU(),                            #Activation function: rectified linear unit (ReLU) applied element-wise to each hidden neuron
            torch.nn.Dropout(0.2),                      #randomly turns off 20% of neurons during each training. --> Why? To prevent overfitting. 
            torch.nn.Linear(64, 32),                    #Hidden Layer (layer 2): 64 neurons/nodes --> Second Hidden layer: 32 neurons/nodes
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),                     #Second Hidden Layer (layer 3): 32 neurons/nodes --> Output layer (target): 1 neuron/node (1 output) 
            torch.nn.Sigmoid()                          #Squashes any number into a value between 0 and 1. Express the answer as a probability
        )
        
    def forward(self, _x):
        return self.neuralNetwork(_x)


class PricePredictorLSTM(torch.nn.Module):
    """
    Long Short Term Memory (LSTM)-based binary classifier for predicting next-bar price direction.

    Wraps a PyTorch LSTM followed by a small feedforward classifier head.
    Expects flat feature vectors from createFeaturesAndLabels() and reshapes
    them internally into (batch, lookback, numberFeatures) sequences before
    passing them to the LSTM.
    
    Short: Reshapes the flat feature vector back into a sequence (lookback, num_features)
    so the LSTM can learn patterns across timesteps — something the feedforward
    model cannot do since it treats all timesteps as independent inputs.

    Architecture:
        Input  : flat vector of shape (batch, lookback * numberFeatures)
        Reshape: (batch, numberFeatures, lookback) → transpose → (batch, lookback, numberFeatures)
        LSTM   : _numberLayers stacked LSTM layers, hidden size _hiddenSize, dropout=0.2
        Head   : Linear(_hiddenSize → 32) → ReLU → Dropout(0.2) → Linear(32 → 1) → Sigmoid
        Output : scalar in (0, 1) — probability that next bar closes up

    Args:
        _numberFeatures (int): Number of features per timestep (F in the feature matrix). (= total input size / lookback).
                            Must match the F used in createFeaturesAndLabels().
        _lookback       (int): Number of timesteps per sequence window.
                            Must match CONFIG["LOOKBACK"].
        _hiddenSize     (int, optional): LSTM hidden state size. Defaults to 64.
        _numberLayers   (int, optional): Number of stacked LSTM layers. Defaults to 2.

    Raises:
        RuntimeError: If the flat input vector length does not equal
                    _lookback * _numberFeatures — the reshape in forward()
                    will produce an incorrect tensor shape and the LSTM
                    will reject it.
        RuntimeError: If _numberLayers > 1 and dropout=0.2 is applied —
                    PyTorch emits a warning (not an error) if _numberLayers=1
                    since dropout has no effect between layers that don't exist.

    Example:
        >>> numberFeatures = 7        # returns, volatility, RSI, MA, momentum, barRange, OFI
        >>> lookback = 30
        >>> model = PricePredictorLSTM(_numberFeatures=numberFeatures, _lookback=lookback)
        >>> features, labels = createFeaturesAndLabels(closes, _orderFlowImbalance=ofi, _lookback=lookback)
        >>> Xtrain, Xtest, yTrain, yTest = trainTestSplit(features, labels)
        >>> history = trainModel(model, Xtrain, yTrain, Xtest, yTest, _epochs=100)
        >>> print(model)
        PricePredictorLSTM(lstm=LSTM(7, 64, num_layers=2, ...), classifier=Sequential(...))
    """    
    def __init__(self, _numberFeatures, _lookback, _hiddenSize=64, _numberLayers=2):
        super().__init__()
        self.numberFeatures = _numberFeatures
        self.lookback = _lookback
        
        self.lstm = torch.nn.LSTM(
            input_size=_numberFeatures,
            hidden_size=_hiddenSize,
            num_layers=_numberLayers,
            batch_first=True,
            dropout=0.2
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(_hiddenSize, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, _x):
        # _x: (batch, lookback * numberFeatures) - flat vector from createFeaturesAndLabels
        # Feature are laid out as: [f1_t0..f1_t7, f2_t0..f2_t7, ...]
        # Reshape to (batch, numberFeatures, lookback) then transpose to 
        # (batch, lookback, numberFeatures) - the format LSTM expects 
        x = _x.view(_x.size(0), self.numberFeatures, self.lookback)
        x = x.transpose(1, 2)                           # (batch, lookback, numberFeatures)
        
        lstmOut, _ = self.lstm(x)                       # (batch, lookback, hidden_size) 
        lastHidden = lstmOut[:, -1, :]                  # last timestamp: (batch, hidden_size)

        return self.classifier(lastHidden)


#-------------------------------
#Training: Train the model (our Neural Network)
#-------------------------------
def trainModel(_model, _Xtrain, _yTrain, _Xtest, _yTest, _epochs=100, _learningRate=0.001, _batchSize=32): 
    """
    Trains the PricePredictor model using mini-batch gradient descent, BCELoss, and Adam optimizer.

    Each epoch iterates over shuffled mini-batches via DataLoader. Loss and accuracy
    are averaged across all batches in the epoch before being recorded. Early stopping
    monitors test loss and halts training if no improvement is seen for `patience` epochs,
    then restores the best weights before returning.

    Args:
        _model        (torch.nn.Module): The neural network to train (PricePredictor or PricePredictorLSTM).
        _Xtrain       (numpy.ndarray):   Training features of shape (N, lookback * F).
        _yTrain       (numpy.ndarray):   Training labels of shape (N,) with values 0 or 1.
        _Xtest        (numpy.ndarray):   Test features used for early stopping evaluation.
        _yTest        (numpy.ndarray):   Test labels used for early stopping evaluation.
        _epochs       (int, optional):   Maximum number of full passes over the training data. Defaults to 100.
        _learningRate (float, optional): Step size for Adam optimizer. Defaults to 0.001.
        _batchSize    (int, optional):   Number of samples per mini-batch. DataLoader shuffles

    Returns:
        dict: Training history with keys:
            - "loss"     (list[float]): Epoch-averaged training loss across all mini-batches.
            - "accuracy" (list[float]): Epoch-averaged training accuracy across all mini-batches.

    Raises:
        RuntimeError:      If the feature dimension of _Xtrain does not match the model's
                           expected input size (PyTorch rejects mismatched shapes on the
                           first forward pass).
        ValueError:        If _Xtrain and _yTrain have different numbers of rows
                           (BCELoss will fail when comparing predictions to labels).
        UnboundLocalError: If _epochs=0 — the training loop never runs, so
                           bestModelWeights is never assigned and load_state_dict()
                           raises on the restore step after the loop.
        
    Example:
        >>> model = PricePredictorLSTM(_numberFeatures=7, _lookback=30)
        >>> XtrainData, XtestData, yTrainData, yTestData = trainTestSplit(features, labels)
        >>> history = trainModel(model, XtrainData, yTrainData, XtestData, yTestData, _epochs=100, _batchSize=32)
        --------------------
        Training...
        --------------------
        Epoch/Iteration   0: Loss=0.7124, Accuracy=0.4823
        Epoch/Iteration  20: Loss=0.6891, Accuracy=0.5241
        Epoch/Iteration  40: Loss=0.6543, Accuracy=0.5780
        Early stopping at epoch/iteration 63
        --------------------
    """
    XtrainData = torch.FloatTensor(_Xtrain)
    yTrainData = torch.FloatTensor(_yTrain).reshape(-1, 1) 
    XtestData = torch.FloatTensor(_Xtest)
    yTestData = torch.FloatTensor(_yTest).reshape(-1, 1)
    
    # DataLoader handles shuffling and splitting into mini-batch each epoch 
    dataset = torch.utils.data.TensorDataset(XtrainData, yTrainData)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=_batchSize, shuffle=True)
    
    optimizer = torch.optim.Adam(_model.parameters(), lr=_learningRate)
    criterion = torch.nn.BCELoss()
    
    history = {"loss": [], "accuracy": []}
    
    print("\n" + "-"*20)
    print("Training...")
    print("-"*20)
    
    #Early stopping - stop training when test accuracy stops improving. Than running all fixed epochs/iterations.
    bestLoss = float("inf")
    patience = 15 #40 #50           #Stop if no improvement for 50 epochs 
    noImpove = 0
    
    for item in range(_epochs):
        _model.train()
        
        batchLosses = []
        batchAccuracies = []
        
        #Mini-batch loop 
        for Xbatch, yBatch in dataLoader:
        
            #Forward pass
            predictions = _model(Xbatch)
            loss = criterion(predictions, yBatch)
        
            #Backward pass
            optimizer.zero_grad()       #Reset/Clear all gradients 
            loss.backward()             #partial derivative of dLoss/dmodel = ..., dLoss/dy = ... [chain rule applied]
            optimizer.step()            #Performs a single optimization step
        
            #Calculate accuracy 
            accuracy = ( (predictions > 0.5).float() == yBatch ).float().mean()
            batchLosses.append(loss.item())
            batchAccuracies.append(accuracy.item()) 
        
        #Average loss and accuracy across all batches in this epoch/"a full cycle iteration"
        epochLoss = sum(batchLosses) / len(batchLosses) 
        epochAccuracy = sum(batchAccuracies) / len(batchAccuracies)
        
        history["loss"].append(epochLoss)
        history["accuracy"].append(epochAccuracy)
        
        if item % 20 == 0:
            print(f"Epoch/Iteration {item:3d}: Loss={epochLoss:.4f}, Accuracy={epochAccuracy:.4f} ") 
            #print(f"    Layer 1: w={_model.neuralNetwork[0].weight.data.shape}, b={_model.neuralNetwork[0].bias.data.shape}") #Our model wraps the Sequential inside `self.neuralNetwork` --> So, For Sequential model we use: model.neuralNetwork[0] the first Linear(x, y) layer. Layer 1 --> shape: x rows, y column 
            #print(f"    Layer 2: w={_model.neuralNetwork[3].weight.data.shape}, b={_model.neuralNetwork[3].bias.data.shape}") #model.neuralNetwork[3] the second Linear(x, y) layer. Layer 2 --> shape: x row, y columns  
            #print(f"    Layer 3: w={_model.neuralNetwork[6].weight.data.shape}, b={_model.neuralNetwork[6].bias.data.shape}") #model.neuralNetwork[6] the second Linear(x, y) layer. Layer 3 --> shape: x row, y columns  
    
            
        #Evaluate on TEST data for early stopping (Optimizer)  
        _model.eval()
        with torch.no_grad():
            testPredictions = _model(XtestData)
            testLoss = criterion(testPredictions, yTestData).item()  
            
        if testLoss < bestLoss:
            bestLoss = testLoss
            noImpove = 0
            bestModelWeights = copy.deepcopy(_model.state_dict())   #Save best. Only save when improved 
        else:
            noImpove += 1
            if noImpove >= patience:
                print(f"Early stopping at epoch/iteration {item:3d}")
                break
            
    #Restore the best weights 
    _model.load_state_dict(bestModelWeights)
    
    print("-"*20 + "\n")
    
    return history 


#-------------------------------
#Evaluation 
#-------------------------------
def evaluateModel(_model, _Xtest, _yTest): 
    """
    Evaluate model on test set.
    Evaluates the trained model on unseen test data and returns classification metrics.

    Runs inference with Dropout disabled (model.eval() + no_grad), then computes
    accuracy, baseline, precision, recall, F1, and a confusion matrix.

    Args:
        _model (PricePredictor): The trained neural network.
        _Xtest (numpy.ndarray): Test features of shape (N, lookback).
        _yTest (numpy.ndarray): Test labels of shape (N,) with values 0 or 1.

    Returns:
        dict: Evaluation metrics with keys:
            - "accuracy"        (float): % of correct predictions.
            - "baseline"        (float): % accuracy of always guessing the most common class.
            - "precision"       (float): Of all predicted UP, how many were actually UP.
            - "recall"          (float): Of all actual UP, how many were correctly predicted.
            - "f1"              (float): Harmonic mean of precision and recall.
            - "confusionMatrix" (dict):  truePositive, falsePositive, trueNegative, falseNegative.

    Raises:
        RuntimeError: If the feature dimension of _Xtest does not match the model's
                      _inputSize (PyTorch Linear layer rejects mismatched shapes on
                      the forward pass).
        ValueError:   If _Xtest and _yTest have different numbers of rows (comparison
                      of predictedClass == yTestData will silently misalign or raise
                      a shape broadcast error).
        AttributeError: If _yTest is not a numpy array (baseline calls _yTest.mean(),
                        which is a numpy method — a plain Python list will fail).

    Example:
        >>> history = trainModel(model, XtrainData, yTrainData, _epochs=100)
        >>> metrics = evaluateModel(model, XtestData, yTestData)
        >>> print(metrics["accuracy"])   # 0.58  — model correct 58% of the time
        >>> print(metrics["baseline"])   # 0.53  — naive guess correct 53% of the time
        >>> print(metrics["f1"])         # 0.61  — balanced precision/recall score
    """
    _model.eval() 
    
    with torch.no_grad():
        XtestData = torch.FloatTensor(_Xtest)
        yTestData = torch.FloatTensor(_yTest).reshape(-1, 1)   #The `-1` means "figure this dimension out yourself." Telling NumPy/PyTorch: "I know I want 1 column — you calculate how many rows are needed."
        
        #Predictions
        predictions = _model(XtestData)
        predictedClass = (predictions > 0.5).float() 
        
        #Metrics 
        accuracy = (predictedClass == yTestData).float().mean().item()
        
        #Baseline: always predict most common class 
        baseline = max(_yTest.mean(), 1 - _yTest.mean()) 
        
        #True positives, false positives, etc.
        truePositive = ( (predictedClass == 1) & (yTestData == 1) ).sum().item()
        falsePositive = ( (predictedClass == 1) & (yTestData == 0) ).sum().item()
        trueNegative = ( (predictedClass == 0) & (yTestData == 0) ).sum().item()
        falseNegative = ( (predictedClass == 0) & (yTestData == 1) ).sum().item()

        precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0
        recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) > 0 else 0
        f1 = 2*(precision*recall) / (precision + recall) if (precision + recall) > 0 else 0 

    
    return {
        "accuracy": accuracy,
        "baseline": baseline,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusionMatrix": {
            "truePositive": truePositive, 
            "falsePositive": falsePositive,
            "trueNegative": trueNegative,
            "falseNegative": falseNegative
        }
    }

def printEvaluation(_metrics): 
    """
    Prints evaluation metrics from evaluateModel() in a formatted report (readable format).

    Displays accuracy, baseline comparison, improvement over baseline,
    precision, recall, F1 score, and a confusion matrix.

    Args:
        _metrics (dict): Output from evaluateModel() containing keys:
            "accuracy", "baseline", "precision", "recall", "f1", "confusionMatrix".

    Returns:
        None

    Raises:
        KeyError: If _metrics is missing any of the expected keys — "accuracy",
                  "baseline", "precision", "recall", "f1", or "confusionMatrix"
                  (each is accessed directly with no default fallback).
        KeyError: If _metrics["confusionMatrix"] is missing any of its sub-keys —
                  "truePositive", "falsePositive", "trueNegative", or "falseNegative".
        TypeError: If any metric value is not a number (f-string :.4f format spec
                   will fail on non-numeric types).

    Example:
        >>> metrics = evaluateModel(model, XtestData, yTestData)
        >>> printEvaluation(metrics)
        xxxxxxxxxxxxxxxxxxxx
        Evaluation results
        xxxxxxxxxxxxxxxxxxxx
        Test Accuracy:          0.5800
        Baseline (majority):    0.5300
        Improvement:            0.0500

        Precision:              0.6100
        Recall:                 0.5500
        F1 Score:               0.5783

        Confusion Matrix:
            True Positive:  45      False Positive: 29
            False Negative: 37      True Negative:  51
        xxxxxxxxxxxxxxxxxxxx
    """
    print("\n" + "x"*20)
    print("Evaluation results")
    print("x"*20)
    print(f"Test Accuracy:          {_metrics["accuracy"]:.4f}")
    print(f"Baseline (majority):    {_metrics["baseline"]:.4f}")
    print(f"Improvement:            {(_metrics["accuracy"] - _metrics["baseline"]):.4f}")
    print(f"\nPrecision:            {_metrics["precision"]:.4f}")
    print(f"Recall:                 {_metrics["recall"]:.4f}")
    print(f"F1 Score:               {_metrics["f1"]:.4f}")
    
    confusionMatrix = _metrics["confusionMatrix"]
    print(f"\nConfusion Matrix: ")
    print(f"    True Positive:  {confusionMatrix["truePositive"]:<6}  False Positive: {confusionMatrix["falsePositive"]}")
    print(f"    False Negative: {confusionMatrix["falseNegative"]:<6} True Negative:  {confusionMatrix["trueNegative"]}")
    print("x"*20 + "\n")
    
    
#-------------------------------
#Visualization 
#-------------------------------
def plotTrainingHistory(_history): 
    """
    Plots training loss and accuracy over epochs/iterations as a side-by-side chart.

    Left panel  — loss curve (should decrease over time).
    Right panel — accuracy curve with a red dashed line at 0.5 (random guess baseline).
    Saves the chart to "Training-history.png" and displays it.

    Args:
        _history (dict): Output from trainModel() with keys:
            - "loss"     (list): Loss value per epoch/iteration.
            - "accuracy" (list): Accuracy value per epoch/iteration.

    Returns:
        None

    Raises:
        KeyError:  If _history is missing "loss" or "accuracy" keys (both are
                   accessed directly with no default fallback).
        ValueError: If _history["loss"] and _history["accuracy"] are empty lists
                    (matplotlib will render a blank plot with no error, but the
                    saved PNG will be meaningless).
        OSError:   If the current working directory is not writable — matplotlib's
                   savefig() will fail when saving "Training-history.png".

    Example:
        >>> history = trainModel(model, XtrainData, yTrainData, _epochs=100)
        >>> plotTrainingHistory(history)
        --> Training history plot saved to Training-history.png
    """
    figure, (axis1, axis2) = plotLibrary.subplots(1, 2, figsize=(12, 4))
    
    #Loss plot 
    axis1.plot(_history["loss"])
    axis1.set_title("Training loss")
    axis1.set_xlabel("Epoch/iteration")
    axis1.set_ylabel("BTC Loss")
    axis1.grid(True, alpha=0.3)
    
    #Accuracy plot 
    axis2.plot(_history["accuracy"])
    axis2.set_title("Training Accuracy")
    axis2.set_xlabel("Epoch/iteration")
    axis2.set_ylabel("Accuracy")
    axis2.axhline(y=0.5, color="red", linestyle="--", label="Random guess")
    axis2.legend()
    axis2.grid(True, alpha=0.3)
    
    plotLibrary.tight_layout()
    plotLibrary.savefig("Training-history.png", dpi=150, bbox_inches="tight")
    print("--> Training history plot saved to training-history.png")
    plotLibrary.show()


def backtest(_model, _XtestData, _testBarReturns, _fee=0.0015, _threshold=0.5): 
    """
    Simulate trades on the test set and compute financial performance metrics.

    For each test bar, runs the model in eval mode to produce a UP probability.
    If the probability exceeds _threshold, the strategy enters a long position
    and holds for PREDICTION_HOLD_BARS bars, collecting bar returns minus a
    round-trip fee on entry. If the model predicts DOWN, the position is flat
    (0% return that bar). After the hold period the loop advances and re-evaluates.

    Fee is deducted only once per new entry (2 * _fee on the first bar of each
    trade). Consecutive UP predictions within an active hold period do not incur
    an additional fee — the position is already open.

    Args:
        _model          (torch.nn.Module): Trained model in any state — set to eval()
                                           internally before inference.
        _XtestData      (numpy.ndarray):   Test features of shape (N, _lookback * F).
                                           Converted to FloatTensor internally.
        _testBarReturns (list or numpy.ndarray): Per-bar returns for the test period,
                                                 used to compute P&L during held positions
                                                 and the buy-and-hold benchmark.
        _fee            (float, optional): One-way transaction fee as a decimal
                                           (e.g., 0.0015 = 0.15%). Applied as 2 * _fee
                                           on entry bar. Defaults to 0.0015.
        _threshold      (float, optional): Probability cutoff for a UP prediction.
                                           Higher values = fewer, higher-confidence trades.
                                           Defaults to 0.5.

    Returns:
        dict with the following keys:
            - "cumulativeStrategy" (numpy.ndarray): Equity curve of the strategy (starts at 1.0).
            - "cumulativeBuyHold"  (numpy.ndarray): Equity curve of buy-and-hold (starts at 1.0).
            - "strategyReturns"    (numpy.ndarray): Per-bar return series after fees.
            - "sharpe"             (float):         Annualized Sharpe ratio — assumes
                                                    ~47,000 dollar bars per year
                                                    (ETH/USD at $25K threshold, 2023–2026).
                                                    Returns 0.0 if return std is zero.
            - "maxDrawdown"        (float):         Worst peak-to-trough loss as a negative
                                                    decimal (e.g., -0.12 = -12%).
            - "totalReturn"        (float):         Strategy total return as a decimal
                                                    (e.g., 0.08 = +8%).
            - "numberTrades"       (int):           Number of new entries (fee-incurring trades).

    Raises:
        RuntimeError: If _XtestData feature dimension does not match the model's
                    expected input size — PyTorch will raise on the forward pass.
        ValueError:   If _testBarReturns and _XtestData have different lengths —
                    strategyReturns is sized to len(predictions) but barReturns
                    is indexed up to endBar, causing an out-of-bounds access.
        
    Example:
        >>> results = backtest(model, XtestData, testBarReturns, _fee=0.0, _threshold=0.52)
        >>> printBacktestResults(results)
        xxxxxxxxxxxxxxxxxxxx
        Backtest result
        xxxxxxxxxxxxxxxxxxxx
        Total return (strategy):    191.58%
        Total return (buy & hold):  30.24%
        Sharp ratio:                1.305
        Max drawdown:               -67.17%
        Number of trades:           5340
        xxxxxxxxxxxxxxxxxxxx
        >>> # Threshold sweep — compare net vs gross returns
        >>> for thresh in [0.50, 0.52, 0.55, 0.58]:
        ...     net   = backtest(model, XtestData, testBarReturns, _fee=0.0015, _threshold=thresh)
        ...     gross = backtest(model, XtestData, testBarReturns, _fee=0.0,    _threshold=thresh)
        ...     print(f"{thresh:.2f}  trades={net['numberTrades']}  net={net['totalReturn']:.2%}  gross={gross['totalReturn']:.2%}")
        0.50  trades=5876   net=-100.00%  gross=75.82%
        0.52  trades=5340   net=-100.00%  gross=191.58%
        0.55  trades=1616   net= -99.20%  gross=3.07%
        0.58  trades=19     net=  -7.01%  gross=-1.54%
    """
    _model.eval()
    with torch.no_grad():
        probs = _model(torch.tensor(_XtestData, dtype=torch.float32)).squeeze().numpy()
        
    predictions = (probs > _threshold).astype(int)
    barReturns = numpy.array(_testBarReturns)
    strategyReturns = numpy.zeros(len(predictions))
    previousPosition = 0
    numberTrades = 0
    
    i = 0
    while i < len(predictions):
        if predictions[i] == 1: 
            #Hold N bars 
            endBar = min(i + CONFIG["PREDICTION_HOLD_BARS"], len(barReturns))
            entering = (previousPosition == 0)
            if entering: 
                numberTrades += 1
            for j in range(i, endBar):
                strategyReturns[j] = barReturns[j] - (2*_fee if j == i and entering else 0.0)
            previousPosition = 1
            i = endBar      #Jump past the hold period 
        else:
            strategyReturns[i] = 0.0 
            previousPosition = 0
            i += 1
        
    cumulativeStrategy = numpy.cumprod(1 + strategyReturns)
    cumulativeBuyHold = numpy.cumprod(1 + barReturns)
    
    #Annualized Sharp: ~47K dollar bars/year at $25K threshold (ETH/USD, 2023-2026)
    barsPerYear = 47_000
    sharpe = (
        numpy.mean(strategyReturns)/numpy.std(strategyReturns)*numpy.sqrt(barsPerYear)
        if numpy.std(strategyReturns) > 0 else 0.0 
    )
    
    peak = numpy.maximum.accumulate(cumulativeStrategy)
    maxDrawdown = float( ( (cumulativeStrategy - peak) / peak).min() )
    
    return {
        "cumulativeStrategy": cumulativeStrategy,
        "cumulativeBuyHold": cumulativeBuyHold,
        "strategyReturns": strategyReturns,
        "sharpe": sharpe,
        "maxDrawdown": maxDrawdown,
        "totalReturn": float(cumulativeStrategy[-1] - 1.0),
        "numberTrades": numberTrades
    }
    

def printBacktestResults(_results):
    """
    Print backtset summary: return, Sharp, drawdown, trade count vd buy-and-hold.
    """
    buyAndHoldReturn = float(_results["cumulativeBuyHold"][-1] - 1.0)
    print("\n" + "x"*20)
    print("Backtest result")
    print("x"*20)
    print(f"Total return (strategy):    {_results["totalReturn"]:.2%}")
    print(f"Total return (buy & hold):  {buyAndHoldReturn:.2%}")
    print(f"Sharp ratio:                {_results["sharpe"]:.3f}")
    print(f"Max drawdown:               {_results["maxDrawdown"]:.2%}")
    print(f"Number of trades:           {_results["numberTrades"]}")
    print("x"*20 + "\n")
    

def plotBacktest(_results):
    """
    Plots strategy equity curve vs buy-and-hold. Saves to Backtest.png.
    """
    figure, axis = plotLibrary.subplots(1, 1, figsize=(12, 5))
    axis.plot(_results["cumulativeStrategy"], label="Strategy", color="steelblue")
    axis.plot(_results["cumulativeBuyHold"], label="Buy & Hold ETH", color="darkorange")
    axis.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Break-even")
    axis.set_title("Backtest: Strategy vs Buy & Hold ETH")
    axis.set_xlabel("Test bar (dollar bar index)")
    axis.set_ylabel("Equity (1.0 = starting capital)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    plotLibrary.tight_layout()
    plotLibrary.savefig("Backtest.png", dpi=150, bbox_inches="tight")
    print("--> Backtest plot saved to Backtest.png")
    plotLibrary.show()
   

def monteCarloPermutationTest(_model, _XtestData, _testBarReturns, _iterations=1000, _threshold=0.52):
    """
    Validate backtest results against a null distribution using Monte Carlo permutation testing.

    Runs the trained model once to get real predictions, computes the real gross return
    at _threshold, then builds a null distribution by shuffling the bar returns _iterations
    times while keeping the prediction pattern fixed. The p-value is the fraction of
    null runs that match or beat the real return — a low p-value means the result is
    unlikely to be luck.

    The null hypothesis: the strategy's return is no better than applying the same
    prediction pattern to randomly ordered bar returns (i.e., the model has no real
    edge — it just got lucky with which bars it happened to hold).

    Fee is excluded (gross return) so the test isolates signal quality from cost structure.

    Args:
        _model          (torch.nn.Module):          Trained model — set to eval() internally.
        _XtestData      (numpy.ndarray):            Test features of shape (N, _lookback * F).
                                                    Converted to FloatTensor internally.
        _testBarReturns (list or numpy.ndarray):    Per-bar returns for the test period.
                                                    Same array used in backtest().
        _iterations     (int, optional):            Number of random shuffles to build the
                                                    null distribution. Defaults to 1000.
                                                    Higher = more precise p-value.
        _threshold      (float, optional):          Probability cutoff for a UP prediction.
                                                    Should match the threshold used in backtest()
                                                    for a consistent comparison. Defaults to 0.52.

    Returns:
        dict with keys:
            - "realReturn"  (float):          Gross total return of the real strategy.
            - "nullReturns" (numpy.ndarray):  Shape (_iterations,) — gross returns of each
                                              permuted null run.
            - "pValue"      (float):          Fraction of null runs >= realReturn.
                                              p < 0.05 → result is statistically significant.

    Raises:
        RuntimeError: If _XtestData feature dimension does not match the model's expected
                      input size — PyTorch will raise on the forward pass.
        ValueError:   If _testBarReturns is empty — numpy.cumprod on an empty array returns
                      an empty array and the [-1] index will raise.

    Example:
        >>> results = monteCarloPermutationTest(model, XtestData, testBarReturns,
        ...                                     _iterations=1000, _threshold=0.52)
        xxxxxxxxxxxxxxxxxxxx
        Monte Carlo Permutation Test
        xxxxxxxxxxxxxxxxxxxx
        Real gross return:          191.58%
        Null mean ± std:             27.22% ± 76.50%
        Null 95th percentile:       166.57%
        p-value:                    0.0350
        Significant (p < 0.05):     YES
        xxxxxxxxxxxxxxxxxxxx
        >>> print(results["pValue"])
        0.035   # real return beat 96.5% of null runs — edge is real, not luck
    """
    _model.eval()
    with torch.no_grad():
        probs = _model(torch.tensor(_XtestData, dtype=torch.float32)).squeeze().numpy()
    predictions = (probs > _threshold).astype(int)
    barReturns = numpy.array(_testBarReturns)
    
    #Real gross return at this threshold
    realReturn = backtest(_model, _XtestData, _testBarReturns, _fee=0.0, _threshold=_threshold)["totalReturn"]
    
    #Null destribution: same prediction pattern, shuffled bar returns
    nullReturns = []
    for i in range(_iterations):
        shuffled = numpy.random.permutation(barReturns)
        strategyReturns = numpy.where(predictions == 1, shuffled, 0.0)
        nullReturns.append(float(numpy.cumprod(1 + strategyReturns)[-1] - 1.0))
        
    nullReturns = numpy.array(nullReturns)
    pValue = (numpy.sum(nullReturns >= realReturn) + 1) / (_iterations + 1)
    
    print("\n" + "x"*20)
    print("Monte Carlo Permutation Test")
    print("x"*20)
    print(f"Real gross return:          {realReturn:.2%}")
    print(f"Null mean ± std:            {numpy.mean(nullReturns):.2%} ± {numpy.std(nullReturns):.2%}")
    print(f"Null 95th percentile:       {numpy.percentile(nullReturns, 95):.2%}")
    print(f"p-value:                    {pValue:.4f}")
    print(f"Significant (p < 0.05):     {'YES' if pValue < 0.05 else 'NO'}")
    print(f"x"*20 + "\n")
    
    return {"realReturn": realReturn, "nullReturns": nullReturns, "pValue": pValue}
        
    
#-------------------------------
#Main Execution 
#-------------------------------    
def main():
    """
    Main execution pipeline. 
    """
    print("\n" + "-"*20)
    print("Crypto price predictor")
    print("-"*20 + "\n")
    
    #1. Fetch data 
    print("Step 1: Fetching data...")
    bars = fetchCryptoData(
        CONFIG["SYMBOL"],
        CONFIG["TIMEFRAME"],
        CONFIG["START_DATE"],
        CONFIG["END_DATE"]
    )
    
    #Fetch BTC reference bars for cross-asset feature 
    btcTimeBars = None 
    if CONFIG.get("USE_CROSS_ASSET", False) and CONFIG["SYMBOL"] != "BTC/USD":
        btcBars = fetchCryptoData("BTC/USD", CONFIG["TIMEFRAME"], CONFIG["START_DATE"], CONFIG["END_DATE"])
        btcTimeBars = formatBarsToDictionaryList(btcBars, "BTC/USD")
    
    for threshold in [25_000]: #[25_000, 10_000]: #[500_000, 100_000, 50_000, 25_000, 10_000]:
        CONFIG["DOLLAR_THRESHOLD"] = threshold
        print(f"\n DOLLAR_THRESHOLD: {threshold}\n")
    
        orderFlowImbalance = None 
        if CONFIG["BAR_MODE"] == "dollar_bars":
            timeBars = formatBarsToDictionaryList(bars, CONFIG["SYMBOL"])
            dollarBars = createDollarBars(timeBars, CONFIG["DOLLAR_THRESHOLD"])
            barStart, time, openPrice, highPrice, lowPrice, closePrice, dollarVolume, orderFlowImbalanceList = formatDollarBarsToList(dollarBars)
            if CONFIG["USE_ORDER_FLOW_IMBALANCE"]: 
                orderFlowImbalance = numpy.array(orderFlowImbalanceList)
            crossAssetRatio = None
            if btcTimeBars is not None:
                btcTimestamps = [item["time"] for item in btcTimeBars]
                btcCloses     = [item["close"] for item in btcTimeBars]
                crossAssetRatio = []
                for i in range(len(closePrice)):
                    index = bisect.bisect_right(btcTimestamps, barStart[i]) - 1
                    btcPrice = btcCloses[max(0, index)]
                    crossAssetRatio.append(closePrice[i]/btcPrice)
        else:   #CONFIG["BAR_MODE"] == "time_bars" 
            time, openPrice, highPrice, lowPrice, closePrice, volume = formatDataToLists(bars, CONFIG["SYMBOL"]) #Version 1
            
        for lookback in [32]: #[8]: #[16, 8]: #[16, 12, 8]: #[34, 32, 30, 20, 16]:
            CONFIG["LOOKBACK"] = lookback
            print(f"    LOOKBACK: {lookback}\n")
                
            #2. Feature engineering 
            print("\nStep 2: Creating features...")
            X, y, trimmedReturns = createFeaturesAndLabels(closePrice, 
                                           _openPrices=openPrice,
                                           _highPrices=highPrice,
                                           _lowPrices=lowPrice,
                                           _volumes=dollarVolume,
                                           _orderFlowImbalance=orderFlowImbalance, 
                                           _crossAssetRatio=crossAssetRatio,
                                           _lookback=CONFIG["LOOKBACK"]
                                           )
            print(f"--> Features shape: {X.shape}")
            print(f"--> Labels shape: {y.shape}")
            print(f"--> Positive class ratio: {y.mean():.2%}")

            #3. Train/Test split 
            print("\nStep 3: Splitting data...")
            XtrainData, XtestData, yTrainData, yTestData = trainTestSplit(
                X, y, _trainRatio=CONFIG["TRAIN_SPLIT"] 
            )
            
            splitIndex = int(CONFIG["TRAIN_SPLIT"] * len(X))
            testBarReturns = trimmedReturns[splitIndex + CONFIG["LOOKBACK"]:]
            
            #Clip extreme outlier values (volume ratio can spike 50x+ during news events)
            #X = numpy.clip(X, -10, 10)
            scaler = StandardScaler()                           #Normalize the features 
            XtrainData = scaler.fit_transform(XtrainData)       #Fit on train data only 
            XtestData = scaler.transform(XtestData)             #Apply same scale to test 
            print(f"--> Train samples: {len(XtrainData)}")
            print(f"--> Test samples:  {len(XtestData)}")
            
            #4. Build model 
            print("\nStep 4: Building model...")
            #One run
            '''random.seed(42)
            numpy.random.seed(42)
            torch.manual_seed(42) #Remove or comment out, when you want variance across multiple runs (for the `runMultipleTimes(_iterations)` ).
            model = PricePredictor(_inputSize=X.shape[1])
            totalParameters = sum(_parameter.numel() for _parameter in model.parameters())
            print(f"--> Model created with {totalParameters} parameters")
            
            #5. Train 
            print("\nStep 5: Training model...")
            history = trainModel(model, XtrainData, yTrainData, XtestData, yTestData,
                                _epochs=CONFIG["EPOCHS"], 
                                _learningRate=CONFIG["LEARNING_RATE"])
            
            #6. Evaluate 
            print("\nStep 6: Evaluating the model...")
            metrics = evaluateModel(model, XtestData, yTestData)
            printEvaluation(metrics)'''
            
            #multiple runs
            results = [] 
            for itemSeed in [42, 43, 44, 45, 46]:
                random.seed(itemSeed)
                numpy.random.seed(itemSeed)
                torch.manual_seed(itemSeed)
                numberFeatures = X.shape[1] // CONFIG["LOOKBACK"]
                if CONFIG["MODEL"] == "lstm":
                    model = PricePredictorLSTM(_numberFeatures=numberFeatures, _lookback=CONFIG["LOOKBACK"])
                else: 
                    model = PricePredictor(_inputSize=X.shape[1])
                
                totalParameters = sum(_parameter.numel() for _parameter in model.parameters())
                print(f"--> Model created with {totalParameters} parameters")
                
                #5. Train 
                print("\nStep 5: Training model...")
                history = trainModel(model, XtrainData, yTrainData, XtestData, yTestData,
                                    _epochs=CONFIG["EPOCHS"], 
                                    _learningRate=CONFIG["LEARNING_RATE"],
                                    _batchSize=CONFIG["BATCH_SIZE"])
                
                #6. Evaluate 
                print("\nStep 6: Evaluating the model...")
                metrics = evaluateModel(model, XtestData, yTestData)
                printEvaluation(metrics)
                
                results.append(metrics["accuracy"])     #test accuracy
                print(f"Seed {itemSeed}: {metrics["accuracy"]:.4f}")
                
        #print(f"\nMean: {numpy.mean(results):.4f} ± {numpy.std(results):.4f}")
        print(f"\nThreshold ${threshold:,.0f}:\n"
              f"    Mean: {numpy.mean(results):.4f} ± {numpy.std(results):.4f}")
        
        backtestResults = backtest(model, XtestData, testBarReturns)
        printBacktestResults(backtestResults)
        plotBacktest(backtestResults)
        print("\nThreshold sweep (net vs gross):")
        print(f"{"Threshold":>10} {"Trades":>7} {"Net return":>12} {"Gross return":>13}")
        for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
            net = backtest(model, XtestData, testBarReturns, _fee=0.0015, _threshold=thresh)
            gross = backtest(model, XtestData, testBarReturns, _fee=0.0, _threshold=thresh)
            print(f"{thresh:>10.2f} {net["numberTrades"]:>7} {net["totalReturn"]:>12.2%} {gross["totalReturn"]:>13.2%}")
        
        monteCarloPermutationTest(model, XtestData, testBarReturns, _threshold=0.52)
        
    #7. Visualize 
    print("\nStep 7: Generating plots...")
    plotTrainingHistory(history)
    
    #8. Save model 
    print("\nStep 8: Saving the model...")
    torch.save(model.state_dict(), "pricePredictor.pth")
    print("--> Model saved to pricePredictor.pth\n")
    
    print("-"*20)
    print("Pipeline completed!")
    print("-"*20 + "\n")
     
    
if __name__ == "__main__":
    main()
      

#-----
#Small manual tests, for each of the function examples and class examples   
#-----
'''returns = calculateReturns([100, 105, 98, 110])
print(returns)'''

'''prices = [100, 102, 101, 105, 103, 107]  # 6 prices → 5 returns
features, labels = createFeaturesAndLabels(prices, _lookback=3)
print(features.shape)   # (2, 3)  — 2 windows of 3 returns each
print(labels)           # [1, 0]  — next move up, then down '''

'''closePrices = [100, 102, 101, 105, 103, 107]  # 6 prices → 5 returns
features, labels = createFeaturesAndLabels(closePrices, _lookback=30)
XtrainData, XtestData, yTrainData, yTestData = trainTestSplit(features, labels, _trainRatio=0.8)
print(XtrainData.shape)   # (N*0.8, 30)
print(XtestData.shape)    # (N*0.2, 30)'''


'''closePrices = [100, 102, 101, 105, 103, 107, 87, 99, 120, 103, 76, 83, 79, 90, 92, 95, 89, 99, 111, 103, 107, 89, 99, 102, 84, 85, 87, 108, 103, 121, 119, 117]  # 32 prices → 31 returns. Explaination: 30 (lookback) + 1 (label) = 31 returns, so we need 32 prices. "What happened next?" this is the prediction... 
model = PricePredictor(_inputSize=30)       # 30 = lookback window
features, labels = createFeaturesAndLabels(closePrices, _lookback=30)
inputTensor = torch.tensor(features[0], dtype=torch.float32)
prediction = model(inputTensor)
print(prediction)'''

'''closePrices = [100, 102, 101, 105, 103, 107, 87, 99, 120, 103, 76, 83, 79, 90, 92, 95, 89, 99, 111, 103, 107, 89, 99, 102, 84, 85, 87, 108, 103, 121, 119, 117]  # 32 prices → 31 returns. Explaination: 30 (lookback) + 1 (label) = 31 returns, so we need 32 prices. "What happened next?" this is the prediction... 
model = PricePredictor(_inputSize=30)       # 30 = lookback window
features, labels = createFeaturesAndLabels(closePrices, _lookback=30)
inputTensor = torch.tensor(features[0], dtype=torch.float32)
prediction = model(inputTensor)
print(prediction)'''

'''closePrices = [100, 102, 101, 105, 103, 107, 87, 99, 120, 103, 76, 83, 79, 90, 92, 95, 89, 99, 111, 103, 107, 89, 99, 102, 84, 85, 87, 108, 103, 121, 119, 117]  # 32 prices → 31 returns. Explaination: 30 (lookback) + 1 (label) = 31 returns, so we need 32 prices. "What happened next?" this is the prediction... 
X, y = createFeaturesAndLabels(closePrices, _lookback=CONFIG["LOOKBACK"])

XtrainData, XtestData, yTrainData, yTestData = trainTestSplit(
        X, y, _trainRatio=CONFIG["TRAIN_SPLIT"] 
    )

model = PricePredictor(_inputSize=X.shape[1])

history = trainModel(model, XtrainData, yTrainData, _epochs=100) #_epochs=1 Just for testing so it runs the example code. Note! you need more epochs/iterations to get good values...
metrics = evaluateModel(model, XtestData, yTestData)
print(metrics["accuracy"])   # 0.58  — model correct 58% of the time
print(metrics["baseline"])   # 0.53  — naive guess correct 53% of the time
print(metrics["f1"])         # 0.61  — balanced precision/recall score '''

'''metrics = evaluateModel(model, XtestData, yTestData)
printEvaluation(metrics)'''

'''history = trainModel(model, XtrainData, yTrainData, _epochs=100)
plotTrainingHistory(history)'''
