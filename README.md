# Financial ML - Dollar Bars & Order Flow Imbalance

A Financial ML pipeline built on cryptocurrency data from the Alpaca Market API.
Progresses from raw candlestick data → dollar bars → order flow features → neural network prediction → backtesting → statistical validation.

--> Implementation of Lopez de Prado's *Advances in Financial Machine Learning* concepts:
information-based sampling and market microstructure features for cryptocurrency price prediction.

**Key Results:** 9.21x data compression | 53.08% directional accuracy | 191.58% gross alpha | Monte Carlo validated (p=0.035)

**Tech Stack:** Python · PyTorch · NumPy · Alpaca API

&nbsp;

## About

This project demonstrates:
- **Academic implementation:** Translating research papers to production code
- **Systems thinking:** Building modular, reusable components
- **Progressive learning:** Each module built from first principles
- **Production quality:** Comprehensive docstrings, error handling, clean APIs

Built as self-directed learning project implementing Lopez de Prado's 
*Advances in Financial Machine Learning*.

**Author:** Leo  

&nbsp;

## Quick Start

Two versions below:

**Quick Start (Recommended)** — simplified, uses pre-built `dollarBarsWithOFI` module  
**Quick Start version 1** — detailed, step-by-step with separate modules (see details)

<details>
<summary> Quick Start version 1 </summary> 

## Quick Start version 1

```python
# 1. Fetch and create dollar bars
from dollarBarsRefactoredFull import fetchCryptoData, createDollarBars, formatDollarBarsToList
from datetime import datetime
from alpaca.data.timeframe import TimeFrame

start = datetime(2025, 5, 1)
end   = datetime(2026, 4, 1)

# Fetch time bars
bars = fetchCryptoData("BTC/USD", TimeFrame.Day, start, end)
timeBars = formatBarsToDictionaryList(bars, "BTC/USD")

# Create dollar bars
dollarBars = createDollarBars(timeBars, _dollarThreshold=25_000)

# Format for ML
dates, opens, highs, lows, closes, volumes = formatDollarBarsToList(dollarBars)

# 2. Calculate Order Flow Imbalance
from orderFlow import fetchCryptoTradesChunked, aggregateOrderFlowToBars

trades = fetchCryptoTradesChunked("BTC/USD", start, end)
ofi = aggregateOrderFlowToBars(trades, barStarts, barEnds)

# 3. Train ML model
from pricePredictorWithOFI import PricePredictorLSTM, PricePredictor, createFeaturesAndLabels, trainTestSplit, trainModel

# Build features and labels
features, labels = createFeaturesAndLabels(
    closes, _openPrices=opens, _highPrices=highs, _lowPrices=lows,
    _volumes=volumes, _orderFlowImbalance=ofi, _lookback=30
)

# Split chronologically (no shuffling)
Xtrain, Xtest, yTrain, yTest = trainTestSplit(features, labels)

# Feedforward baseline
model = PricePredictor(_inputSize=300) # 10 features * lookback 30

# LSTM model (sequence-aware)
model = PricePredictorLSTM(_numberFeatures=10, _lookback=30)

history = trainModel(model, Xtrain, yTrain, Xtest, yTest, _epochs=1000, _batchSize=32)

# Inspect results
print(f"Accuracy: {history['accuracy'][-1]:.3f}")
print(f"Loss:     {history['loss'][-1]:.3f}")
```

</details>

&nbsp;

### Quick Start (Recommended)

```python
# 1. Fetch and create dollar bars with built-in OFI
from dollarBarsWithOFI import fetchCryptoData, createDollarBars, formatDollarBarsToList, formatBarsToDictionaryList
from datetime import datetime
from alpaca.data.timeframe import TimeFrame

# Fetch time bars
bars = fetchCryptoData("ETH/USD", TimeFrame.Day, datetime(2024, 1, 1), datetime(2026, 5, 7))
timeBars = formatBarsToDictionaryList(bars, "ETH/USD")

# Create dollar bars (OFI computed per bar)
dollarBars = createDollarBars(timeBars, _dollarThreshold=25_000)

# Format for ML — 8 return values including OFI
barStarts, times, opens, highs, lows, closes, volumes, ofi = formatDollarBarsToList(dollarBars)

# 2. Train ML model
from pricePredictorWithOFI import PricePredictorLSTM, PricePredictor, createFeaturesAndLabels, trainTestSplit, trainModel, evaluateModel, backtest, monteCarloPermutationTest

# Build features and labels (10 feature arrays: returns, vol, RSI, MA20, momentum,
# barRange, VWAP, OFI, MA200, ADX). Returns trimmed bar returns for backtesting.
features, labels, trimmedReturns = createFeaturesAndLabels(
    closes, _openPrices=opens, _highPrices=highs, _lowPrices=lows,
    _volumes=volumes, _orderFlowImbalance=ofi, _lookback=32
)

# Split chronologically (no shuffling)
Xtrain, Xtest, yTrain, yTest = trainTestSplit(features, labels)

# Feedforward baseline
#model = PricePredictor(_inputSize=300)      # 10 features * lookback 30

# LSTM model (sequence-aware)
model = PricePredictorLSTM(_numberFeatures=10, _lookback=32)

history = trainModel(model, Xtrain, yTrain, Xtest, yTest, _epochs=1000, _batchSize=2048)

# Evaluate on test set
results = evaluateModel(model, Xtest, yTest)
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")

# 3. Backtest the strategy
splitIndex    = int(0.8 * len(features))
testBarReturns = trimmedReturns[splitIndex + 32:]

backtestResults = backtest(model, Xtest, testBarReturns, _fee=0.0, _threshold=0.52)
print(f"Gross return: {backtestResults['totalReturn']:.2%}")
print(f"Sharpe ratio: {backtestResults['sharpe']:.3f}")

# 4. Validate with Monte Carlo permutation test
monteCarloPermutationTest(model, Xtest, testBarReturns, _iterations=1000, _threshold=0.52)
```

&nbsp;

## Project Structure

### Core Pipeline

- `dollarBarsRefactoredFull.py` # Information-based sampling
- `orderFlow.py`                # Order Flow Imbalance calculation
- `stockPredictor.py`           # Neural network with price features
- `pricePredictorWithOFI.py`    # Enhanced with order flow features, backtesting, Monte Carlo
- `timebarsRefactoredFull.py`   # Time-based OHLCV data (Alpaca API)
- `main.py`                     # Pipeline orchestration

<details>
<summary>📁 Supporting & Development Files</summary>

### Files

### timebarsRefactoredFull.py — Full refactored version
The final production-ready version. Adds:
- `barsToDataframe()` — converts OHLCV lists to a pandas DataFrame
- `plotCandlesticksWithVolume()` — two-panel chart (price + volume)
- `exportToCSV()` — saves data to CSV
- `printStatistics()` — summary stats (price range, change %, total volume)
- Full docstrings with examples on all functions

### timebarsWithOFI.py — Time bars + Order Flow Imbalance
Extends timebarsRefactoredFull.py with OFI features, preparing time-bar data
for the OFI-enhanced ML pipeline.

### dollarBarsRefactoredFull.py — Dollar bars (production)
Full implementation with:
- `fetchCryptoData()` — Alpaca API calls
- `createDollarBars()` — core bar construction (threshold: $500K/bar)
- `formatBarsToDictionaryList()` / `formatDollarBarsToList()` — data formatting
- Comprehensive docstrings throughout

### dollarBarsWithOFI.py — Dollar bars + Order Flow Imbalance
Combines dollar bar construction with OFI metrics. Approximates OFI from OHLCV:
`OFI = (close - open) / (high - low)`, capturing intra-bar buy/sell pressure.
The primary data source for the OFI-enhanced predictor.

### orderFlow.py — Order Flow Imbalance module
Standalone module for OFI calculation:
- `fetchCryptoTrades()` — fetches individual tick-level trades from Alpaca
- `fetchCryptoTradesChunked()` — handles large date ranges with weekly chunking
- Computes OFI = buy-initiated volume − sell-initiated volume per bar
- Captures market microstructure: who is driving price?

### stockPredictor.py — Neural network price direction predictor
Predicts whether the next candle closes up (1) or down (0).
Full supervised ML pipeline:
1. Fetch BTC/USD data via Alpaca (dollar bars mode)
2. Feature engineering: returns, rolling volatility, RSI, price-relative-to-MA, volume features
3. Build sliding-window features (lookback=32) + binary labels
4. Train/test split (80/20, chronological — no shuffling)
5. `PricePredictor` — 3-layer neural network (Linear → ReLU → Dropout) × 2 + Sigmoid output
6. Train with Adam optimizer + BCELoss
7. Evaluate with accuracy, precision, recall, F1, confusion matrix
8. Plot training history (loss + accuracy curves)

### pricePredictorWithOFI.py — Neural network + LSTM with OFI features
Extends stockPredictor.py with two model options and OFI-enriched dollar bars as input:
- `PricePredictor` — feedforward baseline (3-layer MLP)
- `PricePredictorLSTM` — stacked LSTM (2 layers, hidden=64) + classifier head;
  processes feature windows as sequences (batch, lookback, features) to capture
  temporal patterns that the feedforward model cannot
- `trainModel` — mini-batch training via DataLoader with shuffling, epoch-averaged
  loss/accuracy, and early stopping on test loss (`patience=15`) with best-weight restore
- Feature set (10 arrays, all CONFIG-gated): returns, volatility, RSI, MA(20) deviation, intra-bar momentum, bar range, VWAP deviation, OFI, MA(200) long-term context, ADX trend strength
- `backtest` — walk-forward simulation: long on predicted up, flat on predicted down. N-bar hold periods, configurable confidence threshold, round-trip fee modeling. Returns equity curve, Sharpe ratio, max drawdown, trade count
- `monteCarloPermutationTest` — validates gross return against a null distribution of 1,000 random bar-return shuffles. Tests whether the model's market timing produces better-than-random returns (p < 0.05 = statistically significant edge)

### main.py — Entry point
Wires together the pipeline components.

</details>

&nbsp;

## Progression

```
timebarsRefactoredFull.py    →  timebarsWithOFI.py
(full feature set)           →  (+ OFI features)
↓
stockPredictor.py
(ML model — time bars)


dollarBarsRefactoredFull.py  →  dollarBarsWithOFI.py
(production-ready)           →  (+ OFI features)
↓
orderFlow.py (OFI calculation) ────────────→  pricePredictorWithOFI.py
                                              (ML model - feedforward baseline + LSTM — OFI enhanced + backtest + Monte Carlo)
```

### Data Flow

```
1. Alpaca API → Time Bars (OHLCV, minute resolution)
                    ↓
2. Dollar Bars (information-based sampling, $25K threshold)
                    ↓
3. Order Flow Imbalance (buy/sell pressure per bar)
                    ↓
4. Feature Engineering (returns, volatility, RSI, MA deviation, OFI, ADX)
                    ↓
5. LSTM Neural Network — sequence-aware binary classifier
                    ↓
6. Binary classification (next bar up/down) → 53.08% accuracy
                    ↓
7. Walk-forward Backtest → 191.58% gross return (threshold=0.52, no fees)
                    ↓
8. Monte Carlo Permutation Test → p=0.035 (statistically significant edge)
```

## Implementation Approach

Built progressively to understand each concept deeply:

1. **Time bars** → Fetch OHLCV data from Alpaca
2. **Dollar bars** → Information-based sampling (Lopez de Prado)
3. **Order flow** → Calculate buy/sell pressure per bar
4. **ML pipeline** → Neural network prediction (baseline vs OFI-enhanced)
5. **Long Short Term Memory (LSTM)** → Sequence-aware model capturing temporal patterns across lookback window
6. **Backtesting** → Walk-forward simulation with realistic fee modeling and threshold sweep
7. **Monte Carlo** → Statistical validation of gross alpha against null distribution

**Development Process:**  
Each concept implemented progressively: prototype → refactored → production-ready with comprehensive docstrings and error handling.

**Learning Philosophy:**  
Build from first principles to understand deeply, not just use libraries.


## Output files
- `Training-history.png` — loss and accuracy plot generated by `plotTrainingHistory()`
- `Backtest.png` — strategy vs buy-and-hold equity curve from `plotBacktest()`
- `pricePredictor.pth` — saved model weights from the last training run

## Installation

```bash
# Clone repository
git clone https://github.com/FinancialML-dev/projects.git
cd projects

# Install dependencies
uv sync

# This installs: `torch`, `numpy`, `pandas`, `plotly`, `matplotlib`, `alpaca-py`, `scikit-learn` 

# Or with pip
pip install torch numpy pandas plotly matplotlib alpaca-py scikit-learn
```

**Run pipeline:**
```bash
python main.py

python uv run pricePredictorWithOFI.py 
```

## Results

### Dollar Bars Performance
- **Compression:** 9.21x (1,436,281 minute bars → 155,944 dollar bars)
- **Benefit:** Uniform information density vs variable in time bars
- **Use case:** Better feature distributions for ML models

### Order Flow Imbalance
- **Captures:** Intra-bar buy/sell pressure via OHLCV approximation: `(close - open) / (high - low)`
- **Range:** -1.0 (all selling) to +1.0 (all buying)
- **Value:** Shows WHO is driving price movement, not just WHAT happened

### ML Model
- **Feedforward (`PricePredictor`):** 3-layer neural network (64→32→1 neurons)
- **LSTM (`PricePredictorLSTM`):** 2-layer stacked LSTM (hidden=64) + classifier head (32→1)
- **Training:** Mini-batch gradient descent (batch=2048, shuffled) + early stopping (patience=15)
- **Features (up to 10 arrays):** Returns, volatility, RSI, MA(20) deviation, intra-bar momentum,
  bar range, VWAP deviation, OFI, MA(200) long-term context, ADX trend strength
- **Task:** Binary classification (next candle-bar up/down)
- **Baseline accuracy:** ~51-52% (price features only, feedforward)
- **Best result BTC/USD:** ~52.4% mean (LSTM + 200MA + ADX, BTC/USD $25K threshold)
- **Best accuracy:** 53.08% ± 0.0007 mean over 5 seeds (LSTM, ETH/USD, $25K threshold, lookback=32, 200MA + ADX)
- **Evaluation:** Accuracy, precision, recall, F1, confusion matrix

### Backtesting
- **Method:** Walk-forward simulation — long when model predicts up, flat when down
- **Fee model:** 0.30% round-trip per trade (0.15% entry + 0.15% exit, Alpaca taker)
- **Best gross return:** 191.58% vs 30.24% buy-and-hold (threshold=0.52, zero fees)
- **Net return at retail fees:** -100% (0.30% × 5,340 trades eliminates all alpha)
- **Finding:** The directional edge is real but not extractable at retail crypto fees — the signal is distributed across too many small bars for 0.30% friction to survive

### Monte Carlo Permutation Test
- **Method:** Fix model prediction pattern, shuffle bar returns 1,000 times, compare real gross return to null distribution
- **Result:** p=0.035 — only 3.5% of random shuffles matched or beat the 191.58% gross return
- **Null distribution:** mean=27.22% ± std=76.50%, 95th percentile=166.57%
- **Conclusion:** The model's market timing produces statistically significant returns (p < 0.05). The edge is real, not luck.

## Future Improvements - Aspiring to 

**Planned enhancements:**
- Multi-timeframe features (combine $10K and $50K dollar bars)
- Additional order flow metrics (trade size distribution, depth imbalance)
- Ensemble methods (multiple models, voting)
- Backtesting framework with realistic slippage/fees

**For production deployment:**
- Real-time data pipeline
- Model monitoring and drift detection
- Risk management layer

## Author

Leo<br>
Learning financial ML 

## License

MIT
