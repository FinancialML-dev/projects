# Financial ML - Dollar Bars & Order Flow Imbalance

A Financial ML pipeline built on cryptocurrency data from the Alpaca Market API.
Progresses from raw candlestick data → dollar bars → order flow features → neural network prediction.

--> Implementation of Lopez de Prado's *Advances in Financial Machine Learning* concepts:
information-based sampling and market microstructure features for cryptocurrency price prediction.

**Key Results:** 3.25x data compression | Order Flow Imbalance | Neural network classification

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

```python
# 1. Fetch and create dollar bars
from dollarBarsRefactoredFull import fetchCryptoData, createDollarBars, formatDollarBarsToLists
from datetime import datetime
from alpaca.data.timeframe import TimeFrame

# Fetch time bars
bars = fetchCryptoData("BTC/USD", TimeFrame.Day, datetime(2026, 3, 1), datetime(2026, 4, 1))
timeBars = formatBarsToDictionaryList(bars, "BTC/USD")

# Create dollar bars
dollarBars = createDollarBars(timeBars, threshold=500_000)

# Format for ML
dates, opens, highs, lows, closes, volumes = formatDollarBarsToList(dollarBars)

# 2. Calculate Order Flow Imbalance
from orderFlow import fetchCryptoTradesChunked, aggregateOrderFlowToBars

trades = fetchCryptoTradesChunked("BTC/USD", start, end)
ofi = aggregateOrderFlowToBars(trades, barStarts, barEnds)

# 3. Train ML model
from pricePredictorWithOFI import PricePredictor, trainModel

model = PricePredictor(input_features=128)
history = trainModel(model, features, labels, epochs=1000)
```

&nbsp;

## Project Structure

### Core Pipeline

- `dollarBarsRefactoredFull.py` # Information-based sampling
- `orderFlow.py`                # Order Flow Imbalance calculation
- `stockPredictor.py`           # Neural network with price features
- `pricePredictorWithOFI.py`    # Enhanced with order flow features
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
Combines dollar bar construction with OFI metrics. Integrates individual trade data
(taker side: B=buy, S=sell) to enrich each bar with buy/sell pressure signals.
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

### pricePredictorWithOFI.py — Neural network with OFI features
Same architecture as stockPredictor.py but uses OFI-enriched dollar bars as input.
Adds order flow imbalance as a feature alongside returns, volatility, and RSI.

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
(ML model — OFI enhanced)
```

### Data Flow

```
1. Alpaca API → Time Bars (OHLCV)
                    ↓
2. Dollar Bars (information-based sampling)
                    ↓
3. Order Flow Imbalance (buy/sell pressure)
                    ↓
4. Feature Engineering (returns, volatility, RSI, OFI)
                    ↓
5. Neural Network (price direction prediction)
```

## Implementation Approach

Built progressively to understand each concept deeply:

1. **Time bars** → Fetch OHLCV data from Alpaca
2. **Dollar bars** → Information-based sampling (Lopez de Prado)
3. **Order flow** → Calculate buy/sell pressure per bar
4. **ML pipeline** → Neural network prediction (baseline vs OFI-enhanced)

**Development Process:**  
Each concept implemented progressively: prototype → refactored → production-ready with comprehensive docstrings and error handling.

**Learning Philosophy:**  
Build from first principles to understand deeply, not just use libraries.


## Output files
- `Training-history.png` — loss and accuracy plot generated by `plotTrainingHistory()`
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
```

## Results

### Dollar Bars Performance
- **Compression:** 3.25x (26 time bars → 8 dollar bars)
- **Benefit:** Uniform information density vs variable in time bars
- **Use case:** Better feature distributions for ML models

### Order Flow Imbalance
- **Captures:** Buy vs sell aggressor volume per bar
- **Range:** -1.0 (all selling) to +1.0 (all buying)
- **Value:** Shows WHO is driving price movement, not just WHAT happened

### ML Model
- **Architecture:** 3-layer neural network (64→32→1 neurons)
- **Features:** Returns, volatility, RSI, MA deviation, OFI
- **Task:** Binary classification (next candle up/down)
- **Baseline accuracy:** ~51-52% (price features only)
- **With OFI:** [Experimental] (order flow features added)
- **Evaluation:** Accuracy, precision, recall, F1, confusion matrix

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
