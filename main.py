"""
Financial ML Pipeline - Main Entry Point

Full pipeline from raw Alpaca data to trained model, backtesting, and
statistical validation:

1. Fetch ETH/USD minute bars from Alpaca (2023–2026)
2. Create dollar bars — information-based sampling (Lopez de Prado)
3. Feature engineering — returns, volatility, RSI, MA(20), intra-bar momentum,
bar range, VWAP deviation, OFI, MA(200) long-term context, ADX trend strength
4. Train LSTM binary classifier — predicts next bar direction (up/down)
5. Evaluate — accuracy, precision, recall, F1, confusion matrix
6. Backtest — walk-forward simulation with fee modeling and threshold sweep
7. Monte Carlo permutation test — statistical validation of gross alpha

Best result: ETH/USD, LSTM, lookback=32, $25K threshold, 200MA + ADX
    Accuracy:     53.08% ± 0.0007 (mean over 5 seeds)
    Gross return: 191.58% vs 30.24% buy-and-hold (threshold=0.52, no fees)
    Monte Carlo:  p=0.035 — statistically significant edge

Author: Leo
Date: 2026-05-19
Reference: Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
"""

from pricePredictorWithOFI import main

if __name__ == "__main__":
    main()