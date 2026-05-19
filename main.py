"""
Financial ML Pipeline - Main Entry Point

Demonstrates the complete pipeline from raw data to trained model:
1. Fetch time bars from Alpaca
2. Create dollar bars (information-based sampling)
3. Fetch trades and calculate Order Flow Imbalance
4. Train neural network price predictor

Author: Leonard 
Date: 2026-05-19
"""

from datetime import datetime
from alpaca.data.timeframe import TimeFrame

# Import our modules
from dollarBarsRefactoredFull import (
    fetchCryptoData,
    formatBarsToDictionaryList,
    createDollarBars,
    formatDollarBarsToList
)
from orderFlow import (
    fetchCryptoTradesChunked,
    aggregateOrderFlowToBars
)

def main():
    """
    Run the complete financial ML pipeline.
    
    Demonstrates:
    - Data fetching from Alpaca API
    - Dollar bars construction (Lopez de Prado)
    - Order Flow Imbalance calculation
    - Data preparation for ML model
    """
    print("="*60)
    print("Financial ML Pipeline")
    print("="*60)
    
    # Configuration
    SYMBOL = "BTC/USD"
    START_DATE = datetime(2026, 4, 1)
    END_DATE = datetime(2026, 4, 27)
    DOLLAR_THRESHOLD = 500_000  # $500K per bar
    
    print(f"\nConfiguration:")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"  Dollar threshold: ${DOLLAR_THRESHOLD:,}")
    
    # Step 1: Fetch time bars
    print("\n" + "-"*60)
    print("Step 1: Fetching time bars from Alpaca...")
    print("-"*60)
    
    bars = fetchCryptoData(
        SYMBOL,
        TimeFrame.Day,
        START_DATE,
        END_DATE
    )
    
    # Step 2: Format to dictionary list
    print("\n" + "-"*60)
    print("Step 2: Formatting time bars...")
    print("-"*60)
    
    timeBars = formatBarsToDictionaryList(bars, SYMBOL)
    
    # Step 3: Create dollar bars
    print("\n" + "-"*60)
    print("Step 3: Creating dollar bars...")
    print("-"*60)
    
    dollarBars = createDollarBars(timeBars, DOLLAR_THRESHOLD)
    
    # Step 4: Format dollar bars for ML
    print("\n" + "-"*60)
    print("Step 4: Formatting dollar bars for ML pipeline...")
    print("-"*60)
    
    dates, opens, highs, lows, closes, volumes = formatDollarBarsToList(dollarBars)
    
    print(f"\nData Summary:")
    print(f"  Time bars: {len(timeBars)}")
    print(f"  Dollar bars: {len(dollarBars)}")
    print(f"  Compression: {len(timeBars)/len(dollarBars):.2f}x")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    # Step 5: Fetch trades for OFI (optional - can be slow)
    # Uncomment to include Order Flow Imbalance calculation:
    """
    print("\n" + "-"*60)
    print("Step 5: Fetching individual trades for OFI...")
    print("-"*60)
    
    trades = fetchCryptoTradesChunked(SYMBOL, START_DATE, END_DATE, _chunkDays=7)
    
    # Extract bar timestamps for OFI aggregation
    barStarts = [bar.get("bar_start", bar["timestamp"]) for bar in dollarBars]
    barEnds = [bar["timestamp"] for bar in dollarBars]
    
    print("\n" + "-"*60)
    print("Step 6: Calculating Order Flow Imbalance...")
    print("-"*60)
    
    ofi = aggregateOrderFlowToBars(trades, barStarts, barEnds)
    
    print(f"\nOFI Summary:")
    print(f"  Bars with OFI: {len(ofi['order_flow_imbalance'])}")
    print(f"  OFI range: [{ofi['order_flow_imbalance'].min():.3f}, {ofi['order_flow_imbalance'].max():.3f}]")
    print(f"  Mean OFI: {ofi['order_flow_imbalance'].mean():.3f}")
    """
    
    # Summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run stock_predictor_baseline.py for ML training")
    print("  2. Run stock_predictor_ofi.py for OFI-enhanced training")
    print("  3. Compare model performance")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()