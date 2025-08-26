# LSTM Stock Direction Predictor 📈

A deep learning model using LSTM (Long Short-Term Memory) networks to predict weekly stock price direction. The model predicts whether this Friday's closing price will be higher or lower than last Friday's closing price.

## 🎯 Project Overview

This project implements a binary classification model that:
- Fetches historical stock data using Alpaca API
- Engineers technical indicators as features
- Uses LSTM layers to capture temporal patterns
- Predicts weekly price direction (UP/DOWN) for Friday-to-Friday movements

## 🏗️ Model Architecture

```
Input (10 weeks of features)
    ↓
LSTM Layer (64 units, return sequences)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (32 units, return sequences)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (16 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (8 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (1 unit, Sigmoid)
```

## 📊 Features Used

The model uses multiple technical indicators:

- **Price-based**: Returns, log returns, price position in daily range
- **Volume**: Volume change, volume ratio to SMA
- **Moving Averages**: SMA (5, 10, 20, 50 periods)
- **Momentum**: RSI (14 period)
- **Trend**: MACD, Signal, Histogram
- **Volatility**: 20-day rolling standard deviation, Bollinger Bands

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas
pip install tensorflow
pip install alpaca-py
pip install scikit-learn
pip install plotly
```

### API Setup

1. Get your Alpaca API keys from [Alpaca Markets](https://alpaca.markets/)
2. Update the configuration in the script:

```python
class Config:
    API_KEY = "YOUR_API_KEY"
    SECRET_KEY = "YOUR_SECRET_KEY"
```

### Running the Model

```bash
python lstm_stock_predictor.py
```

## 📈 Performance Metrics

The model evaluates using:
- **Accuracy**: Overall prediction accuracy
- **Precision/Recall**: For both UP and DOWN movements
- **AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed prediction breakdown

## 🧠 Key Concepts Explained

### Why LSTM for Stocks?

LSTMs are ideal for stock prediction because they:
1. **Remember long-term dependencies**: Can learn from patterns that span multiple time periods
2. **Handle sequential data**: Natural fit for time series
3. **Selective memory**: Gates mechanism helps focus on relevant information

### The Friday-to-Friday Strategy

Predicting weekly movements (Friday close to Friday close) instead of daily:
- Reduces noise from daily fluctuations
- Captures weekly market cycles
- More actionable for swing traders
- Better signal-to-noise ratio

### Feature Engineering Importance

Technical indicators provide different market perspectives:
- **Trend indicators** (SMA, MACD): Market direction
- **Momentum indicators** (RSI): Overbought/oversold conditions
- **Volatility indicators** (Bollinger Bands): Price stability
- **Volume indicators**: Market participation

## ⚠️ Important Disclaimers

### Model Limitations

1. **Historical patterns may not repeat**: Markets are influenced by countless factors
2. **Overfitting risk**: Model might memorize training data patterns
3. **Market regime changes**: Model trained on bull market may fail in bear market
4. **Black swan events**: Cannot predict unprecedented events

### Trading Risks

**⚠️ WARNING**: This model is for educational purposes only!
- Never trade with money you can't afford to lose
- Always backtest extensively
- Consider transaction costs, slippage, and taxes
- Past performance doesn't guarantee future results

## 📁 Project Structure

```
lstm-stock-predictor/
│
├── lstm_stock_predictor.py    # Main model implementation
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── notebooks/
│   └── analysis.ipynb         # Data exploration and analysis
├── models/
│   └── saved_models/          # Trained model files
└── results/
    ├── plots/                 # Visualization outputs
    └── metrics/               # Performance metrics
```

## 🔧 Configuration Options

Key parameters you can tune:

```python
LOOKBACK_DAYS = 10      # Weeks of history to use
BATCH_SIZE = 32         # Training batch size
EPOCHS = 100            # Training epochs
LEARNING_RATE = 0.001   # Adam optimizer learning rate
TRAIN_TEST_SPLIT = 0.8  # Train/test data split
```

## 📊 Sample Results

Expected output format:
```
Classification Report:
              precision    recall  f1-score   support
        Down       0.55      0.58      0.57        24
          Up       0.61      0.58      0.59        26

Confusion Matrix:
[[14 10]
 [11 15]]

Next Friday Prediction:
Probability of UP movement: 67.3%
Prediction: UP
```

## 🔄 Future Improvements

Potential enhancements to explore:

1. **Additional Features**
   - Sentiment analysis from news/social media
   - Market breadth indicators
   - Sector performance correlation
   - Options flow data

2. **Model Enhancements**
   - Attention mechanisms
   - Bidirectional LSTMs
   - Ensemble methods
   - Transfer learning from larger datasets

3. **Risk Management**
   - Position sizing algorithms
   - Stop-loss optimization
   - Portfolio integration
   - Kelly criterion implementation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Alpaca Markets API Documentation](https://docs.alpaca.markets/)
- [Technical Analysis Library Documentation](https://technical-analysis-library-in-python.readthedocs.io/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by various Instagram and social media ML trading tutorials
- Thanks to the Alpaca Markets team for the free API
- TensorFlow/Keras community for excellent documentation

---

**Remember**: The stock market is complex and unpredictable. This model is a learning tool, not financial advice. Always do your own research and consider consulting with financial professionals before making investment decisions.
