"""
LSTM Model for Weekly Stock Direction Prediction
Predicts if this Friday's close will be higher than last Friday's close
"""

import numpy as np
import pandas as pd
import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data import TimeFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =====================================
# Configuration
# =====================================
class Config:
    # API Configuration
    API_KEY = "YOUR_API_KEY"
    SECRET_KEY = "YOUR_SECRET_KEY"
    
    # Model Parameters
    SYMBOL = "AAPL"  # Stock to predict
    LOOKBACK_DAYS = 10  # Number of weeks to look back
    TRAIN_TEST_SPLIT = 0.8
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Feature Engineering
    SMA_PERIODS = [5, 10, 20, 50]  # Multiple SMAs
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

# =====================================
# Data Collection
# =====================================
def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch historical stock data from Alpaca
    """
    client = StockHistoricalDataClient(Config.API_KEY, Config.SECRET_KEY)
    
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    
    # Ensure we have the columns we need
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    return df

# =====================================
# Feature Engineering
# =====================================
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicators"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Price position within the day's range
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Multiple SMAs
    for period in Config.SMA_PERIODS:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], Config.RSI_PERIOD)
    
    # MACD
    macd, signal, histogram = calculate_macd(
        df['close'], 
        Config.MACD_FAST, 
        Config.MACD_SLOW, 
        Config.MACD_SIGNAL
    )
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_histogram'] = histogram
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (2 * std_20)
    df['bb_lower'] = sma_20 - (2 * std_20)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df

def create_weekly_features(df):
    """
    Convert daily data to weekly features for Friday-to-Friday prediction
    """
    # Add day of week
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Filter to only Fridays (day_of_week == 4)
    fridays = df[df['day_of_week'] == 4].copy()
    
    if len(fridays) < 2:
        raise ValueError("Not enough Friday data points")
    
    # Create target: 1 if next Friday's close > this Friday's close
    fridays['target'] = (fridays['close'].shift(-1) > fridays['close']).astype(int)
    
    # Select features for modeling
    feature_columns = [
        'returns', 'log_returns', 'volume_ratio', 'volatility', 
        'price_position', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_position'
    ] + [f'sma_{p}_ratio' for p in Config.SMA_PERIODS]
    
    # Remove rows with NaN values
    fridays = fridays.dropna(subset=feature_columns + ['target'])
    
    return fridays, feature_columns

# =====================================
# Data Preparation for LSTM
# =====================================
def prepare_lstm_data(df, feature_columns, lookback=10):
    """
    Prepare data for LSTM model
    Creates sequences of lookback weeks for prediction
    """
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])
    
    X, y = [], []
    
    for i in range(lookback, len(scaled_features) - 1):  # -1 because we need target
        X.append(scaled_features[i-lookback:i])
        y.append(df['target'].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

# =====================================
# Model Creation and Training
# =====================================
def create_lstm_model(input_shape, learning_rate=0.001):
    """
    Create LSTM model for binary classification
    """
    model = Sequential([
        # First LSTM layer
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        
        # Third LSTM layer
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(8, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val):
    """
    Train the LSTM model with callbacks
    """
    model = create_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        learning_rate=Config.LEARNING_RATE
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

# =====================================
# Evaluation and Visualization
# =====================================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate accuracy for each class
    accuracy_down = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    accuracy_up = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    print(f"\nAccuracy for 'Down' predictions: {accuracy_down:.2%}")
    print(f"Accuracy for 'Up' predictions: {accuracy_up:.2%}")
    
    return y_pred, y_pred_prob

def plot_results(history, y_test, y_pred_prob, fridays_df):
    """
    Create visualization of results
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Training History', 'Prediction Distribution', 'Weekly Price & Predictions'],
        vertical_spacing=0.1
    )
    
    # Plot 1: Training history
    fig.add_trace(
        go.Scatter(x=list(range(len(history.history['loss']))), 
                   y=history.history['loss'], name='Train Loss'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(history.history['val_loss']))), 
                   y=history.history['val_loss'], name='Val Loss'),
        row=1, col=1
    )
    
    # Plot 2: Prediction distribution
    fig.add_trace(
        go.Histogram(x=y_pred_prob.flatten(), nbinsx=20, name='Predictions'),
        row=2, col=1
    )
    
    # Plot 3: Price with predictions (last 52 weeks)
    recent_fridays = fridays_df.tail(52).copy()
    fig.add_trace(
        go.Scatter(x=recent_fridays['timestamp'], 
                   y=recent_fridays['close'], 
                   mode='lines+markers',
                   name='Friday Close Prices'),
        row=3, col=1
    )
    
    fig.update_layout(height=900, title_text=f"LSTM Stock Direction Prediction - {Config.SYMBOL}")
    fig.show()

# =====================================
# Main Execution
# =====================================
def main():
    print("="*50)
    print("LSTM Weekly Stock Direction Predictor")
    print("="*50)
    
    # 1. Fetch Data
    print("\n1. Fetching stock data...")
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365*3)  # 3 years of data
    
    df = fetch_stock_data(Config.SYMBOL, start_date, end_date)
    print(f"   Fetched {len(df)} daily records")
    
    # 2. Add Technical Indicators
    print("\n2. Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # 3. Create Weekly Features
    print("\n3. Creating weekly features (Friday-to-Friday)...")
    fridays_df, feature_columns = create_weekly_features(df)
    print(f"   Created {len(fridays_df)} weekly records")
    print(f"   Using {len(feature_columns)} features")
    
    # 4. Prepare LSTM Data
    print("\n4. Preparing LSTM sequences...")
    X, y, scaler = prepare_lstm_data(fridays_df, feature_columns, Config.LOOKBACK_DAYS)
    print(f"   Created {len(X)} sequences with lookback of {Config.LOOKBACK_DAYS} weeks")
    
    # 5. Split Data
    print("\n5. Splitting data...")
    split_idx = int(len(X) * Config.TRAIN_TEST_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # 6. Train Model
    print("\n6. Training LSTM model...")
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # 7. Evaluate Model
    print("\n7. Evaluating model...")
    y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    
    # 8. Visualize Results
    print("\n8. Creating visualizations...")
    plot_results(history, y_test, y_pred_prob, fridays_df)
    
    # 9. Make Next Week Prediction
    print("\n9. Next Friday Prediction:")
    last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
    next_prediction = model.predict(last_sequence)[0][0]
    
    print(f"   Probability of UP movement: {next_prediction:.2%}")
    print(f"   Prediction: {'UP' if next_prediction > 0.5 else 'DOWN'}")
    
    # Save model
    print("\n10. Saving model...")
    model.save('lstm_stock_predictor.h5')
    print("    Model saved as 'lstm_stock_predictor.h5'")
    
    return model, scaler, feature_columns

if __name__ == "__main__":
    model, scaler, feature_columns = main()
