# app.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Union


# Set page configuration
st.set_page_config(
    page_title="Market Seasonality Predictor",
    page_icon="📈",
    layout="wide"
)

# CSS for styling
st.markdown("""
<style>
.main {
    background-color:rgb(0, 2, 4);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: gray;
    border-radius: 4px;
    padding: 10px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.metric-card {
    background-color: gray;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}
.prediction-positive {
    color: #28a745;
    font-weight: bold;
}
.prediction-negative {
    color: #dc3545;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# Function to get historical data
def get_data(ticker:str, period:str="5y"):
    """Fetch historical data for a given ticker"""
    try:
        data = get_local_data(ticker, period)
        return data
    except FileNotFoundError as e:
        # st.error(f":primary[Error fetching data for {ticker}: {e}]")
        st.error(f"Error fetching data for {ticker}: {e}")
        data = yf.download(ticker, period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def get_local_data(ticker:str, period:str) -> Union[pd.DataFrame, None]:
    try:
        data = pd.read_csv(f'data/{ticker.upper()}Raw.txt')
        data['Date'] = pd.to_datetime(data.Date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None


# Function to create features with ATR and excursion targets
def create_features(data):
    """Create time-based features and ATR-normalized excursion targets"""
    if data is None or len(data) == 0:
        return None
    
    df = data.copy()
    
    # Reset index to work with date as a column
    df = df.reset_index()
    
    # Basic date features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    
    # Business day features
    df['trading_day_of_month'] = df.groupby(df['Date'].dt.to_period('M')).cumcount() + 1
    
    # Calculate trading days left in month
    def trading_days_left(date):
        # Get the last day of the month
        next_month = date.replace(day=28) + timedelta(days=4)
        last_day = next_month - timedelta(days=next_month.day)
        
        # Count business days between current date and end of month
        days_left = np.busday_count(date.date(), last_day.date())
        return days_left
    
    df['trading_days_left'] = df['Date'].apply(trading_days_left)
    
    # Calculate ATR (Average True Range)
    df['high_low'] = df['High'] - df['Low']
    df['high_close'] = np.abs(df['High'] - df['Close'].shift(1))
    df['low_close'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    
    # Calculate returns (for reference)
    df['return'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_20d'] = df['Close'].pct_change(20)
    df['return_60d'] = df['Close'].pct_change(60)
    
    # Calculate forward-looking excursions for different time periods
    lookforward_periods = [1, 5, 20, 60]
    
    for period in lookforward_periods:
        # Calculate max favorable and adverse excursions
        rolling_max = df['Close'].rolling(window=period, min_periods=1).max().shift(-period)
        rolling_min = df['Close'].rolling(window=period, min_periods=1).min().shift(-period)
        
        # Handle the last rows where we don't have full forward data
        # by using the available data only
        for i in range(1, period):
            mask = df.index >= (len(df) - i)
            if i == 1:
                # For the very last row, use current values (no prediction)
                rolling_max.loc[mask] = df.loc[mask, 'Close']
                rolling_min.loc[mask] = df.loc[mask, 'Close']
            else:
                # For other near-end rows, use shorter rolling windows
                temp_max = df.loc[mask, 'Close'].rolling(window=period-i+1, min_periods=1).max()
                temp_min = df.loc[mask, 'Close'].rolling(window=period-i+1, min_periods=1).min()
                rolling_max.loc[mask] = temp_max
                rolling_min.loc[mask] = temp_min
        
        # Calculate favorable and adverse excursions (in points)
        df[f'favorable_exc_{period}d'] = rolling_max - df['Close']
        df[f'adverse_exc_{period}d'] = df['Close'] - rolling_min
        
        # Normalize by ATR
        df[f'favorable_exc_{period}d_atr'] = df[f'favorable_exc_{period}d'] / df['atr_14']
        df[f'adverse_exc_{period}d_atr'] = df[f'adverse_exc_{period}d'] / df['atr_14']
        
        # Calculate net excursion (favorable - adverse) normalized by ATR
        df[f'net_exc_{period}d_atr'] = df[f'favorable_exc_{period}d_atr'] - df[f'adverse_exc_{period}d_atr']
        
        # Create binary target variables based on ATR-normalized net excursion
        # If net excursion > 0, it's favorable overall (BUY)
        df[f'target_{period}d'] = np.where(df[f'net_exc_{period}d_atr'] > 0, 1, 0)
    
    # Rename for consistency with previous code
    df['target_daily'] = df['target_1d']
    df['target_monthly'] = df['target_20d']
    df['target_quarterly'] = df['target_60d']
    
    # Drop NA values
    df = df.dropna()
    
    return df

# Train XGBoost models
def train_xgboost_models(df):
    """Train XGBoost models for different time horizons"""
    if df is None or len(df) < 100:  # Need enough data to train
        return None, None, None
    
    # Define features
    features = [
        'day_of_week', 'month', 'quarter', 'week_of_year',
        'trading_day_of_month', 'trading_days_left'
    ]
    
    # Train test split - use earlier data for training
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Train daily model
    X_train = train_df[features]
    y_train_daily = train_df['target_daily']
    daily_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    daily_model.fit(X_train, y_train_daily)
    
    # Train monthly model
    y_train_monthly = train_df['target_monthly']
    monthly_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    monthly_model.fit(X_train, y_train_monthly)
    
    # Train quarterly model
    y_train_quarterly = train_df['target_quarterly']
    quarterly_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    quarterly_model.fit(X_train, y_train_quarterly)
    
    # Evaluate models
    X_test = test_df[features]
    
    # Calculate model accuracy
    if len(test_df) > 0:
        daily_acc = daily_model.score(X_test, test_df['target_daily'])
        monthly_acc = monthly_model.score(X_test, test_df['target_monthly'])
        quarterly_acc = quarterly_model.score(X_test, test_df['target_quarterly'])
        
        print(f"Daily model accuracy: {daily_acc:.4f}")
        print(f"Monthly model accuracy: {monthly_acc:.4f}")
        print(f"Quarterly model accuracy: {quarterly_acc:.4f}")
    
    return daily_model, monthly_model, quarterly_model

# Generate predictions for current and future periods
def generate_predictions(model, current_features, horizon):
    """Generate prediction with probability for the specified time horizon"""
    if model is None:
        return None, None
    
    # Predict class and probability
    pred_class = model.predict(current_features)[0]
    pred_proba = model.predict_proba(current_features)[0]
    
    # Get confidence (probability of the predicted class)
    confidence = pred_proba[1] if pred_class == 1 else pred_proba[0]
    
    # Convert to buy/sell signal with magnitude
    signal = "BUY" if pred_class == 1 else "SELL"
    magnitude = abs(confidence - 0.5) * 2  # Scale to 0-1 range
    
    return signal, magnitude

def calculate_expected_excursion(df, model, current_features, period):
    """Calculate expected favorable and adverse excursions based on model prediction"""
    if model is None or df is None:
        return None, None
    
    # Get prediction and probability
    pred_class = model.predict(current_features)[0]
    pred_proba = model.predict_proba(current_features)[0]
    
    # Calculate conditional expectations based on historical data
    if pred_class == 1:  # BUY prediction
        # Filter historical data where the target was 1 (BUY)
        filtered_df = df[df[f'target_{period}d'] == 1]
        prob = pred_proba[1]
    else:  # SELL prediction
        # Filter historical data where the target was 0 (SELL)
        filtered_df = df[df[f'target_{period}d'] == 0]
        prob = pred_proba[0]
    
    # Calculate average favorable and adverse excursions for this prediction
    if len(filtered_df) > 0:
        avg_favorable = filtered_df[f'favorable_exc_{period}d_atr'].mean()
        avg_adverse = filtered_df[f'adverse_exc_{period}d_atr'].mean()
    else:
        # Fallback to full dataset if filtered data is empty
        avg_favorable = df[f'favorable_exc_{period}d_atr'].mean()
        avg_adverse = df[f'adverse_exc_{period}d_atr'].mean()
    
    # Adjust expectation by probability
    expected_favorable = avg_favorable * prob
    expected_adverse = avg_adverse * prob
    
    return expected_favorable, expected_adverse

# Function to create the dashboard
def create_dashboard(ticker, data, daily_model, monthly_model, quarterly_model):
    """Create the dashboard with predictions and visualizations"""
    if data is None or daily_model is None:
        return
    
    # Get latest features for prediction
    latest_data = data.iloc[-1:].copy()
    features = [
        'day_of_week', 'month', 'quarter', 'week_of_year',
        'trading_day_of_month', 'trading_days_left',
        'atr_14'  # Include ATR as a feature for display purposes
    ]
    prediction_features = [
        'day_of_week', 'month', 'quarter', 'week_of_year',
        'trading_day_of_month', 'trading_days_left'
    ]
    current_features = latest_data[prediction_features]
    
    # Generate predictions
    daily_signal, daily_magnitude = generate_predictions(daily_model, current_features, "daily")
    monthly_signal, monthly_magnitude = generate_predictions(monthly_model, current_features, "monthly")
    quarterly_signal, quarterly_magnitude = generate_predictions(quarterly_model, current_features, "quarterly")
    
    # Header
    col1, col2 = st.columns([2, 3])
    with col1:
        st.title(f"Seasonality Prediction: {ticker}")
        latest_price = data.iloc[-1]['Close']
        prev_price = data.iloc[-2]['Close']
        price_change = (latest_price - prev_price) / prev_price * 100
        
        st.metric(
            label="Latest Price", 
            value=f"${latest_price:.2f}", 
            delta=f"{price_change:.2f}%"
        )
    
    with col2:
        # Basic chart
        fig = px.line(
            data.reset_index(), 
            x='Date', y='Close',
            title=f"{ticker} Price History"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Get current ATR value for display
    current_atr = latest_data['atr_14'].values[0]
    
    # Calculate expected excursions
    daily_favorable, daily_adverse = calculate_expected_excursion(data, daily_model, current_features, 1)
    monthly_favorable, monthly_adverse = calculate_expected_excursion(data, monthly_model, current_features, 20)
    quarterly_favorable, quarterly_adverse = calculate_expected_excursion(data, quarterly_model, current_features, 60)
    
    # Prediction KPIs with expected excursions
    st.subheader("Seasonality-Based Predictions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Daily Forecast</h3>
                <p class="prediction-{'positive' if daily_signal == 'BUY' else 'negative'}">
                    {daily_signal} (Strength: {daily_magnitude:.2f})
                </p>
                <hr style="margin: 10px 0; border-color: #eee;">
                <p><small>Expected Favorable: {daily_favorable:.2f} ATR ({daily_favorable * current_atr:.2f} points)</small></p>
                <p><small>Expected Adverse: {daily_adverse:.2f} ATR ({daily_adverse * current_atr:.2f} points)</small></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Monthly Forecast</h3>
                <p class="prediction-{'positive' if monthly_signal == 'BUY' else 'negative'}">
                    {monthly_signal} (Strength: {monthly_magnitude:.2f})
                </p>
                <hr style="margin: 10px 0; border-color: #eee;">
                <p><small>Expected Favorable: {monthly_favorable:.2f} ATR ({monthly_favorable * current_atr:.2f} points)</small></p>
                <p><small>Expected Adverse: {monthly_adverse:.2f} ATR ({monthly_adverse * current_atr:.2f} points)</small></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Quarterly Forecast</h3>
                <p class="prediction-{'positive' if quarterly_signal == 'BUY' else 'negative'}">
                    {quarterly_signal} (Strength: {quarterly_magnitude:.2f})
                </p>
                <hr style="margin: 10px 0; border-color: #eee;">
                <p><small>Expected Favorable: {quarterly_favorable:.2f} ATR ({quarterly_favorable * current_atr:.2f} points)</small></p>
                <p><small>Expected Adverse: {quarterly_adverse:.2f} ATR ({quarterly_adverse * current_atr:.2f} points)</small></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    # Display current ATR value
    st.info(f"Current ATR (14): {current_atr:.2f} points")
    
    # Feature importance
    st.subheader("Model Insights")
    tab1, tab2, tab3 = st.tabs(["Daily Model", "Monthly Model", "Quarterly Model"])
    
    with tab1:
        plot_feature_importance(daily_model, features, "Daily Model")
    
    with tab2:
        plot_feature_importance(monthly_model, features, "Monthly Model")
    
    with tab3:
        plot_feature_importance(quarterly_model, features, "Quarterly Model")
    
    # Historical seasonality patterns
    st.subheader("Historical Seasonality Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_returns = analyze_monthly_seasonality(data.reset_index())
        fig = px.bar(
            monthly_returns, 
            x='Month', y='Average Return',
            title="Average Returns by Month",
            color='Average Return',
            color_continuous_scale=['red', 'green'],
            text='Average Return'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        dow_returns = analyze_day_of_week_seasonality(data.reset_index())
        fig = px.bar(
            dow_returns, 
            x='Day', y='Average Return',
            title="Average Returns by Day of Week",
            color='Average Return',
            color_continuous_scale=['red', 'green'],
            text='Average Return'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# Plotting functions
def plot_feature_importance(model, feature_names, title):
    """Plot feature importance for a given model"""
    if model is None:
        st.write("No model available")
        return
    
    importance = model.feature_importances_
    indices = np.argsort(importance)
    
    fig = px.bar(
        x=importance[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        title=f"Feature Importance - {title}"
    )
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def analyze_monthly_seasonality(data):
    """Analyze returns by month"""
    df = data.copy()
    df['Month'] = df['Date'].dt.month
    df['Month Name'] = df['Date'].dt.strftime('%b')
    df['Return'] = df['Close'].pct_change()
    
    monthly_returns = df.groupby('Month').agg({
        'Return': 'mean',
        'Month Name': 'first'
    }).reset_index()
    
    monthly_returns = monthly_returns.rename(columns={'Return': 'Average Return'})
    monthly_returns = monthly_returns.sort_values('Month')
    
    # Ensure proper month ordering
    month_order = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    monthly_returns['Month'] = monthly_returns['Month Name'].map(month_order)
    monthly_returns = monthly_returns.sort_values('Month')
    
    return monthly_returns

def analyze_day_of_week_seasonality(data):
    """Analyze returns by day of week"""
    df = data.copy()
    df['Day'] = df['Date'].dt.dayofweek
    df['Day Name'] = df['Date'].dt.strftime('%a')
    df['Return'] = df['Close'].pct_change()
    
    dow_returns = df.groupby('Day').agg({
        'Return': 'mean',
        'Day Name': 'first'
    }).reset_index()
    
    dow_returns = dow_returns.rename(columns={'Return': 'Average Return'})
    
    # Ensure proper day ordering
    day_order = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4}
    dow_returns['Order'] = dow_returns['Day Name'].map(day_order)
    dow_returns = dow_returns.sort_values('Order')
    
    return dow_returns

# Main app function
def main():
    # Sidebar
    st.sidebar.title("Market Seasonality Predictor")
    
    # Input for ticker symbol
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., SPY, QQQ, GLD):", "SPY").upper()
    
    # Data period selection
    period = st.sidebar.selectbox(
        "Select training data period:",
        options=["2y", "3y", "5y", "10y", "max"],
        index=2
    )
    
    # Process when ticker is provided
    if ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            # Get historical data
            data = get_data(ticker, period)
            
            if data is not None and len(data) > 0:
                # Create features
                features_df = create_features(data)
                
                if features_df is not None and len(features_df) > 100:
                    # Train models
                    daily_model, monthly_model, quarterly_model = train_xgboost_models(features_df)
                    
                    # Create dashboard
                    create_dashboard(ticker, features_df, daily_model, monthly_model, quarterly_model)
                else:
                    st.error(f"Not enough data for {ticker} to create reliable predictions.")
            else:
                st.error(f"Unable to fetch data for {ticker}. Please check the ticker symbol.")
    else:
        # Display welcome message when no ticker is entered
        st.title("Welcome to the Market Seasonality Predictor")
        st.write("""
        This app analyzes historical market data to predict buy/sell signals based on seasonality patterns.
        
        Enter a ticker symbol in the sidebar to get started.
        """)
        
        # Example tickers for quick selection
        st.subheader("Try these popular tickers:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("SPY (S&P 500)"):
                st.session_state.ticker = "SPY"
                st.experimental_rerun()
        with col2:
            if st.button("QQQ (Nasdaq)"):
                st.session_state.ticker = "QQQ"
                st.experimental_rerun()
        with col3:
            if st.button("GLD (Gold)"):
                st.session_state.ticker = "GLD"
                st.experimental_rerun()
        with col4:
            if st.button("TLT (Long-Term Treasury)"):
                st.session_state.ticker = "TLT"
                st.experimental_rerun()

if __name__ == "__main__":
    main()