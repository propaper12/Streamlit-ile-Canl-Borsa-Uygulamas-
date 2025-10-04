import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas_ta as ta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
import requests
from textblob import TextBlob
import json
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from prophet import Prophet
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

st.title("Fiyat Analizi ve Tahmini")
st.write("Seçilen BIST 100 hisse senedi için geçmiş verileri kullanarak analiz ve tahminler yapın.")
st.write("---")

bist_100_tickers = {
    "AKBANK": "AKBNK.IS",
    "GARANTİ BBVA": "GARAN.IS",
    "TÜPRAŞ": "TUPRS.IS",
    "THY": "THYAO.IS",
    "KOÇ HOLDİNG": "KCHOL.IS",
    "SASA": "SASA.IS",
    "ASELSAN": "ASELS.IS",
    "BİM": "BIMAS.IS",
    "EREĞLİ DEMİR ÇELİK": "EREGL.IS",
    "FORD OTOSAN": "FROTO.IS"
}


@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, interval="1d")
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Veri çekilirken bir hata oluştu: {e}")
        return None


@st.cache_data(ttl=3600)
def get_sentiment_data(ticker):
    FINNHUB_API_KEY = "YOUR_REAL_FINNHUB_API KEY"
    if FINNHUB_API_KEY == "YOUR_REAL_FINNHUB_API KEY":
        st.warning("Lütfen Finnhub API anahtarınızı girin.")
        return pd.DataFrame()
    today = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={today}&token={FINNHUB_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        news = response.json()

        if not isinstance(news, list):
            st.warning(
                "API'den geçerli bir haber listesi alınamadı. Lütfen API anahtarınızı veya kullanım limitinizi kontrol edin.")
            return pd.DataFrame()

        sentiment_scores = {}
        for item in news:
            try:
                date = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')
                headline = item['headline']
                analysis = TextBlob(headline)
                sentiment_score = analysis.sentiment.polarity
                if date not in sentiment_scores:
                    sentiment_scores[date] = []
                sentiment_scores[date].append(sentiment_score)
            except Exception as e:
                continue

        if not sentiment_scores:
            st.warning("Hiçbir haber öğesi işlenemedi. Duygu verisi boş.")
            return pd.DataFrame()

        sentiment_df = pd.DataFrame(
            [(date, np.mean(scores)) for date, scores in sentiment_scores.items()],
            columns=['Date', 'Sentiment']
        )
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        sentiment_df.set_index('Date', inplace=True)
        return sentiment_df

    except requests.exceptions.HTTPError as errh:
        st.warning(f"Duygu verisi çekilirken bir HTTP hatası oluştu: {errh}. Lütfen API anahtarınızı kontrol edin.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as err:
        st.warning(f"Duygu verisi çekilirken bir hata oluştu: {err}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        st.warning("API yanıtı JSON formatında değil. API anahtarınızı veya kullanım limitinizi kontrol edin.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Duygu verisi işlenirken beklenmedik bir hata oluştu: {e}")
        return pd.DataFrame()


def prepare_data(df, lags=[1, 7, 14, 21], rolling_windows=[7, 14, 21]):
    df_copy = df.copy()

    if 'Close' not in df_copy.columns or df_copy['Close'].empty:
        st.warning("Veri setinde 'Close' sütunu bulunmuyor veya boş.")
        return pd.DataFrame()

    df_copy.index = pd.to_datetime(df_copy.index)
    df_copy['dayofweek'] = df_copy.index.dayofweek.astype(int)
    df_copy['dayofmonth'] = df_copy.index.day.astype(int)
    df_copy['month'] = df_copy.index.month.astype(int)
    for lag in lags:
        df_copy[f'lag_{lag}'] = df_copy['Close'].shift(lag)
    for window in rolling_windows:
        df_copy[f'SMA_{window}'] = df_copy['Close'].rolling(window).mean()
        df_copy[f'EMA_{window}'] = df_copy['Close'].ewm(span=window, adjust=False).mean()
        df_copy[f'rolling_min_{window}'] = df_copy['Close'].rolling(window).min().shift(1)
        df_copy[f'rolling_max_{window}'] = df_copy['Close'].rolling(window).max().shift(1)
        df_copy[f'rolling_mean_{window}'] = df_copy['Close'].rolling(window).mean().shift(1)

    try:
        df_copy.ta.bbands(close=df_copy['Close'], append=True, length=20)
        df_copy.ta.rsi(close=df_copy['Close'], append=True, length=14)
        df_copy.ta.macd(close=df_copy['Close'], append=True)
    except Exception as e:
        st.warning(f"Teknik göstergeler eklenirken bir hata oluştu: {e}")

    df_copy.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1, inplace=True, errors='ignore')
    df_copy.dropna(inplace=True)
    return df_copy


def add_explainability(model, X_test, model_name):
    if shap is None:
        return

    st.subheader(f"{model_name} Modeli Açıklanabilirlik Analizi")
    st.info("Bu grafik, modelin tahminlerinde hangi özelliklerin en etkili olduğunu gösterir.")
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        st.write("Modelin Tahminlerinde Özelliklerin Etkisi")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"SHAP analizi sırasında bir hata oluştu: {e}")


def plot_actual_vs_predicted(y_test, y_pred, dates, model_name):
    st.subheader(f"Gerçek vs. Tahmin Grafiği ({model_name})")
    st.info(
        "Bu grafik, modelin test verisi üzerindeki performansını gösterir ve gelecek tahmin gün sayınızdan bağımsızdır.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_test, mode='lines', name='Gerçek Fiyat', line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=y_pred.index, y=y_pred, mode='lines', name='Model Tahmini', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title='Model Performansı (Test Verisi)',
        xaxis_title='Tarih',
        yaxis_title='Fiyat (TL)',
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)


def train_model(model_name, ohlc_df):
    min_data_points = 50
    if len(ohlc_df) < min_data_points:
        st.error(f"Model eğitimi için yeterli veri yok. En az {min_data_points} veri noktası gerekli.")
        return None, None, None

    if model_name in ["ARIMA", "SARIMA"]:
        train_size = int(len(ohlc_df) * 0.8)
        train_df = ohlc_df['Close'].iloc[:train_size]
        test_df = ohlc_df['Close'].iloc[train_size:]
        if len(test_df) == 0:
            st.error("Test veri seti boş. Lütfen daha fazla veri çekmeyi deneyin.")
            return None, None, None

        if model_name == "ARIMA":
            try:
                model = ARIMA(train_df, order=(5, 1, 0))
                model_fit = model.fit()
                y_pred = model_fit.predict(start=len(train_df), end=len(ohlc_df) - 1, typ='levels')
            except Exception as e:
                st.error(f"ARIMA modeli eğitimi sırasında hata oluştu: {e}")
                return None, None, None

        elif model_name == "SARIMA":
            try:
                model = sm.tsa.statespace.SARIMAX(train_df,
                                                  order=(1, 1, 1),
                                                  seasonal_order=(1, 1, 1, 12),
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                y_pred = model_fit.predict(start=len(train_df), end=len(ohlc_df) - 1, dynamic=True)
            except Exception as e:
                st.error(f"SARIMA modeli eğitimi sırasında hata oluştu: {e}")
                return None, None, None

        y_test = test_df
        y_pred.index = y_test.index

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.success(f"**{model_name}** modeli eğitimi tamamlandı!")
        st.info(f"RMSE: **{rmse:.2f}**, MAE: **{mae:.2f}**, R2 Skoru: **{r2:.2f}**")
        plot_actual_vs_predicted(y_test, y_pred, y_test.index, model_name)
        return model_fit, y_pred, y_test

    elif model_name == "Prophet":
        try:
            df_prophet = ohlc_df[['Close']].reset_index()
            df_prophet.columns = ['ds', 'y']

            train_size = int(len(df_prophet) * 0.8)
            train_df = df_prophet.iloc[:train_size].copy()
            test_df = df_prophet.iloc[train_size:].copy()

            model = Prophet()
            model.fit(train_df)

            future = model.make_future_dataframe(periods=len(test_df), include_history=False)
            forecast = model.predict(future)

            y_test = test_df['y']
            y_pred = forecast['yhat']
            y_pred.index = y_test.index

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.success(f"**{model_name}** modeli eğitimi tamamlandı! 🎉")
            st.info(f"RMSE: **{rmse:.2f}**, MAE: **{mae:.2f}**, R2 Skoru: **{r2:.2f}**")
            plot_actual_vs_predicted(y_test, y_pred, y_test.index, model_name)

            return model, y_pred, y_test
        except Exception as e:
            st.error(f"Prophet modeli eğitimi sırasında hata oluştu: {e}")
            return None, None, None

    else:
        features_df = prepare_data(ohlc_df.copy())
        if features_df.empty:
            st.error("Özellik mühendisliği sonrası veri seti boş kaldı. Veri miktarını artırın.")
            return None, None, None

        if isinstance(features_df.columns, pd.MultiIndex):
            features_df.columns = features_df.columns.droplevel(1)

        features_df.columns = features_df.columns.str.strip()

        X = features_df.drop('Close', axis=1)
        y = features_df['Close']

        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X, y)

        selected_features = X.columns[selector.get_support(indices=True)]
        selected_features_str = [str(col) for col in selected_features]
        st.info(f"Seçilen En İyi Özellikler: {', '.join(selected_features_str)}")

        features_df = features_df[['Close'] + selected_features_str]

        train_size = int(len(features_df) * 0.8)
        train_df = features_df.iloc[:train_size].copy()
        test_df = features_df.iloc[train_size:].copy()

        if len(test_df) == 0:
            st.error("Test veri seti boş. Lütfen daha fazla veri çekmeyi deneyin.")
            return None, None, None

        X_train = train_df.drop('Close', axis=1)
        y_train = train_df['Close']
        X_test = test_df.drop('Close', axis=1)
        y_test = test_df['Close']
        test_dates = test_df.index

        if model_name == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = pd.Series(model.predict(X_test), index=test_dates)
            add_explainability(model, X_test, model_name)
        elif model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = pd.Series(model.predict(X_test), index=test_dates)
            add_explainability(model, X_test, model_name)
        elif model_name == "XGBoost":
            try:
                model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train)
                y_pred = pd.Series(model.predict(X_test), index=test_dates)
                add_explainability(model, X_test, model_name)
            except Exception as e:
                st.error(f"XGBoost modeli eğitimi sırasında hata oluştu: {e}")
                return None, None, None

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.success(f"**{model_name}** modeli eğitimi tamamlandı!")
        st.info(f"RMSE: **{rmse:.2f}**, MAE: **{mae:.2f}**, R2 Skoru: **{r2:.2f}**")
        plot_actual_vs_predicted(y_test, y_pred, y_test.index, model_name)
        return model, y_pred, y_test


def run_prediction(trained_model, model_name, historical_data, forecast_steps):
    if model_name in ["ARIMA", "SARIMA"]:
        try:
            forecast_result = trained_model.get_forecast(steps=forecast_steps)
            forecast_list = forecast_result.predicted_mean.tolist()
            conf_int = forecast_result.conf_int().values
            forecast_dates = [historical_data.index[-1] + timedelta(days=i + 1) for i in range(forecast_steps)]

            valid_forecast_dates = []
            for d in forecast_dates:
                while d.dayofweek >= 5:
                    d += timedelta(days=1)
                valid_forecast_dates.append(d)

            return forecast_list, valid_forecast_dates, conf_int
        except Exception as e:
            st.error(f"ARIMA/SARIMA tahmini sırasında bir hata oluştu: {e}")
            return [], [], None

    elif model_name == "Prophet":
        try:
            future = trained_model.make_future_dataframe(periods=forecast_steps, include_history=False)
            forecast_result = trained_model.predict(future)

            forecast_list = forecast_result['yhat'].tolist()
            forecast_dates = pd.to_datetime(forecast_result['ds']).tolist()

            conf_int_df = forecast_result[['yhat_lower', 'yhat_upper']]
            conf_int = conf_int_df.values

            return forecast_list, forecast_dates, conf_int
        except Exception as e:
            st.error(f"Prophet tahmini sırasında bir hata oluştu: {e}")
            return [], [], None
    else:
        forecast_list = []
        forecast_dates = []
        confidence_intervals = None

        if historical_data.empty:
            return [], [], None

        features_df = prepare_data(historical_data.copy())
        if features_df.empty:
            st.error("Özellik mühendisliği sonrası veri seti boş kaldı. Veri miktarını artırın.")
            return [], [], None

        last_features = features_df.iloc[-1].to_frame().T
        last_features.columns = last_features.columns.droplevel(1)

        last_features.columns = last_features.columns.str.strip()

        last_date = historical_data.index[-1]

        for i in range(forecast_steps):
            X_future = last_features.drop('Close', axis=1)
            prediction = trained_model.predict(X_future)[0]

            new_date = last_date + timedelta(days=i + 1)
            while new_date.dayofweek >= 5:
                new_date += timedelta(days=1)

            forecast_list.append(prediction)
            forecast_dates.append(new_date)

            last_features['lag_1'] = prediction
            for lag in range(21, 1, -1):
                if f'lag_{lag}' in last_features.columns and f'lag_{lag - 1}' in last_features.columns:
                    last_features[f'lag_{lag}'] = last_features[f'lag_{lag - 1}']

        return forecast_list, forecast_dates, confidence_intervals


def plot_data(historical_df, forecast_list, forecast_dates, confidence_intervals, model_name):
    historical_df.index = pd.to_datetime(historical_df.index)
    forecast_df = pd.DataFrame({'price': forecast_list, 'timestamp': forecast_dates})
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(historical_df.index, historical_df['Close'], label='Tarihsel Fiyat', color='b', linewidth=2)
    ax.plot(forecast_df['timestamp'], forecast_df['price'], label='Tahmin', color='r', linestyle='--', marker='o',
            markersize=4)

    if confidence_intervals is not None:
        ax.fill_between(forecast_dates, confidence_intervals[:, 0], confidence_intervals[:, 1], color='r', alpha=0.1,
                        label='Tahmin Aralığı (%95 CI)')

    ax.set_title(f'{st.session_state.selected_name} Fiyat Tahmini', fontsize=16)
    ax.set_xlabel('Tarih', fontsize=12)
    ax.set_ylabel('Fiyat (TL)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


def plot_volatility_analysis(df, title):
    st.subheader(f"{title} Volatilite Analizi (GARCH)")
    st.info("Bu grafik, hisse senedinin fiyat oynaklığının gelecekte nasıl değişeceğini gösterir.")

    try:
        from arch import arch_model
    except ImportError:
        st.warning("GARCH analizi için 'arch' kütüphanesi gerekli. Lütfen `pip install arch` komutunu çalıştırın.")
        return

    returns = 100 * df['Close'].pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        st.warning("Volatilite analizi için yeterli getiri verisi yok. Lütfen daha fazla veri çekin.")
        return

    try:
        am = arch_model(returns, vol='Garch', p=1, q=1)
        res = am.fit(disp='off')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(res.conditional_volatility, color='blue', label='Koşullu Volatilite')
        ax.set_title(f'Koşullu Volatilite - {title}', fontsize=16)
        ax.set_xlabel('Tarih')
        ax.set_ylabel('Oynaklık')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"GARCH analizi sırasında bir hata oluştu: {e}")


def calculate_and_display_risk_metrics(df, title, risk_free_rate=0.01):
    st.subheader(f"{title} Risk ve Getiri Metrikleri")
    st.info(
        "Bu metrikler, varlığın riskine karşılık ne kadar getiri sağladığını ölçer. Yüksek değerler, daha iyi performans anlamına gelir.")

    returns = df['Close'].pct_change().dropna()
    if returns.empty:
        st.warning("Risk metrikleri için yeterli getiri verisi yok.")
        return

    annualized_returns = returns.mean() * 252
    annualized_std_dev = returns.std() * np.sqrt(252)
    downside_returns = returns[returns < 0]

    if downside_returns.empty:
        annualized_downside_std_dev = np.nan
        sortino_ratio = np.nan
    else:
        annualized_downside_std_dev = downside_returns.std() * np.sqrt(252)
        if annualized_downside_std_dev.item() == 0:
            sortino_ratio = np.nan
        else:
            sortino_ratio = (annualized_returns.item() - risk_free_rate) / annualized_downside_std_dev.item()

    if annualized_std_dev.item() == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (annualized_returns.item() - risk_free_rate) / annualized_std_dev.item()

    metrics_df = pd.DataFrame({
        'Metrik': ['Yıllık Getiri', 'Yıllık Volatilite', 'Sharpe Oranı', 'Sortino Oranı'],
        'Değer': [annualized_returns.item(), annualized_std_dev.item(), sharpe_ratio, sortino_ratio]
    })

    def format_value(x):
        if pd.isna(x) or not np.isfinite(x):
            return "NaN"
        return f"{x:.2f}"

    metrics_df['Değer'] = metrics_df['Değer'].apply(format_value)

    st.dataframe(metrics_df)


sorted_coin_names = sorted(list(bist_100_tickers.keys()))
default_index = sorted_coin_names.index("THY") if "THY" in sorted_coin_names else 0
if 'selected_name' not in st.session_state:
    st.session_state.selected_name = sorted_coin_names[default_index]
st.session_state.selected_name = st.sidebar.selectbox("Hisse Senedi Seçin", sorted_coin_names, index=default_index)

models = ["Random Forest", "Linear Regression", "ARIMA", "SARIMA", "Prophet", "XGBoost"]
model_name = st.sidebar.selectbox("Model Seçin", models)
forecast_steps = st.sidebar.number_input("Kaç günlük tahmin istersiniz?", min_value=1, max_value=30, value=7)

if st.button("Analiz ve Tahmin Yap"):
    with st.spinner('Veriler çekiliyor ve model eğitiliyor...'):
        selected_ticker = bist_100_tickers.get(st.session_state.selected_name)
        ohlc_df = get_stock_data(selected_ticker, period="5y")

        if ohlc_df is not None and not ohlc_df.empty:
            sentiment_df = get_sentiment_data(selected_ticker)
            if not sentiment_df.empty:
                ohlc_df = ohlc_df.join(sentiment_df, how='left')
                ohlc_df['Sentiment'] = ohlc_df['Sentiment'].ffill().bfill()

            calculate_and_display_risk_metrics(ohlc_df, st.session_state.selected_name)
            plot_volatility_analysis(ohlc_df, st.session_state.selected_name)

            trained_model, y_pred, y_test = train_model(model_name, ohlc_df)

            if trained_model is not None:
                st.subheader("Gelecek Fiyat Tahmini")
                forecast, forecast_dates, conf_int = run_prediction(trained_model, model_name, ohlc_df, forecast_steps)

                if forecast:
                    plot_data(ohlc_df, forecast, forecast_dates, conf_int, model_name)
                    st.write("### Tahminler")
                    forecast_table = pd.DataFrame({
                        'Tarih': [d.strftime('%Y-%m-%d') for d in forecast_dates],
                        'Tahmin': [f'₺{val:.2f}' for val in forecast]
                    })
                    st.dataframe(forecast_table)
                else:
                    st.warning("Tahmin yapılamadı.")
            else:
                st.error("Model eğitilemedi. Lütfen ayarları kontrol edin.")
        else:
            st.error("Veri çekilemedi. Lütfen ayarları kontrol edin.")