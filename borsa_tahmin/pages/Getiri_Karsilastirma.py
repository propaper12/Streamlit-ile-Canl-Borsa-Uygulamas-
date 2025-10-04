import streamlit as st
import yfinance as yf
import pycoingecko
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns
import pandas_ta as ta

st.set_page_config(layout="wide")

try:
    cg = pycoingecko.CoinGeckoAPI()
except Exception as e:
    st.error(f"CoinGecko bağlantı hatası: {e}")
    cg = None

all_tickers = {
    "AKBANK": {"yfinance": "AKBNK.IS", "is_stock": True, "category": "Hisse Senedi"},
    "GARANTİ BBVA": {"yfinance": "GARAN.IS", "is_stock": True, "category": "Hisse Senedi"},
    "TÜPRAŞ": {"yfinance": "TUPRS.IS", "is_stock": True, "category": "Hisse Senedi"},
    "THY": {"yfinance": "THYAO.IS", "is_stock": True, "category": "Hisse Senedi"},
    "KOÇ HOLDİNG": {"yfinance": "KCHOL.IS", "is_stock": True, "category": "Hisse Senedi"},
    "SASA": {"yfinance": "SASA.IS", "is_stock": True, "category": "Hisse Senedi"},
    "ASELSAN": {"yfinance": "ASELS.IS", "is_stock": True, "category": "Hisse Senedi"},
    "BİM": {"yfinance": "BIMAS.IS", "is_stock": True, "category": "Hisse Senedi"},
    "EREĞLİ DEMİR ÇELİK": {"yfinance": "EREGL.IS", "is_stock": True, "category": "Hisse Senedi"},
    "FORD OTOSAN": {"yfinance": "FROTO.IS", "is_stock": True, "category": "Hisse Senedi"},
    "BIST 30": {"yfinance": "XU030.IS", "is_stock": True, "category": "Endeks"},
    "BIST 100": {"yfinance": "XU100.IS", "is_stock": True, "category": "Endeks"},
    "BIST TUM": {"yfinance": "XUTUM.IS", "is_stock": True, "category": "Endeks"},
    "Dolar (USD)": {"yfinance": "USDTRY=X", "is_stock": False, "category": "Döviz"},
    "Euro (EUR)": {"yfinance": "EURTRY=X", "is_stock": False, "category": "Döviz"},
    "Altın (Gram)": {"yfinance_gold_usd": "GC=F", "yfinance_usd_try": "USDTRY=X", "type": "calculated_gold",
                     "category": "Emtia"},
    "Brent Petrol": {"yfinance": "BZ=F", "is_stock": False, "category": "Emtia"},
    "Bitcoin (BTC)": {"coingecko": "bitcoin", "type": "crypto", "category": "Kripto"}
}

PERIODS = {
    "1 Ay": "1mo",
    "3 Ay": "3mo",
    "6 Ay": "6mo",
    "1 Yıl": "1y",
    "5 Yıl": "5y"
}

TURKISH_INFLATION_RATE_YOY = 32.95

if 'show_return_analysis' not in st.session_state:
    st.session_state.show_return_analysis = False
if 'show_correlation' not in st.session_state:
    st.session_state.show_correlation = False
if 'show_fundamentals' not in st.session_state:
    st.session_state.show_fundamentals = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.comparison_results = []
    st.session_state.df_for_timeseries = []
    st.session_state.fundamentals_list = []
    st.session_state.income_statements = {}
    st.session_state.balance_sheets = {}
    st.session_state.cash_flows = {}
if 'show_daily_changes' not in st.session_state:
    st.session_state.show_daily_changes = False
if 'show_technical_analysis' not in st.session_state:
    st.session_state.show_technical_analysis = False


def set_section_state(section):
    st.session_state.show_return_analysis = False
    st.session_state.show_correlation = False
    st.session_state.show_fundamentals = False
    st.session_state.show_daily_changes = False
    st.session_state.show_technical_analysis = False
    if section == 'return_analysis':
        st.session_state.show_return_analysis = True
    elif section == 'correlation':
        st.session_state.show_correlation = True
    elif section == 'fundamentals':
        st.session_state.show_fundamentals = True
    elif section == 'daily_changes':
        st.session_state.show_daily_changes = True
    elif section == 'technical_analysis':
        st.session_state.show_technical_analysis = True


def load_data_and_set_state(assets, period):
    with st.spinner('Veriler çekiliyor ve analiz ediliyor...'):
        comparison_results = []
        df_for_timeseries = []
        fundamentals_list = []
        income_statements = {}
        balance_sheets = {}
        cash_flows = {}

        for name in assets:
            asset_info = all_tickers.get(name)
            if asset_info:
                df = get_historical_data(name, asset_info, period)
                df_for_timeseries.append((name, df))

                if asset_info.get("is_stock"):
                    fundamentals = get_stock_fundamentals(asset_info["yfinance"])

                    fundamentals_list.append({
                        'Varlık': name,
                        'Piyasa Değeri': fundamentals.get('marketCap'),
                        'Kurumsal Değer': fundamentals.get('enterpriseValue'),
                        'F/K Oranı': fundamentals.get('trailingPE'),
                        'PD/DD Oranı': fundamentals.get('priceToBook'),
                        'PEG Oranı': fundamentals.get('pegRatio'),
                        'Temettü Verimi': fundamentals.get('dividendYield'),
                        'Beta': fundamentals.get('beta'),
                        'FD/Satışlar': fundamentals.get('enterpriseToRevenue'),
                        'FD/FAVÖK': fundamentals.get('enterpriseToEbitda'),
                        'Fiyat/Serbest Nakit Akışı': fundamentals.get('priceToFreeCashflow')
                    })

                if df is not None and not df.empty:
                    percentage, tl_gain = calculate_percentage_change(df)
                    if percentage is not None:
                        comparison_results.append({
                            "Varlık": name,
                            "Yüzde Değişim": percentage,
                            "TL Kazanç": tl_gain
                        })
                else:
                    st.warning(f"{name} için veri çekilemedi. Lütfen tekrar deneyin.")
            else:
                st.warning(f"'{name}' için sembol bilgisi bulunamadı.")

        st.session_state.comparison_results = comparison_results
        st.session_state.df_for_timeseries = df_for_timeseries
        st.session_state.fundamentals_list = fundamentals_list
        st.session_state.data_loaded = True


@st.cache_data(ttl=3600)
def get_historical_data(asset_name, asset_info, period):
    try:
        if "coingecko" in asset_info and cg:
            end_date_unix = int(datetime.datetime.now().timestamp())

            if period == "1mo":
                start_date_unix = int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp())
            elif period == "3mo":
                start_date_unix = int((datetime.datetime.now() - datetime.timedelta(days=90)).timestamp())
            elif period == "6mo":
                start_date_unix = int((datetime.datetime.now() - datetime.timedelta(days=180)).timestamp())
            elif period == "1y":
                start_date_unix = int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp())
            elif period == "5y":
                start_date_unix = int((datetime.datetime.now() - datetime.timedelta(days=365 * 5)).timestamp())

            data = cg.get_coin_market_chart_range_by_id(
                id=asset_info["coingecko"],
                vs_currency="try",
                from_timestamp=start_date_unix,
                to_timestamp=end_date_unix
            )

            prices = data.get('prices', [])
            if prices:
                df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return None

        elif "yfinance_gold_usd" in asset_info:
            gold_usd_df = yf.download(asset_info["yfinance_gold_usd"], period=period, interval="1d")
            usd_try_df = yf.download(asset_info["yfinance_usd_try"], period=period, interval="1d")

            if not gold_usd_df.empty and not usd_try_df.empty:
                combined_df = pd.merge(gold_usd_df, usd_try_df, left_index=True, right_index=True,
                                       suffixes=('_gold', '_usd_try'))
                combined_df['Close'] = combined_df['Close_gold'] * combined_df['Close_usd_try']
                combined_df['Close'] = combined_df['Close'] / 31.1035
                return combined_df
            else:
                st.warning(f"Altın verisi çekilirken bir sorun oluştu.")
                return None

        elif "yfinance" in asset_info:
            df = yf.download(asset_info["yfinance"], period=period, interval="1d")
            return df if not df.empty else None

        return None
    except Exception as e:
        st.error(f"Veri çekilirken bir hata oluştu: {asset_name} - {e}")
        return None


@st.cache_data(ttl=3600 * 24)
def get_stock_fundamentals(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return info
    except Exception as e:
        st.warning(f"{ticker_symbol} için temel veri çekilirken hata: {e}")
        return {}


def calculate_percentage_change(df):
    if df.empty:
        return None, None

    start_price = df['Close'].iloc[0].item()
    end_price = df['Close'].iloc[-1].item()

    if start_price == 0:
        return 0, 0

    percentage_change = ((end_price - start_price) / start_price) * 100
    tl_gain = end_price - start_price

    return percentage_change, tl_gain


def plot_absolute_return_chart(comparison_df):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))

    sorted_df = comparison_df.sort_values(by='Yüzde Değişim', ascending=False)

    bars = ax.bar(sorted_df.index, sorted_df['Yüzde Değişim'], color='#4c72b0')

    ax.set_title("Mutlak Getiri Karşılaştırması", fontsize=18, fontweight='bold')
    ax.set_xlabel("Varlık", fontsize=14)
    ax.set_ylabel("Yüzde Değişim (%)", fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.axhline(y=TURKISH_INFLATION_RATE_YOY, color='r', linestyle='--', linewidth=2, label='Türkiye Enflasyonu')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}%',
                ha='center', va='bottom' if yval > 0 else 'top',
                fontsize=11, fontweight='bold')

    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)


def plot_inflation_adjusted_chart(comparison_df):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))

    sorted_df = comparison_df.sort_values(by='Enflasyon Üzeri Getiri', ascending=False)

    bar_colors = ['#55a868' if val >= 0 else '#c44e52' for val in sorted_df['Enflasyon Üzeri Getiri']]

    bars = ax.bar(sorted_df.index, sorted_df['Enflasyon Üzeri Getiri'], color=bar_colors)

    ax.set_title("Enflasyon Üzeri/Altı Getiri (%)", fontsize=18, fontweight='bold')
    ax.set_xlabel("Varlık", fontsize=14)
    ax.set_ylabel("Enflasyon Üzeri/Altı Getiri (%)", fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}%',
                ha='center', va='bottom' if yval > 0 else 'top',
                fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)


def plot_pe_ratio_chart(fundamentals_df):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))

    sorted_df = fundamentals_df.dropna(subset=['F/K Oranı']).sort_values(by='F/K Oranı', ascending=True)

    if sorted_df.empty:
        st.warning("F/K oranı grafiği için yeterli veri bulunamadı.")
        return

    bars = ax.bar(sorted_df['Varlık'], sorted_df['F/K Oranı'],
                  color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(sorted_df))))

    ax.set_title("Hisse Senetleri Fiyat/Kazanç (F/K) Oranı Karşılaştırması", fontsize=18, fontweight='bold')
    ax.set_xlabel("Hisse Senedi", fontsize=14)
    ax.set_ylabel("F/K Oranı", fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)


def plot_market_cap_chart(fundamentals_df):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))

    sorted_df = fundamentals_df.dropna(subset=['Piyasa Değeri']).sort_values(by='Piyasa Değeri', ascending=False)

    if sorted_df.empty:
        st.warning("Piyasa değeri grafiği için yeterli veri bulunamadı.")
        return

    sorted_df['Piyasa Değeri (Milyon ₺)'] = sorted_df['Piyasa Değeri'] / 1_000_000

    bars = ax.bar(sorted_df['Varlık'], sorted_df['Piyasa Değeri (Milyon ₺)'],
                  color=plt.cm.coolwarm(np.linspace(0, 1, len(sorted_df))))

    ax.set_title("Hisse Senetleri Piyasa Değeri Karşılaştırması (Milyon ₺)", fontsize=18, fontweight='bold')
    ax.set_xlabel("Hisse Senedi", fontsize=14)
    ax.set_ylabel("Piyasa Değeri (Milyon ₺)", fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.ticklabel_format(style='plain', axis='y')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'₺{yval:.2f}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)


def plot_100_tl_return_chart(comparison_df):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))

    df_to_process = comparison_df.copy()
    df_to_process['100 TL ile Yıl Sonu Değeri'] = 100 * (1 + df_to_process['Yüzde Değişim'] / 100)
    df_to_process.loc['Türkiye Enflasyonu', '100 TL ile Yıl Sonu Değeri'] = 100 * (1 + TURKISH_INFLATION_RATE_YOY / 100)

    sorted_df = df_to_process.sort_values(by='100 TL ile Yıl Sonu Değeri', ascending=False)

    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_df)))
    bars = ax.bar(sorted_df.index, sorted_df['100 TL ile Yıl Sonu Değeri'], color=colors)

    ax.set_title("100 TL'nin Dönem Sonu Değeri Karşılaştırması", fontsize=18, fontweight='bold')
    ax.set_xlabel("Varlık", fontsize=14)
    ax.set_ylabel("Dönem Sonu Değeri (TL)", fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'₺{yval:.2f}',
                ha='center', va='bottom' if yval > 0 else 'top',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)


def plot_time_series_performance(df_list, period_text):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, df in df_list:
        if df is not None and not df.empty:
            normalized_prices = (df['Close'] / df['Close'].iloc[0]) * 100
            ax.plot(normalized_prices, label=name, linewidth=2)

    ax.set_title(f"100 TL'nin Zaman İçindeki Değeri ({period_text})", fontsize=18, fontweight='bold')
    ax.set_xlabel("Tarih", fontsize=14)
    ax.set_ylabel("Değer (₺)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)


def plot_correlation_matrix(df_list):
    combined_df = pd.DataFrame()
    for name, df in df_list:
        if df is not None and not df.empty:
            combined_df[name] = df['Close']

    combined_df.dropna(inplace=True)

    if combined_df.empty or len(combined_df.columns) < 2:
        st.warning("Korelasyon matrisi için yeterli veri bulunamadı. Lütfen en az iki varlık seçin.")
        return

    returns_df = combined_df.pct_change().dropna()
    correlation_matrix = returns_df.corr()

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, linewidths=.5)
    ax.set_title("Varlık Getirileri Korelasyon Matrisi", fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)


def show_daily_change_rates(df_list):
    st.subheader("Günlük Değişim Oranları")
    daily_changes = {}
    for name, df in df_list:
        if df is not None and not df.empty:
            daily_pct_change = df['Close'].pct_change() * 100
            last_change_series = daily_pct_change.dropna()

            if not last_change_series.empty:
                daily_changes[name] = last_change_series.iloc[-1].item()
            else:
                daily_changes[name] = np.nan
        else:
            daily_changes[name] = np.nan

    if daily_changes:
        daily_changes_df = pd.DataFrame(daily_changes.items(), columns=['Varlık', 'Son Günlük Değişim (%)'])
        daily_changes_df.set_index('Varlık', inplace=True)
        plot_daily_change_rates_chart(daily_changes_df)

        st.write("### Günlük Değişim Tablosu")
        st.dataframe(daily_changes_df.style.format({
            'Son Günlük Değişim (%)': '{:.2f}%'
        }, na_rep="-"))
    else:
        st.warning("Günlük değişim oranları için yeterli veri bulunamadı.")


def plot_daily_change_rates_chart(df):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))

    sorted_df = df.sort_values(by='Son Günlük Değişim (%)', ascending=False)

    bar_colors = ['#55a868' if val >= 0 else '#c44e52' for val in sorted_df['Son Günlük Değişim (%)']]

    bars = ax.bar(sorted_df.index, sorted_df['Son Günlük Değişim (%)'], color=bar_colors)

    ax.set_title("Varlıkların Son Günlük Değişim Oranları (%)", fontsize=18, fontweight='bold')
    ax.set_xlabel("Varlık", fontsize=14)
    ax.set_ylabel("Günlük Değişim (%)", fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}%',
                ha='center', va='bottom' if yval > 0 else 'top',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)


def plot_technical_indicators(df_list):
    st.subheader("Teknik Analiz Göstergeleri")

    for name, df in df_list:
        if df is not None and not df.empty:
            st.markdown(f"#### {name}")

            df_ta = df.copy()
            if isinstance(df_ta.columns, pd.MultiIndex):
                df_ta.columns = df_ta.columns.get_level_values(0)
            df_ta.dropna(inplace=True)

            if 'Close' not in df_ta.columns:
                st.warning(f"Teknik analiz için {name} varlığında 'Close' sütunu bulunamadı.")
                continue

            df_ta.ta.bbands(close=df_ta['Close'], append=True)
            df_ta.ta.rsi(close=df_ta['Close'], append=True)
            df_ta.ta.sma(close=df_ta['Close'], length=20, append=True)
            df_ta.ta.sma(close=df_ta['Close'], length=50, append=True)

            fig, ax1 = plt.subplots(figsize=(12, 7))
            ax1.set_title(f"{name} Fiyat, Bollinger Bantları ve Hareketli Ortalamalar", fontsize=16)
            ax1.set_xlabel("Tarih")
            ax1.set_ylabel("Fiyat (₺)")
            ax1.plot(df_ta.index, df_ta['Close'], label='Kapanış Fiyatı', color='black', linewidth=1.5)

            if 'BBL_20_2.0' in df_ta.columns and 'BBM_20_2.0' in df_ta.columns and 'BBU_20_2.0' in df_ta.columns:
                ax1.plot(df_ta.index, df_ta['BBL_20_2.0'], label='Alt Bant', color='red', linestyle='--', linewidth=1)
                ax1.plot(df_ta.index, df_ta['BBM_20_2.0'], label='Orta Bant', color='blue', linestyle='--', linewidth=1)
                ax1.plot(df_ta.index, df_ta['BBU_20_2.0'], label='Üst Bant', color='red', linestyle='--', linewidth=1)

            ax1.plot(df_ta.index, df_ta['SMA_20'], label='20-günlük MA', color='green', linewidth=1)
            ax1.plot(df_ta.index, df_ta['SMA_50'], label='50-günlük MA', color='purple', linewidth=1)
            ax1.legend()
            ax1.grid(True)
            plt.tight_layout()
            st.pyplot(fig)

            fig, ax2 = plt.subplots(figsize=(12, 3))
            ax2.set_title(f"{name} Göreceli Güç Endeksi (RSI)", fontsize=16)
            ax2.set_xlabel("Tarih")
            ax2.set_ylabel("RSI Değeri")

            if 'RSI_14' in df_ta.columns:
                ax2.plot(df_ta.index, df_ta['RSI_14'], label='RSI (14)', color='blue')
                ax2.axhline(70, linestyle='--', color='red', label='Aşırı Alım (70)')
                ax2.axhline(30, linestyle='--', color='green', label='Aşırı Satım (30)')

            ax2.legend()
            ax2.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("---")
        else:
            st.warning(f"{name} için teknik analiz verisi bulunamadı.")


st.title("💰 Getiri Karşılaştırması")
st.write(
    "Seçtiğiniz varlıkların getirisini TL ve yüzde bazlı olarak Türkiye enflasyonu ile karşılaştırın.")
st.write("---")

all_categories = sorted(list(set(info["category"] for info in all_tickers.values())))

st.sidebar.header("Ayarlar")

selected_period_text = st.sidebar.selectbox(
    "Zaman Aralığı Seçin",
    options=list(PERIODS.keys()),
    index=3
)

selected_period = PERIODS[selected_period_text]

selected_categories = st.sidebar.multiselect(
    "Varlık Kategorisi Seçin (Boş bırakırsanız tümü seçilir)",
    options=all_categories,
    default=all_categories
)

filtered_assets = {name: info for name, info in all_tickers.items() if info["category"] in selected_categories}
filtered_tickers = list(filtered_assets.keys())

selected_stocks = st.sidebar.multiselect(
    "Karşılaştırmak istediğiniz varlıkları seçin",
    options=filtered_tickers,
    default=[name for name, info in filtered_assets.items() if
             info.get("yfinance") in ["THYAO.IS", "KCHOL.IS", "USDTRY=X", "EURTRY=X", "XU100.IS"] or info.get(
                 "type") in ["crypto", "calculated_gold", "yfinance_gold_usd"]]
)

if st.button("Analiz ve Grafikleri Göster"):
    load_data_and_set_state(selected_stocks, selected_period)

if st.session_state.data_loaded:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Getiri Analizi", use_container_width=True):
            set_section_state('return_analysis')

    with col2:
        if st.button("Korelasyon", use_container_width=True):
            set_section_state('correlation')

    with col3:
        if st.button("Temel Veriler", use_container_width=True):
            set_section_state('fundamentals')

    with col4:
        if st.button("Günlük Değişim", use_container_width=True):
            set_section_state('daily_changes')

    with col5:
        if st.button("Teknik Analiz", use_container_width=True):
            set_section_state('technical_analysis')

    st.markdown("---")

    if st.session_state.show_return_analysis:
        if st.session_state.comparison_results:
            comparison_df = pd.DataFrame(st.session_state.comparison_results)
            comparison_df.set_index('Varlık', inplace=True)
            comparison_df.loc['Türkiye Enflasyonu'] = [TURKISH_INFLATION_RATE_YOY, np.nan]
            comparison_df = comparison_df.sort_values(by='Yüzde Değişim', ascending=False)
            comparison_df['Sıra'] = np.arange(1, len(comparison_df) + 1)
            comparison_df = comparison_df[['Sıra', 'Yüzde Değişim', 'TL Kazanç']]

            inflation_adjusted_df = comparison_df.copy()
            inflation_adjusted_df['Enflasyon Üzeri Getiri'] = inflation_adjusted_df[
                                                                  'Yüzde Değişim'] - TURKISH_INFLATION_RATE_YOY
            inflation_adjusted_df.drop(['Sıra', 'TL Kazanç'], axis=1, inplace=True)
            inflation_adjusted_df.sort_values(by='Enflasyon Üzeri Getiri', ascending=False, inplace=True)
        else:
            comparison_df = pd.DataFrame()
            inflation_adjusted_df = pd.DataFrame()

        st.subheader("Getiri Analizi")
        plot_time_series_performance(st.session_state.df_for_timeseries, selected_period_text)
        plot_absolute_return_chart(comparison_df)
        plot_inflation_adjusted_chart(inflation_adjusted_df)
        plot_100_tl_return_chart(comparison_df)
        st.write("### Varlık Getiri Tablosu")
        st.dataframe(comparison_df.style.format({
            'Sıra': '{:.0f}',
            'Yüzde Değişim': '{:.2f}%',
            'TL Kazanç': '₺{:.2f}'
        }, na_rep="-"))

    if st.session_state.show_correlation:
        st.subheader("Korelasyon Matrisi")
        plot_correlation_matrix(st.session_state.df_for_timeseries)

    if st.session_state.show_fundamentals and st.session_state.fundamentals_list:
        st.subheader("Temel Finansal Veriler")
        st.markdown("#### Temel Oranlar ve Metrikler")
        fundamentals_df = pd.DataFrame(st.session_state.fundamentals_list)
        fundamentals_df.set_index('Varlık', inplace=True)
        st.dataframe(fundamentals_df.style.format({
            'Piyasa Değeri': '₺{:.2f}',
            'Kurumsal Değer': '₺{:.2f}',
            'F/K Oranı': '{:.2f}',
            'PD/DD Oranı': '{:.2f}',
            'PEG Oranı': '{:.2f}',
            'Temettü Verimi': '{:.2%}',
            'Beta': '{:.2f}',
            'FD/Satışlar': '{:.2f}',
            'FD/FAVÖK': '{:.2f}',
            'Fiyat/Serbest Nakit Akışı': '{:.2f}'
        }, na_rep='-'))

        plot_market_cap_chart(fundamentals_df.reset_index())

        st.markdown("---")
        if st.button("F/K Oranı Grafiğini Göster"):
            plot_pe_ratio_chart(fundamentals_df.reset_index())
    elif st.session_state.show_fundamentals:
        st.warning("Temel finansal veriler için yeterli veri yok.")

    if st.session_state.show_daily_changes:
        show_daily_change_rates(st.session_state.df_for_timeseries)

    if st.session_state.show_technical_analysis:
        plot_technical_indicators(st.session_state.df_for_timeseries)