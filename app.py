import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
from io import BytesIO
import statsmodels.api as sm
from arch import arch_model 

# Page Configuration
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHING FUNCTIONS (inchangées) ---

@st.cache_data
def calculate_portfolio_metrics(returns):
    """Calculates mean returns and covariance matrix (Historical)"""
    mean_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252     # Annualized
    return mean_returns, cov_matrix

@st.cache_data
def get_stock_data(tickers, start, end):
    """Downloads and pre-processes stock data, calculates returns and initial metrics."""
    
    data_raw = yf.download(tickers, start=start, end=end)
    
    if 'Close' in data_raw.columns:
        data = data_raw['Close']
    elif isinstance(data_raw.columns, pd.MultiIndex):
        data = data_raw['Close']
    else:
        data = data_raw
    
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
        
    data = data.dropna(how='all')
    if data.empty or len(data) < 20:
        raise ValueError("Insufficient data retrieved for the selected period.")

    data = data.fillna(method='ffill').dropna()
    
    returns = data.pct_change().dropna()
    
    if returns.empty:
        raise ValueError("No valid trading days found after synchronizing data.")

    mean_returns, cov_matrix = calculate_portfolio_metrics(returns)
    
    return data, returns, mean_returns, cov_matrix # mean_returns et cov_matrix sont HISTORIQUES ici

@st.cache_data
def predict_ml_metrics(returns):
    """
    Prédit les rendements annuels moyens et la matrice de covariance
    en utilisant les modèles ARMA(1,1) et GARCH(1,1) sur les rendements quotidiens.
    """
    
    predicted_annual_returns = []
    predicted_daily_volatility = []

    for ticker in returns.columns:
        r = returns[ticker].dropna()
        
        try:
            am = arch_model(100 * r, vol='Garch', p=1, q=1, mean='AR', lags=1, dist='normal')
            res = am.fit(disp='off')
            
            forecast = res.forecast(horizon=1, method='simulation')
            
            # Note: Pour éviter la volatilité explosive observée, nous limitons la volatilité prédite
            daily_variance_pred = forecast.variance.iloc[-1, 0] / 10000
            daily_volatility_pred = np.sqrt(daily_variance_pred)
            
            # Capping de la volatilité à 150% pour empêcher la simulation de s'effondrer
            if daily_volatility_pred * np.sqrt(252) > 1.5:
                 daily_volatility_pred = 1.5 / np.sqrt(252)
            
            mean_daily_return_pred = res.params['mu'] + res.params['ar[1]'] * r.iloc[-1]
            
        except Exception as e:
            daily_volatility_pred = r.std()
            mean_daily_return_pred = r.mean()

        predicted_annual_returns.append(mean_daily_return_pred * 252)
        predicted_daily_volatility.append(daily_volatility_pred)

    mean_returns_pred = pd.Series(predicted_annual_returns, index=returns.columns)
    
    historical_daily_std = returns.std()
    scale_factors = np.array(predicted_daily_volatility) / historical_daily_std.values
    
    corr_matrix = returns.corr()
    V_matrix = np.diag(scale_factors)
    
    cov_matrix_daily_adjusted = V_matrix @ corr_matrix.values @ V_matrix
    
    cov_matrix_annual_pred = pd.DataFrame(
        cov_matrix_daily_adjusted * 252,
        index=returns.columns,
        columns=returns.columns
    )
    
    return mean_returns_pred, cov_matrix_annual_pred

@st.cache_data
def perform_optimization(mean_returns, cov_matrix, selected_tickers, risk_free_rate):
    """Performs the full Markowitz optimization and simulation."""
    
    def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio

    def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
        return -portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]

    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(selected_tickers)))
    initial_guess = [1/len(selected_tickers)] * len(selected_tickers)

    num_portfolios = 10000
    results = np.zeros((4, num_portfolios))
    
    np.random.seed(42)
    for i in range(num_portfolios):
        weights = np.random.random(len(selected_tickers))
        weights /= np.sum(weights)
        port_return, port_std, sharpe = portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe
        results[3, i] = i

    max_sharpe_idx = np.argmax(results[2])
    
    opt_sharpe = minimize(
        neg_sharpe,
        initial_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    max_sharpe_weights = opt_sharpe.x
    
    opt_vol = minimize(
        portfolio_volatility,
        initial_guess,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    min_vol_weights = opt_vol.x

    max_sharpe_return, max_sharpe_std, max_sharpe_ratio = portfolio_stats(max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)
    min_vol_return, min_vol_std, min_vol_sharpe = portfolio_stats(min_vol_weights, mean_returns, cov_matrix, risk_free_rate)
    
    return (results, max_sharpe_weights, min_vol_weights, 
            max_sharpe_return, max_sharpe_std, max_sharpe_ratio,
            min_vol_return, min_vol_std, min_vol_sharpe)

@st.cache_data
def run_monte_carlo_simulation(weights, mean_returns, cov_matrix, years, num_simulations):
    """
    Exécute les simulations de Monte Carlo pour un portefeuille donné.
    """
    
    portfolio_mean_return = np.sum(weights * mean_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    T = 252 * years
    
    results = np.zeros((num_simulations, T + 1))
    
    drift = portfolio_mean_return - (0.5 * portfolio_std_dev**2)
    
    drift_daily = drift / 252
    volatility_daily = portfolio_std_dev / np.sqrt(252)
    
    results[:, 0] = 1.0 
    
    np.random.seed(42) # Pour la reproductibilité des simulations
    for i in range(num_simulations):
        
        daily_returns_factor = np.exp(
            drift_daily + volatility_daily * np.random.normal(0, 1, T)
        )
        
        results[i, 1:] = np.cumprod(daily_returns_factor)
    
    final_values = results[:, -1]
    
    P50 = np.percentile(final_values, 50)
    P90 = np.percentile(final_values, 90)
    P10 = np.percentile(final_values, 10)
    
    return results, portfolio_mean_return, portfolio_std_dev, P10, P50, P90

# --- BACKTESTING / METRICS FUNCTIONS (inchangées) ---

def get_rebalance_periods(returns_index, frequency):
    """Calculates the time points where rebalancing should occur. (FIXED CLOSED ARGUMENT)"""
    if frequency == 'None (Buy & Hold)':
        return returns_index[[0]]
    
    offset_map = {'Monthly': 'M', 'Quarterly': 'Q', 'Annually': 'A'}
    offset = offset_map.get(frequency)
    
    if not offset:
        return returns_index[[0]]

    periods = pd.date_range(start=returns_index.min(), end=returns_index.max(), freq=offset, inclusive='right')
    
    rebalance_dates = []
    if not returns_index.empty:
        rebalance_dates.append(returns_index[0]) 
        
        for p in periods:
            closest_date = returns_index[returns_index >= p].min()
            if closest_date is not pd.NaT:
                rebalance_dates.append(closest_date)
                
    rebalance_dates = sorted(list(set(rebalance_dates)))
    return pd.to_datetime(rebalance_dates)

def backtest_with_rebalancing(returns, initial_weights, frequency):
    # ... (inchangé)
    if returns.empty:
        return pd.Series([1.0], index=[datetime.now().date()])

    rebalance_dates = get_rebalance_periods(returns.index, frequency)
    rebalance_dates = [d for d in rebalance_dates if d in returns.index]
    
    if not rebalance_dates:
        frequency = 'None (Buy & Hold)'
        rebalance_dates = returns.index[[0]]

    cumulative_returns = pd.Series(1.0, index=[returns.index[0] - timedelta(days=1)])
    current_weights = initial_weights
    start_date = returns.index[0]
    
    for i in range(len(rebalance_dates)):
        rebalance_point = rebalance_dates[i]
        end_date = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else returns.index[-1]
        
        period_returns = returns.loc[start_date:end_date]
        
        if period_returns.empty:
            break
        
        period_portfolio_returns = (period_returns * current_weights).sum(axis=1)
        period_cumulative = (1 + period_portfolio_returns).cumprod()
        
        last_value = cumulative_returns.iloc[-1]
        period_cumulative_scaled = period_cumulative * last_value
        
        if start_date != returns.index[0]:
            cumulative_returns = pd.concat([cumulative_returns, period_cumulative_scaled[1:]])
        else:
             cumulative_returns = pd.concat([cumulative_returns, period_cumulative_scaled])

        if i + 1 < len(rebalance_dates):
            current_weights = initial_weights
            start_date = rebalance_dates[i+1]
        
    return cumulative_returns.iloc[1:]

def calculate_metrics(returns_series, risk_free_rate):
    # ... (inchangé)
    if returns_series.empty:
        return {
            'Total Return': 0.0, 'Annual Return': 0.0, 'Annual Volatility': 0.0, 
            'Sharpe Ratio': 0.0, 'Max Drawdown': 0.0
        }
    
    total_return = (1 + returns_series).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns_series)) - 1
    annual_vol = returns_series.std() * np.sqrt(252)
    
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else 0
    
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    }

@st.cache_data
def create_excel_report(data, returns_df, correlation, sharpe_allocation, vol_allocation, 
    metrics_df, initial_investment, final_max_sharpe, final_min_vol, final_equal, 
    start_date, end_date, benchmark_ticker, risk_free_rate_pct):
    # ... (inchangé)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_data = {
            'Analysis Date': [datetime.now().strftime('%Y-%m-%d')],
            'Period': [f"{start_date} to {end_date}"],
            'Number of Stocks': [len(data.columns)],
            'Trading Days': [len(returns_df)],
            'Risk-Free Rate': [f"{risk_free_rate_pct:.2%}"],
            'Benchmark': [benchmark_ticker]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        returns_df.to_excel(writer, sheet_name='Stock Returns', index=False)
        correlation.to_excel(writer, sheet_name='Correlation Matrix')
        sharpe_allocation.to_excel(writer, sheet_name='Max Sharpe Portfolio', index=False)
        vol_allocation.to_excel(writer, sheet_name='Min Vol Portfolio', index=False)
        metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=True)
        data.to_excel(writer, sheet_name='Historical Prices')
        investment_calc = pd.DataFrame({
            'Portfolio': ['Max Sharpe', 'Min Volatility', 'Equal Weight'],
            'Initial Investment': [initial_investment] * 3,
            'Final Value': [final_max_sharpe, final_min_vol, final_equal],
            'Total Gain': [final_max_sharpe - initial_investment, final_min_vol - initial_investment, final_equal - initial_investment],
            'Return %': [(final_max_sharpe/initial_investment - 1) * 100, (final_min_vol/initial_investment - 1) * 100, (final_equal/initial_investment - 1) * 100]
        })
        investment_calc.to_excel(writer, sheet_name='Investment Calculation', index=False)
    output.seek(0)
    return output

# --- MAIN APPLICATION START ---

st.title("Portfolio Optimizer Dashboard")
st.markdown("---")

st.markdown("""
**Welcome to the Portfolio Optimizer!**
This application uses **Modern Portfolio Theory (Markowitz)**, enhanced with optional **Machine Learning** forecasting, to help you build and optimize investment portfolios.
""")

st.markdown("---")

# Sidebar - Parameter Selection
st.sidebar.header("Configuration")

# --- NOUVEAU: MODE D'ANALYSE ---
st.sidebar.subheader("Analysis Mode")
analysis_mode = st.sidebar.selectbox(
    "Select Optimization Basis:",
    options=['Classic MPT (Historical)', 'ML Forecast (ARMA/GARCH)'],
    index=0,
    help="Classic MPT uses historical averages. ML Forecast uses a model to predict the next period's expected return and volatility."
)
# Déterminer si les champs MC doivent être désactivés
mc_disabled = (analysis_mode == 'ML Forecast (ARMA/GARCH)')

if analysis_mode == 'ML Forecast (ARMA/GARCH)':
    st.sidebar.warning("ML Prediction requires the 'arch' library (pip install arch) and can be time-consuming.")

# --- PARAMÈTRES D'INVESTISSEMENT ---
st.sidebar.subheader("Investment Parameters")

# Taux Sans Risque
risk_free_rate_pct = st.sidebar.number_input(
    "Risk-Free Rate (Annual, %):",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    format="%.2f",
    help="Used to calculate the Sharpe Ratio. (e.g., 3-month T-Bill yield)."
) / 100
risk_free_rate = risk_free_rate_pct

# Benchmark
benchmark_options = {'S&P 500': '^GSPC', 'NASDAQ 100': '^IXIC', 'CAC 40': '^FCHI', 'DAX': '^GDAXI', 'None': None}
benchmark_selected_name = st.sidebar.selectbox(
    "Select Benchmark Index:",
    options=list(benchmark_options.keys()),
    index=0
)
benchmark_ticker = benchmark_options[benchmark_selected_name]


# --- LISTES PRÉDÉFINIES ---
asset_lists = {
    "US Tech Giants": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    "US Blue Chips": ['JPM', 'V', 'PG', 'KO', 'UNH', 'JNJ', 'WMT', 'DIS', 'HD'],
    "European Blue Chips": ['MC.PA', 'OR.PA', 'BNP.PA', 'AIR.PA', 'SAN.PA', 'NESN.SW', 'AD.AS', 'SAP'],
    "Other Global": ['EFA', 'EEM', 'GLD', 'BND']
}

# --- SÉLECTION DES ACTIFS ---
st.sidebar.subheader("Asset Selection (Tickers)")

# 1. ZONE DE TEXTE POUR LES TICKERS PERSONNALISÉS
custom_tickers_input = st.sidebar.text_input(
    "Enter Tickers (comma-separated):",
    value='AAPL, MSFT',
    help="Enter stock symbols (e.g., AAPL, JPM, MC.PA). Max 15 tickers."
)
custom_tickers = [t.strip().upper() for t in custom_tickers_input.split(',') if t.strip()]

st.sidebar.markdown("---")
st.sidebar.markdown("**OR select from predefined lists:**")

# 2. LISTES PRÉDÉFINIES (MultiSelects)
multiselect_tickers = []
for category, tickers in asset_lists.items():
    
    # AFFICHAGE DU TITRE DE LA CATÉGORIE
    st.sidebar.markdown(f"**{category}**")
    
    selected = st.sidebar.multiselect(
        f"Select from {category}:",
        options=tickers,
        default=[],
        key=category.replace(" ", "_"),
        label_visibility="collapsed"
    )
    multiselect_tickers.extend(selected)

# 3. FUSION ET NETTOYAGE
combined_tickers_set = set(custom_tickers + multiselect_tickers)

if benchmark_ticker in combined_tickers_set:
    combined_tickers_set.remove(benchmark_ticker)

# Tickers finaux pour l'optimisation
selected_tickers = sorted(list(combined_tickers_set))


# Afficher la liste finale
st.sidebar.markdown("---")
st.sidebar.write(f"**Assets to Optimize ({len(selected_tickers)}):** {', '.join(selected_tickers)}")


# Sélection de la période
st.sidebar.subheader("Time Period")
end_date = datetime.now()
start_date_default = end_date - timedelta(days=3*365)

start_date = st.sidebar.date_input("Start date:", start_date_default)
end_date = st.sidebar.date_input("End date:", end_date)

# Fréquence de Rebalancing
st.sidebar.subheader("Rebalancing Settings")
rebalance_frequency = st.sidebar.selectbox(
    "Automatic Rebalancing Frequency:",
    options=['None (Buy & Hold)', 'Monthly', 'Quarterly', 'Annually'],
    index=2,
    help="How often the portfolio weights are reset to their initial optimal values."
)

# Paramètres Monte Carlo
st.sidebar.subheader("Monte Carlo Simulation")
st.sidebar.caption("Available only in 'Classic MPT' mode.")
mc_years = st.sidebar.number_input(
    "Simulation Horizon (Years):",
    min_value=1,
    max_value=30,
    value=10,
    step=1,
    help="How many years into the future to forecast the portfolio.",
    disabled=mc_disabled # DÉSACTIVATION CONDITIONNELLE
)
mc_simulations = st.sidebar.number_input(
    "Number of Simulations:",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="The number of random paths to generate for forecasting.",
    disabled=mc_disabled # DÉSACTIVATION CONDITIONNELLE
)

# Button to run analysis
run_analysis = st.sidebar.button("Run Analysis", type="primary")

if run_analysis:
    st.session_state.analysis_run = True

if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False


# Le bloc principal d'analyse
if st.session_state.analysis_run:
    if len(selected_tickers) < 2:
        st.error("Please select at least 2 stocks for optimization!")
        st.session_state.analysis_run = False
    elif start_date >= end_date:
        st.error("Start date must be before end date!")
        st.session_state.analysis_run = False
    else:
        st.success(f"Analysis running for {len(selected_tickers)} stocks...")
        
        # --- DATA RETRIEVAL ---
        
        tickers_to_download = selected_tickers[:]
        if benchmark_ticker and benchmark_ticker not in tickers_to_download:
            tickers_to_download.append(benchmark_ticker)
            
        with st.spinner("Downloading and processing data..."):
            try:
                # 1. Télécharger les données historiques (BASE)
                data_all, returns_all, mean_returns_hist, cov_matrix_hist = get_stock_data(
                    tickers_to_download, start_date, end_date
                )
                
                # Filtrer les données pour le portefeuille
                data = data_all[selected_tickers]
                returns = returns_all[selected_tickers]
                
                # Données Benchmark
                returns_benchmark = None
                if benchmark_ticker in returns_all.columns:
                    returns_benchmark = returns_all[benchmark_ticker].rename('Benchmark')

            except ValueError as e:
                st.error(f"Data Error: {str(e)}")
                st.session_state.analysis_run = False
                st.stop()
            except Exception as e:
                st.error(f"An unexpected error occurred during data download: {str(e)}")
                st.session_state.analysis_run = False
                st.stop()
        
        # --- 2. CHOIX DU MODÈLE D'ESTIMATION ---
        
        if analysis_mode == 'ML Forecast (ARMA/GARCH)':
            with st.spinner("Running Machine Learning model (ARMA/GARCH) for prediction..."):
                try:
                    mean_returns_pred, cov_matrix_pred = predict_ml_metrics(returns)
                    
                    mean_returns = mean_returns_pred
                    cov_matrix = cov_matrix_pred
                    
                    st.success("ML Forecasting complete! Optimization uses predicted metrics.")
                except Exception as e:
                    st.error(f"ML Model execution failed. Falling back to Classic MPT. Error: {e}")
                    mean_returns = mean_returns_hist[selected_tickers]
                    cov_matrix = cov_matrix_hist.loc[selected_tickers, selected_tickers]
                    analysis_mode = 'Classic MPT (Historical)' # Change mode for display
        else:
            # Mode Classique: Utiliser les données historiques
            mean_returns = mean_returns_hist[selected_tickers]
            cov_matrix = cov_matrix_hist.loc[selected_tickers, selected_tickers]
        
        
        # --- 3. OPTIMIZATION ---
        with st.spinner(f"Running Markowitz Optimization using {analysis_mode} metrics..."):
            (results, max_sharpe_weights, min_vol_weights, 
             max_sharpe_return, max_sharpe_std, max_sharpe_ratio,
             min_vol_return, min_vol_std, min_vol_sharpe) = perform_optimization(
                 mean_returns, cov_matrix, selected_tickers, risk_free_rate
             )
        
        # --- DISPLAY SECTIONS ---
        
        # ============================================================
        # SECTION 1 : HISTORICAL PRICES
        # ============================================================
        st.subheader("Historical Prices")
        
        normalized_data = (data / data.iloc[0]) * 100
        
        fig = go.Figure()
        for ticker in normalized_data.columns:
            fig.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data[ticker], mode='lines', name=ticker, line=dict(width=2)))
        
        if returns_benchmark is not None:
             normalized_benchmark = (data_all[benchmark_ticker] / data_all[benchmark_ticker].iloc[0]) * 100
             fig.add_trace(go.Scatter(x=normalized_benchmark.index, y=normalized_benchmark, mode='lines', name=f"Benchmark ({benchmark_ticker})", line=dict(width=2, dash='dash', color='purple')))

        
        fig.update_layout(
            title="Normalized Prices (Base 100)",
            xaxis_title="Date",
            yaxis_title="Price (Base 100)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ============================================================
        # SECTION 2 : RETURNS
        # ============================================================
        st.markdown("---")
        st.subheader(f"Returns Analysis ({analysis_mode} Basis)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Trading Days (Historical)", len(returns))
        with col2:
            st.metric("Period", f"{data.index[0].date()} to {data.index[-1].date()}")
        with col3:
            st.metric("Optimization Basis", analysis_mode)
        with col4:
            st.metric("Risk-Free Rate", f"{risk_free_rate_pct:.2%}")

        
        st.write("**Annualized Input Metrics by Asset:**")
        returns_df = pd.DataFrame({
            'Stock': mean_returns.index,
            'Annual Return (Input)': mean_returns.values,
            'Annual Volatility (Input)': np.sqrt(np.diag(cov_matrix))
        })
        returns_df['Sharpe Ratio (Input)'] = (returns_df['Annual Return (Input)'] - risk_free_rate) / returns_df['Annual Volatility (Input)']
        returns_df = returns_df.sort_values('Annual Return (Input)', ascending=False)
        
        def color_returns(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}'
        
        styled_returns = returns_df.style.format({
            'Annual Return (Input)': '{:.2%}',
            'Annual Volatility (Input)': '{:.2%}',
            'Sharpe Ratio (Input)': '{:.2f}'
        }).applymap(color_returns, subset=['Annual Return (Input)'])
        
        st.dataframe(styled_returns, use_container_width=True)
        
        # ============================================================
        # SECTION 3 : CORRELATION MATRIX
        # ============================================================
        st.markdown("---")
        st.subheader(f"Correlation Matrix (Used in Optimization - based on {analysis_mode})")
        
        correlation = pd.DataFrame(cov_matrix.values / (np.sqrt(np.diag(cov_matrix))[:, None] * np.sqrt(np.diag(cov_matrix))[None, :]),
                                    index=cov_matrix.index, columns=cov_matrix.columns)
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation.values, x=correlation.columns, y=correlation.columns, colorscale='RdBu', zmid=0, 
            text=correlation.values, texttemplate='%{text:.2f}', textfont={"size": 12}, colorbar=dict(title="Correlation")
        ))
        
        fig_corr.update_layout(title="Correlation Matrix of Input Metrics", height=500, width=500)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.write("**Interpretation:**")
            st.write(f"""
            This matrix shows the correlation used for optimization. 
            When using **ML Forecast**, the correlation structure is based on historical correlation, but the volatility component is adjusted by the predicted GARCH volatility.
            """)
            
            corr_pairs = []
            for i in range(len(correlation.columns)):
                for j in range(i+1, len(correlation.columns)):
                    corr_pairs.append({'Pair': f"{correlation.columns[i]} - {correlation.columns[j]}", 'Correlation': correlation.iloc[i, j]})
            
            corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
            
            st.write("**Most Correlated:**")
            st.dataframe(corr_pairs_df.head(3), use_container_width=True)
            
            st.write("**Least Correlated:**")
            st.dataframe(corr_pairs_df.tail(3), use_container_width=True)
        
        # ============================================================
        # SECTION 4 : RETURNS DISTRIBUTION (HISTORICAL - inchangé)
        # ============================================================
        st.markdown("---")
        st.subheader("Historical Returns Distribution (Used for ML Training)")
        
        fig_dist = go.Figure()
        for ticker in returns.columns:
            fig_dist.add_trace(go.Histogram(x=returns[ticker], name=ticker, opacity=0.7, nbinsx=50))
        
        fig_dist.update_layout(title="Daily Returns Distribution", xaxis_title="Daily Return", yaxis_title="Frequency", barmode='overlay', height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
        # 

        # ============================================================
        # SECTION 5 : PORTFOLIO OPTIMIZATION
        # ============================================================
        st.markdown("---")
        st.subheader(f"Portfolio Optimization ({analysis_mode})")
        
        st.write("### Optimal Portfolios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Maximum Sharpe Ratio Portfolio")
            st.metric("Return (Input)", f"{max_sharpe_return:.2%}")
            st.metric("Volatility (Input)", f"{max_sharpe_std:.2%}")
            st.metric("Sharpe Ratio (Input)", f"{max_sharpe_ratio:.2f}")
        
        with col2:
            st.markdown("#### Minimum Volatility Portfolio")
            st.metric("Return (Input)", f"{min_vol_return:.2%}")
            st.metric("Volatility (Input)", f"{min_vol_std:.2%}")
            st.metric("Sharpe Ratio (Input)", f"{min_vol_sharpe:.2f}")
        
        st.write("### Efficient Frontier")
        
        fig_frontier = go.Figure()
        
        fig_frontier.add_trace(go.Scatter(
            x=results[1, :], y=results[0, :], mode='markers', 
            marker=dict(size=4, color=results[2, :], colorscale='Viridis', showscale=True, colorbar=dict(title=f"Sharpe<br>Ratio (RFR={risk_free_rate_pct:.1%})")),
            text=[f"Return: {r:.2%}<br>Volatility: {v:.2%}<br>Sharpe: {s:.2f}" for r, v, s in zip(results[0, :], results[1, :], results[2, :])],
            hovertemplate='%{text}<extra></extra>', name='Portfolios'
        ))
        
        fig_frontier.add_trace(go.Scatter(x=[max_sharpe_std], y=[max_sharpe_return], mode='markers', marker=dict(size=15, color='red', symbol='star', line=dict(color='white', width=2)), name='Max Sharpe', hovertemplate=f'Max Sharpe Ratio<br>Return: {max_sharpe_return:.2%}<br>Volatility: {max_sharpe_std:.2%}<br>Sharpe: {max_sharpe_ratio:.2f}<extra></extra>'))
        fig_frontier.add_trace(go.Scatter(x=[min_vol_std], y=[min_vol_return], mode='markers', marker=dict(size=15, color='green', symbol='diamond', line=dict(color='white', width=2)), name='Min Volatility', hovertemplate=f'Min Volatility<br>Return: {min_vol_return:.2%}<br>Volatility: {min_vol_std:.2%}<br>Sharpe: {min_vol_sharpe:.2f}<extra></extra>'))
        
        for i, ticker in enumerate(selected_tickers):
            ann_return = mean_returns[ticker]
            ann_std = np.sqrt(cov_matrix.iloc[i, i])
            fig_frontier.add_trace(go.Scatter(x=[ann_std], y=[ann_return], mode='markers+text', marker=dict(size=10, color='orange', symbol='square'), text=[ticker], textposition='top center', name=ticker, hovertemplate=f'{ticker}<br>Return: {ann_return:.2%}<br>Volatility: {ann_std:.2%}<extra></extra>'))
        
        fig_frontier.update_layout(title=f"Efficient Frontier ({analysis_mode}) - Risk vs Return", xaxis_title="Volatility (Input)", yaxis_title="Expected Return (Input)", hovermode='closest', height=600, showlegend=True)
        st.plotly_chart(fig_frontier, use_container_width=True)
        # 
        
        st.info("""
        **Note:** The metrics on this chart are the **inputs** to the optimization (Historical Averages or ML Predictions).
        """)

        # ============================================================
        # SECTION 6 : OPTIMAL PORTFOLIO ALLOCATIONS
        # ============================================================
        st.markdown("---")
        st.write("### Portfolio Allocations")
        
        col1, col2 = st.columns(2)
        
        sharpe_allocation = pd.DataFrame({'Stock': selected_tickers, 'Weight (%)': max_sharpe_weights * 100}).sort_values('Weight (%)', ascending=False)
        vol_allocation = pd.DataFrame({'Stock': selected_tickers, 'Weight (%)': min_vol_weights * 100}).sort_values('Weight (%)', ascending=False)
        
        with col1:
            st.markdown("#### Maximum Sharpe Ratio Allocation")
            st.dataframe(sharpe_allocation.style.format({'Weight (%)': '{:.2f}%'}).background_gradient(cmap='Greens'), use_container_width=True)
            
            fig_pie1 = go.Figure(data=[go.Pie(labels=sharpe_allocation['Stock'], values=sharpe_allocation['Weight (%)'], hole=0.3, marker=dict(colors=px.colors.qualitative.Set3))])
            fig_pie1.update_layout(title="Allocation Breakdown", height=400)
            st.plotly_chart(fig_pie1, use_container_width=True)
            
            st.markdown("**Investment Calculator**")
            investment_amount_sharpe = st.number_input("Amount to invest ($):", min_value=100, value=10000, step=100, key='sharpe_investment')
            
            st.write("**Allocation in dollars:**")
            allocation_dollars_sharpe = pd.DataFrame({'Stock': selected_tickers, 'Amount ($)': max_sharpe_weights * investment_amount_sharpe}).sort_values('Amount ($)', ascending=False)
            st.dataframe(allocation_dollars_sharpe.style.format({'Amount ($)': '${:,.2f}'}), use_container_width=True)
        
        with col2:
            st.markdown("#### Minimum Volatility Allocation")
            st.dataframe(vol_allocation.style.format({'Weight (%)': '{:.2f}%'}).background_gradient(cmap='Blues'), use_container_width=True)
            
            fig_pie2 = go.Figure(data=[go.Pie(labels=vol_allocation['Stock'], values=vol_allocation['Weight (%)'], hole=0.3, marker=dict(colors=px.colors.qualitative.Pastel))])
            fig_pie2.update_layout(title="Allocation Breakdown", height=400)
            st.plotly_chart(fig_pie2, use_container_width=True)
            
            st.markdown("**Investment Calculator**")
            investment_amount_vol = st.number_input("Amount to invest ($):", min_value=100, value=10000, step=100, key='vol_investment')
            
            st.write("**Allocation in dollars:**")
            allocation_dollars_vol = pd.DataFrame({'Stock': selected_tickers, 'Amount ($)': min_vol_weights * investment_amount_vol}).sort_values('Amount ($)', ascending=False)
            st.dataframe(allocation_dollars_vol.style.format({'Amount ($)': '${:,.2f}'}), use_container_width=True)
        
        st.markdown("---")
        st.write("### Portfolio Comparison")
        
        comparison_df = pd.DataFrame({
            'Metric': ['Expected Return (Input)', 'Volatility (Input)', 'Sharpe Ratio (Input)'],
            'Max Sharpe Portfolio': [f"{max_sharpe_return:.2%}", f"{max_sharpe_std:.2%}", f"{max_sharpe_ratio:.2f}"],
            'Min Volatility Portfolio': [f"{min_vol_return:.2%}", f"{min_vol_std:.2%}", f"{min_vol_sharpe:.2f}"]
        })
        st.dataframe(comparison_df, use_container_width=True)
        
        st.success("""
        **Which portfolio should you choose?**
        - **Max Sharpe Ratio**: Best risk-adjusted returns (higher Sharpe means better return per unit of risk).
        - **Min Volatility**: Lowest absolute risk.
        """)
        
        # ============================================================
        # SECTION 7 : SIMULATION MONTE CARLO (FORECASTING)
        # ============================================================
        st.markdown("---")
        st.subheader("Portfolio Forecasting (Monte Carlo)")

        initial_investment = 10000 
        
        # Condition pour l'exécution de Monte Carlo
        if analysis_mode == 'Classic MPT (Historical)':
            st.info(f"""
            Running **{mc_simulations}** Monte Carlo simulations over **{mc_years} years** using the **Maximum Sharpe Ratio** portfolio weights 
            and **{analysis_mode}** metrics for simulation inputs.
            """)

            try:
                # Exécution de Monte Carlo UNIQUEMENT en mode Classique
                mc_results, mc_mean, mc_std, P10, P50, P90 = run_monte_carlo_simulation(
                    max_sharpe_weights, mean_returns, cov_matrix, mc_years, mc_simulations
                )
                
                # Affichage des métriques clés
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Expected Annual Return", f"{mc_mean:.2%}")
                with col2:
                    st.metric("Expected Annual Volatility", f"{mc_std:.2%}")
                with col3:
                    st.metric("Worst Case (10%) Value", f"${P10 * initial_investment:,.0f}")
                with col4:
                    st.metric("Median Case (50%) Value", f"${P50 * initial_investment:,.0f}")
                with col5:
                    st.metric("Best Case (90%) Value", f"${P90 * initial_investment:,.0f}")

                # Graphique de la simulation
                st.write("### Monte Carlo Path Visualization")
                
                df_mc = pd.DataFrame(mc_results).T
                
                fig_mc = go.Figure()
                
                num_paths_to_plot = min(50, mc_simulations)
                for i in range(num_paths_to_plot):
                    fig_mc.add_trace(go.Scatter(
                        y=df_mc.iloc[:, i] * initial_investment, 
                        mode='lines', 
                        opacity=0.2, 
                        line=dict(width=1, color='lightblue'),
                        showlegend=False
                    ))

                def calculate_percentile_path(mc_results, percentile):
                    return np.percentile(mc_results, percentile, axis=0)

                time_index = np.linspace(0, mc_years, mc_results.shape[1])
                
                P90_path = calculate_percentile_path(mc_results, 90) * initial_investment
                P50_path = calculate_percentile_path(mc_results, 50) * initial_investment
                P10_path = calculate_percentile_path(mc_results, 10) * initial_investment
                
                fig_mc.add_trace(go.Scatter(y=P50_path, x=time_index, mode='lines', line=dict(color='green', width=3), name='Median (P50)'))
                fig_mc.add_trace(go.Scatter(y=P90_path, x=time_index, mode='lines', line=dict(color='blue', width=2), name='Optimistic (P90)'))
                fig_mc.add_trace(go.Scatter(y=P10_path, x=time_index, mode='lines', line=dict(color='red', width=2), name='Pessimistic (P10)'))
                
                fig_mc.update_layout(
                    title=f"Monte Carlo Forecast for {mc_years} Years (Based on {analysis_mode})",
                    xaxis_title="Time (Years)",
                    yaxis_title=f"Portfolio Value ($) - Initial: ${initial_investment:,.0f}",
                    height=600
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
                
                
                st.markdown("""
                **Interpretation of Percentiles:**
                - The **P50 (Median)** line shows that, based on historical risk and return, there is a 50% chance the portfolio will perform better than this path.
                - The **P10** line indicates that in 90% of the simulations, the portfolio ended up with a value *higher* than this line. It represents a potential worst-case scenario.
                - The **P90** line indicates an optimistic scenario, where the portfolio ended up with a value *higher* than this line in only 10% of the simulations.
                """)
                
            except Exception as e:
                st.error(f"Error during Monte Carlo simulation: {e}")
                st.warning("Monte Carlo simulation failed. Check input data and historical volatility.")
        
        else:
            # Mode ML sélectionné, Monte Carlo est désactivé
            st.info("Monte Carlo simulation is disabled in **ML Forecast** mode. The analysis focuses on predicted one-period metrics (Optimization Inputs).")


        # ============================================================
        # SECTION 8 : BACKTESTING AVEC BENCHMARK ET REBALANCING
        # ============================================================
        st.markdown("---")
        st.subheader("Backtesting & Performance Comparison (Historical)")
        
        st.warning("Note: Backtesting always uses **historical returns**, regardless of the optimization mode (ML/Classic).")

        if rebalance_frequency != 'None (Buy & Hold)':
            st.info(f"Portfolio performance calculated with **{rebalance_frequency} Rebalancing**.")
            
            cumulative_max_sharpe = backtest_with_rebalancing(returns, max_sharpe_weights, rebalance_frequency)
            portfolio_max_sharpe = cumulative_max_sharpe.pct_change().dropna()
            
            cumulative_min_vol = backtest_with_rebalancing(returns, min_vol_weights, rebalance_frequency)
            portfolio_min_vol = cumulative_min_vol.pct_change().dropna()
            
            equal_weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
            cumulative_equal = backtest_with_rebalancing(returns, equal_weights, rebalance_frequency)
            portfolio_equal = cumulative_equal.pct_change().dropna()
            
        else:
            st.info("Portfolio performance calculated using a **Buy & Hold** strategy (no rebalancing).")
            portfolio_max_sharpe = (returns * max_sharpe_weights).sum(axis=1)
            cumulative_max_sharpe = (1 + portfolio_max_sharpe).cumprod()
            
            portfolio_min_vol = (returns * min_vol_weights).sum(axis=1)
            cumulative_min_vol = (1 + portfolio_min_vol).cumprod()
            
            equal_weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
            portfolio_equal = (returns * equal_weights).sum(axis=1)
            cumulative_equal = (1 + portfolio_equal).cumprod()

        cumulative_stocks = (1 + returns).cumprod()
        
        if returns_benchmark is not None:
            cumulative_benchmark = (1 + returns_benchmark).cumprod()
            metrics_benchmark = calculate_metrics(returns_benchmark, risk_free_rate)
        
        # Performance Chart
        fig_backtest = go.Figure()
        
        fig_backtest.add_trace(go.Scatter(x=cumulative_max_sharpe.index, y=cumulative_max_sharpe.values, mode='lines', name='Max Sharpe Portfolio (Optimized)', line=dict(color='red', width=3)))
        fig_backtest.add_trace(go.Scatter(x=cumulative_min_vol.index, y=cumulative_min_vol.values, mode='lines', name='Min Volatility Portfolio (Optimized)', line=dict(color='green', width=3)))
        fig_backtest.add_trace(go.Scatter(x=cumulative_equal.index, y=cumulative_equal.values, mode='lines', name='Equal Weight Portfolio', line=dict(color='gray', width=2, dash='dash')))
        
        if returns_benchmark is not None:
             fig_backtest.add_trace(go.Scatter(x=cumulative_benchmark.index, y=cumulative_benchmark.values, mode='lines', name=f"Benchmark ({benchmark_ticker})", line=dict(color='purple', width=3, dash='dot')))
        
        colors = px.colors.qualitative.Plotly
        for i, ticker in enumerate(selected_tickers):
            fig_backtest.add_trace(go.Scatter(x=cumulative_stocks.index, y=cumulative_stocks[ticker].values, mode='lines', name=ticker, line=dict(color=colors[i % len(colors)], width=1.5, dash='dot'), opacity=0.6))
        
        fig_backtest.update_layout(title="Cumulative Returns Comparison (Base = $1)", xaxis_title="Date", yaxis_title="Portfolio Value ($)", hovermode='x unified', height=600, showlegend=True)
        st.plotly_chart(fig_backtest, use_container_width=True)
        # 

        # Performance Metrics Table
        st.write("### Performance Metrics (Historical Backtest)")
        
        metrics_max_sharpe = calculate_metrics(portfolio_max_sharpe, risk_free_rate)
        metrics_min_vol = calculate_metrics(portfolio_min_vol, risk_free_rate)
        metrics_equal = calculate_metrics(portfolio_equal, risk_free_rate)
        
        data_for_df = {
            'Max Sharpe': [f"{metrics_max_sharpe['Total Return']:.2%}", f"{metrics_max_sharpe['Annual Return']:.2%}", f"{metrics_max_sharpe['Annual Volatility']:.2%}", f"{metrics_max_sharpe['Sharpe Ratio']:.2f}", f"{metrics_max_sharpe['Max Drawdown']:.2f}"],
            'Min Volatility': [f"{metrics_min_vol['Total Return']:.2%}", f"{metrics_min_vol['Annual Return']:.2%}", f"{metrics_min_vol['Annual Volatility']:.2%}", f"{metrics_min_vol['Sharpe Ratio']:.2f}", f"{metrics_min_vol['Max Drawdown']:.2f}"],
            'Equal Weight': [f"{metrics_equal['Total Return']:.2%}", f"{metrics_equal['Annual Return']:.2%}", f"{metrics_equal['Annual Volatility']:.2%}", f"{metrics_equal['Sharpe Ratio']:.2f}", f"{metrics_equal['Max Drawdown']:.2f}"]
        }
        
        if returns_benchmark is not None:
            data_for_df[benchmark_selected_name] = [
                f"{metrics_benchmark['Total Return']:.2%}", f"{metrics_benchmark['Annual Return']:.2%}", f"{metrics_benchmark['Annual Volatility']:.2%}",
                f"{metrics_benchmark['Sharpe Ratio']:.2f}", f"{metrics_benchmark['Max Drawdown']:.2f}"
            ]
        
        metrics_df = pd.DataFrame(data_for_df, index=['Total Return', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown'])
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Investment Growth
        st.write("### Investment Growth (Historical Backtest)")
        
        final_max_sharpe = initial_investment * cumulative_max_sharpe.iloc[-1] if not cumulative_max_sharpe.empty else initial_investment
        final_min_vol = initial_investment * cumulative_min_vol.iloc[-1] if not cumulative_min_vol.empty else initial_investment
        final_equal = initial_investment * cumulative_equal.iloc[-1] if not cumulative_equal.empty else initial_investment
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Sharpe Portfolio", f"${final_max_sharpe:,.2f}", f"+${final_max_sharpe - initial_investment:,.2f}")
        
        with col2:
            st.metric("Min Volatility Portfolio", f"${final_min_vol:,.2f}", f"+${final_min_vol - initial_investment:,.2f}")
        
        with col3:
            st.metric("Equal Weight Portfolio", f"${final_equal:,.2f}", f"+${final_equal - initial_investment:,.2f}")
        
        if returns_benchmark is not None:
            final_benchmark = initial_investment * cumulative_benchmark.iloc[-1] if not cumulative_benchmark.empty else initial_investment
            st.metric(f"Benchmark ({benchmark_ticker})", f"${final_benchmark:,.2f}", f"+${final_benchmark - initial_investment:,.2f}")
        
        st.success(f"""
        **Conclusion:**
        - An initial investment of **${initial_investment:,}** would have grown to:
          - **Max Sharpe:** ${final_max_sharpe:,.2f} (+{((final_max_sharpe/initial_investment - 1) * 100):.1f}%)
          - **Min Volatility:** ${final_min_vol:,.2f} (+{((final_min_vol/initial_investment - 1) * 100):.1f}%)
        """ + (f"- **{benchmark_selected_name}:** ${final_benchmark:,.2f} (+{((final_benchmark/initial_investment - 1) * 100):.1f}%)" if returns_benchmark is not None else ""))

        # ============================================================
        # EXPORT EXCEL 
        # ============================================================
        st.markdown("---")
        st.write("### Export Results")
        
        excel_file = create_excel_report(
            data, returns_df, correlation, sharpe_allocation, vol_allocation, 
            metrics_df, initial_investment, final_max_sharpe, final_min_vol, final_equal,
            start_date, end_date, benchmark_ticker if benchmark_ticker else 'None', risk_free_rate_pct
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            st.download_button(
                label="Download Excel Report",
                data=excel_file,
                file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.caption("Includes all analysis, allocations, and performance metrics")
        
        st.info("Next step: Further refinement or new features!")
                
else:
    st.info("Configure your portfolio and parameters in the sidebar and click 'Run Analysis'")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | Portfolio Optimizer v1.0</p>
</div>
""", unsafe_allow_html=True)