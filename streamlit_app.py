# ==========================================================
# Análise Didática B3 + ML — v12
# Mantém tudo da v11 e adiciona:
# - LSTM multivariado (opcional)
# - Tuning com Optuna (opcional)
# - Indicadores ADX, MACD, Bollinger
# - Exportação de relatório em PDF
# - ABA de Indicadores com explicações didáticas (NOVO nesta entrega)
# ==========================================================
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta

# Data & gráficos
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Utilitários B3 do projeto
# (certifique-se de que b3_utils.py está presente como nas versões anteriores)
from b3_utils import load_b3_tickers, ensure_sa_suffix, is_known_b3_ticker, search_b3

# ML tradicional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss, roc_curve
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# NeuralProphet (opcional)
try:
    from neuralprophet import NeuralProphet
    _NP_AVAILABLE = True
except Exception:
    _NP_AVAILABLE = False

# LSTM (opcional)
try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False

# Optuna (opcional)
try:
    import optuna
    _OPTUNA_AVAILABLE = True
except Exception:
    _OPTUNA_AVAILABLE = False

# PDF (opcional)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    _PDF_AVAILABLE = True
except Exception:
    _PDF_AVAILABLE = False

# ----------------------------------------------------------
# Page config + tema escuro padrão
# ----------------------------------------------------------
st.set_page_config(page_title="B3 + ML — v12", page_icon="✨", layout="wide")
pio.templates.default = "plotly_dark"
st.markdown("<style>body, .stApp {background-color: #0e1117; color: #e5e5e5;}</style>", unsafe_allow_html=True)
st.markdown("""
<style>
.chip { font-size: 0.85rem; display: inline-block; }
.kpi-caption { opacity: 0.75; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Helpers básicos de dados e indicadores + avançados
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    df = yf.download(ensure_sa_suffix(ticker), start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index()
    return df

def sma(s, w):
    return s.rolling(window=w, min_periods=w).mean()

def rsi(s, w=14):
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(w).mean()
    ma_down = down.rolling(w).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# --- Indicadores Avançados (MACD, Bollinger, ADX) ---
def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close, window=20, num_std=2):
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return ma, upper, lower

def adx(high, low, close, window=14):
    # Cálculo simples do ADX sem TA-Lib
    df = pd.DataFrame({"high": high, "low": low, "close": close}).copy()
    df["+DM"] = (df["high"] - df["high"].shift(1)).clip(lower=0)
    df["-DM"] = (df["low"].shift(1) - df["low"]).clip(lower=0)
    df.loc[df["+DM"] < df["-DM"], "+DM"] = 0
    df.loc[df["-DM"] < df["+DM"], "-DM"] = 0

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window).mean()
    plus_di = 100 * (df["+DM"].rolling(window).sum() / atr)
    minus_di = 100 * (df["-DM"].rolling(window).sum() / atr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.rolling(window).mean()
    return plus_di, minus_di, adx_val

def add_indicators(df, want_sma50=False, want_sma200=False, want_adv=False):
    if df.empty:
        return df
    df = df.copy()
    df["SMA20"] = sma(df["Close"], 20)
    if want_sma50:
        df["SMA50"] = sma(df["Close"], 50)
    if want_sma200:
        df["SMA200"] = sma(df["Close"], 200)
    df["RSI14"] = rsi(df["Close"])

    # Avançados (opcionais)
    if want_adv:
        macd_line, macd_signal, macd_hist = macd(df["Close"])
        bb_mid, bb_up, bb_low = bollinger(df["Close"])
        pdi, mdi, adx_val = adx(df["High"], df["Low"], df["Close"])
        df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd_line, macd_signal, macd_hist
        df["BB_MID"], df["BB_UP"], df["BB_LOW"] = bb_mid, bb_up, bb_low
        df["+DI"], df["-DI"], df["ADX"] = pdi, mdi, adx_val
    return df

def annotate_events(df):
    ev = pd.DataFrame(index=df.index)
    ev["far_below"] = (df["Close"]/df["SMA20"] - 1) <= -0.07
    ev["rsi_os"]    = df["RSI14"] <= 30
    return ev

# ----------------------------------------------------------
# Feature engineering para ML clássico
# ----------------------------------------------------------
def build_features(df, horizon=1):
    d = df.copy()
    d["ret_1"]  = d["Close"].pct_change(1)
    d["ret_3"]  = d["Close"].pct_change(3)
    d["ret_5"]  = d["Close"].pct_change(5)
    d["ret_10"] = d["Close"].pct_change(10)

    if "SMA50" not in d.columns:  d["SMA50"]  = sma(d["Close"],50)
    if "SMA200" not in d.columns: d["SMA200"] = sma(d["Close"],200)

    d["dist_sma20"]  = d["Close"]/d["SMA20"]  - 1
    d["dist_sma50"]  = d["Close"]/d["SMA50"]  - 1
    d["dist_sma200"] = d["Close"]/d["SMA200"] - 1
    d["vol_5"]  = d["Close"].pct_change().rolling(5).std()
    d["vol_10"] = d["Close"].pct_change().rolling(10).std()
    d["rsi"]    = d["RSI14"]

    d["future_ret"] = d["Close"].shift(-horizon)/d["Close"] - 1.0
    d["target_up"]  = (d["future_ret"] > 0).astype(int)

    feat_cols = ["ret_1","ret_3","ret_5","ret_10","dist_sma20","dist_sma50","dist_sma200","vol_5","vol_10","rsi"]
    d = d.dropna(subset=feat_cols + ["target_up","future_ret"]).reset_index(drop=True)

    X = d[feat_cols].values
    y = d["target_up"].values
    future_ret = d["future_ret"].values
    return d, X, y, future_ret, feat_cols

def fit_calibrated(model, X_train, y_train, frac_calib=0.2, method="sigmoid"):
    n = len(X_train)
    if n < 40:
        m = clone(model); m.fit(X_train, y_train); return m
    n_cal = max(int(n * frac_calib), 50) if n >= 100 else max(int(n * 0.1), 20)
    n_cal = min(n_cal, n-20) if n > 40 else max(5, n-5)
    m = clone(model); m.fit(X_train[:-n_cal], y_train[:-n_cal])
    cal = CalibratedClassifierCV(m, method=method, cv="prefit")
    cal.fit(X_train[-n_cal:], y_train[-n_cal:])
    return cal

def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    equity = (1 + pd.Series(returns)).cumprod()
    peak = equity.cummax()
    dd = equity/peak - 1.0
    return float(dd.min())

def best_threshold_by_return(proba, rets):
    if len(proba) != len(rets) or len(proba) == 0:
        return 0.5
    grid = np.linspace(0.4, 0.7, 61)
    best_thr, best_ret = 0.5, -1e9
    for thr in grid:
        sig = (proba >= thr).astype(int)
        cum = (1 + pd.Series(rets * sig)).prod() - 1
        if cum > best_ret:
            best_ret = float(cum); best_thr = float(thr)
    return best_thr

def best_threshold_by_sharpe(proba, rets):
    if len(proba) != len(rets) or len(proba) == 0:
        return 0.5
    grid = np.linspace(0.4, 0.7, 61)
    best_thr, best_s = 0.5, -1e9
    for thr in grid:
        sig = (proba >= thr).astype(int)
        strat = rets * sig
        mu, sigma = np.nanmean(strat), np.nanstd(strat) + 1e-12
        sharpe = mu / sigma
        if sharpe > best_s:
            best_s, best_thr = float(sharpe), float(thr)
    return best_thr

def best_threshold_by_calmar(proba, rets):
    if len(proba) != len(rets) or len(proba) == 0:
        return 0.5
    grid = np.linspace(0.4, 0.7, 61)
    best_thr, best_c = 0.5, -1e9
    for thr in grid:
        sig = (proba >= thr).astype(int)
        strat = rets * sig
        ret = (1 + pd.Series(strat)).prod() - 1
        dd = abs(max_drawdown(strat)) + 1e-12
        calmar = ret / dd if dd > 0 else -1e-9
        if calmar > best_c:
            best_c, best_thr = float(calmar), float(thr)
    return best_thr

def safe_tscv_params(n_samples, n_splits, test_size_min):
    max_splits = max(1, n_samples // max(1, test_size_min) - 1)
    adj_splits = min(n_splits, max_splits)
    adj_test = test_size_min
    while adj_splits < 2 and adj_test > 20:
        adj_test = max(20, adj_test // 2)
        max_splits = max(1, n_samples // max(1, adj_test) - 1)
        adj_splits = min(n_splits, max_splits)
    return adj_splits, adj_test

def tscv_with_embargo(n, n_splits, test_size, embargo):
    start = n - n_splits*test_size
    if start < 0: start = 0
    for s in range(start, n, test_size):
        test_start, test_end = s, min(s + test_size, n)
        train_end = max(0, test_start - embargo)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) > 0 and len(test_idx) > 0 and test_end - test_start >= max(5, embargo//2 + 1):
            yield train_idx, test_idx

def time_series_cv_ensemble(X, y, future_ret, n_splits=5, test_size_min=60, seed=42, thr_method="youden", embargo=0):
    n = len(X)
    if n < 80:
        return {"note": "Poucos dados para CV robusta (mín. ~80 amostras)."}, None, None, None, None, None
    n_splits_safe, test_size_safe = safe_tscv_params(n, n_splits, test_size_min)
    if n_splits_safe < 2:
        return {"note": f"Amostra insuficiente para dividir {n_splits}x com teste={test_size_min}. Reduza o período, o 'test_size' ou os 'splits'."}, None, None, None, None, None

    y_pred_proba = np.full(n, np.nan, dtype=float)
    thresholds = []
    last_models = None

    splitter = (
        tscv_with_embargo(n, n_splits_safe, test_size_safe, embargo)
        if embargo > 0 else
        TimeSeriesSplit(n_splits=n_splits_safe, test_size=test_size_safe).split(range(n))
    )

    for train_idx, test_idx in splitter:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        rets_te = future_ret[test_idx]

        hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=500, random_state=seed, early_stopping=True)
        xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=seed, tree_method="hist")
        lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1)

        use_lgb = (len(X_tr) >= 150)
        hgb_cal = fit_calibrated(hgb, X_tr, y_tr, method="sigmoid")
        xgb_cal  = fit_calibrated(xgb, X_tr, y_tr, method="sigmoid")
        models = [hgb_cal, xgb_cal]
        if use_lgb:
            lgb_cal = fit_calibrated(lgb, X_tr, y_tr, method="sigmoid")
            models.append(lgb_cal)

        probs = [m.predict_proba(X_te)[:,1] for m in models]
        proba = np.mean(probs, axis=0)

        if thr_method == "retorno":
            thr_fold = best_threshold_by_return(proba, rets_te)
        elif thr_method == "sharpe":
            thr_fold = best_threshold_by_sharpe(proba, rets_te)
        elif thr_method == "calmar":
            thr_fold = best_threshold_by_calmar(proba, rets_te)
        else:
            fpr, tpr, thr = roc_curve(y_te, proba); j = tpr - fpr
            thr_fold = thr[int(np.argmax(j))]

        thresholds.append(float(thr_fold))
        y_pred_proba[test_idx] = proba
        last_models = models

    mask = ~np.isnan(y_pred_proba)
    if mask.sum() == 0:
        return {"note": "Falha ao gerar previsões OOS."}, None, None, None, None, None

    y_true = y[mask]
    y_prob = y_pred_proba[mask]
    metrics = {
        "accuracy": float(accuracy_score(y_true, (y_prob>=0.5).astype(int))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, (y_prob>=0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n_oos": int(mask.sum()),
        "threshold_avg": float(np.nanmean(thresholds)) if thresholds else 0.5,
        "adj_splits": int(n_splits_safe),
        "adj_test_size": int(test_size_safe)
    }
    return metrics, y_prob, y_true, thresholds, last_models, mask

def compute_reliability_curve(y_true, y_prob, n_bins=10, strategy="quantile"):
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
        return prob_true, prob_pred
    except Exception:
        return None, None

def plot_reliability_plotly(prob_true, prob_pred, title="Curva de Confiabilidade (Calibration)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfeito", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Modelo"))
    fig.update_layout(title=title, xaxis_title="Probabilidade prevista", yaxis_title="Frequência observada")
    return fig

def make_trades_table_from_signals(dates, prices, signals, min_hold=1):
    if dates is None or prices is None or signals is None:
        return pd.DataFrame()
    s = pd.Series(signals).fillna(0).astype(int).values
    p = pd.Series(prices).astype(float).values
    d = pd.to_datetime(pd.Series(dates))
    entries, exits, ret, dur = [], [], [], []
    in_pos = False
    entry_idx = None
    for i in range(1, len(s)):
        if not in_pos and s[i-1] == 0 and s[i] == 1:
            in_pos = True; entry_idx = i
        elif in_pos and s[i-1] == 1 and s[i] == 0:
            if i - entry_idx >= min_hold:
                in_pos = False
                entries.append(d.iloc[entry_idx]); exits.append(d.iloc[i])
                r = (p[i] / p[entry_idx] - 1.0) * 100.0
                ret.append(r); dur.append(i - entry_idx)
    if in_pos and entry_idx is not None and len(p) - 1 - entry_idx >= min_hold:
        entries.append(d.iloc[entry_idx]); exits.append(d.iloc[-1])
        r = (p[-1] / p[entry_idx] - 1.0) * 100.0
        ret.append(r); dur.append(len(p) - 1 - entry_idx)

    df_trades = pd.DataFrame({"Entrada": entries, "Saída": exits, "Retorno (%)": ret, "Duração (dias)": dur})
    return df_trades.sort_values("Retorno (%)", ascending=False).reset_index(drop=True)

# ----------------------------------------------------------
# Sidebar: estado, presets, parâmetros
# ----------------------------------------------------------
b3 = load_b3_tickers()
st.sidebar.header("⚙️ Configurações")

defaults = dict(
    show_sma50=False, show_sma200=False, show_adv_ind=False,
    horizon=1, splits=5, test_size=60,
    thr_method_label="Sharpe OOS", min_prob=0.55, neutral_band=0.05,
    use_trend=True, allow_contrarian=True, contrarian_max_dist=-0.05,
    cost_bps=6, slip_bps=3, min_hold=2,
    # flags ML + storage
    use_ml=False, ml_trained=False, ml_proba_next=None, ml_metrics=None,
    ml_sig=None, ml_dates=None, ml_rets_oos=None, ml_cum_strat=None, ml_cum_bh=None,
    ml_dd_strat=None, ml_vol_strat=None,
)
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

q = st.sidebar.text_input("Buscar empresa ou ticker", "", key="search_q",
    help="Digite o código (ex.: PETR4) ou parte do nome (ex.: Petrobras).")
res = search_b3(q) if q else b3
ticker = st.sidebar.selectbox("Selecione o ticker", res["ticker"], key="ticker_select",
    help="Somente tickers da B3 (.SA).")

st.sidebar.markdown("---")
quick = st.sidebar.selectbox("Período rápido", ["Personalizado", "6M", "1A", "YTD"], index=2, key="quick_period")
today = date.today()
if quick == "6M":
    start_default = today - timedelta(days=182)
elif quick == "1A":
    start_default = today - timedelta(days=365)
elif quick == "YTD":
    start_default = date(today.year, 1, 1)
else:
    start_default = today - timedelta(days=365)
start = st.sidebar.date_input("Início", start_default, key="start_date")
end = st.sidebar.date_input("Fim", today, key="end_date")

st.sidebar.markdown("---")
st.sidebar.markdown("**Médias/Indicadores no gráfico:**")
st.sidebar.checkbox("Mostrar SMA50 (médio prazo)", value=st.session_state["show_sma50"], key="show_sma50")
st.sidebar.checkbox("Mostrar SMA200 (longo prazo)", value=st.session_state["show_sma200"], key="show_sma200")
st.sidebar.checkbox("Indicadores avançados (MACD, Bollinger, ADX)", value=st.session_state["show_adv_ind"], key="show_adv_ind",
                    help="Exibe MACD, Bandas de Bollinger e ADX em abas específicas.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Previsão (ML) — pesada**")
simple_mode = st.sidebar.checkbox("Modo simples (usar presets)", value=st.session_state.get("simple_mode", True), key="simple_mode")

if simple_mode:
    st.sidebar.markdown("**Presets de estratégia**")
    preset_choice = st.sidebar.selectbox("Escolha um preset", ["Conservador","Balanceado","Agressivo"], key="preset_choice")
    colp1, colp2 = st.sidebar.columns(2)
    if colp1.button("Aplicar preset", key="btn_apply_preset"):
        if preset_choice == "Conservador":
            st.session_state.update(dict(
                thr_method_label="Calmar OOS", min_prob=0.62, neutral_band=0.06,
                use_trend=True, allow_contrarian=False, contrarian_max_dist=-0.03,
                splits=5, test_size=60, cost_bps=8, slip_bps=5, min_hold=3
            ))
        elif preset_choice == "Balanceado":
            st.session_state.update(dict(
                thr_method_label="Sharpe OOS", min_prob=0.58, neutral_band=0.05,
                use_trend=True, allow_contrarian=True, contrarian_max_dist=-0.05,
                splits=5, test_size=60, cost_bps=6, slip_bps=3, min_hold=2
            ))
        else:
            st.session_state.update(dict(
                thr_method_label="Retorno OOS (backtest)", min_prob=0.54, neutral_band=0.03,
                use_trend=False, allow_contrarian=True, contrarian_max_dist=-0.08,
                splits=4, test_size=40, cost_bps=6, slip_bps=3, min_hold=1
            ))
        st.session_state["ml_trained"] = False
        st.rerun()
    if colp2.button("Aplicar + Treinar", key="btn_apply_and_train"):
        st.session_state["ml_trained"] = False
        st.session_state["train_now"] = True
else:
    st.session_state["train_now"] = False

st.sidebar.checkbox("Ativar previsão com ML", value=st.session_state["use_ml"], key="use_ml")
st.sidebar.markdown("*Básico*")
st.sidebar.selectbox("Horizonte (dias)", [1, 5, 10], key="horizon")
st.sidebar.selectbox("Método do limiar", ["Sharpe OOS","Calmar OOS","Retorno OOS (backtest)","Youden (acerto)"], key="thr_method_label")
st.sidebar.slider("Confiança mínima (min_prob)", 0.50, 0.75, st.session_state["min_prob"], 0.01, key="min_prob")

if not simple_mode:
    with st.sidebar.expander("Opções avançadas"):
        st.slider("Banda neutra (± p.p. em torno de 50%)", 0.00, 0.12, st.session_state["neutral_band"], 0.01, key="neutral_band")
        st.checkbox("Operar long apenas se Preço > SMA200", value=st.session_state["use_trend"], key="use_trend")
        st.checkbox("Permitir contrarian em sobrevenda (RSI≤30)", value=st.session_state["allow_contrarian"], key="allow_contrarian")
        st.slider("Distância máx. à SMA20 (contrarian) — negativo = abaixo", -0.20, 0.00, st.session_state["contrarian_max_dist"], 0.01, key="contrarian_max_dist")
        d1, d2 = st.columns(2)
        d1.slider("Nº de divisões (walk-forward CV)", 3, 8, st.session_state["splits"], 1, key="splits")
        d2.slider("Tamanho do bloco de teste (dias)", 20, 120, st.session_state["test_size"], 5, key="test_size")
        e1, e2, e3 = st.columns(3)
        e1.number_input("Custo por trade (bps)", 0, 50, st.session_state["cost_bps"], key="cost_bps")
        e2.number_input("Slippage (bps)", 0, 50, st.session_state["slip_bps"], key="slip_bps")
        e3.number_input("Dias mínimos em posição", 0, 10, st.session_state["min_hold"], key="min_hold")

# ----------------------------------------------------------
# Carregamento e resumo
# ----------------------------------------------------------
st.title("📊 Análise Didática de Ações da B3 — v12")
st.caption("Somente tickers da B3 (.SA) — dados do Yahoo Finance.")

if not is_known_b3_ticker(st.session_state["ticker_select"]):
    st.error("Ticker fora da lista da B3."); st.stop()

with st.spinner("Baixando dados..."):
    df = fetch_data(st.session_state["ticker_select"], st.session_state["start_date"], st.session_state["end_date"])
if df.empty:
    st.warning("Sem dados disponíveis para este período."); st.stop()

df = add_indicators(
    df,
    want_sma50=st.session_state["show_sma50"],
    want_sma200=st.session_state["show_sma200"],
    want_adv=st.session_state["show_adv_ind"]
)
events = annotate_events(df)

price = float(df["Close"].iloc[-1])
sma20 = float(df["SMA20"].iloc[-1])
rsi_val = float(df["RSI14"].iloc[-1])
delta20 = (price/sma20 - 1) * 100 if sma20 else np.nan

regime = None
if "SMA200" in df.columns and not df["SMA200"].isna().iloc[-1]:
    regime = "Acima da SMA200 (tendência longa positiva)" if price > df["SMA200"].iloc[-1] else "Abaixo da SMA200 (tendência longa negativa)"

def chip(txt, color):
    return f"<span class='chip' style='background:{color};padding:4px 8px;border-radius:999px;color:#111;font-weight:600;margin-right:6px;'>{txt}</span>"

chip_sma = ("Bem abaixo da média", "#FCA5A5") if delta20 <= -7 else \
           ("Abaixo da média", "#FDE68A")     if delta20 < -2  else \
           ("Perto da média", "#A7F3D0")      if delta20 <  2  else \
           ("Acima da média", "#93C5FD")      if delta20 <  7  else \
           ("Bem acima da média", "#C4B5FD")

chip_rsi = ("Sobrevenda (≤30)", "#86EFAC") if rsi_val <= 30 else \
           ("Neutro (30–70)", "#E5E7EB")    if rsi_val < 70  else \
           ("Sobrecompra (≥70)", "#FCA5A5")

st.markdown("### ⚡ Resumo em 10s", unsafe_allow_html=True)
cA, cB = st.columns([0.62, 0.38])
with cA:
    st.markdown(
        chip(chip_sma[0], chip_sma[1]) +
        chip(chip_rsi[0], chip_rsi[1]) +
        (chip("Longo prazo ↑", "#BBF7D0") if regime and "positiva" in regime else chip("Longo prazo ↓", "#FECACA") if regime else ""),
        unsafe_allow_html=True
    )
with cB:
    if st.session_state.get("ml_trained") and st.session_state.get("ml_proba_next") is not None:
        figg = go.Figure(go.Indicator(mode="gauge+number",
            value=st.session_state["ml_proba_next"]*100, number={'suffix': "%"},
            gauge={'axis': {'range': [0,100]}, 'bar': {'thickness': 0.25},
                   'threshold': {'line': {'color': "white",'width': 2}, 'value': 50}},
            domain={'x':[0,1],'y':[0,1]}, title={'text': f"Prob. de alta ({int(st.session_state['horizon'])}d)"}
        ))
        figg.update_layout(height=160, margin=dict(l=10,r=10,t=40,b=0))
        st.plotly_chart(figg, use_container_width=True)
    else:
        st.caption("Ative e treine o ML para ver a probabilidade de alta.")

k1, k2, k3 = st.columns(3)
k1.metric("Fechamento", f"R$ {price:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
k2.metric("Δ vs SMA20", f"{delta20:+.2f}%")
k3.metric("RSI(14)", f"{rsi_val:.1f}")
if regime:
    st.caption(regime)

# ----------------------------------------------------------
# Abas principais
# ----------------------------------------------------------
base_tabs = ["📈 Gráfico", "📚 Indicadores", "🤖 ML", "🧪 Backtest",
             "ℹ️ Glossário", "📊 Confiabilidade & Trades", "🧠 NeuralProphet"]
extra_tabs = ["🔮 LSTM", "🧭 Optuna", "🗂️ PDF"]

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(base_tabs + extra_tabs)

# ---- Tab 1: Gráfico
with tab1:
    def plot_price(df, show_sma50, show_sma200, show_adv):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Preço"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA20"))
        if show_sma50 and "SMA50" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50"))
        if show_sma200 and "SMA200" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA200"], name="SMA200"))
        # Eventos
        ext_idxs = df.index[events["far_below"]]
        if len(ext_idxs) > 0:
            fig.add_trace(go.Scatter(
                x=df.loc[ext_idxs,"Date"], y=df.loc[ext_idxs,"Close"],
                mode="markers", marker=dict(size=8, symbol="triangle-up"),
                name="Afastado da SMA20 (−7% ou mais)"
            ))
        # Sinais ML (se já treinado)
        if st.session_state.get("ml_trained") and st.session_state.get("ml_sig") is not None:
            sig = st.session_state["ml_sig"]
            dates_oos = st.session_state["ml_dates"]
            idxs = np.where(sig==1)[0]
            if len(idxs) > 0:
                close_on_oos = df.set_index("Date").reindex(dates_oos)["Close"].values
                fig.add_trace(go.Scatter(
                    x=dates_oos[idxs], y=close_on_oos[idxs],
                    mode="markers", marker=dict(size=9, symbol="star"),
                    name="Sinal ML (long)"
                ))
        fig.update_layout(title=f"{st.session_state['ticker_select']} — Preço e Médias", xaxis_title="Data", yaxis_title="Preço (R$)")
        st.plotly_chart(fig, use_container_width=True)

        # Se indicadores avançados estiverem ativos, mostrar subgráficos
        if show_adv:
            c1, c2 = st.columns(2)
            if {"MACD","MACD_SIGNAL","MACD_HIST"}.issubset(df.columns):
                with c1:
                    figm = go.Figure()
                    figm.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
                    figm.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], name="Signal"))
                    figm.add_trace(go.Bar(x=df["Date"], y=df["MACD_HIST"], name="Hist", opacity=0.5))
                    figm.update_layout(title="MACD", barmode="overlay")
                    st.plotly_chart(figm, use_container_width=True)
            if {"BB_MID","BB_UP","BB_LOW"}.issubset(df.columns):
                with c2:
                    figb = go.Figure()
                    figb.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
                    figb.add_trace(go.Scatter(x=df["Date"], y=df["BB_MID"], name="BB Mid", line=dict(dash="dot")))
                    figb.add_trace(go.Scatter(x=df["Date"], y=df["BB_UP"], name="BB Up", line=dict(dash="dash")))
                    figb.add_trace(go.Scatter(x=df["Date"], y=df["BB_LOW"], name="BB Low", line=dict(dash="dash")))
                    figb.update_layout(title="Bandas de Bollinger")
                    st.plotly_chart(figb, use_container_width=True)
            if {"+DI","-DI","ADX"}.issubset(df.columns):
                figd = go.Figure()
                figd.add_trace(go.Scatter(x=df["Date"], y=df["+DI"], name="+DI"))
                figd.add_trace(go.Scatter(x=df["Date"], y=df["-DI"], name="-DI"))
                figd.add_trace(go.Scatter(x=df["Date"], y=df["ADX"], name="ADX"))
                figd.update_layout(title="ADX (+DI / -DI / ADX)")
                st.plotly_chart(figd, use_container_width=True)

    def plot_rsi_tab(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI(14)"))
        fig.add_hline(y=70, line_dash="dash"); fig.add_hline(y=30, line_dash="dash")
        os_idxs = df.index[events["rsi_os"]]
        if len(os_idxs) > 0:
            fig.add_trace(go.Scatter(x=df.loc[os_idxs,"Date"], y=df.loc[os_idxs,"RSI14"],
                                     mode="markers", marker=dict(size=7, symbol="diamond"), name="RSI≤30"))
        fig.update_layout(title=f"{st.session_state['ticker_select']} — RSI(14)", xaxis_title="Data", yaxis_title="RSI")
        st.plotly_chart(fig, use_container_width=True)

    plot_price(df, st.session_state["show_sma50"], st.session_state["show_sma200"], st.session_state["show_adv_ind"])
    plot_rsi_tab(df)
    st.info("Dica: cole PETR4, VALE3, ITUB4... Se faltar .SA, o app adiciona automaticamente.")

# ---- Tab 2: Indicadores Técnicos (didático + explicações)
with tab2:
    st.markdown("### 💡 O que o gráfico está tentando te contar")
    st.markdown("#### 🪜 1. Entendendo a SMA20 — “a linha da média”")
    st.markdown(
        "A **SMA20** é como a média dos últimos 20 preços de fechamento — a **linha de equilíbrio** que mostra a direção geral do preço.\n\n"
        "• Se o **preço está acima** da linha, há **força** (tendência de alta).\n"
        "• Se **está abaixo**, há **fraqueza** (tendência de queda)."
    )
    st.markdown(f"👉 Em **{st.session_state['ticker_select']}**, o preço atual é **R$ {price:,.2f}**, cerca de **{delta20:+.2f}%** vs. SMA20.".replace(",", "X").replace(".", ",").replace("X","."))

    if delta20 <= -2:
        st.warning("A ação está **abaixo da média** — fraqueza de curto prazo.")
    elif delta20 < 2:
        st.info("O preço está **próximo da média** — mercado **em equilíbrio**.")
    else:
        st.success("O preço está **acima da média** — **força** de curto prazo.")

    st.caption("O preço pode ficar afastado da média; isso pode indicar **exagero** (como uma corda esticada).")

    st.markdown("#### ⚖️ 2. Entendendo o RSI(14) — “o termômetro da força”")
    st.markdown("O **RSI** (0–100) indica quem domina: compradores ou vendedores.")
    st.table(pd.DataFrame({
        "Faixa": ["70 a 100", "50", "0 a 30"],
        "Situação": ["Sobrecompra", "Neutro", "Sobrevenda"],
        "O que significa": ["Subiu rápido demais — pode corrigir pra baixo.", "Equilíbrio entre compra e venda.", "Caiu rápido demais — pode reagir pra cima."]
    }))
    st.markdown(f"Para **{st.session_state['ticker_select']}**, o **RSI(14)** está em **{rsi_val:.1f}**.")
    if rsi_val >= 70:
        st.warning("**Sobrecompra** — pode corrigir.")
    elif rsi_val <= 30:
        st.success("**Sobrevenda** — pode reagir.")
    else:
        st.info("**Neutro** — equilíbrio.")

    st.markdown("#### 🧩 3. Juntando as duas informações")
    if (delta20 <= -2) and (rsi_val <= 35):
        st.info("“Caiu bastante e **pode dar um respiro** em breve.” (pressão de venda diminuindo)")
    elif (delta20 >= 2) and (rsi_val >= 65):
        st.warning("“Subiu bastante e **pode descansar**.” (compra esticada)")
    else:
        st.info("“Quadro **equilibrado** — sem sinal forte de excesso.”")

    st.markdown("#### 💬 Em resumo")
    resumo_rows = []
    resumo_rows.append(["SMA20", "Preço vs. média 20d",
        "Bem abaixo — pressionada." if delta20 <= -7 else
        "Abaixo — tendência fraca." if delta20 < -2 else
        "Perto — equilíbrio." if delta20 < 2 else
        "Acima — força." if delta20 < 7 else
        "Bem acima — atenção a exageros."
    ])
    resumo_rows.append(["RSI(14)", "Energia do mercado (0–100)",
        "Sobrevenda (≤30) — pode reagir." if rsi_val <= 30 else
        "Neutro (30–70) — equilíbrio." if rsi_val < 70 else
        "Sobrecompra (≥70) — pode corrigir."
    ])
    resumo_rows.append(["Conclusão geral", "Preço + RSI",
        "Fraca, mas pode haver repique." if (delta20 <= -2 and rsi_val <= 35) else
        "Forte, atenção a correções." if (delta20 >= 2 and rsi_val >= 65) else
        "Equilíbrio — sem sinal forte."
    ])
    st.table(pd.DataFrame(resumo_rows, columns=["Indicador","O que está mostrando","Significado prático"]))

    with st.expander("🕯️ Como ler candles (clique para ver)"):
        st.markdown("""
- **Candle** mostra Abertura, Máxima, Mínima e Fechamento do período.
- Corpo cheio: fechou **acima** da abertura (alta). Corpo vazio/escuro: **abaixo** (baixa).
- **Pavios** indicam onde o preço foi mas **não ficou**.
- Sequências fortes indicam **impulso**; sombras longas sugerem **reversões**.
""")

    st.subheader("📘 Indicadores Técnicos (inclui extras)")
    st.caption("Nesta aba você encontra explicações simples sobre cada indicador usado no gráfico.")

    st.markdown("""
### 📈 SMA20 / SMA50 / SMA200
Médias móveis de **curto (20)**, **médio (50)** e **longo prazo (200)**.  
Mostram a **tendência geral** dos preços — se o preço está **acima**, o ativo tende a estar forte; se está **abaixo**, tende a estar fraco.

### ⚖️ RSI(14)
O **Índice de Força Relativa** funciona como um **termômetro de força**.  
- **Acima de 70** → sobrecompra (ativo pode estar caro).  
- **Abaixo de 30** → sobrevenda (ativo pode estar barato).  
Ajuda a detectar **exageros** no curto prazo.

### 📊 MACD (Moving Average Convergence Divergence)
Mede o **momentum** do preço.  
É composto por:
- **Linha MACD** (diferença entre médias rápidas e lentas),
- **Linha de sinal** (média da MACD),
- **Histograma** (MACD − sinal).  
Quando a linha MACD cruza **acima da linha de sinal**, pode indicar **início de alta**.  
Quando cruza **para baixo**, pode sinalizar **fraqueza**.

### 💡 Bandas de Bollinger
Mostram o **nível de volatilidade** do ativo.  
- **Bandas se expandem** → volatilidade alta.  
- **Bandas se contraem** → mercado mais calmo.  
Quando o preço toca as bordas, pode indicar **exageros temporários**.

### 🧭 ADX (+DI / −DI)
Mede a **força da tendência** e sua **direção**:  
- **ADX > 25** → tendência forte.  
- **+DI** representa pressão **compradora**.  
- **−DI** representa pressão **vendedora**.

---

💬 **Dica:** Combine indicadores para interpretações mais robustas — por exemplo:  
📉 RSI < 30 + preço abaixo da SMA20 → possível sinal de reversão (ação “barata” demais).  
📈 RSI > 70 + preço acima da SMA200 → possível sinal de exaustão (ação “esticada” demais).
""")


# ---- Tab 3: ML (treino/validação, KPIs, callouts)
with tab3:
    st.subheader("Previsão (ML)")
    st.caption(f"Horizonte: **{int(st.session_state['horizon'])}d** • Limiar: **{st.session_state['thr_method_label']}** • min_prob: **{st.session_state['min_prob']:.2f}**")
    if st.session_state.get("train_now", False) or st.button("Treinar/Atualizar modelo", type="primary"):
        with st.spinner("Treinando e validando (walk-forward)..."):
            d, X, y, future_ret, feat_cols = build_features(df, horizon=int(st.session_state["horizon"]))
            finite_rows = np.isfinite(X).all(axis=1)
            d, X, y, future_ret = d.loc[finite_rows].reset_index(drop=True), X[finite_rows], y[finite_rows], future_ret[finite_rows]
            if len(X) < 80:
                st.warning("Poucos dados úteis. Aumente o período, reduza o horizonte ou ajuste test_size/splits.")
            else:
                thr_lbl = st.session_state["thr_method_label"]
                key = "retorno" if thr_lbl.startswith("Retorno") else ("sharpe" if thr_lbl.startswith("Sharpe") else ("calmar" if thr_lbl.startswith("Calmar") else "youden"))
                embargo = int(st.session_state["horizon"])
                metrics, y_prob, y_true, thresholds, last_models, oos_mask = time_series_cv_ensemble(
                    X, y, future_ret, n_splits=int(st.session_state["splits"]),
                    test_size_min=int(st.session_state["test_size"]), thr_method=key, embargo=embargo
                )
                if isinstance(metrics, dict) and "note" in metrics and y_prob is None:
                    st.warning(metrics["note"] + " — Tente período maior, test_size menor ou menos splits.")
                else:
                    proba_next = None
                    if last_models is not None and len(d) > 0:
                        x_next = d[feat_cols].values[-1:].copy()
                        proba_next = float(np.mean([m.predict_proba(x_next)[:,1] for m in last_models]))
                    thr_avg = metrics["threshold_avg"]
                    prob_oos = y_prob
                    rets_oos = future_ret[oos_mask]
                    dates_oos = d.loc[oos_mask, "Date"].values
                    px_oos    = d.loc[oos_mask, "Close"].values
                    sma200_oos= d.loc[oos_mask, "SMA200"].values
                    rsi_oos   = d.loc[oos_mask, "rsi"].values
                    dist20_oos= d.loc[oos_mask, "dist_sma20"].values

                    sig = (prob_oos >= thr_avg).astype(int)
                    low_b, high_b = 0.5 - st.session_state["neutral_band"], 0.5 + st.session_state["neutral_band"]
                    neutral = (prob_oos >= low_b) & (prob_oos <= high_b)
                    sig[neutral] = 0
                    sig[prob_oos < st.session_state["min_prob"]] = 0
                    if st.session_state["use_trend"]:
                        above_trend = np.isfinite(sma200_oos) & (px_oos > sma200_oos)
                        contrarian = (rsi_oos < 30) & (dist20_oos <= st.session_state["contrarian_max_dist"])
                        allow = above_trend | (st.session_state["allow_contrarian"] & contrarian)
                        sig = sig * allow.astype(int)
                    if st.session_state["min_hold"] > 0 and len(sig) > 0:
                        locked = 0
                        for i in range(len(sig)):
                            if sig[i] == 1 and locked == 0: locked = int(st.session_state["min_hold"])
                            if locked > 0: sig[i] = 1; locked -= 1
                    changes = np.diff(np.concatenate([[0], sig.astype(int)]))
                    per_side_cost = (st.session_state["cost_bps"] + st.session_state["slip_bps"]) / 10000.0
                    txn_costs = np.zeros_like(prob_oos, dtype=float)
                    for t, ch in enumerate(changes):
                        if ch != 0: txn_costs[t] -= per_side_cost
                    strat = rets_oos * sig + txn_costs
                    cum_strat = (1 + pd.Series(strat)).cumprod() - 1
                    cum_bh    = (1 + pd.Series(rets_oos)).cumprod() - 1
                    dd_strat  = max_drawdown(strat)
                    vol_strat = float(np.nanstd(strat))
                    st.session_state.update(dict(
                        ml_trained=True, ml_proba_next=proba_next, ml_metrics=metrics, ml_sig=sig,
                        ml_dates=dates_oos, ml_rets_oos=rets_oos, ml_cum_strat=cum_strat.values, ml_cum_bh=cum_bh.values,
                        ml_dd_strat=dd_strat, ml_vol_strat=vol_strat,
                        ml_oos_y_true=y_true, ml_oos_y_prob=y_prob, oos_prices=px_oos, oos_dates=dates_oos,
                    ))
    if st.session_state.get("ml_trained") and st.session_state.get("ml_metrics"):
        m = st.session_state["ml_metrics"]
        colA, colB, colC, colD, colE = st.columns(5)
        colA.metric("Acurácia (OOS)", f"{m['accuracy']*100:.1f}%")
        colB.metric("Balanced Acc.", f"{m['balanced_accuracy']*100:.1f}%")
        colC.metric("ROC AUC", f"{m['roc_auc']:.3f}")
        colD.metric("Brier (↓ melhor)", f"{m['brier']:.3f}")
        colE.metric("OOS", f"{m['n_oos']}")
        st.caption(f"CV: splits={m['adj_splits']} • test_size={m['adj_test_size']} • Limiar: {st.session_state['thr_method_label']} • Embargo: {int(st.session_state['horizon'])}d")
        proba_next = st.session_state.get("ml_proba_next", None)
        if proba_next is not None:
            st.metric(f"Prob. de alta em {int(st.session_state['horizon'])} dia(s)", f"{proba_next*100:.1f}%")
        auc = m['roc_auc']; brier = m['brier']
        msg_auc = "vantagem forte" if auc >= 0.65 else "vantagem moderada" if auc >= 0.60 else "vantagem pequena" if auc >= 0.53 else "sinal fraco (≈ acaso)"
        msg_brier = "probabilidades bem calibradas" if brier < 0.23 else "probabilidades razoáveis" if brier < 0.26 else "probabilidades pouco informativas"
        if auc < 0.55:
            st.warning(f"Sinal **fraco** ({msg_auc}; {msg_brier}). Considere **aumentar min_prob**, usar **Sharpe/Calmar**, ativar **tendência** e **holding**.")
        else:
            st.success(f"Sinal com **{msg_auc}** ({msg_brier}). Ajuste **min_prob** e **banda neutra** para dosar seletividade vs. nº de trades.")

# ---- Tab 4: Backtest
with tab4:
    if not (st.session_state.get("ml_trained") and st.session_state.get("ml_cum_strat") is not None):
        st.info("Treine o modelo na aba **ML** para ver o backtest.")
    else:
        perf_df = pd.DataFrame({
            "Data": st.session_state["ml_dates"],
            "Estratégia (long nos sinais)": st.session_state["ml_cum_strat"],
            "Buy & Hold (OOS)": st.session_state["ml_cum_bh"],
        }).melt("Data", var_name="Série", value_name="Retorno Acumulado")
        figp = px.line(perf_df, x="Data", y="Retorno Acumulado", color="Série", title="Backtest — Retorno Acumulado (fora da amostra)")
        st.plotly_chart(figp, use_container_width=True)
        dd = st.session_state["ml_dd_strat"]; vol = st.session_state["ml_vol_strat"]
        c1, c2 = st.columns(2)
        c1.metric("Máx. drawdown (estratégia)", f"{dd*100:.1f}%")
        c2.metric("Vol (por passo)", f"{vol*100:.2f}%")
        st.caption("Inclui custos por lado e holding.")

# ---- Tab 5: Glossário
with tab5:
    st.markdown("### 📚 Glossário rápido")
    st.markdown("""
- **Candle**: Abertura, Máxima, Mínima, Fechamento.
- **SMA**: média de fechamentos; tendência.
- **RSI**: velocidade das altas/quedas (0–100).
- **Sobrecompra/Sobrevenda**: ≥70 / ≤30.
- **Embargo**: separa treino e teste.
- **Sharpe/Calmar**: retorno ajustado a risco/drawdown.
- **Holding**: dias mínimos na posição.
- **bps**: basis points; 10 bps = 0,10%.
""")

# ---- Tab 6: Confiabilidade & Trades
with tab6:
    st.subheader("📊 Confiabilidade & Trades")
    st.caption("Valide probabilidades (calibração) e veja as operações OOS.")
    y_true = st.session_state.get("ml_oos_y_true")
    y_prob = st.session_state.get("ml_oos_y_prob")
    st.markdown("### Curva de Confiabilidade (Calibration)")
    if y_true is None or y_prob is None or len(y_true) < 20 or len(y_true) != len(y_prob):
        st.info("Treine o modelo na aba **ML** para gerar probabilidades OOS.")
    else:
        pt, pp = compute_reliability_curve(np.array(y_true), np.array(y_prob), n_bins=10)
        if pt is None:
            st.warning("Não foi possível calcular a curva.")
        else:
            fig = plot_reliability_plotly(pt, pp)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Linha pontilhada = ideal; Linha do modelo = comportamento real das probabilidades.")
    st.markdown("---")
    st.markdown("### Tabela de Trades (OOS)")
    dates_oos = st.session_state.get("oos_dates")
    prices_oos = st.session_state.get("oos_prices")
    signals_oos = st.session_state.get("ml_sig")
    min_hold = int(st.session_state.get("min_hold", 1))
    trades_df = make_trades_table_from_signals(dates_oos, prices_oos, signals_oos, min_hold=min_hold)
    if trades_df is None or trades_df.empty:
        st.info("Sem trades OOS. Ajuste período/limiar e treine.")
    else:
        st.dataframe(trades_df.head(50), use_container_width=True)
        colA, colB = st.columns(2)
        with colA: st.metric("Trades", len(trades_df))
        with colB: st.metric("Retorno médio/trade", f"{trades_df['Retorno (%)'].mean():.2f}%")
        csv = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar trades (CSV)", data=csv, file_name="oos_trades.csv", mime="text/csv")

# ---- Tab 7: NeuralProphet (histórico + futuro + métricas e scatter)
with tab7:
    st.subheader("🧠 NeuralProphet — previsão de tendência")
    st.caption("Previsões históricas (in-sample) e futuras (out-of-sample) com métricas R²/MAE/MAPE.")
    np_h = st.number_input("Dias para prever (futuro)", min_value=1, max_value=365, value=30, step=1)
    if not _NP_AVAILABLE:
        st.warning("NeuralProphet não instalado. Adicione `neuralprophet` ao requirements.txt.")
        st.code("neuralprophet>=0.6.0", language="bash")
    else:
        try:
            from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
            price_df = df.copy()
            if "Date" in price_df.columns: price_df = price_df.rename(columns={"Date": "ds"})
            if "Close" in price_df.columns: price_df = price_df.rename(columns={"Close": "y"})
            if "Adj Close" in price_df.columns and "y" not in price_df.columns: price_df = price_df.rename(columns={"Adj Close": "y"})
            np_df = price_df[["ds", "y"]].dropna().copy(); np_df["ds"] = pd.to_datetime(np_df["ds"])
            last_hist_date = np_df["ds"].max()

            m = NeuralProphet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            _ = m.fit(np_df, freq="B")
            df_future = m.make_future_dataframe(np_df, periods=int(np_h), n_historic_predictions=True)
            forecast = m.predict(df_future)
            fc = forecast.copy(); fc["ds"] = pd.to_datetime(fc["ds"])
            hist_pred = fc[fc["ds"] <= last_hist_date]; fut_pred = fc[fc["ds"] > last_hist_date]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np_df["ds"], y=np_df["y"], mode="lines", name="Fechamento (real)", line=dict(width=2)))
            if not hist_pred.empty:
                fig.add_trace(go.Scatter(x=hist_pred["ds"], y=hist_pred["yhat1"], mode="lines", name="Previsão (histórico)", line=dict(width=2, dash="dot", color="#8ecae6")))
            if not fut_pred.empty:
                fig.add_trace(go.Scatter(x=fut_pred["ds"], y=fut_pred["yhat1"], mode="lines", name="Previsão (futuro)", line=dict(width=3, color="#ffd166")))
            fig.add_vline(x=last_hist_date, line=dict(color="#aaaaaa", width=1, dash="dash"))
            fig.update_layout(title="NeuralProphet — Histórico vs Futuro", xaxis_title="Data", yaxis_title="Preço")
            st.plotly_chart(fig, use_container_width=True)

            hist_merge = pd.merge(np_df[["ds","y"]], hist_pred[["ds","yhat1"]], on="ds", how="inner").dropna()
            if not hist_merge.empty:
                r2_level = r2_score(hist_merge["y"], hist_merge["yhat1"])
                mae = mean_absolute_error(hist_merge["y"], hist_merge["yhat1"])
                mape = mean_absolute_percentage_error(hist_merge["y"], hist_merge["yhat1"])
                ret_df = hist_merge[["y","yhat1"]].pct_change().dropna()
                r2_return = r2_score(ret_df["y"], ret_df["yhat1"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R² (nível)", f"{r2_level:.3f}")
                c2.metric("R² (retornos)", f"{r2_return:.3f}")
                c3.metric("MAE", f"{mae:.4f}")
                c4.metric("MAPE", f"{mape*100:.2f}%")
            st.markdown("### 🔍 Dispersão (real vs previsto)")
            col1, col2 = st.columns(2)
            if not hist_merge.empty:
                with col1:
                    fig_scatter1 = px.scatter(hist_merge, x="y", y="yhat1", title="Preço real vs Previsto (nível)",
                                              labels={"y": "Preço real", "yhat1": "Previsto"})
                    fig_scatter1.add_trace(go.Scatter(x=[hist_merge["y"].min(), hist_merge["y"].max()],
                                                      y=[hist_merge["y"].min(), hist_merge["y"].max()],
                                                      mode="lines", line=dict(color="red", dash="dash"), name="Ideal"))
                    st.plotly_chart(fig_scatter1, use_container_width=True)
            if 'ret_df' in locals() and not ret_df.empty:
                with col2:
                    fig_scatter2 = px.scatter(ret_df, x="y", y="yhat1", title="Retornos reais vs previstos",
                                              labels={"y": "Retorno real", "yhat1": "Previsto"})
                    fig_scatter2.add_trace(go.Scatter(x=[ret_df["y"].min(), ret_df["y"].max()],
                                                      y=[ret_df["y"].min(), ret_df["y"].max()],
                                                      mode="lines", line=dict(color="red", dash="dash"), name="Ideal"))
                    st.plotly_chart(fig_scatter2, use_container_width=True)

            last_close = float(np_df.iloc[-1]["y"])
            last_forecast = float(fut_pred.tail(1)["yhat1"].values[0]) if not fut_pred.empty else float(fc.tail(1)["yhat1"].values[0])
            pct = (last_forecast/last_close - 1.0)*100.0
            st.metric("Tendência prevista (fim do horizonte)", "alta" if pct>=0 else "baixa", f"{pct:.2f}%")
        except Exception as e:
            st.error(f"Falha ao rodar NeuralProphet: {e}")

# ---- Tab 8: LSTM multivariado (experimental)
with tab8:
    st.subheader("🔮 LSTM Multivariado (experimental)")
    st.caption("Prevê retorno futuro usando janelas temporais e múltiplas variáveis (Close, RSI, distâncias às médias etc.).")

    if not _TF_AVAILABLE:
        st.warning("TensorFlow não instalado. Adicione ao requirements.txt:")
        st.code("tensorflow-cpu>=2.12", language="bash")
    else:
        lookback = st.slider("Lookback (dias por janela)", 10, 120, 30, 5,
                             help="Quantos dias o modelo olha para trás em cada amostra.")
        horizon_lstm = st.selectbox("Horizonte de previsão (dias à frente)", [1,5,10], index=0)
        epochs = st.slider("Épocas (treino)", 5, 100, 30, 5)
        units = st.slider("Unidades LSTM", 8, 128, 32, 8)
        dropout = st.slider("Dropout", 0.0, 0.6, 0.2, 0.05)
        do_train = st.button("Treinar LSTM")

        def make_lstm_dataset(dfin, lookback=30, horizon=1):
            d = dfin.copy()
            d["ret_1"]  = d["Close"].pct_change(1)
            d["ret_3"]  = d["Close"].pct_change(3)
            d["ret_5"]  = d["Close"].pct_change(5)
            d["dist20"] = d["Close"]/d["SMA20"] - 1
            if "SMA200" not in d.columns: d["SMA200"] = sma(d["Close"], 200)
            d["dist200"] = d["Close"]/d["SMA200"] - 1
            d["rsi"] = d["RSI14"]
            feats = ["ret_1","ret_3","ret_5","dist20","dist200","rsi"]
            d = d.dropna(subset=feats).reset_index(drop=True)
            d["y_target"] = d["Close"].shift(-horizon)/d["Close"] - 1.0
            d = d.dropna(subset=["y_target"]).reset_index(drop=True)

            Xseq, y = [], []
            for i in range(len(d) - lookback):
                Xseq.append(d.loc[i:i+lookback-1, feats].values)
                y.append(d.loc[i+lookback, "y_target"])
            Xseq, y = np.array(Xseq, dtype=np.float32), np.array(y, dtype=np.float32)
            dates = d.loc[lookback:, "Date"].values if "Date" in d.columns else None
            last_feats = d.loc[len(d)-lookback:, feats].values
            return Xseq, y, dates, last_feats, feats

        if do_train:
            with st.spinner("Treinando LSTM..."):
                Xseq, yseq, dates_seq, last_feats, feat_names = make_lstm_dataset(df, lookback=lookback, horizon=int(horizon_lstm))
                n = len(Xseq)
                if n < 200:
                    st.warning("Poucos dados após preparação. Aumente período ou diminua lookback/horizonte.")
                else:
                    split = int(n*0.8)
                    Xtr, Xte = Xseq[:split], Xseq[split:]
                    ytr, yte = yseq[:split], yseq[split:]

                    model = keras.Sequential([
                        keras.layers.Input(shape=(lookback, Xseq.shape[-1])),
                        keras.layers.LSTM(units, return_sequences=False),
                        keras.layers.Dropout(dropout),
                        keras.layers.Dense(1)  # retorno futuro
                    ])
                    model.compile(optimizer="adam", loss="mae")
                    es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                    _ = model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs, batch_size=32, callbacks=[es], verbose=0)

                    # Métricas simples no conjunto de teste
                    ypred = model.predict(Xte, verbose=0).ravel()
                    mae = float(np.mean(np.abs(ypred - yte)))
                    mape = float(np.mean(np.abs((ypred - yte) / (np.abs(yte)+1e-9))))
                    ss_res = float(np.sum((yte - ypred)**2))
                    ss_tot = float(np.sum((yte - np.mean(yte))**2) + 1e-12)
                    r2_ret = 1.0 - ss_res/ss_tot

                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE (retorno)", f"{mae*100:.2f} p.p.")
                    c2.metric("MAPE (retorno)", f"{mape*100:.2f}%")
                    c3.metric("R² (retornos)", f"{r2_ret:.3f}")

                    # Previsão out-of-sample (próximo ponto)
                    x_next = np.array([Xseq[-1]], dtype=np.float32)
                    y_next = float(model.predict(x_next, verbose=0).ravel()[0])
                    st.metric(f"Retorno previsto (próximos {int(horizon_lstm)}d)", f"{y_next*100:.2f}%")
                    st.caption("Sinal: positivo sugere alta; negativo sugere baixa no horizonte escolhido.")

# ---- Tab 9: Optuna (tuning simples do LSTM)
with tab9:
    st.subheader("🧭 Optuna — Tuning de Hiperparâmetros (LSTM)")
    st.caption("Busca automática por unidades/lookup/dropout/epochs. Use com parcimônia (pode ser pesado).")

    if not _OPTUNA_AVAILABLE:
        st.warning("Optuna não instalado. Adicione ao requirements.txt:")
        st.code("optuna>=3.0.0", language="bash")
    elif not _TF_AVAILABLE:
        st.warning("TensorFlow não instalado. Necessário para treinar o LSTM.")
    else:
        n_trials = st.slider("Nº de estudos (trials)", 5, 40, 10, 1)
        max_epochs = st.slider("Épocas máx. por trial", 10, 80, 30, 5)
        horizon_opt = st.selectbox("Horizonte alvo", [1,5,10], index=0)
        go_opt = st.button("Rodar estudo (experimental)")

        def prepare_for_optuna(df_, horizon=1):
            d = df_.copy()
            d["ret_1"]  = d["Close"].pct_change(1)
            d["ret_3"]  = d["Close"].pct_change(3)
            d["ret_5"]  = d["Close"].pct_change(5)
            d["dist20"] = d["Close"]/d["SMA20"] - 1
            if "SMA200" not in d.columns: d["SMA200"] = sma(d["Close"],200)
            d["dist200"] = d["Close"]/d["SMA200"] - 1
            d["rsi"] = d["RSI14"]
            feats = ["ret_1","ret_3","ret_5","dist20","dist200","rsi"]
            d = d.dropna(subset=feats).reset_index(drop=True)
            d["y_target"] = d["Close"].shift(-horizon)/d["Close"] - 1.0
            d = d.dropna(subset=["y_target"]).reset_index(drop=True)
            return d, feats

        if go_opt:
            data_for_opt, feats = prepare_for_optuna(df, horizon=int(horizon_opt))
            if len(data_for_opt) < 300:
                st.warning("Dados insuficientes para estudo. Amplie o período.")
            else:
                def objective(trial):
                    lookback = trial.suggest_int("lookback", 15, 80, step=5)
                    units = trial.suggest_int("units", 16, 96, step=16)
                    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
                    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
                    epochs = max_epochs

                    # monta janelas
                    d = data_for_opt.copy()
                    # janelas com lookback do trial
                    Xseq, yseq = [], []
                    for i in range(len(d) - lookback):
                        Xseq.append(d.loc[i:i+lookback-1, feats].values)
                        yseq.append(d.loc[i+lookback, "y_target"])
                    Xseq = np.array(Xseq, dtype=np.float32); yseq = np.array(yseq, dtype=np.float32)
                    n = len(Xseq)
                    split = int(n*0.8); Xtr, Xte = Xseq[:split], Xseq[split:]; ytr, yte = yseq[:split], yseq[split:]

                    model = keras.Sequential([
                        keras.layers.Input(shape=(lookback, Xseq.shape[-1])),
                        keras.layers.LSTM(units),
                        keras.layers.Dropout(dropout),
                        keras.layers.Dense(1)
                    ])
                    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mae")
                    es = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
                    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs, batch_size=32, verbose=0, callbacks=[es])
                    ypred = model.predict(Xte, verbose=0).ravel()
                    mae = float(np.mean(np.abs(ypred - yte)))
                    return mae

                with st.spinner("Rodando Optuna..."):
                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=n_trials)

                st.success(f"Melhor trial: {study.best_trial.number} — MAE={study.best_value:.6f}")
                st.json(study.best_trial.params)

# ---- Tab 10: Exportação PDF (corrigido sem 'nonlocal')
with tab10:
    st.subheader("🗂️ Exportar relatório em PDF")
    st.caption("Gera um sumário didático (preço, SMA/RSI, KPIs de ML, configurações e observações).")

    if not _PDF_AVAILABLE:
        st.warning("ReportLab não instalado. Adicione ao requirements.txt:")
        st.code("reportlab>=3.6.12", language="bash")
    else:
        title = st.text_input("Título do relatório", value=f"Relatório — {st.session_state['ticker_select']}")
        obs = st.text_area("Observações (opcional)", placeholder="Anote insights, restrições, alertas, etc.")

        if st.button("Gerar PDF"):
            file_path = f"relatorio_{st.session_state['ticker_select']}.pdf"
            c = canvas.Canvas(file_path, pagesize=A4)
            W, H = A4

            # Helper que retorna o novo y (sem nonlocal)
            def draw_line(ca, text, y_pos, dy=0.8*cm, size=11, bold=False):
                if bold:
                    ca.setFont("Helvetica-Bold", size)
                else:
                    ca.setFont("Helvetica", size)
                ca.drawString(2*cm, y_pos, text)
                return y_pos - dy

            y_pos = H - 2*cm  # posição inicial

            # Cabeçalho
            y_pos = draw_line(c, title, y_pos, size=14, bold=True, dy=1.0*cm)
            y_pos = draw_line(c, f"Ticker: {st.session_state['ticker_select']}", y_pos)
            y_pos = draw_line(c, f"Período: {st.session_state['start_date']} a {st.session_state['end_date']}", y_pos)
            y_pos = draw_line(c, f"Fechamento: R$ {price:,.2f}".replace(",", "X").replace(".", ",").replace("X","."), y_pos)
            y_pos = draw_line(c, f"Δ vs SMA20: {delta20:+.2f}% — RSI(14): {rsi_val:.1f}", y_pos)
            if regime:
                y_pos = draw_line(c, f"Regime: {regime}", y_pos)

            # Config de ML
            y_pos = draw_line(c, "—", y_pos, dy=0.5*cm)
            y_pos = draw_line(c, "Configurações de ML", y_pos, bold=True)
            y_pos = draw_line(c, f"Horizonte: {int(st.session_state['horizon'])}d • Limiar: {st.session_state['thr_method_label']} • min_prob: {st.session_state['min_prob']:.2f}", y_pos)
            y_pos = draw_line(c, f"Splits/Test_size: {int(st.session_state['splits'])}/{int(st.session_state['test_size'])} • Banda neutra: ±{st.session_state['neutral_band']:.2f}", y_pos)
            y_pos = draw_line(c, f"Tendência(SMA200): {'ON' if st.session_state['use_trend'] else 'OFF'} • Contrarian(RSI<=30): {'ON' if st.session_state['allow_contrarian'] else 'OFF'}", y_pos)
            y_pos = draw_line(c, f"Custo/Slippage (bps): {int(st.session_state['cost_bps'])}/{int(st.session_state['slip_bps'])} • Holding mínimo: {int(st.session_state['min_hold'])}d", y_pos)

            # KPIs (se houver)
            if st.session_state.get("ml_trained") and st.session_state.get("ml_metrics"):
                m = st.session_state["ml_metrics"]
                y_pos = draw_line(c, "—", y_pos, dy=0.5*cm)
                y_pos = draw_line(c, "KPIs (fora da amostra)", y_pos, bold=True)
                y_pos = draw_line(c, f"Acurácia: {m['accuracy']*100:.1f}% — Balanced Acc: {m['balanced_accuracy']*100:.1f}%", y_pos)
                y_pos = draw_line(c, f"ROC AUC: {m['roc_auc']:.3f} — Brier: {m['brier']:.3f} — OOS: {m['n_oos']}", y_pos)

            # Observações
            if obs:
                y_pos = draw_line(c, "—", y_pos, dy=0.5*cm)
                y_pos = draw_line(c, "Observações", y_pos, bold=True)
                for line in obs.splitlines():
                    y_pos = draw_line(c, line, y_pos, dy=0.6*cm, size=10)

            c.showPage()
            c.save()
            with open(file_path, "rb") as f:
                st.download_button("Baixar PDF", data=f.read(), file_name=file_path, mime="application/pdf")
            st.success("PDF gerado.")
