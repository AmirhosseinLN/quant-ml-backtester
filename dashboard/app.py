from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# =========================================================
# Page Setup & Theme Constants
# =========================================================
st.set_page_config(
    page_title="ML Strategy Research | OOS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Theme
# =========================================================
BG = "#05070A"
SURFACE = "rgba(13, 17, 23, 0.82)"
ACCENT = "#00D1FF"
ACCENT_GLOW = "rgba(0, 209, 255, 0.10)"
BORDER = "rgba(48, 54, 61, 0.60)"
TEXT = "#E6EDF3"
TEXT_MUTED = "#8B949E"
ERROR_RED = "#FF4B4B"

C_BASELINE = "#5B7FA3"
C_LOGISTIC = "#00D1FF"
C_RF = "#E59A5A"

# =========================================================
# Global CSS
# =========================================================
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;600;700;800&display=swap');

        .stApp {{
            background: radial-gradient(circle at 20% 20%, #0d1117 0%, {BG} 100%);
            color: {TEXT};
            font-family: 'Inter', sans-serif;
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            background: radial-gradient(circle at 20% 20%, #0d1117 0%, {BG} 100%) !important;
        }}

        [data-testid="stHeader"] {{
            background: {BG};
            border-bottom: 1px solid {BORDER};
        }}

        .block-container {{
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 2.2rem;
        }}

        .topbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.15rem 0 1.2rem 0;
            border-bottom: 1px solid {BORDER};
            margin-bottom: 1.7rem;
            margin-top: 0.8rem;
        }}

        .topbar-title {{
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: {TEXT};
            font-family: 'Inter', sans-serif;
        }}

        .topbar-link a {{
            color: {ACCENT};
            text-decoration: none;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            font-weight: 500;
            border: 1px solid {ACCENT};
            padding: 6px 14px;
            border-radius: 4px;
            transition: all 0.25s ease;
            letter-spacing: 0.05em;
        }}

        .topbar-link a:hover {{
            background: {ACCENT};
            color: {BG};
            box-shadow: 0 0 20px {ACCENT_GLOW};
        }}

        .panel {{
            background: {SURFACE};
            backdrop-filter: blur(12px);
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 20px;
            height: 100%;
        }}

        .panel-title {{
            font-size: 0.92rem;
            font-weight: 700;
            color: {TEXT};
            margin-bottom: 12px;
            letter-spacing: 0.04em;
            font-family: 'JetBrains Mono', monospace;
        }}

        .panel-note {{
            color: {TEXT_MUTED};
            font-size: 0.84rem;
            line-height: 1.6;
            margin-bottom: 14px;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
            margin-bottom: 20px;
        }}

        .kpi-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            padding: 20px;
            border-radius: 12px;
        }}

        .kpi-label {{
            font-family: 'JetBrains Mono', monospace;
            color: {TEXT_MUTED};
            font-size: 0.64rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            margin-bottom: 8px;
        }}

        .kpi-value {{
            font-size: 2.15rem;
            font-weight: 700;
            color: {ACCENT};
            line-height: 1;
        }}

        .kpi-sub {{
            color: {TEXT_MUTED};
            font-size: 0.76rem;
            margin-top: 7px;
        }}

        .note-bar {{
            border-left: 2px solid {ACCENT};
            padding-left: 18px;
            margin-top: 8px;
            margin-bottom: 20px;
        }}

        .note-label {{
            font-family: 'JetBrains Mono', monospace;
            color: {ACCENT};
            font-size: 0.64rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            margin-bottom: 6px;
        }}

        .note-body {{
            color: {TEXT_MUTED};
            font-size: 0.86rem;
            line-height: 1.65;
        }}

        .model-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }}

        .model-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 18px;
            min-height: 165px;
        }}

        .model-badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.60rem;
            padding: 3px 8px;
            border-radius: 4px;
            margin-bottom: 10px;
            letter-spacing: 0.08em;
        }}

        .badge-benchmark {{
            color: #7da7d1;
            border: 1px solid #35506c;
            background: rgba(91,127,163,0.10);
        }}

        .badge-linear {{
            color: {ACCENT};
            border: 1px solid {ACCENT};
            background: {ACCENT_GLOW};
        }}

        .badge-ensemble {{
            color: #E59A5A;
            border: 1px solid #9d663b;
            background: rgba(229,154,90,0.10);
        }}

        .model-title {{
            color: {TEXT};
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 7px;
        }}

        .model-body {{
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            line-height: 1.6;
        }}

        section[data-testid="stSidebar"] {{
            background-color: #080C10 !important;
            border-right: 1px solid {BORDER};
        }}

        .stMultiSelect, .stSelectbox {{
            margin-bottom: 18px;
        }}

        .sidebar-title {{
            font-size: 1rem;
            color: {ACCENT};
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.08em;
            margin-bottom: 16px;
        }}

        .footnote {{
            border-left: 2px solid {ACCENT};
            padding-left: 20px;
            margin-top: 28px;
        }}

        .footnote p {{
            color: {TEXT_MUTED};
            font-size: 0.84rem;
            line-height: 1.65;
            margin: 0;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Paths / Data
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@st.cache_data
def load_data():
    metrics = pd.read_parquet(ARTIFACTS_DIR / "dashboard_metrics_table.parquet")
    curves = pd.read_parquet(ARTIFACTS_DIR / "dashboard_curves.parquet")
    drawdowns = pd.read_parquet(ARTIFACTS_DIR / "dashboard_drawdowns.parquet")
    log_importance = pd.read_parquet(ARTIFACTS_DIR / "logistic_feature_importance.parquet")
    rf_importance = pd.read_parquet(ARTIFACTS_DIR / "rf_feature_importance.parquet")

    curves["date"] = pd.to_datetime(curves["date"], errors="coerce")
    drawdowns["date"] = pd.to_datetime(drawdowns["date"], errors="coerce")

    curves = (
        curves.dropna(subset=["date", "strategy_name", "cum_return"])
        .sort_values(["strategy_name", "date"])
        .reset_index(drop=True)
    )
    drawdowns = (
        drawdowns.dropna(subset=["date", "strategy_name", "drawdown"])
        .sort_values(["strategy_name", "date"])
        .reset_index(drop=True)
    )

    return metrics, curves, drawdowns, log_importance, rf_importance


# =========================================================
# Helpers
# =========================================================
def strategy_label_map():
    return {
        "momentum_baseline": "Momentum Baseline",
        "logistic_top2": "Logistic Regression",
        "rf_top3": "Random Forest",
    }


def line_color_map():
    return {
        "momentum_baseline": C_BASELINE,
        "logistic_top2": C_LOGISTIC,
        "rf_top3": C_RF,
    }


def _chart_layout(height: int, y_title: str):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, family="JetBrains Mono", size=10),
        margin=dict(l=10, r=10, t=24, b=10),
        height=height,
        hovermode="x unified",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False, title=None),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False, title=y_title),
        legend=dict(orientation="h", y=1.10, x=0),
    )


def build_equity_chart(curves_f: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    names = strategy_label_map()
    colors = line_color_map()

    for strategy in curves_f["strategy_name"].unique():
        tmp = curves_f[curves_f["strategy_name"] == strategy]
        fig.add_trace(
            go.Scatter(
                x=tmp["date"],
                y=tmp["cum_return"],
                mode="lines",
                name=names.get(strategy, strategy),
                line=dict(width=2.1, color=colors.get(strategy, TEXT)),
                hovertemplate="<b>%{fullData.name}</b><br>%{x|%d %b %Y}<br>Portfolio Value: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(**_chart_layout(470, "Portfolio Value"))
    return fig


def build_drawdown_chart(drawdowns_f: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    names = strategy_label_map()
    colors = line_color_map()

    for strategy in drawdowns_f["strategy_name"].unique():
        tmp = drawdowns_f[drawdowns_f["strategy_name"] == strategy]
        color_hex = colors.get(strategy, "#999999")
        rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))

        fig.add_trace(
            go.Scatter(
                x=tmp["date"],
                y=tmp["drawdown"] * 100,
                mode="lines",
                name=names.get(strategy, strategy),
                line=dict(width=1.8, color=color_hex),
                fill="tozeroy",
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08)",
                hovertemplate="<b>%{fullData.name}</b><br>%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(**_chart_layout(430, "Drawdown %"))
    return fig


def build_feature_importance_chart(df_imp: pd.DataFrame, model_name: str) -> go.Figure:
    tmp = df_imp.head(10).copy().sort_values("abs_importance")
    bar_colors = [ACCENT if x >= 0 else "#E59A5A" for x in tmp["importance"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tmp["importance"],
            y=tmp["feature"],
            orientation="h",
            marker=dict(color=bar_colors),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    layout = _chart_layout(430, "")
    layout["xaxis"]["title"] = f"{model_name} Importance"
    fig.update_layout(**layout)
    fig.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.08)")
    return fig


def interpret_metrics(metrics_f: pd.DataFrame) -> str:
    names = strategy_label_map()
    best_sharpe = metrics_f.sort_values("sharpe", ascending=False).iloc[0]
    best_return = metrics_f.sort_values("annual_return", ascending=False).iloc[0]
    best_dd = metrics_f.sort_values("max_drawdown", ascending=False).iloc[0]

    return (
        f"{names.get(best_sharpe['strategy_name'], best_sharpe['strategy_name'])} delivers the strongest "
        f"risk-adjusted result, while {names.get(best_return['strategy_name'], best_return['strategy_name'])} "
        f"achieves the highest annualized return. The shallowest drawdown is observed in "
        f"{names.get(best_dd['strategy_name'], best_dd['strategy_name'])}. Both ML architectures improve "
        f"materially on the momentum benchmark over the out-of-sample evaluation window."
    )


def format_metrics_table(metrics_f: pd.DataFrame) -> pd.DataFrame:
    names = strategy_label_map()
    out = metrics_f.copy()

    out["strategy_name"] = out["strategy_name"].map(names).fillna(out["strategy_name"])
    out["annual_return"] = out["annual_return"] * 100
    out["annual_volatility"] = out["annual_volatility"] * 100
    out["max_drawdown"] = out["max_drawdown"] * 100

    out = out.rename(
        columns={
            "strategy_name": "Strategy",
            "annual_return": "Annual Return %",
            "annual_volatility": "Annual Volatility %",
            "sharpe": "Sharpe",
            "max_drawdown": "Max Drawdown %",
        }
    )

    return out[
        ["Strategy", "Annual Return %", "Annual Volatility %", "Sharpe", "Max Drawdown %"]
    ].copy()


def render_metrics_table_html(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        dd_class = "value-negative" if row["Max Drawdown %"] < 0 else "value-positive"
        rows.append(
            f"""
            <tr>
                <td>{row['Strategy']}</td>
                <td class="value-positive">{row['Annual Return %']:.2f}</td>
                <td>{row['Annual Volatility %']:.2f}</td>
                <td class="value-positive">{row['Sharpe']:.3f}</td>
                <td class="{dd_class}">{row['Max Drawdown %']:.2f}</td>
            </tr>
            """
        )

    return f"""
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            background: transparent;
            font-family: Inter, sans-serif;
        }}
        .custom-table-wrap {{
            width: 100%;
            overflow-x: auto;
            border: 1px solid {BORDER};
            border-radius: 8px;
            background: rgba(255,255,255,0.01);
        }}
        .custom-table {{
            width: 100%;
            min-width: 560px;
            border-collapse: collapse;
        }}
        .custom-table thead tr {{
            background: rgba(255,255,255,0.03);
        }}
        .custom-table th {{
            text-align: left;
            padding: 11px 12px;
            color: {TEXT_MUTED};
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border-bottom: 1px solid {BORDER};
            white-space: nowrap;
        }}
        .custom-table td {{
            padding: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            color: {TEXT};
            font-size: 13px;
            white-space: nowrap;
        }}
        .value-positive {{
            color: {ACCENT};
            font-family: 'JetBrains Mono', monospace;
        }}
        .value-negative {{
            color: #E59A5A;
            font-family: 'JetBrains Mono', monospace;
        }}
    </style>
    </head>
    <body>
        <div class="custom-table-wrap">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>Annual Return %</th>
                        <th>Annual Volatility %</th>
                        <th>Sharpe</th>
                        <th>Max Drawdown %</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """


def render_importance_table_html(df: pd.DataFrame) -> str:
    tmp = df.head(10).copy()
    rows = []
    for _, row in tmp.iterrows():
        cls = "value-positive" if row["importance"] >= 0 else "value-negative"
        rows.append(
            f"""
            <tr>
                <td>{row['feature']}</td>
                <td class="{cls}">{row['importance']:.4f}</td>
                <td>{row['abs_importance']:.4f}</td>
            </tr>
            """
        )

    return f"""
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            background: transparent;
            font-family: Inter, sans-serif;
        }}
        .custom-table-wrap {{
            width: 100%;
            overflow-x: auto;
            border: 1px solid {BORDER};
            border-radius: 8px;
            background: rgba(255,255,255,0.01);
        }}
        .custom-table {{
            width: 100%;
            min-width: 480px;
            border-collapse: collapse;
        }}
        .custom-table thead tr {{
            background: rgba(255,255,255,0.03);
        }}
        .custom-table th {{
            text-align: left;
            padding: 11px 12px;
            color: {TEXT_MUTED};
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border-bottom: 1px solid {BORDER};
            white-space: nowrap;
        }}
        .custom-table td {{
            padding: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            color: {TEXT};
            font-size: 13px;
            white-space: nowrap;
        }}
        .value-positive {{
            color: {ACCENT};
            font-family: 'JetBrains Mono', monospace;
        }}
        .value-negative {{
            color: #E59A5A;
            font-family: 'JetBrains Mono', monospace;
        }}
    </style>
    </head>
    <body>
        <div class="custom-table-wrap">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                        <th>|Importance|</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """


# =========================================================
# Load Data
# =========================================================
try:
    metrics, curves, drawdowns, log_importance, rf_importance = load_data()
except Exception as e:
    st.error(f"Failed to load dashboard artifacts: {e}")
    st.stop()

# =========================================================
# Header
# =========================================================
st.markdown(
    f"""
    <div class="topbar">
        <div class="topbar-left">
            <span class="topbar-title">ML STRATEGY RESEARCH <span style="color:{TEXT_MUTED}; font-weight:300;">/ OOS_EVAL</span></span>
        </div>
        <div class="topbar-link">
            <a href="https://www.linkedin.com/in/amirhossein-latifinavid-5923272a7" target="_blank">
                CONNECT: AMIRHOSSEIN LATIFINAVID
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown(
    f"<div class='sidebar-title'>CONTROL CENTER</div>",
    unsafe_allow_html=True,
)

strategy_options = sorted(curves["strategy_name"].unique().tolist())
selected_strategies = st.sidebar.multiselect(
    "Displayed Strategies",
    options=strategy_options,
    default=strategy_options,
    format_func=lambda x: strategy_label_map().get(x, x),
)

feature_model = st.sidebar.selectbox(
    "Feature Model",
    options=["logistic_top2", "rf_top3"],
    format_func=lambda x: strategy_label_map().get(x, x),
)

if not selected_strategies:
    st.warning("Select at least one strategy.")
    st.stop()

curves_f = curves[curves["strategy_name"].isin(selected_strategies)].copy()
drawdowns_f = drawdowns[drawdowns["strategy_name"].isin(selected_strategies)].copy()
metrics_f = metrics[metrics["strategy_name"].isin(selected_strategies)].copy()

# =========================================================
# KPI Strip
# =========================================================
best_sharpe = metrics_f.sort_values("sharpe", ascending=False).iloc[0]
best_return = metrics_f.sort_values("annual_return", ascending=False).iloc[0]
best_dd = metrics_f.sort_values("max_drawdown", ascending=False).iloc[0]

st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Best Sharpe Ratio</div>
            <div class="kpi-value">{best_sharpe['sharpe']:.3f}</div>
            <div class="kpi-sub">{strategy_label_map().get(best_sharpe['strategy_name'], best_sharpe['strategy_name'])}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Annualized Return</div>
            <div class="kpi-value">{best_return['annual_return'] * 100:.2f}%</div>
            <div class="kpi-sub">{strategy_label_map().get(best_return['strategy_name'], best_return['strategy_name'])}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Peak Drawdown</div>
            <div class="kpi-value" style="color:{ERROR_RED};">{best_dd['max_drawdown'] * 100:.2f}%</div>
            <div class="kpi-sub">{strategy_label_map().get(best_dd['strategy_name'], best_dd['strategy_name'])}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Interpretation
# =========================================================
st.markdown(
    f"""
    <div class="note-bar">
        <div class="note-label">Research Interpretation</div>
        <div class="note-body">{interpret_metrics(metrics_f)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Model Explanation Row
# =========================================================
st.markdown(
    """
    <div class="model-grid">
        <div class="model-card">
            <div class="model-badge badge-benchmark">Benchmark</div>
            <div class="model-title">Momentum Baseline</div>
            <div class="model-body">
                A classical rules-based allocation process ranking assets by recent relative strength. Used as the benchmark to test whether machine learning adds usable signal beyond trend following.
            </div>
        </div>
        <div class="model-card">
            <div class="model-badge badge-linear">Linear Model</div>
            <div class="model-title">Logistic Regression</div>
            <div class="model-body">
                A probabilistic linear classifier estimating the likelihood of positive forward returns from engineered market and macro features. Interpretable, stable, and well suited to controlled signal analysis.
            </div>
        </div>
        <div class="model-card">
            <div class="model-badge badge-ensemble">Ensemble Model</div>
            <div class="model-title">Random Forest</div>
            <div class="model-body">
                A nonlinear ensemble of decision trees capable of learning interaction effects and more complex regime behavior. Useful when market structure is not well captured by linear assumptions.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Main Charts Row
# =========================================================
col1, col2 = st.columns([1.5, 1], gap="medium")

with col1:
    st.markdown(
        '<div class="panel"><div class="panel-title">CUMULATIVE PERFORMANCE</div><div class="panel-note">Portfolio value through time across the out-of-sample window, highlighting compounding behavior and separation between strategies.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(build_equity_chart(curves_f), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        '<div class="panel"><div class="panel-title">PERFORMANCE METRICS</div><div class="panel-note">Annualized return, annualized volatility, Sharpe ratio, and maximum drawdown for selected strategies.</div>',
        unsafe_allow_html=True,
    )
    metrics_table = format_metrics_table(metrics_f)
    components.html(render_metrics_table_html(metrics_table), height=260, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Bottom Row
# =========================================================
col3, col4 = st.columns([1.2, 1.2], gap="medium")

with col3:
    st.markdown(
        '<div class="panel"><div class="panel-title">DRAWDOWN PROFILE</div><div class="panel-note">Peak-to-trough loss through time. Shallower drawdowns and faster recoveries indicate stronger downside control.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(build_drawdown_chart(drawdowns_f), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    model_title = "Logistic Regression" if feature_model == "logistic_top2" else "Random Forest"
    importance_df = log_importance if feature_model == "logistic_top2" else rf_importance

    st.markdown(
        f'<div class="panel"><div class="panel-title">FEATURE ATTRIBUTION / {model_title.upper()}</div><div class="panel-note">Top-ranked explanatory features supporting the current model. Positive and negative importance values indicate directional contribution to signal strength.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(build_feature_importance_chart(importance_df, model_title), use_container_width=True)
    components.html(render_importance_table_html(importance_df), height=340, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Footer Note
# =========================================================
st.markdown(
    f"""
    <div class="footnote">
        <p>
            <b>RESEARCH NOTE:</b> The momentum portfolio is retained as the classical benchmark throughout the study.
            Both ML architectures improve materially out of sample. Random Forest currently delivers the strongest
            Sharpe ratio and annualized return, while Logistic Regression remains the more transparent and interpretable
            signal engine with cleaner explanatory structure.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)