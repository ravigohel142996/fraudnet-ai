"""
ui.charts
=========
All Plotly figure-creation functions.

Each function accepts plain data objects (DataFrames, arrays, dicts) and
returns a :class:`plotly.graph_objects.Figure`.  No Streamlit calls are made
here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

import config


# ---------------------------------------------------------------------------
# Colour palette helpers
# ---------------------------------------------------------------------------


def _fraud_colorscale() -> List[list]:
    return [[0, config.THEME_NORMAL_COLOR], [1, config.THEME_FRAUD_COLOR]]


# ---------------------------------------------------------------------------
# Fraud probability distribution
# ---------------------------------------------------------------------------


def fraud_probability_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of ``fraud_probability`` split by true label."""
    fig = go.Figure()
    for label, colour, name in [
        (0, config.THEME_NORMAL_COLOR, "Normal"),
        (1, config.THEME_FRAUD_COLOR, "Fraud"),
    ]:
        subset = df[df["is_fraud"] == label]["fraud_probability"]
        fig.add_trace(
            go.Histogram(
                x=subset,
                nbinsx=50,
                name=name,
                marker_color=colour,
                opacity=0.75,
            )
        )
    fig.update_layout(
        title="Fraud Probability Distribution",
        xaxis_title="Fraud Probability",
        yaxis_title="Count",
        barmode="overlay",
        legend=dict(orientation="h", y=-0.2),
        template="plotly_dark",
        height=380,
        margin=dict(t=50, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix_heatmap(cm: np.ndarray) -> go.Figure:
    """Annotated heatmap for a 2 × 2 confusion matrix."""
    labels = ["Normal", "Fraud"]
    text = [
        [f"TN: {cm[0, 0]:,}", f"FP: {cm[0, 1]:,}"],
        [f"FN: {cm[1, 0]:,}", f"TP: {cm[1, 1]:,}"],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_dark",
        height=350,
        margin=dict(t=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def feature_importance_chart(importances: pd.Series) -> go.Figure:
    """Horizontal bar chart of Random-Forest feature importances."""
    fig = go.Figure(
        go.Bar(
            x=importances.values,
            y=importances.index.tolist(),
            orientation="h",
            marker_color=config.THEME_NORMAL_COLOR,
        )
    )
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=350,
        margin=dict(t=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Fraud trend over time
# ---------------------------------------------------------------------------


def fraud_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Dual-axis line chart: transaction volume + fraud rate over time."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["total"],
            name="Total Transactions",
            line=dict(color=config.THEME_NORMAL_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.15)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["fraud_rate"] * 100,
            name="Fraud Rate (%)",
            line=dict(color=config.THEME_FRAUD_COLOR, width=2, dash="dot"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Fraud Trend Over Time",
        legend=dict(orientation="h", y=-0.2),
        template="plotly_dark",
        height=380,
        margin=dict(t=50, b=60),
    )
    fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
    fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# High-risk accounts
# ---------------------------------------------------------------------------


def high_risk_accounts_chart(accounts_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top high-risk accounts."""
    fig = go.Figure(
        go.Bar(
            x=accounts_df["avg_fraud_probability"],
            y=accounts_df["sender_account"],
            orientation="h",
            marker=dict(
                color=accounts_df["avg_fraud_probability"],
                colorscale=[[0, config.THEME_WARNING_COLOR], [1, config.THEME_FRAUD_COLOR]],
                showscale=True,
                colorbar=dict(title="Risk"),
            ),
            text=[f"{v:.2%}" for v in accounts_df["avg_fraud_probability"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Top High-Risk Accounts",
        xaxis_title="Avg Fraud Probability",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=420,
        margin=dict(t=50, r=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Transaction network graph
# ---------------------------------------------------------------------------


def transaction_network_graph(
    graph: nx.DiGraph,
    ring_accounts: Optional[set] = None,
) -> go.Figure:
    """Plotly Scatter-based visualisation of the transaction graph.

    Parameters
    ----------
    graph:
        The (sub)graph to render.
    ring_accounts:
        Set of account IDs that participate in fraud rings (rendered in
        orange).
    """
    ring_accounts = ring_accounts or set()

    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="Transaction Network (no data)", template="plotly_dark")
        return fig

    # Layout
    try:
        pos = nx.spring_layout(graph, seed=42, k=1.5)
    except Exception:
        pos = {n: (0, 0) for n in graph.nodes()}

    # --- edges ---
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for src, dst in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        name="Transactions",
    )

    # --- nodes ---
    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        risk = data.get("risk_score", 0.0)
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Risk: {risk:.2%}")

        if node in ring_accounts:
            node_color.append(config.THEME_WARNING_COLOR)
            node_size.append(14)
        elif risk >= 0.5:
            node_color.append(config.THEME_FRAUD_COLOR)
            node_size.append(12)
        else:
            node_color.append(config.THEME_NORMAL_COLOR)
            node_size.append(8)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="#fff"),
        ),
        name="Accounts",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Transaction Network Graph",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_dark",
        height=520,
        margin=dict(t=50),
        annotations=[
            dict(
                text=(
                    "<span style='color:#3B82F6'>● Normal</span>  "
                    "<span style='color:#EF4444'>● High Risk</span>  "
                    "<span style='color:#F59E0B'>● Ring Account</span>"
                ),
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                font=dict(size=12),
            )
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Anomaly score scatter
# ---------------------------------------------------------------------------


def anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter of transaction amount vs anomaly score, coloured by label."""
    if "anomaly_score" not in df.columns:
        return go.Figure()

    sample = df.sample(min(2000, len(df)), random_state=42)
    colors = sample["is_anomaly"].map({0: config.THEME_NORMAL_COLOR, 1: config.THEME_FRAUD_COLOR})

    fig = go.Figure(
        go.Scatter(
            x=sample["transaction_amount"],
            y=sample["anomaly_score"],
            mode="markers",
            marker=dict(
                color=colors,
                size=5,
                opacity=0.6,
                line=dict(width=0.3, color="#fff"),
            ),
            text=sample["sender_account"],
            hovertemplate="Account: %{text}<br>Amount: $%{x:.2f}<br>Anomaly Score: %{y:.4f}",
        )
    )
    fig.update_layout(
        title="Transaction Anomaly Scores",
        xaxis_title="Transaction Amount ($)",
        yaxis_title="Anomaly Score",
        template="plotly_dark",
        height=380,
        margin=dict(t=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Hourly heatmap
# ---------------------------------------------------------------------------


def hourly_heatmap(hourly_df: pd.DataFrame) -> go.Figure:
    """Bar chart of fraud rate by hour of day."""
    fig = go.Figure(
        go.Bar(
            x=hourly_df["hour"],
            y=hourly_df["fraud_rate"] * 100,
            marker=dict(
                color=hourly_df["fraud_rate"],
                colorscale=[[0, config.THEME_OK_COLOR], [1, config.THEME_FRAUD_COLOR]],
                showscale=False,
            ),
        )
    )
    fig.update_layout(
        title="Fraud Rate by Hour of Day",
        xaxis_title="Hour (0–23)",
        yaxis_title="Fraud Rate (%)",
        template="plotly_dark",
        height=320,
        margin=dict(t=50),
    )
    return fig
