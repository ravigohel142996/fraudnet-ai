"""
ui/charts.py — Plotly chart factory for FraudNet AI dashboard.

Each function accepts pre-computed data (DataFrames / arrays / dicts) and
returns a fully configured :class:`plotly.graph_objects.Figure`.
Keeping all chart logic here prevents the dashboard module from becoming
cluttered with visualisation boilerplate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    COLOR_FRAUD,
    COLOR_NORMAL,
    COLOR_SUCCESS,
    COLOR_WARNING,
    PLOTLY_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _base_layout(title: str, **kwargs) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16)),
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=20, t=50, b=40),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fraud probability histogram
# ---------------------------------------------------------------------------

def fraud_probability_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of fraud probability scores split by true label.

    Parameters
    ----------
    df:
        Must contain ``fraud_probability`` and ``is_fraud`` columns.
    """
    fig = px.histogram(
        df,
        x="fraud_probability",
        color=df["is_fraud"].map({0: "Legitimate", 1: "Fraud"}),
        nbins=50,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={"Legitimate": COLOR_NORMAL, "Fraud": COLOR_FRAUD},
        labels={"fraud_probability": "Fraud Probability", "color": "Label"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(**_base_layout("Fraud Probability Distribution"))
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_heatmap(cm: np.ndarray) -> go.Figure:
    """Heat-map visualisation of a 2×2 confusion matrix.

    Parameters
    ----------
    cm:
        2-D array [[TN, FP], [FN, TP]] as returned by sklearn.
    """
    labels = ["Legitimate", "Fraud"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            text=cm,
            texttemplate="%{text}",
            colorscale="RdBu",
            reversescale=True,
            showscale=False,
        )
    )
    fig.update_layout(**_base_layout("Confusion Matrix"))
    return fig


# ---------------------------------------------------------------------------
# Feature importance bar chart
# ---------------------------------------------------------------------------

def feature_importance_chart(importance: pd.Series) -> go.Figure:
    """Horizontal bar chart of Random Forest feature importances.

    Parameters
    ----------
    importance:
        pandas Series indexed by feature name, valued by importance score.
    """
    sorted_imp = importance.sort_values()
    fig = go.Figure(
        go.Bar(
            x=sorted_imp.values,
            y=sorted_imp.index,
            orientation="h",
            marker_color=COLOR_NORMAL,
        )
    )
    fig.update_layout(
        **_base_layout("Feature Importance"),
        xaxis_title="Importance",
        yaxis_title="",
    )
    return fig


# ---------------------------------------------------------------------------
# Fraud trend line chart
# ---------------------------------------------------------------------------

def fraud_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: fraud count (bars) and fraud rate (line).

    Parameters
    ----------
    trend_df:
        DataFrame with columns ``date``, ``total``, ``fraud_count``,
        ``fraud_rate`` (output of :func:`analytics.fraud_metrics.compute_fraud_trend`).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=trend_df["date"],
            y=trend_df["fraud_count"],
            name="Fraud Count",
            marker_color=COLOR_FRAUD,
            opacity=0.7,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["fraud_rate"] * 100,
            name="Fraud Rate (%)",
            line=dict(color=COLOR_WARNING, width=2),
            mode="lines+markers",
            marker_size=4,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        **_base_layout("Fraud Trend Over Time"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Fraud Count", secondary_y=False)
    fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Transaction network graph
# ---------------------------------------------------------------------------

def network_graph_chart(
    graph,  # nx.DiGraph
    node_risk: dict[str, float],
    max_nodes: int = 150,
) -> go.Figure:
    """Plotly scatter chart visualising the transaction network.

    Nodes are coloured by risk score (blue=normal → red=high risk).
    Edge width scales with transaction volume.

    Parameters
    ----------
    graph:
        NetworkX DiGraph.
    node_risk:
        Dict mapping node_id → risk score [0, 1].
    max_nodes:
        Cap node count for rendering performance.
    """
    import networkx as nx  # local import to keep module-level imports clean

    # Sample nodes if the graph is very large
    nodes = list(graph.nodes())
    if len(nodes) > max_nodes:
        # Prefer high-risk nodes
        ranked = sorted(nodes, key=lambda n: node_risk.get(n, 0), reverse=True)
        nodes = ranked[:max_nodes]
    sub = graph.subgraph(nodes)

    try:
        pos = nx.spring_layout(sub, seed=42, k=0.5)
    except Exception:  # noqa: BLE001
        pos = {n: (i % 10, i // 10) for i, n in enumerate(sub.nodes())}

    # Edge traces
    edge_x, edge_y = [], []
    for u, v in sub.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
    )

    # Node traces
    node_x = [pos[n][0] for n in sub.nodes()]
    node_y = [pos[n][1] for n in sub.nodes()]
    risk_scores = [node_risk.get(n, 0.0) for n in sub.nodes()]
    node_text = [f"{n}<br>Risk: {node_risk.get(n, 0):.2f}" for n in sub.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            color=risk_scores,
            colorscale=[[0, COLOR_NORMAL], [0.5, COLOR_WARNING], [1, COLOR_FRAUD]],
            size=8,
            colorbar=dict(title="Risk Score", thickness=12),
            line=dict(width=0.5, color="white"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            **_base_layout("Transaction Network Graph"),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode="closest",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Anomaly score scatter
# ---------------------------------------------------------------------------

def anomaly_scatter_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter of log_amount vs anomaly_score, coloured by is_anomaly.

    Parameters
    ----------
    df:
        Must contain ``log_amount``, ``anomaly_score``, ``is_anomaly``.
    """
    color_map = {0: COLOR_NORMAL, 1: COLOR_FRAUD}
    label_map = {0: "Normal", 1: "Anomaly"}
    df_plot = df.copy()
    df_plot["label"] = df_plot["is_anomaly"].map(label_map)

    fig = px.scatter(
        df_plot,
        x="log_amount",
        y="anomaly_score",
        color="label",
        color_discrete_map={"Normal": COLOR_NORMAL, "Anomaly": COLOR_FRAUD},
        opacity=0.5,
        labels={"log_amount": "Log(Amount)", "anomaly_score": "Anomaly Score"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(**_base_layout("Transaction Anomaly Scores"))
    return fig


# ---------------------------------------------------------------------------
# High-risk accounts bar chart
# ---------------------------------------------------------------------------

def high_risk_accounts_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of the top high-risk accounts.

    Parameters
    ----------
    risk_df:
        Output of :func:`analytics.fraud_metrics.compute_high_risk_accounts`.
    """
    top = risk_df.head(15)
    fig = go.Figure(
        go.Bar(
            x=top["fraud_rate"] * 100,
            y=top["account"],
            orientation="h",
            marker_color=COLOR_FRAUD,
            text=[f"{r:.1f}%" for r in top["fraud_rate"] * 100],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_base_layout("Top High-Risk Accounts (Fraud Rate %)"),
        xaxis_title="Fraud Rate (%)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Category fraud rate chart
# ---------------------------------------------------------------------------

def category_fraud_chart(cat_df: pd.DataFrame) -> go.Figure:
    """Bar chart of fraud rate by merchant category.

    Parameters
    ----------
    cat_df:
        Output of
        :func:`analytics.fraud_metrics.compute_category_fraud_rates`.
    """
    fig = px.bar(
        cat_df.sort_values("fraud_rate"),
        x="fraud_rate",
        y="merchant_category",
        orientation="h",
        color="fraud_rate",
        color_continuous_scale=[[0, COLOR_NORMAL], [1, COLOR_FRAUD]],
        labels={"fraud_rate": "Fraud Rate", "merchant_category": "Category"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(**_base_layout("Fraud Rate by Merchant Category"))
    return fig
