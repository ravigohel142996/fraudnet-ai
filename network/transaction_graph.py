"""
network/transaction_graph.py — NetworkX transaction graph construction.

Models the transaction dataset as a directed weighted graph where:
  * Nodes  = financial accounts
  * Edges  = individual transactions (directed, sender → receiver)
  * Edge attributes include amount, timestamp, and fraud label.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from config import HIGH_RISK_DEGREE_THRESHOLD
from utils.helpers import timed

logger = logging.getLogger(__name__)


class TransactionGraph:
    """Directed weighted transaction graph.

    Parameters
    ----------
    df:
        Transaction DataFrame (output of
        :func:`data.transaction_generator.generate_transactions`).

    Attributes
    ----------
    graph:
        The underlying :class:`networkx.DiGraph`.
    node_risk:
        Dict mapping account_id → risk score in [0, 1].
    high_risk_nodes:
        Set of account IDs flagged as high-risk.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.graph: nx.DiGraph = nx.DiGraph()
        self.node_risk: dict[str, float] = {}
        self.high_risk_nodes: set[str] = set()
        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed
    def _build(self) -> None:
        """Construct the graph from the transaction DataFrame."""
        logger.info("Building transaction graph from %d rows…", len(self.df))

        for _, row in self.df.iterrows():
            sender = row["sender_account"]
            receiver = row["receiver_account"]
            amount = float(row["amount"])
            is_fraud = int(row.get("is_fraud", 0))

            # Accumulate edge weight (total amount) for multi-edges
            if self.graph.has_edge(sender, receiver):
                self.graph[sender][receiver]["weight"] += amount
                self.graph[sender][receiver]["count"] += 1
                self.graph[sender][receiver]["fraud_count"] += is_fraud
            else:
                self.graph.add_edge(
                    sender, receiver,
                    weight=amount,
                    count=1,
                    fraud_count=is_fraud,
                )

        self._compute_node_risk()
        logger.info(
            "Graph built: %d nodes, %d edges, %d high-risk nodes",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(self.high_risk_nodes),
        )

    def _compute_node_risk(self) -> None:
        """Assign a risk score to every node based on fraud-edge density."""
        for node in self.graph.nodes():
            out_edges = list(self.graph.out_edges(node, data=True))
            in_edges = list(self.graph.in_edges(node, data=True))
            all_edges = out_edges + in_edges

            if not all_edges:
                self.node_risk[node] = 0.0
                continue

            total = sum(d.get("count", 0) for _, _, d in all_edges)
            fraud = sum(d.get("fraud_count", 0) for _, _, d in all_edges)
            degree = self.graph.degree(node)

            fraud_ratio = fraud / max(total, 1)
            degree_penalty = min(degree / HIGH_RISK_DEGREE_THRESHOLD, 1.0)

            self.node_risk[node] = min(0.7 * fraud_ratio + 0.3 * degree_penalty, 1.0)

        threshold = 0.4
        self.high_risk_nodes = {n for n, r in self.node_risk.items() if r >= threshold}

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def subgraph_around(self, node: str, radius: int = 2) -> nx.DiGraph:
        """Return the ego-graph centred on *node* with given *radius*."""
        return nx.ego_graph(self.graph, node, radius=radius)

    def top_risk_nodes(self, k: int = 20) -> list[tuple[str, float]]:
        """Return the *k* accounts with the highest risk scores."""
        return sorted(self.node_risk.items(), key=lambda x: x[1], reverse=True)[:k]

    def summary(self) -> dict:
        """Return a plain-dict summary of the graph."""
        return {
            "nodes": self.n_nodes,
            "edges": self.n_edges,
            "high_risk_nodes": len(self.high_risk_nodes),
            "avg_risk_score": float(np.mean(list(self.node_risk.values()))) if self.node_risk else 0.0,
            "density": nx.density(self.graph),
        }
