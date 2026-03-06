"""
network.transaction_graph
==========================
Builds a directed transaction graph with NetworkX where:

* **Nodes** represent financial accounts.
* **Edges** represent individual transactions (with metadata as attributes).

Node-level risk scores are computed from the proportion of fraudulent
transactions in each account's neighbourhood, enabling visual hot-spot
identification on the dashboard.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

import config


class TransactionGraph:
    """Directed transaction graph built from a transaction DataFrame.

    Parameters
    ----------
    df:
        Annotated transaction DataFrame (must contain ``sender_account``,
        ``receiver_account``, ``transaction_id``, ``amount``,
        ``fraud_probability``, and ``is_fraud`` columns).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.graph: nx.DiGraph = nx.DiGraph()
        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_high_risk_nodes(self, threshold: float = 0.5) -> List[str]:
        """Return accounts whose node risk score exceeds *threshold*."""
        return [
            node
            for node, data in self.graph.nodes(data=True)
            if data.get("risk_score", 0.0) >= threshold
        ]

    def get_node_risk_scores(self) -> Dict[str, float]:
        """Return a mapping of account → risk score."""
        return {
            node: data.get("risk_score", 0.0)
            for node, data in self.graph.nodes(data=True)
        }

    def subgraph(self, nodes: List[str]) -> nx.DiGraph:
        """Return an induced subgraph for the given *nodes*."""
        return self.graph.subgraph(nodes).copy()

    def top_risk_subgraph(self, n: int = config.MAX_GRAPH_DISPLAY_NODES) -> nx.DiGraph:
        """Return a subgraph of the *n* highest-risk nodes."""
        scores = self.get_node_risk_scores()
        top_nodes = sorted(scores, key=scores.get, reverse=True)[:n]  # type: ignore[arg-type]
        return self.subgraph(top_nodes)

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, int | float]:
        """Return basic graph statistics."""
        return {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "n_high_risk": len(self.get_high_risk_nodes()),
            "density": round(nx.density(self.graph), 6),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Populate the graph from self.df."""
        for _, row in self.df.iterrows():
            sender = row["sender_account"]
            receiver = row["receiver_account"]

            # Edges (transactions)
            self.graph.add_edge(
                sender,
                receiver,
                transaction_id=row["transaction_id"],
                amount=float(row["amount"]),
                fraud_probability=float(row.get("fraud_probability", 0.0)),
                is_fraud=int(row["is_fraud"]),
            )

        # Compute per-node risk score = mean fraud_probability of all incident edges
        for node in self.graph.nodes():
            incident_edges = list(self.graph.in_edges(node, data=True)) + \
                             list(self.graph.out_edges(node, data=True))
            if incident_edges:
                probs = [d.get("fraud_probability", 0.0) for _, _, d in incident_edges]
                risk = sum(probs) / len(probs)
            else:
                risk = 0.0
            self.graph.nodes[node]["risk_score"] = round(risk, 4)
