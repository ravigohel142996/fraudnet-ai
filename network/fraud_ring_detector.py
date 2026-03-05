"""
network/fraud_ring_detector.py — Circular money-transfer pattern detection.

Detects fraud rings (cycles) in the transaction graph — patterns like
A → B → C → A — that indicate coordinated layering schemes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import pandas as pd

from config import (
    FRAUD_RING_MAX_CYCLE_LENGTH,
    FRAUD_RING_MIN_CYCLE_LENGTH,
    HIGH_RISK_DEGREE_THRESHOLD,
)
from network.transaction_graph import TransactionGraph
from utils.helpers import timed

logger = logging.getLogger(__name__)


@dataclass
class FraudRing:
    """A detected fraud ring (cycle) in the transaction graph.

    Attributes
    ----------
    nodes:
        Ordered list of account IDs forming the cycle.
    total_amount:
        Cumulative transaction amount flowing around the ring.
    fraud_edge_count:
        Number of edges in the cycle that are labelled as fraudulent.
    risk_score:
        Computed ring-level risk in [0, 1].
    """

    nodes: list[str]
    total_amount: float = 0.0
    fraud_edge_count: int = 0
    risk_score: float = 0.0

    @property
    def length(self) -> int:
        """Number of hops in the ring."""
        return len(self.nodes)

    def __repr__(self) -> str:  # pragma: no cover
        path = " → ".join(self.nodes) + " → " + self.nodes[0]
        return f"FraudRing(len={self.length}, risk={self.risk_score:.2f}, path={path})"


class FraudRingDetector:
    """Detects circular transaction patterns (fraud rings) in a graph.

    Parameters
    ----------
    transaction_graph:
        A fitted :class:`network.transaction_graph.TransactionGraph`.
    min_cycle_length:
        Minimum number of hops to consider as a fraud ring.
    max_cycle_length:
        Maximum cycle length (limits search space).
    """

    def __init__(
        self,
        transaction_graph: TransactionGraph,
        min_cycle_length: int = FRAUD_RING_MIN_CYCLE_LENGTH,
        max_cycle_length: int = FRAUD_RING_MAX_CYCLE_LENGTH,
    ) -> None:
        self.transaction_graph = transaction_graph
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self.fraud_rings: list[FraudRing] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Maximum number of nodes in the condensed subgraph passed to simple_cycles.
    # This prevents exponential blow-up on dense transaction graphs.
    _MAX_SUBGRAPH_NODES: int = 80

    @timed
    def detect(self) -> list[FraudRing]:
        """Run cycle detection and return all flagged fraud rings.

        Returns
        -------
        list[FraudRing]
            Rings sorted by risk score descending.
        """
        graph = self.transaction_graph.graph
        logger.info("Detecting fraud rings (min=%d, max=%d hops)…",
                    self.min_cycle_length, self.max_cycle_length)

        # Work on a condensed subgraph to keep runtime manageable.
        # Prefer high-risk nodes, cap at _MAX_SUBGRAPH_NODES.
        node_risk = self.transaction_graph.node_risk
        active_nodes = sorted(
            [n for n in graph.nodes() if graph.degree(n) >= 2],
            key=lambda n: node_risk.get(n, 0.0),
            reverse=True,
        )[: self._MAX_SUBGRAPH_NODES]

        subgraph = graph.subgraph(active_nodes).copy()

        rings: list[FraudRing] = []

        try:
            for cycle in nx.simple_cycles(subgraph):
                if self.min_cycle_length <= len(cycle) <= self.max_cycle_length:
                    ring = self._score_ring(cycle, graph)
                    rings.append(ring)
        except (RecursionError, MemoryError) as exc:
            logger.warning("Cycle detection interrupted (%s): %s", type(exc).__name__, exc)

        # De-duplicate: two cycles with the same node-set are the same ring
        seen: set[frozenset] = set()
        unique_rings: list[FraudRing] = []
        for ring in rings:
            key = frozenset(ring.nodes)
            if key not in seen:
                seen.add(key)
                unique_rings.append(ring)

        self.fraud_rings = sorted(unique_rings, key=lambda r: r.risk_score, reverse=True)
        logger.info("Detected %d unique fraud rings", len(self.fraud_rings))
        return self.fraud_rings

    def summary(self) -> dict:
        """Plain-dict summary of detected rings."""
        if not self.fraud_rings:
            return {"total_rings": 0, "high_risk_rings": 0, "max_risk_score": 0.0}

        return {
            "total_rings": len(self.fraud_rings),
            "high_risk_rings": sum(r.risk_score >= 0.7 for r in self.fraud_rings),
            "max_risk_score": max(r.risk_score for r in self.fraud_rings),
            "avg_ring_length": sum(r.length for r in self.fraud_rings) / len(self.fraud_rings),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_ring(self, cycle: list[str], graph: nx.DiGraph) -> FraudRing:
        """Compute risk score and financial metrics for a detected cycle."""
        total_amount = 0.0
        fraud_edges = 0

        for i, node in enumerate(cycle):
            next_node = cycle[(i + 1) % len(cycle)]
            if graph.has_edge(node, next_node):
                edge_data = graph[node][next_node]
                total_amount += edge_data.get("weight", 0)
                fraud_edges += edge_data.get("fraud_count", 0)

        # Risk is driven by: fraud edge ratio + node-level risk scores
        edge_fraud_ratio = fraud_edges / max(len(cycle), 1)
        node_risks = [self.transaction_graph.node_risk.get(n, 0.0) for n in cycle]
        avg_node_risk = sum(node_risks) / len(node_risks)

        risk_score = min(0.6 * edge_fraud_ratio + 0.4 * avg_node_risk, 1.0)

        return FraudRing(
            nodes=list(cycle),
            total_amount=total_amount,
            fraud_edge_count=fraud_edges,
            risk_score=risk_score,
        )
