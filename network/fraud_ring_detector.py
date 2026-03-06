"""
network.fraud_ring_detector
============================
Detects circular money-transfer patterns (fraud rings) in a transaction graph.

A *fraud ring* is a directed cycle of length ≥ ``config.MIN_RING_LENGTH`` where
funds flow A → B → … → A, potentially obscuring the true origin.

To prevent exponential runtime on dense graphs the detector caps the working
subgraph to the top-``_MAX_SUBGRAPH_NODES`` highest-risk nodes before running
``nx.simple_cycles``.
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx

import config
from network.transaction_graph import TransactionGraph

_MAX_SUBGRAPH_NODES: int = config.MAX_SUBGRAPH_NODES


class FraudRingDetector:
    """Detect circular money-transfer patterns in a transaction graph.

    Parameters
    ----------
    transaction_graph:
        A fitted :class:`~network.transaction_graph.TransactionGraph` instance.
    min_ring_length:
        Minimum cycle length to report (default: ``config.MIN_RING_LENGTH``).
    max_ring_length:
        Maximum cycle length to report (default: ``config.MAX_RING_LENGTH``).
    """

    def __init__(
        self,
        transaction_graph: TransactionGraph,
        min_ring_length: int = config.MIN_RING_LENGTH,
        max_ring_length: int = config.MAX_RING_LENGTH,
    ) -> None:
        self.transaction_graph = transaction_graph
        self.min_ring_length = min_ring_length
        self.max_ring_length = max_ring_length

        self.rings: List[List[str]] = []
        self.ring_accounts: set[str] = set()
        self._detected: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> "FraudRingDetector":
        """Run ring detection and populate ``self.rings``.

        Returns
        -------
        self
        """
        # Work only on highest-risk nodes to keep runtime bounded
        subgraph = self.transaction_graph.top_risk_subgraph(_MAX_SUBGRAPH_NODES)

        raw_cycles: List[List[str]] = list(nx.simple_cycles(subgraph))

        self.rings = [
            cycle
            for cycle in raw_cycles
            if self.min_ring_length <= len(cycle) <= self.max_ring_length
        ]

        self.ring_accounts = {acct for ring in self.rings for acct in ring}
        self._detected = True
        return self

    def summary(self) -> Dict[str, int]:
        """Return summary statistics for detected rings."""
        self._assert_detected()
        if not self.rings:
            return {"n_rings": 0, "n_ring_accounts": 0, "avg_ring_length": 0}
        avg_len = sum(len(r) for r in self.rings) / len(self.rings)
        return {
            "n_rings": len(self.rings),
            "n_ring_accounts": len(self.ring_accounts),
            "avg_ring_length": round(avg_len, 2),
        }

    def get_rings(self) -> List[List[str]]:
        """Return the list of detected rings (each ring is a list of accounts)."""
        self._assert_detected()
        return self.rings

    def is_ring_account(self, account: str) -> bool:
        """Return True if *account* participates in at least one fraud ring."""
        self._assert_detected()
        return account in self.ring_accounts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_detected(self) -> None:
        if not self._detected:
            raise RuntimeError("FraudRingDetector.detect() must be called before accessing results.")
