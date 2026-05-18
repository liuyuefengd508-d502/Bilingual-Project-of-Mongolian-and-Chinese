"""Lightweight regression tests for H-mean / IoU helpers (no torch).

Loads ``metrics/hmean.py`` directly so ``metrics/__init__.py`` (which imports torch)
is not executed.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_hmean_path = ROOT / "metrics" / "hmean.py"
_spec = importlib.util.spec_from_file_location("_metrics_hmean_under_test", _hmean_path)
assert _spec is not None and _spec.loader is not None
_hmean = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hmean)
HMeanEvaluator = _hmean.HMeanEvaluator
polygon_iou = _hmean.polygon_iou


class TestPolygonIoU(unittest.TestCase):
    def test_identical_square(self) -> None:
        sq = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
        iou = polygon_iou(sq, sq)
        self.assertGreater(iou, 0.99)

    def test_disjoint(self) -> None:
        a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]], dtype=np.float32)
        self.assertLess(polygon_iou(a, b), 1e-6)


class TestHMeanEvaluator(unittest.TestCase):
    def test_perfect_one_to_one(self) -> None:
        gt = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
        ev = HMeanEvaluator(iou_thresh=0.5)
        ev.add([gt], [False], [gt], [1.0])
        m = ev.compute()
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)
        self.assertAlmostEqual(m["hmean"], 1.0, places=5)

    def test_false_positive(self) -> None:
        gt = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
        pred = np.array([[100.0, 100.0], [110.0, 100.0], [110.0, 110.0], [100.0, 110.0]], dtype=np.float32)
        ev = HMeanEvaluator(iou_thresh=0.5)
        ev.add([gt], [False], [pred], [0.9])
        m = ev.compute()
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 1)
        self.assertEqual(m["hmean"], 0.0)


if __name__ == "__main__":
    unittest.main()
