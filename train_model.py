#!/usr/bin/env python3
"""
AutoFix AI - model training pipeline (CPU-only).

What this script does
---------------------
1. Synthesizes SAE J1979-style OBD-II telemetry with fault signatures.
2. Trains a small multi-class XGBoost classifier on 4 states:
       Normal | Fuel System Lean | Cooling Fault | Air Leak
3. Saves a bundle (model + feature order + labels + version) to `model.joblib`.
4. Exports a stdlib-only edge pre-screener (`edge_model.json` +
   `edge_prescreen.py`) so low-power devices can run the same trees without
   xgboost, numpy, scikit-learn or any TensorFlow runtime.

Usage
-----
    python train_model.py                      # defaults
    python train_model.py --samples 4000 --out-dir artifacts

Bundle contract (consumed by the FastAPI service)
-------------------------------------------------
    bundle = joblib.load("model.joblib")
    bundle["model"]     -> fitted xgboost.XGBClassifier
    bundle["features"]  -> ordered list of feature column names
    bundle["labels"]    -> list of class names; index == class id
    bundle["version"]   -> model version string (log it to BigQuery)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("autofix.train")

# -----------------------------------------------------------------------------
# Schema - keep in sync with the API's Pydantic request model.
# -----------------------------------------------------------------------------
MODEL_VERSION = "1.0.0"

# Index in this list == integer class id used by XGBoost.
LABELS: list[str] = ["Normal", "Fuel System Lean", "Cooling Fault", "Air Leak"]
LABEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}

FEATURES: list[str] = [
    "rpm",                  # PID 0x0C  - engine speed (rpm)
    "coolant_temp_c",       # PID 0x05  - engine coolant temperature (deg C)
    "short_fuel_trim_pct",  # PID 0x06  - short-term fuel trim, bank 1 (%)
    "long_fuel_trim_pct",   # PID 0x07  - long-term fuel trim, bank 1 (%)
    "intake_pressure_kpa",  # PID 0x0B  - intake manifold absolute pressure (kPa)
    "engine_load_pct",      # PID 0x04  - calculated engine load (%)
]

# Physical limits from the task spec; engine load is naturally 0-100 %.
BOUNDS: dict[str, tuple[float, float]] = {
    "rpm": (600.0, 4000.0),
    "coolant_temp_c": (70.0, 115.0),
    "short_fuel_trim_pct": (-25.0, 25.0),
    "long_fuel_trim_pct": (-25.0, 25.0),
    "intake_pressure_kpa": (20.0, 100.0),
    "engine_load_pct": (0.0, 100.0),
}


# -----------------------------------------------------------------------------
# 1. Synthetic data
# -----------------------------------------------------------------------------
def _healthy_operating_point(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Generate n physically-coherent healthy readings.

    Load rises with throttle and RPM; manifold pressure (MAP) tracks load.
    That coupling matters: an Air Leak is detectable precisely because MAP
    becomes inconsistent with engine load.
    """
    rpm = rng.uniform(*BOUNDS["rpm"], n)
    rpm_norm = (rpm - 600.0) / 3400.0
    throttle = rng.beta(2.0, 3.0, n)  # mostly light-to-moderate throttle

    load = 12.0 + 65.0 * throttle * (0.45 + 0.55 * rpm_norm) + rng.normal(0, 3, n)
    load = np.clip(load, 5.0, 95.0)

    map_kpa = 22.0 + 0.8 * load + rng.normal(0, 3, n)

    return pd.DataFrame(
        {
            "rpm": rpm,
            "coolant_temp_c": rng.normal(92.0, 4.0, n),   # healthy thermostat band
            "short_fuel_trim_pct": rng.normal(0.0, 3.0, n),
            "long_fuel_trim_pct": rng.normal(0.0, 3.0, n),
            "intake_pressure_kpa": map_kpa,
            "engine_load_pct": load,
        }
    )


def synthesize_telemetry(n_per_class: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Return a shuffled, class-balanced DataFrame of OBD-II telemetry.

    Fault signatures (deliberately overlapping so the task is not trivial):
      * Fuel System Lean : both fuel trims strongly positive, MAP/load normal
                           (ECM is adding fuel to compensate for lean mixture).
      * Cooling Fault    : ~70% overheating (104-115 C), ~30% thermostat stuck
                           open / sensor fault (70-78 C); trims normal.
      * Air Leak         : unmetered air -> MAP elevated vs. load (worst at
                           idle) plus moderately positive trims.
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for label in LABELS:
        df = _healthy_operating_point(rng, n_per_class)
        rpm_norm = (df["rpm"].to_numpy() - 600.0) / 3400.0

        if label == "Fuel System Lean":
            df["short_fuel_trim_pct"] = rng.uniform(5.0, 22.0, n_per_class) + rng.normal(0, 2, n_per_class)
            df["long_fuel_trim_pct"] = rng.uniform(8.0, 25.0, n_per_class) + rng.normal(0, 1.5, n_per_class)

        elif label == "Cooling Fault":
            overheating = rng.random(n_per_class) < 0.70
            hot = rng.uniform(104.0, 115.0, n_per_class)
            cold = rng.uniform(70.0, 78.0, n_per_class)
            df["coolant_temp_c"] = np.where(overheating, hot, cold)

        elif label == "Air Leak":
            # Leak effect is largest at low RPM (high manifold vacuum normally).
            leak = rng.uniform(12.0, 30.0, n_per_class) * (1.0 - 0.5 * rpm_norm)
            df["intake_pressure_kpa"] = df["intake_pressure_kpa"] + leak
            df["short_fuel_trim_pct"] = rng.uniform(3.0, 16.0, n_per_class) + rng.normal(0, 2, n_per_class)
            df["long_fuel_trim_pct"] = rng.uniform(2.0, 14.0, n_per_class) + rng.normal(0, 1.5, n_per_class)

        df["fault_state"] = label
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Enforce physical sensor limits.
    for col, (lo, hi) in BOUNDS.items():
        data[col] = data[col].clip(lo, hi)

    return data.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 2. Training
# -----------------------------------------------------------------------------
def train_classifier(
    data: pd.DataFrame, seed: int = 42
) -> tuple[XGBClassifier, pd.DataFrame, pd.Series]:
    """Train the CPU-friendly XGBoost classifier. Returns (model, X_test, y_test)."""
    X = data[FEATURES].astype("float32")
    y = data["fault_state"].map(LABEL_TO_ID).astype("int32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    model = XGBClassifier(
        objective="multi:softprob",
        n_estimators=50,        # per spec: keep the model tiny
        max_depth=4,            # per spec
        learning_rate=0.2,      # compensates for only 50 rounds
        n_jobs=-1,              # use all laptop cores
        tree_method="hist",     # fastest CPU algorithm
        eval_metric="mlogloss",
        random_state=seed,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    log.info("Hold-out accuracy: %.4f", accuracy_score(y_test, preds))
    log.info(
        "Classification report:\n%s",
        classification_report(y_test, preds, target_names=LABELS, digits=3),
    )
    return model, X_test, y_test


def save_model_bundle(model: XGBClassifier, path: Path) -> None:
    """Persist model + metadata as one joblib file (compressed to stay small)."""
    bundle = {
        "model": model,
        "features": FEATURES,
        "labels": LABELS,
        "version": MODEL_VERSION,
    }
    joblib.dump(bundle, path, compress=3)
    log.info("Saved model bundle -> %s (%.1f KB)", path, path.stat().st_size / 1024)


# -----------------------------------------------------------------------------
# 3. Edge fallback export (no xgboost / numpy / TensorFlow needed at runtime)
# -----------------------------------------------------------------------------
# This source is written verbatim to `edge_prescreen.py`. It uses only the
# Python standard library, so it can run on a Raspberry Pi / OBD dongle host
# with a bare Python install. It reproduces XGBoost's multi-class inference:
# sum leaf values per class (tree i belongs to class i % n_classes), softmax.
EDGE_RUNTIME_SRC = r'''"""AutoFix AI edge pre-screener (auto-generated by train_model.py).

Pure standard library: no xgboost, numpy, scikit-learn or TensorFlow.
Use it to flag obviously faulty readings locally and escalate to the cloud
service only when needed.
"""
import json
import math


class EdgePrescreener:
    def __init__(self, model_path="edge_model.json"):
        with open(model_path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
        self.features = spec["features"]
        self.labels = spec["labels"]
        self.trees = spec["trees"]
        self.version = spec.get("version", "unknown")

    @staticmethod
    def _walk(node, x):
        """Traverse one tree. XGBoost rule: go to 'yes' child if x < threshold."""
        while "v" not in node:
            value = x[node["f"]]
            if value is None or value != value:  # None or NaN -> default direction
                go_yes = node["m"]
            else:
                go_yes = value < node["t"]
            node = node["y"] if go_yes else node["n"]
        return node["v"]

    def predict_proba(self, reading):
        """reading: dict keyed by feature name. Returns {label: probability}."""
        x = [reading.get(name) for name in self.features]
        n_classes = len(self.labels)
        margins = [0.0] * n_classes
        for i, tree in enumerate(self.trees):
            margins[i % n_classes] += self._walk(tree, x)
        peak = max(margins)  # subtract max for numerical stability
        exps = [math.exp(m - peak) for m in margins]
        total = sum(exps)
        return {label: e / total for label, e in zip(self.labels, exps)}

    def predict(self, reading):
        probs = self.predict_proba(reading)
        label = max(probs, key=probs.get)
        return label, probs[label]

    def prescreen(self, reading, min_confidence=0.80):
        """Return a small decision dict. Escalate to cloud on any fault or low confidence."""
        label, confidence = self.predict(reading)
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "escalate": label != "Normal" or confidence < min_confidence,
        }
'''


def _compact_tree(node: dict, feature_index: dict[str, int]) -> dict:
    """Convert one XGBoost JSON-dump tree into a compact nested dict.

    Leaf     -> {"v": leaf_value}
    Internal -> {"f": feature_idx, "t": threshold, "y": yes_subtree,
                 "n": no_subtree, "m": missing_goes_to_yes}
    """
    if "leaf" in node:
        return {"v": float(node["leaf"])}

    children = {child["nodeid"]: child for child in node["children"]}
    split = node["split"]
    # Booster keeps real names when trained on a DataFrame; else "f0", "f1", ...
    feat_idx = feature_index[split] if split in feature_index else int(str(split).lstrip("f"))

    return {
        "f": feat_idx,
        "t": float(node["split_condition"]),
        "y": _compact_tree(children[node["yes"]], feature_index),
        "n": _compact_tree(children[node["no"]], feature_index),
        "m": node["missing"] == node["yes"],
    }


def export_edge_model(model: XGBClassifier, out_dir: Path) -> tuple[Path, Path]:
    """Export the trained trees for dependency-free edge inference.

    Writes:
        edge_model.json     - compact tree structure + labels + feature order
        edge_prescreen.py   - stdlib-only runtime (EdgePrescreener)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_index = {name: i for i, name in enumerate(FEATURES)}

    dumps = model.get_booster().get_dump(dump_format="json")
    trees = [_compact_tree(json.loads(d), feature_index) for d in dumps]

    spec = {
        "version": MODEL_VERSION,
        "features": FEATURES,
        "labels": LABELS,
        "trees": trees,
    }
    model_path = out_dir / "edge_model.json"
    runtime_path = out_dir / "edge_prescreen.py"

    model_path.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    runtime_path.write_text(EDGE_RUNTIME_SRC, encoding="utf-8")

    log.info("Edge export -> %s (%.1f KB), %s", model_path, model_path.stat().st_size / 1024, runtime_path)
    return model_path, runtime_path


def verify_edge_parity(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    model_path: Path,
    runtime_path: Path,
    n_rows: int = 500,
    min_agreement: float = 0.99,
) -> float:
    """Check the pure-Python predictor agrees with XGBoost on held-out rows."""
    spec = importlib.util.spec_from_file_location("edge_prescreen", runtime_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    edge = module.EdgePrescreener(str(model_path))

    sample = X_test.head(n_rows)
    xgb_pred = model.predict(sample)
    edge_pred = [
        LABEL_TO_ID[edge.predict(row)[0]]
        for row in sample.to_dict(orient="records")
    ]
    agreement = float(np.mean(np.asarray(edge_pred) == xgb_pred))
    log.info("Edge/XGBoost parity on %d rows: %.2f%%", len(sample), agreement * 100)

    if agreement < min_agreement:
        raise RuntimeError(
            f"Edge export parity {agreement:.3f} below threshold {min_agreement:.2f}"
        )
    return agreement


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the AutoFix AI fault classifier.")
    parser.add_argument("--samples", type=int, default=2500, help="Synthetic rows per class (default 2500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Directory for output artifacts")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Synthesizing %d rows (%d per class)...", args.samples * len(LABELS), args.samples)
    data = synthesize_telemetry(n_per_class=args.samples, seed=args.seed)
    log.info("Class distribution:\n%s", data["fault_state"].value_counts().to_string())

    model, X_test, _ = train_classifier(data, seed=args.seed)
    save_model_bundle(model, args.out_dir / "model.joblib")

    model_path, runtime_path = export_edge_model(model, args.out_dir)
    verify_edge_parity(model, X_test, model_path, runtime_path)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
