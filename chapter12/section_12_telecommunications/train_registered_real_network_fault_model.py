"""MLflow training example for genuine labeled network faults.

Requires CSV timestamp, fault_label, throughput_mbps, latency_ms, packet_loss_pct.
Uses chronological holdout; no substitute datasets. Intended as a reference
implementation, not a completed telecom experiment.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURES = ["throughput_mbps", "latency_ms", "packet_loss_pct"]

def main(config_path: str) -> None:
    cfg = json.loads(Path(config_path).read_text())
    source = Path(cfg["network_telemetry_csv"])
    if not source.is_file():
        raise FileNotFoundError("Real labeled network telemetry required")
    d = pd.read_csv(source, parse_dates=["timestamp"])
    required = FEATURES + ["timestamp", "fault_label"]
    if any(c not in d for c in required):
        raise ValueError("Missing required telemetry or fault labels")
    if len(d) < 200 or d[required].isna().any().any():
        raise ValueError("Insufficient or missing measured telemetry")
    if not d.fault_label.isin([0, 1]).all():
        raise ValueError("Fault labels must be binary")
    d = d.sort_values("timestamp", kind="stable")
    n = int(0.75 * len(d))
    train, test = d.iloc[:n], d.iloc[n:]
    if train.fault_label.nunique() != 2 or test.fault_label.nunique() != 2:
        raise ValueError("Both time partitions require positive and negative faults")
    seed = int(cfg.get("seed", 42))
    model = RandomForestClassifier(
        n_estimators=int(cfg.get("n_estimators", 250)),
        min_samples_leaf=3, class_weight="balanced",
        random_state=seed, n_jobs=-1)
    model.fit(train[FEATURES], train.fault_label)
    score = model.predict_proba(test[FEATURES])[:, 1]
    auc = roc_auc_score(test.fault_label, score)
    ap = average_precision_score(test.fault_label, score)
    baseline_ap = test.fault_label.mean()
    mlflow.set_experiment(cfg.get("experiment", "telecom-fault-prediction"))
    with mlflow.start_run():
        mlflow.log_params({"dataset_id": cfg["dataset_id"],
                           "git_commit": os.getenv("GIT_COMMIT", "unknown"),
                           "seed": seed, "chronological_test_rows": len(test)})
        mlflow.log_metrics({"test_roc_auc": auc, "test_ap": ap,
                            "prior_baseline_ap": baseline_ap})
        if auc < float(cfg.get("minimum_auc", 0.75)):
            raise RuntimeError("Model rejected by independent time-holdout gate")
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model",
            registered_model_name=cfg.get(
                "registered_model", "network-fault-classifier"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
