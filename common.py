"""Shared, torch-free utilities: loading the preprocessed slices, patient-level metrics and the
cross-validation driver used by `baselines.py` and `quantum_hybrid.py`.

A "model" here is any function with the signature

    fit_predict(X_train, y_train, X_val, seed) -> (probabilities_on_X_val, n_trainable_params_or_None)
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

METRIC_KEYS = ["slice_acc", "slice_bacc", "slice_auc", "pat_acc", "pat_bacc", "pat_auc"]
TEST_FOLD = -1  # fold id that marks the held-out test patients


def load_slices(path):
    """Load the arrays written by preprocess_brats.py: X, y (1 = HGG), patient ids, fold ids."""
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"].astype(np.int64), d["patient"].astype(str), d["fold"].astype(int)


def aggregate_by_patient(prob, y, patients):
    """Mean slice probability per patient, with each patient's (single) label."""
    ids = np.unique(patients)
    patient_prob = np.array([prob[patients == i].mean() for i in ids])
    patient_label = np.array([y[patients == i][0] for i in ids])
    return patient_prob, patient_label


def evaluate(prob, y, patients):
    """Accuracy, balanced accuracy and AUC at slice level and at patient level."""
    out = {}
    pred = (prob >= 0.5).astype(int)
    out["slice_acc"] = accuracy_score(y, pred)
    out["slice_bacc"] = balanced_accuracy_score(y, pred)
    out["slice_auc"] = roc_auc_score(y, prob)
    patient_prob, patient_label = aggregate_by_patient(prob, y, patients)
    patient_pred = (patient_prob >= 0.5).astype(int)
    out["pat_acc"] = accuracy_score(patient_label, patient_pred)
    out["pat_bacc"] = balanced_accuracy_score(patient_label, patient_pred)
    out["pat_auc"] = roc_auc_score(patient_label, patient_prob)
    return out


def _format(metrics):
    return "  ".join(f"{k}={metrics[k]:.3f}" for k in METRIC_KEYS)


def cross_validate(fit_predict, X, y, patients, fold, n_seeds):
    """Patient-level k-fold CV (folds come from preprocess_brats.py). The test fold is never used."""
    rows, n_params = [], None
    for seed in range(n_seeds):
        for k in sorted(set(fold.tolist()) - {TEST_FOLD}):
            train, val = (fold != k) & (fold != TEST_FOLD), fold == k
            prob, n_params = fit_predict(X[train], y[train], X[val], seed)
            row = evaluate(prob, y[val], patients[val])
            row.update(seed=seed, fold=k)
            rows.append(row)
            print(f"seed {seed} fold {k}: {_format(row)}")
    return rows, n_params


def summarise(rows):
    """Print and return mean and std of every metric over folds x seeds."""
    print("\n=== CV summary (mean +- std over folds x seeds) ===")
    summary = {}
    for key in METRIC_KEYS:
        values = np.array([r[key] for r in rows])
        summary[key] = (float(values.mean()), float(values.std()))
        print(f"{key:10s} {values.mean():.4f} +- {values.std():.4f}")
    return summary


def run_experiment(fit_predict, args, tag):
    """CV, optional single scoring of the held-out test set, and a JSON dump of everything."""
    X, y, patients, fold = load_slices(args.data)
    t0 = time.time()

    rows, n_params = cross_validate(fit_predict, X, y, patients, fold, args.seeds)
    result = {"args": vars(args), "cv_rows": rows, "cv_summary": summarise(rows)}
    if n_params is not None:
        result["n_params"] = n_params
        print(f"trainable params: {n_params:,}")

    if args.eval_test:
        # Train on every non-test patient and score the held-out patients once (seed 0).
        train, test = fold != TEST_FOLD, fold == TEST_FOLD
        prob, _ = fit_predict(X[train], y[train], X[test], 0)
        result["test"] = evaluate(prob, y[test], patients[test])
        print("\n=== HELD-OUT TEST ===")
        print(_format(result["test"]))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved {out_dir / (tag + '.json')}  ({time.time() - t0:.0f}s)")
