"""Evaluate eye-closure / blink detection on Eyeblink8, and train a better detector.

Protocol
--------
* **Leave-one-person-out (LOPO)**: 4 folds, one per person. Every number for a person comes
  from a model that never saw that person. Thresholds for trained models are picked by an
  *inner* LOPO on the 3 training people only, so the test person never influences any choice.
* **Frame level**: target = "at least one eye fully closed" (annotation FC == 'C').
  Frames where an eye is annotated not-visible are excluded from frame metrics.
* **Blink (event) level**: predicted closed frames -> runs (gaps <= 2 frames merged). A
  predicted run is a true positive if it overlaps a ground-truth blink interval (frames
  sharing a blink ID); each ground-truth blink can be matched once.
* **PERCLOS**: 60 s windows, 10 s step; mean absolute error vs. the hand-labelled closure.
* **False alerts**: all subjects are awake (working at a computer), so every microsleep
  (closed >= 1 s) or drowsy (20 s PERCLOS > 15%) alert the logic would raise is a false alarm.

Methods
-------
1. ``fixed_ear_0.2``: raw EAR < 0.2 (the classic textbook rule)
2. ``deployed_rule``: exactly what DriverMonitor ships (calibrated 80th-pct baseline x 0.72, EMA)
3. ``mediapipe_blink``: MediaPipe's learned eyeBlink blendshape > 0.5 (no training by us)
4. ``ear_window_linear``: 13-frame window of calibrated EAR -> logistic regression
   (the "EAR SVM" design of Soukupova & Cech 2016) - trained
5. ``gbm_multifeature``: gradient-boosted trees on EAR windows (static + adaptive
   calibration), MediaPipe eye blendshapes and head pose - trained

    .venv/bin/python -m eval.eyeblink8_eval
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drivemind.config import ROOT
from drivemind.perception.eye_state import ADAPT_N, HALF, window_features

FEAT_DIR = ROOT / "data" / "features"
RESULTS = ROOT / "eval" / "results"
FPS = 30.0


# ----------------------------------------------------------------------------- data
def load() -> list[dict]:
    vids = []
    for p in sorted(FEAT_DIR.glob("eyeblink8_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
        d = dict(np.load(p, allow_pickle=True))
        d["person"], d["video"] = str(d["person"]), str(d["video"])
        vids.append(d)
    return vids


def ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs (frames without a face), then back-fill the start."""
    x = x.astype(np.float64).copy()
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    if np.isnan(x[0]):
        first = np.flatnonzero(~np.isnan(x))
        x[: first[0] if len(first) else len(x)] = x[first[0]] if len(first) else 0.0
    return x


def windows(x: np.ndarray, half: int = HALF) -> np.ndarray:
    pad = np.pad(x, half, mode="edge")
    return sliding_window_view(pad, 2 * half + 1)


def calibration_baseline(ear: np.ndarray, face: np.ndarray) -> float:
    """Same as DriverMonitor: 80th percentile of the first 45 frames with a face."""
    vals = ear[face == 1][:45]
    return float(np.percentile(vals, 80))


def adaptive_baseline(ear_ff: np.ndarray, n: int = ADAPT_N) -> np.ndarray:
    """Causal rolling 85th percentile over the last n frames (~10 s). Adapts to lighting
    and posture changes instead of trusting a single calibration at the start."""
    pad = np.concatenate([np.full(n - 1, ear_ff[0]), ear_ff])
    return np.percentile(sliding_window_view(pad, n)[:, :], 85, axis=1)


def features(d: dict) -> dict:
    face = np.nan_to_num(d["face"]).astype(int)
    ear = (d["ear_l"] + d["ear_r"]) / 2
    ear_ff = ffill(ear)
    base = calibration_baseline(ear, face)
    blink = ffill((d["eyeBlinkLeft"] + d["eyeBlinkRight"]) / 2)
    raw = np.column_stack(
        [
            ear_ff, ffill(d["ear_l"]), ffill(d["ear_r"]), blink,
            ffill((d["eyeSquintLeft"] + d["eyeSquintRight"]) / 2),
            ffill((d["eyeLookDownLeft"] + d["eyeLookDownRight"]) / 2),
            ffill(d["pitch"]), ffill(d["yaw"]),
        ]
    )  # column order = drivemind.perception.eye_state.RAW
    adapt = adaptive_baseline(ear_ff)
    # Edge-pad so every frame has a full 13-frame window, then build features with the
    # *same* function the live DriverMonitor uses (no training/serving skew).
    raw_p = np.pad(raw, ((HALF, HALF), (0, 0)), mode="edge")
    adapt_p = np.pad(adapt, HALF, mode="edge")
    gbm = np.stack([window_features(raw_p[i : i + 2 * HALF + 1], adapt_p[i : i + 2 * HALF + 1], base) for i in range(len(raw))])
    return {"face": face, "ear": ear, "ear_ff": ear_ff, "base": base, "blink": blink, "lin": windows(ear_ff / base), "gbm": gbm}


# -------------------------------------------------------------------- rule methods
def deployed_rule(d: dict, f: dict) -> np.ndarray:
    """Re-implementation of DriverMonitor's per-frame closed decision (verified against the
    class itself in verify_deployed())."""
    th = 0.72 * f["base"]
    closed = np.zeros(len(f["face"]), bool)
    smooth = None
    seen = 0
    for i, (has_face, e) in enumerate(zip(f["face"], f["ear"])):
        if not has_face:
            continue
        smooth = e if smooth is None else 0.6 * e + 0.4 * smooth
        seen += 1
        if seen > 45:  # during calibration the shipped code reports 'calibrating'
            closed[i] = smooth < th
    return closed


# --------------------------------------------------------------------------- metrics
def runs(mask: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def gt_blinks(blink_id: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for b in np.unique(blink_id[blink_id >= 0]):
        idx = np.flatnonzero(blink_id == b)
        out.append((idx[0], idx[-1]))
    return out


def event_metrics(pred: np.ndarray, blink_id: np.ndarray) -> dict:
    p_runs, g = runs(pred), gt_blinks(blink_id)
    matched_gt, tp = set(), 0
    for s, e in p_runs:
        hit = next((j for j, (gs, ge) in enumerate(g) if s <= ge and gs <= e and j not in matched_gt), None)
        if hit is not None:
            matched_gt.add(hit)
            tp += 1
    return {"pred_events": len(p_runs), "gt_blinks": len(g), "tp": tp}


def prf(tp: int, n_pred: int, n_gt: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def perclos_mae(pred: np.ndarray, gt: np.ndarray, win_s: float = 60, step_s: float = 10) -> float:
    w, s = int(win_s * FPS), int(step_s * FPS)
    errs = [abs(pred[i : i + w].mean() - gt[i : i + w].mean()) for i in range(0, max(len(gt) - w, 0) + 1, s)]
    return float(np.mean(errs)) if errs else float("nan")


def alerts(pred: np.ndarray) -> dict:
    """Run the shipped alert logic on a closed/open sequence."""
    micro, drowsy, state = 0, 0, "alert"
    run = 0
    w = int(20 * FPS)
    csum = np.concatenate([[0], np.cumsum(pred)])
    for i, c in enumerate(pred):
        run = run + 1 if c else 0
        if run == int(1.0 * FPS):
            micro += 1
        lo = max(0, i - w + 1)
        perclos = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
        if i + 1 >= w:
            if state == "alert" and perclos > 0.15:
                state, drowsy = "drowsy", drowsy + 1
            elif state == "drowsy" and perclos < 0.08:
                state = "alert"
    return {"microsleep": micro, "drowsy": drowsy}


# -------------------------------------------------------------------------- training
def make_model(kind: str):
    if kind == "lin":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
        class_weight="balanced", random_state=0,
    )


def train_rows(vids, feats, kind):
    X, y = [], []
    for d, f in zip(vids, feats):
        keep = (f["face"] == 1) & ~d["not_visible"]
        X.append(f[kind][keep])
        y.append(d["closed"][keep])
    return np.concatenate(X), np.concatenate(y)


def pick_threshold(vids, feats, kind) -> float:
    """Inner leave-one-person-out on the training people: choose the score threshold that
    maximizes blink-event F1 on out-of-fold predictions."""
    people = sorted({d["person"] for d in vids})
    oof = [None] * len(vids)
    for p in people:
        tr = [i for i, d in enumerate(vids) if d["person"] != p]
        model = make_model(kind).fit(*train_rows([vids[i] for i in tr], [feats[i] for i in tr], kind))
        for i, d in enumerate(vids):
            if d["person"] == p:
                oof[i] = model.predict_proba(feats[i][kind])[:, 1] * (feats[i]["face"] == 1)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        tp = npred = ngt = 0
        for d, s in zip(vids, oof):
            m = event_metrics(s >= t, d["blink_id"])
            tp, npred, ngt = tp + m["tp"], npred + m["pred_events"], ngt + m["gt_blinks"]
        f1 = prf(tp, npred, ngt)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


# ---------------------------------------------------------------------------- main
def evaluate() -> dict:
    vids = load()
    feats = [features(d) for d in vids]
    people = sorted({d["person"] for d in vids})
    methods = ["fixed_ear_0.2", "deployed_rule", "mediapipe_blink", "ear_window_linear", "gbm_multifeature"]
    preds = {m: [None] * len(vids) for m in methods}
    scores = {m: [None] * len(vids) for m in methods}
    thresholds: dict[str, list] = {"ear_window_linear": [], "gbm_multifeature": []}

    for i, (d, f) in enumerate(zip(vids, feats)):
        face = f["face"] == 1
        preds["fixed_ear_0.2"][i] = face & (f["ear_ff"] < 0.2)
        scores["fixed_ear_0.2"][i] = np.where(face, -f["ear_ff"], -1e3)
        preds["deployed_rule"][i] = deployed_rule(d, f)
        scores["deployed_rule"][i] = np.where(face, -f["ear_ff"] / f["base"], -1e3)
        preds["mediapipe_blink"][i] = face & (f["blink"] > 0.5)
        scores["mediapipe_blink"][i] = np.where(face, f["blink"], -1e3)

    for p in people:
        t0 = time.perf_counter()
        tr = [i for i, d in enumerate(vids) if d["person"] != p]
        te = [i for i, d in enumerate(vids) if d["person"] == p]
        for m, kind in (("ear_window_linear", "lin"), ("gbm_multifeature", "gbm")):
            trv, trf = [vids[i] for i in tr], [feats[i] for i in tr]
            thr = pick_threshold(trv, trf, kind)
            model = make_model(kind).fit(*train_rows(trv, trf, kind))
            thresholds[m].append(thr)
            for i in te:
                s = model.predict_proba(feats[i][kind])[:, 1] * (feats[i]["face"] == 1)
                scores[m][i], preds[m][i] = s, s >= thr
        print(f"fold: test person {p} done in {time.perf_counter() - t0:.0f}s")

    hours = sum(len(d["closed"]) for d in vids) / FPS / 3600
    report: dict = {"protocol": __doc__.split("Methods")[0].strip(), "hours_of_video": round(hours, 3), "methods": {}}
    for m in methods:
        tp = npred = ngt = 0
        y_all, p_all, s_all, per_person = [], [], [], {}
        perclos_err, al = [], {"microsleep": 0, "drowsy": 0}
        for i, d in enumerate(vids):
            ev = event_metrics(preds[m][i], d["blink_id"])
            tp, npred, ngt = tp + ev["tp"], npred + ev["pred_events"], ngt + ev["gt_blinks"]
            pp = per_person.setdefault(d["person"], {"tp": 0, "pred_events": 0, "gt_blinks": 0})
            for k in pp:
                pp[k] += ev[k]
            keep = ~d["not_visible"]
            y_all.append(d["closed"][keep])
            p_all.append(preds[m][i][keep])
            s_all.append(scores[m][i][keep])
            perclos_err.append(perclos_mae(preds[m][i].astype(float), d["closed"].astype(float)))
            a = alerts(preds[m][i])
            al = {k: al[k] + a[k] for k in al}
            if any(a.values()):
                report.setdefault("alerts_by_video", {}).setdefault(m, {})[d["video"]] = a
        y, pr, sc = np.concatenate(y_all), np.concatenate(p_all), np.concatenate(s_all)
        report["methods"][m] = {
            "blink": prf(tp, npred, ngt) | {"tp": tp, "pred_events": npred, "gt_blinks": ngt},
            "blink_per_person": {k: prf(v["tp"], v["pred_events"], v["gt_blinks"]) for k, v in per_person.items()},
            "frame_closed": {
                "precision": precision_score(y, pr, zero_division=0),
                "recall": recall_score(y, pr, zero_division=0),
                "f1": f1_score(y, pr, zero_division=0),
                "roc_auc": roc_auc_score(y, sc),
                "pr_auc": average_precision_score(y, sc),
            },
            "perclos_mae_pct_points": 100 * float(np.mean(perclos_err)),
            "false_alerts_per_hour": {k: v / hours for k, v in al.items()},
        }
        if m in thresholds:
            report["methods"][m]["thresholds_per_fold"] = thresholds[m]
    # Robustness check (reporting only, not used for any selection): how much do the trained
    # models' held-out results depend on the decision threshold?
    sens = {}
    for m in ("ear_window_linear", "gbm_multifeature"):
        sens[m] = {}
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            tp = npred = ngt = 0
            perr, micro = [], 0
            for i, d in enumerate(vids):
                pr = scores[m][i] >= t
                ev = event_metrics(pr, d["blink_id"])
                tp, npred, ngt = tp + ev["tp"], npred + ev["pred_events"], ngt + ev["gt_blinks"]
                perr.append(perclos_mae(pr.astype(float), d["closed"].astype(float)))
                micro += alerts(pr)["microsleep"]
            sens[m][str(t)] = {"blink_f1": prf(tp, npred, ngt)["f1"], "perclos_mae_pts": 100 * float(np.mean(perr)), "false_microsleep_per_h": micro / hours}
    report["threshold_sensitivity"] = sens
    gt_al = {"microsleep": 0, "drowsy": 0}
    for d in vids:
        a = alerts(d["closed"])
        gt_al = {k: gt_al[k] + a[k] for k in gt_al}
    report["ground_truth_alerts"] = gt_al
    report["closed_frame_rate"] = float(np.mean(np.concatenate([d["closed"] for d in vids])))
    return report, vids, feats


def train_final(vids, feats) -> dict:
    """Fit the GBM on all four people for deployment; threshold via LOPO over all four."""
    thr = pick_threshold(vids, feats, "gbm")
    model = make_model("gbm").fit(*train_rows(vids, feats, "gbm"))
    out = ROOT / "models" / "eye_state_gbm.joblib"
    joblib.dump({"model": model, "threshold": thr, "half_window": HALF, "trained_on": "Eyeblink8 (4 people)"}, out)
    return {"path": str(out.relative_to(ROOT)), "threshold": thr}


def to_markdown(r: dict) -> str:
    names = {
        "fixed_ear_0.2": "Fixed EAR < 0.2 (textbook rule)",
        "deployed_rule": "Deployed rule (calibrated EAR, before this work)",
        "mediapipe_blink": "MediaPipe eyeBlink blendshape > 0.5",
        "ear_window_linear": "EAR window + logistic regression (trained)",
        "gbm_multifeature": "Gradient-boosted trees, multi-feature (trained)",
    }
    lines = [
        "# Eyeblink8 evaluation",
        "",
        f"{r['hours_of_video']:.2f} h of video, 4 people, leave-one-person-out. "
        f"Closed-eye frames are {100 * r['closed_frame_rate']:.2f}% of all frames.",
        "",
        "| Method | Blink P | Blink R | Blink F1 | Closed-frame F1 | PR-AUC | PERCLOS MAE (pts) | False microsleep /h | False drowsy /h |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k, m in r["methods"].items():
        b, fc, fa = m["blink"], m["frame_closed"], m["false_alerts_per_hour"]
        lines.append(
            f"| {names[k]} | {b['precision']:.3f} | {b['recall']:.3f} | **{b['f1']:.3f}** | {fc['f1']:.3f} | "
            f"{fc['pr_auc']:.3f} | {m['perclos_mae_pct_points']:.2f} | {fa['microsleep']:.1f} | {fa['drowsy']:.1f} |"
        )
    lines += ["", "Per-person blink F1:", "", "| Method | " + " | ".join(sorted(next(iter(r["methods"].values()))["blink_per_person"])) + " |",
              "|---|" + "---|" * 4]
    for k, m in r["methods"].items():
        pp = m["blink_per_person"]
        lines.append(f"| {names[k]} | " + " | ".join(f"{pp[p]['f1']:.3f}" for p in sorted(pp)) + " |")
    lines += ["", "Threshold sensitivity of the trained models (held-out scores; reporting only):", "",
              "| Method | threshold | Blink F1 | PERCLOS MAE (pts) | False microsleep /h |", "|---|---|---|---|---|"]
    for k, by_t in r.get("threshold_sensitivity", {}).items():
        for t, v in by_t.items():
            lines.append(f"| {names[k]} | {t} | {v['blink_f1']:.3f} | {v['perclos_mae_pts']:.2f} | {v['false_microsleep_per_h']:.1f} |")
    lines += ["", f"Ground-truth labels themselves would trigger: {r['ground_truth_alerts']}", "", "## Protocol", "", r["protocol"]]
    return "\n".join(lines)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    report, vids, feats = evaluate()
    report["deployed_model"] = train_final(vids, feats)
    (RESULTS / "eyeblink8.json").write_text(json.dumps(report, indent=2, default=float))
    md = to_markdown(report)
    (RESULTS / "eyeblink8.md").write_text(md)
    print(md)
