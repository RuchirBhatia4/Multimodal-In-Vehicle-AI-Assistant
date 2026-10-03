"""Drowsiness evaluation on UTA-RLDD (one fold part: 6 participants x alert/low-vigilant/drowsy).

Three questions, from most to least product-relevant:

A. **Shipped logic, zero RLDD training.** Run the app's eye-closure detector (trained only
   on Eyeblink8) + its alert rules on every video. Do microsleep/drowsy alerts and PERCLOS
   go up when people report being drowsy? This is an out-of-domain test: different people,
   cameras (phones/webcams, < 30 fps), lighting, and real drowsiness.
B. **Separability.** ROC-AUC of per-video eye metrics for alert (0) vs drowsy (10), and the
   *within-person* view: does each person's drowsy video score higher than their alert one?
C. **Trained drowsiness classifier.** 1-minute windows -> blink/eye features -> classifier,
   leave-one-participant-out. Video label = majority vote of its windows (the paper's
   "video accuracy"). Compared against the RLDD paper's HM-LSTM (65.2% overall, 70% on
   fold 3) and human judges (57.8% overall, 60% on fold 3). Not directly comparable:
   the paper trained on 48 people and tested on 12; we train on 5 and test on 1, using
   only half of fold 3.

    .venv/bin/python -m eval.rldd_eval
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drivemind.config import ROOT
from eval.eyeblink8_eval import FPS, alerts, deployed_rule, features, runs

FEAT_DIR = ROOT / "data" / "features"
RESULTS = ROOT / "eval" / "results"
LABELS = {0: "alert", 5: "low vigilant", 10: "drowsy"}
COLS = ["face", "ear_l", "ear_r", "mar", "yaw", "pitch", "eyeBlinkLeft", "eyeBlinkRight",
        "eyeSquintLeft", "eyeSquintRight", "eyeLookDownLeft", "eyeLookDownRight"]
WIN_S = 60


# ------------------------------------------------------------------------------ data
def fold_of(d: dict) -> str:
    return str(d["fold"]) if "fold" in d else "Fold3_part2"  # extracted before the field existed


def load_videos(cap_fps: float | None = None, folds: set[str] | None = None) -> list[dict]:
    """Group per-file features by (participant, label), concatenating split recordings.
    cap_fps emulates a slower camera by keeping only the first frame in each 1/cap_fps slot:
    the control for frame rate differing between recordings."""
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for p in FEAT_DIR.glob("rldd_*.npz"):
        m = re.match(r"rldd_(\d+)_(\d+)(_\d+)?$", p.stem)
        d = dict(np.load(p, allow_pickle=True))
        if folds and fold_of(d) not in folds:
            continue
        groups[(m.group(1), int(m.group(2)))].append((m.group(3) or "", d))
    vids = []
    for (pid, label), parts in sorted(groups.items()):
        parts.sort(key=lambda x: x[0])
        t_off, cat = 0.0, defaultdict(list)
        for _, d in parts:
            t = d["t"].astype(np.float64)
            cat["t"].append(t - t[0] + t_off)
            t_off = cat["t"][-1][-1] + 1.0 / float(d["fps"])
            for c in COLS:
                cat[c].append(d[c])
        raw = {k: np.concatenate(v) for k, v in cat.items()}
        if cap_fps:
            slot = np.floor(raw["t"] * cap_fps)
            keep = np.concatenate([[True], slot[1:] != slot[:-1]])
            raw = {k: v[keep] for k, v in raw.items()}
        vids.append({"pid": pid, "label": label, "fold": fold_of(parts[0][1]), "fps": float(parts[0][1]["fps"]), **resample_30fps(raw)})
    return vids


def resample_30fps(raw: dict) -> dict:
    """Put every video on a 30 fps grid so the eye model's 13-frame window always spans the
    same 0.43 s it was trained on. Continuous features are linearly interpolated between
    face frames; a grid point has a face only if its nearest source frame did."""
    t = raw["t"]
    grid = np.arange(t[0], t[-1], 1.0 / FPS)
    nearest = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    prev = np.clip(nearest - 1, 0, len(t) - 1)
    nearest = np.where(np.abs(t[prev] - grid) < np.abs(t[nearest] - grid), prev, nearest)
    face = raw["face"] > 0.5
    out = {"face": face[nearest].astype(np.float32)}
    for c in COLS[1:]:
        v = raw[c].astype(np.float64)
        ok = face & ~np.isnan(v)
        out[c] = np.interp(grid, t[ok], v[ok]).astype(np.float32) if ok.sum() > 1 else np.full(len(grid), np.nan, np.float32)
        out[c][~face[nearest]] = np.nan
    return out


# ------------------------------------------------------------------- eye decisions
def eye_methods(d: dict, gbm_bundle) -> tuple[dict, dict]:
    f = features(d)
    face = f["face"] == 1
    p = gbm_bundle["model"].predict_proba(f["gbm"])[:, 1] * face
    return {
        "original_rule": deployed_rule(d, f),
        "mediapipe_blink": face & (f["blink"] > 0.5),
        "trained_gbm": p >= gbm_bundle["threshold"],
    }, f


def blink_stats(closed: np.ndarray) -> dict:
    ev = runs(closed)
    durs = np.array([(e - s + 1) / FPS for s, e in ev]) if ev else np.zeros(0)
    minutes = len(closed) / FPS / 60
    return {
        "perclos": float(closed.mean()),
        "blinks_per_min": len(ev) / minutes if minutes else 0.0,
        "mean_blink_s": float(durs.mean()) if len(durs) else 0.0,
        "long_closures_per_min": float((durs >= 0.5).sum() / minutes) if minutes else 0.0,
    }


def yawns(mar: np.ndarray) -> int:
    open_ = np.nan_to_num(mar) > 0.55
    return sum(1 for s, e in runs(open_, max_gap=3) if (e - s + 1) / FPS > 1.2)


def window_features(closed: np.ndarray, f: dict, d: dict, s: int, e: int) -> list[float]:
    c = closed[s:e]
    b = blink_stats(c)
    ev = runs(c)
    durs = np.array([(q - p + 1) / FPS for p, q in ev]) if ev else np.zeros(1)
    face = f["face"][s:e] == 1
    ear_n = (f["ear_ff"][s:e] / f["base"])[face] if face.any() else np.zeros(1)
    pitch = np.nan_to_num(d["pitch"][s:e])
    return [
        b["perclos"], b["blinks_per_min"], b["mean_blink_s"], float(durs.std()), b["long_closures_per_min"],
        float(np.mean(ear_n)), float(np.std(ear_n)), float(np.nanmean(f["blink"][s:e])),
        float(yawns(d["mar"][s:e])), float(np.mean(pitch)), float(np.std(pitch)), float(face.mean()),
    ]


WINDOW_FEATURES = ["perclos", "blinks_per_min", "mean_blink_s", "std_blink_s", "long_closures_per_min",
                   "mean_ear_norm", "std_ear_norm", "mean_blink_score", "yawns", "pitch_mean", "pitch_std", "face_frac"]


# --------------------------------------------------------------------------- main
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (honest error bars for tiny n)."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main(cap_fps: float | None = None) -> None:
    bundle = joblib.load(ROOT / "models" / "eye_state_gbm.joblib")
    vids = load_videos(cap_fps, folds={"Fold3_part2"})
    report: dict = {"videos": [], "protocol": __doc__.strip(), "cap_fps": cap_fps}
    per_video: dict[str, list] = defaultdict(list)
    windows: list[dict] = []

    for d in vids:
        meths, f = eye_methods(d, bundle)
        hours = len(d["face"]) / FPS / 3600
        row = {"pid": d["pid"], "label": d["label"], "minutes": round(hours * 60, 1), "source_fps": round(d["fps"], 1),
               "face_found": round(float(np.mean(d["face"])), 3), "yawns_per_h": yawns(d["mar"]) / hours}
        for m, closed in meths.items():
            a = alerts(closed)
            row[m] = {**blink_stats(closed), "microsleep_per_h": a["microsleep"] / hours, "drowsy_alerts_per_h": a["drowsy"] / hours}
        report["videos"].append(row)
        per_video[d["pid"]].append(row)
        w = int(WIN_S * FPS)
        for s in range(0, len(d["face"]) - w + 1, w):
            windows.append({"pid": d["pid"], "label": d["label"], "x": window_features(meths["trained_gbm"], f, d, s, s + w)})

    # ---- A + B: shipped logic, per class, AUC and within-person ordering
    summary = {}
    for m in ("original_rule", "mediapipe_blink", "trained_gbm"):
        by_cls = {}
        for lab in (0, 5, 10):
            rows = [r for r in report["videos"] if r["label"] == lab]
            by_cls[LABELS[lab]] = {k: float(np.mean([r[m][k] for r in rows])) for k in rows[0][m]}
        auc, within = {}, {}
        for k in ("perclos", "mean_blink_s", "long_closures_per_min", "microsleep_per_h", "drowsy_alerts_per_h"):
            r0 = [r for r in report["videos"] if r["label"] == 0]
            r10 = [r for r in report["videos"] if r["label"] == 10]
            y = [0] * len(r0) + [1] * len(r10)
            auc[k] = float(roc_auc_score(y, [r[m][k] for r in r0 + r10]))
            pairs = [(next(r for r in rs if r["label"] == 0)[m][k], next(r for r in rs if r["label"] == 10)[m][k])
                     for rs in per_video.values() if {0, 10} <= {r["label"] for r in rs}]
            within[k] = f"{sum(b > a for a, b in pairs)}/{len(pairs)} people higher when drowsy"
        summary[m] = {"by_class_mean": by_cls, "auc_alert_vs_drowsy": auc, "within_person": within}
    report["shipped_logic"] = summary

    # ---- C: trained window classifier, leave-one-participant-out
    X = np.array([w["x"] for w in windows])
    y = np.array([w["label"] for w in windows])
    pids = np.array([w["pid"] for w in windows])
    variants = {}
    for norm in ("global", "per_driver"):
        Xn = X.copy()
        if norm == "per_driver":
            # Label-free normalization: z-score each driver's windows against *their own*
            # recordings. Needs an unlabeled history of the driver, as a car would have.
            for p in np.unique(pids):
                m_ = pids == p
                Xn[m_] = (Xn[m_] - Xn[m_].mean(0)) / (Xn[m_].std(0) + 1e-6)
        for name, make in (
            ("logreg", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))),
            ("gbm", lambda: HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05, class_weight="balanced", random_state=0)),
        ):
            proba = np.zeros((len(y), 3))
            for p in np.unique(pids):
                tr, te = pids != p, pids == p
                model = make().fit(Xn[tr], y[tr])
                proba[te] = model.predict_proba(Xn[te])
            pred = np.array([0, 5, 10])[proba.argmax(1)]
            vid_correct, vid_rows = 0, []
            for p in np.unique(pids):
                for lab in (0, 5, 10):
                    m_ = (pids == p) & (y == lab)
                    if not m_.any():
                        continue
                    vote = np.array([0, 5, 10])[proba[m_].mean(0).argmax()]  # soft vote over windows
                    vid_correct += vote == lab
                    vid_rows.append((p, lab, int(vote)))
            ad = (y == 0) | (y == 10)
            lo, hi = wilson(int(vid_correct), len(vid_rows))
            variants[f"{name}_{norm}"] = {
                "window_accuracy_3class": float((pred == y).mean()),
                "video_accuracy_3class": f"{vid_correct}/{len(vid_rows)} = {vid_correct / len(vid_rows):.1%} (95% CI {lo:.0%}-{hi:.0%})",
                "window_auc_alert_vs_drowsy": float(roc_auc_score(y[ad] == 10, proba[ad, 2] - proba[ad, 0])),
                "video_predictions": vid_rows,
            }
    r0 = [v for v in report["videos"] if v["label"] == 0]
    r10 = [v for v in report["videos"] if v["label"] == 10]
    report["confound_source_fps_auc_alert_vs_drowsy"] = float(roc_auc_score([0] * len(r0) + [1] * len(r10), [v["source_fps"] for v in r0 + r10]))
    report["window_classifier"] = {"n_windows": int(len(y)), "features": WINDOW_FEATURES, "results": variants}
    report["paper_baselines"] = {"hm_lstm_video_accuracy": {"all_folds": 0.652, "fold3": 0.70},
                                 "human_judgment_video_accuracy": {"all_folds": 0.578, "fold3": 0.60}, "chance": 1 / 3}

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"_fps{int(cap_fps)}" if cap_fps else ""
    (RESULTS / f"rldd_fold3_part2{tag}.json").write_text(json.dumps(report, indent=2, default=float))
    (RESULTS / f"rldd_fold3_part2{tag}.md").write_text(to_markdown(report))
    print(to_markdown(report))


def to_markdown(r: dict) -> str:
    names = {"original_rule": "Original calibrated-EAR rule", "mediapipe_blink": "MediaPipe eyeBlink > 0.5", "trained_gbm": "Trained eye model (shipped)"}
    vids = r["videos"]
    title = "# UTA-RLDD drowsiness evaluation (fold 3, part 2: 6 participants)"
    if r.get("cap_fps"):
        title += f", every video degraded to {r['cap_fps']:.0f} fps (frame-rate control)"
    L = [title, "",
         f"Frame-rate confound check: source fps alone separates alert vs drowsy videos with ROC-AUC "
         f"{r['confound_source_fps_auc_alert_vs_drowsy']:.2f} (0.5 = no confound).", "",
         f"{len(vids)} videos, {sum(v['minutes'] for v in vids) / 60:.1f} h, source frame rates "
         f"{min(v['source_fps'] for v in vids):.0f}-{max(v['source_fps'] for v in vids):.0f} fps, face found "
         f"{min(v['face_found'] for v in vids):.0%}-{max(v['face_found'] for v in vids):.0%} of frames.", "",
         "## A. Shipped logic, no RLDD training (means per self-reported state)", "",
         "| Eye detector | State | PERCLOS | Blinks/min | Mean blink (s) | Long closures/min | Microsleep alerts/h | Drowsy alerts/h |",
         "|---|---|---|---|---|---|---|---|"]
    for m, s in r["shipped_logic"].items():
        for st, v in s["by_class_mean"].items():
            L.append(f"| {names[m]} | {st} | {v['perclos']:.1%} | {v['blinks_per_min']:.1f} | {v['mean_blink_s']:.2f} | "
                     f"{v['long_closures_per_min']:.2f} | {v['microsleep_per_h']:.1f} | {v['drowsy_alerts_per_h']:.1f} |")
    L += ["", "## B. Alert vs. drowsy separability (per video; 6 vs 6 videos, so treat as indicative)", "",
          "| Eye detector | Metric | ROC-AUC | Within-person |", "|---|---|---|---|"]
    for m, s in r["shipped_logic"].items():
        for k, a in s["auc_alert_vs_drowsy"].items():
            L.append(f"| {names[m]} | {k} | {a:.2f} | {s['within_person'][k]} |")
    L += ["", "## C. Trained drowsiness classifier (1-min windows, leave-one-participant-out)", "",
          f"{r['window_classifier']['n_windows']} windows. Paper baselines (video accuracy, 3-class): HM-LSTM 65.2% "
          "overall / 70% on fold 3; human judges 57.8% / 60%; chance 33%.", "",
          "| Model | Window acc. (3-class) | Video acc. (3-class) | Window AUC alert vs drowsy |", "|---|---|---|---|"]
    for k, v in r["window_classifier"]["results"].items():
        L.append(f"| {k} | {v['window_accuracy_3class']:.1%} | {v['video_accuracy_3class']} | {v['window_auc_alert_vs_drowsy']:.2f} |")
    L += ["", "## Per-video detail (trained eye model)", "",
          "| Participant | State | Minutes | Source fps | Face found | PERCLOS | Mean blink (s) | Microsleep/h | Drowsy alerts/h | Yawns/h |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for v in vids:
        g = v["trained_gbm"]
        L.append(f"| {v['pid']} | {LABELS[v['label']]} | {v['minutes']} | {v['source_fps']} | {v['face_found']:.0%} | {g['perclos']:.1%} | "
                 f"{g['mean_blink_s']:.2f} | {g['microsleep_per_h']:.1f} | {g['drowsy_alerts_per_h']:.1f} | {v['yawns_per_h']:.1f} |")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-fps", type=float, default=None, help="degrade every video to this frame rate first")
    main(ap.parse_args().cap_fps)
