"""Design a better drowsiness alert on development data, freeze it, then test it once on
people it has never seen.

Why: on RLDD fold 3 part 2, the shipped alert (PERCLOS > 15% over 20 s, from the
Eyeblink8-trained eye model) rose with drowsiness on average but fired more when drowsy for
only 3 of 6 people, while the *duration* of eye closures separated states much better.
Candidate: alert when a driver has >= K eye closures of >= L seconds within 60 s.

Protocol (pre-registered):
1. ``select``: choose detector (MediaPipe score vs trained model), L and K using ONLY the
   development people (Fold3_part2) plus Eyeblink8 (awake people) as a false-alarm check.
   Primary objective: false alerts <= 1 per hour on both the alert videos and Eyeblink8
   (the Eyeblink8 limit was added after the first selection run picked a rule that
   alerted 4.5 times/h on awake people; no test data had been touched); then most drowsy
   videos with >= 1 alert; then most people alerted more when drowsy; then fewest false
   alerts; then the highest alert rate on drowsy videos. The most sensitive rule
   without the Eyeblink8 limit is also frozen, as a clearly labelled *exploratory* comparison.
   Also fix the drowsiness classifier configuration (logistic regression on 1-min windows).
   Everything is written to eval/results/preregistration.json and committed to git
   *before* the test data is processed.
2. ``test``: apply the frozen rule and classifier, unchanged, to new people (Fold5_part1)
   and report the shipped baseline next to it. No further tuning.

    .venv/bin/python -m eval.drowsy_rule select
    .venv/bin/python -m eval.drowsy_rule test
"""

from __future__ import annotations

import json
import sys
from collections import deque
from datetime import date

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drivemind.config import ROOT
from eval.eyeblink8_eval import FPS, alerts, runs
from eval.eyeblink8_eval import features as eb8_features
from eval.eyeblink8_eval import load as load_eyeblink8
from eval.rldd_eval import WIN_S, eye_methods, load_videos, wilson, window_features

DEV, TEST = "Fold3_part2", "Fold5_part1"
PREREG = ROOT / "eval" / "results" / "preregistration.json"
WINDOW_S, COOLDOWN_S, MICROSLEEP_S = 60.0, 60.0, 1.0


# ----------------------------------------------------------------------------- rules
def long_closure_alerts(closed: np.ndarray, min_dur_s: float, k: int) -> list[float]:
    """Alert times (s): >= k closures lasting >= min_dur_s within the last WINDOW_S, at most
    one alert per COOLDOWN_S. Microsleeps (closure >= 1 s) alert immediately, as before."""
    out, recent, last = [], deque(), -1e9
    for s, e in runs(closed):
        dur, t = (e - s + 1) / FPS, (e + 1) / FPS
        if dur < min_dur_s:
            continue
        recent.append(t)
        while recent and t - recent[0] > WINDOW_S:
            recent.popleft()
        if (len(recent) >= k or dur >= MICROSLEEP_S) and t - last >= COOLDOWN_S:
            out.append(t)
            last = t
    return out


def shipped_alerts(closed: np.ndarray) -> int:
    a = alerts(closed)  # PERCLOS > 15% / 20 s drowsy state + microsleeps, as shipped
    return a["microsleep"] + a["drowsy"]


def score_rule(videos: list[dict], fire) -> dict:
    """fire(video) -> number of alerts. Summarize per state and within person."""
    per = {}
    for v in videos:
        n = fire(v)
        per[(v["pid"], v["label"])] = (n, len(v["face"]) / FPS / 3600, v)
    alert_h = sum(h for (p, lab), (n, h, _) in per.items() if lab == 0)
    false_alerts = sum(n for (p, lab), (n, h, _) in per.items() if lab == 0)
    pids = sorted({p for p, _ in per})
    detected = sum(per[(p, 10)][0] > 0 for p in pids if (p, 10) in per)
    within = sum(per[(p, 10)][0] / per[(p, 10)][1] > per[(p, 0)][0] / per[(p, 0)][1] for p in pids if (p, 0) in per and (p, 10) in per)
    rate = {lab: sum(n for (p, l2), (n, h, _) in per.items() if l2 == lab) / max(sum(h for (p, l2), (n, h, _) in per.items() if l2 == lab), 1e-9) for lab in (0, 5, 10)}
    return {
        "false_alerts_per_h_on_alert_videos": false_alerts / alert_h,
        "drowsy_videos_with_alert": f"{detected}/{len(pids)}",
        "drowsy_detected": detected,
        "people_alerted_more_when_drowsy": f"{within}/{len(pids)}",
        "within": within,
        "alerts_per_h_by_state": {"alert": rate[0], "low vigilant": rate[5], "drowsy": rate[10]},
        "per_video": {f"{p}/{lab}": n for (p, lab), (n, h, _) in sorted(per.items())},
    }


def prepare(folds: set[str], bundle) -> list[dict]:
    vids = load_videos(folds=folds)
    for v in vids:
        meths, f = eye_methods(v, bundle)
        v["closed"], v["f"] = meths, f
    return vids


def eyeblink8_closed(bundle) -> dict[str, list[np.ndarray]]:
    """Per-detector eye-closed sequences for the Eyeblink8 videos (awake people)."""
    out = {"mediapipe_blink": [], "trained_gbm": []}
    for d in load_eyeblink8():
        f = eb8_features(d)
        face = f["face"] == 1
        out["mediapipe_blink"].append(face & (f["blink"] > 0.5))
        out["trained_gbm"].append((bundle["model"].predict_proba(f["gbm"])[:, 1] * face) >= bundle["threshold"])
    return out


def eyeblink8_false_alerts(eb8: dict, detector: str, min_dur_s: float, k: int) -> float:
    n = sum(len(long_closure_alerts(c, min_dur_s, k)) for c in eb8[detector])
    return n / (sum(len(c) for c in eb8[detector]) / FPS / 3600)


# ------------------------------------------------------------------------ classifier
def make_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))


def windows_of(vids: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, pid = [], [], []
    w = int(WIN_S * FPS)
    for v in vids:
        for s in range(0, len(v["face"]) - w + 1, w):
            X.append(window_features(v["closed"]["trained_gbm"], v["f"], v, s, s + w))
            y.append(v["label"])
            pid.append(v["pid"])
    return np.array(X), np.array(y), np.array(pid)


# ---------------------------------------------------------------------------- modes
def select() -> None:
    bundle = joblib.load(ROOT / "models" / "eye_state_gbm.joblib")
    dev = prepare({DEV}, bundle)
    baseline = score_rule(dev, lambda v: shipped_alerts(v["closed"]["trained_gbm"]))
    eb8 = eyeblink8_closed(bundle)
    grid = []
    for det in ("mediapipe_blink", "trained_gbm"):
        for L in (0.3, 0.4, 0.5, 0.7, 1.0):
            for K in (1, 2, 3, 4, 5, 6, 8):
                s = score_rule(dev, lambda v, det=det, L=L, K=K: len(long_closure_alerts(v["closed"][det], L, K)))
                grid.append({"detector": det, "min_closure_s": L, "k_per_60s": K, **s,
                             "eyeblink8_false_alerts_per_h": eyeblink8_false_alerts(eb8, det, L, K)})

    def key(g):
        return (g["drowsy_detected"], g["within"], -g["false_alerts_per_h_on_alert_videos"],
                -g["eyeblink8_false_alerts_per_h"], g["alerts_per_h_by_state"]["drowsy"])

    def as_rule(g):
        return {"detector": g["detector"], "min_closure_s": g["min_closure_s"], "k_per_60s": g["k_per_60s"],
                "window_s": WINDOW_S, "cooldown_s": COOLDOWN_S, "microsleep_s": MICROSLEEP_S}

    ok = [g for g in grid if g["false_alerts_per_h_on_alert_videos"] <= 1.0 and g["eyeblink8_false_alerts_per_h"] <= 1.0]
    best = max(ok, key=key)
    exploratory = max((g for g in grid if g["false_alerts_per_h_on_alert_videos"] <= 1.0), key=key)
    rule = as_rule(best)
    prereg = {
        "date": str(date.today()),
        "development_data": f"UTA-RLDD {DEV} (participants {sorted({v['pid'] for v in dev})})",
        "test_data": f"UTA-RLDD {TEST} (not yet downloaded/processed when this file was written)",
        "frozen_rule": rule,
        "exploratory_rule": as_rule(exploratory),
        "dev_result_exploratory_rule": {k: v for k, v in exploratory.items() if k not in ("detector", "min_closure_s", "k_per_60s")},
        "selection_objective": "false alerts <= 1/h on dev alert videos AND on Eyeblink8 awake people; then max drowsy videos alerted; then max people alerted more when drowsy; then fewest false alerts; then highest drowsy alert rate",
        "objective_revision_note": "First selection run (dev-only limit) picked mediapipe_blink L=0.7 K=1, which alerted 4.5/h on Eyeblink8 awake people; the Eyeblink8 limit was added before any test data was downloaded or processed. That rule is kept as the exploratory comparison.",
        "dev_result_frozen_rule": {k: v for k, v in best.items() if k not in ("detector", "min_closure_s", "k_per_60s")},
        "dev_result_shipped_baseline": baseline,
        "eyeblink8_awake_false_alerts_per_h": best["eyeblink8_false_alerts_per_h"],
        "frozen_classifier": {"model": "StandardScaler + LogisticRegression(C=0.5, class_weight=balanced)",
                              "features": "eval.rldd_eval.WINDOW_FEATURES on 60 s windows (trained eye model)",
                              "train_on": DEV, "video_decision": "soft vote (mean class probability over windows)"},
        "grid_size": len(grid),
        "candidates_meeting_false_alert_limit": len(ok),
    }
    PREREG.write_text(json.dumps(prereg, indent=2, default=float))
    show = lambda d: {k: d[k] for k in ("false_alerts_per_h_on_alert_videos", "eyeblink8_false_alerts_per_h", "drowsy_videos_with_alert", "people_alerted_more_when_drowsy", "alerts_per_h_by_state") if k in d}  # noqa: E731
    print(json.dumps({"frozen_rule": rule, "dev": show(best), "exploratory_rule": prereg["exploratory_rule"], "dev_exploratory": show(exploratory),
                      "shipped_baseline_dev": show(baseline)}, indent=2, default=float))


def test(test_fold: str = TEST, extra_train: tuple[str, ...] = ()) -> None:
    """Evaluate the frozen rules on `test_fold`. The classifier is always trained on the
    development people; with `extra_train`, a second classifier is also trained on the
    development + those already-tested people (a learning-curve point: does more data help?)."""
    if not PREREG.exists():
        sys.exit("Run `select` first: the rule must be frozen before looking at test data.")
    prereg = json.loads(PREREG.read_text())
    rule = prereg["frozen_rule"]
    bundle = joblib.load(ROOT / "models" / "eye_state_gbm.joblib")
    dev, tst = prepare({DEV}, bundle), prepare({test_fold}, bundle)
    seen = dev + (prepare(set(extra_train), bundle) if extra_train else [])
    assert tst and not {v["pid"] for v in seen} & {v["pid"] for v in tst}, "test people must be new"

    new = score_rule(tst, lambda v: len(long_closure_alerts(v["closed"][rule["detector"]], rule["min_closure_s"], rule["k_per_60s"])))
    ex = prereg["exploratory_rule"]
    expl = score_rule(tst, lambda v: len(long_closure_alerts(v["closed"][ex["detector"]], ex["min_closure_s"], ex["k_per_60s"])))
    base = score_rule(tst, lambda v: shipped_alerts(v["closed"]["trained_gbm"]))

    Xt, yt, pt = windows_of(tst)
    train_sets = {f"{len({v['pid'] for v in dev})} development people": dev}
    if extra_train:
        train_sets[f"{len({v['pid'] for v in seen})} people (development + earlier test)"] = seen
    classifiers = {}
    for name, vids in train_sets.items():
        Xd, yd, _ = windows_of(vids)
        proba = make_classifier().fit(Xd, yd).predict_proba(Xt)
        rows, correct = [], 0
        for p in np.unique(pt):
            for lab in (0, 5, 10):
                m = (pt == p) & (yt == lab)
                if m.any():
                    vote = int(np.array([0, 5, 10])[proba[m].mean(0).argmax()])
                    correct += vote == lab
                    rows.append((p, lab, vote))
        lo, hi = wilson(int(correct), len(rows))
        ad = (yt == 0) | (yt == 10)
        classifiers[name] = {
            "video_accuracy_3class": f"{correct}/{len(rows)} = {correct / len(rows):.1%} (95% CI {lo:.0%}-{hi:.0%})",
            "window_auc_alert_vs_drowsy": float(roc_auc_score(yt[ad] == 10, proba[ad, 2] - proba[ad, 0])),
            "video_predictions": rows,
        }
    first = next(iter(classifiers.values()))
    result = {
        "test_fold": test_fold,
        "preregistration": prereg,
        "test_people": sorted({v["pid"] for v in tst}),
        "test_hours": sum(len(v["face"]) for v in tst) / FPS / 3600,
        "test_source_fps": {f"{v['pid']}/{v['label']}": round(v["fps"], 1) for v in tst},
        "frozen_rule_on_test": new,
        "shipped_baseline_on_test": base,
        "exploratory_rule_on_test": expl,
        "classifier_on_test": first,
        "classifiers_on_test": classifiers,
    }
    stem = "rldd_heldout_test" if test_fold == TEST else f"rldd_heldout_test_{test_fold}"
    (ROOT / "eval" / "results" / f"{stem}.json").write_text(json.dumps(result, indent=2, default=float))
    md = to_markdown(result)
    (ROOT / "eval" / "results" / f"{stem}.md").write_text(md)
    print(md)


def pooled() -> None:
    """Combine every held-out test run so far (each test person counted once)."""
    runs = [json.loads(p.read_text()) for p in sorted((ROOT / "eval" / "results").glob("rldd_heldout_test*.json"))]
    L = ["# Frozen drowsiness rules, pooled over all held-out tests", "",
         f"{len(runs)} test sets, {sum(len(r['test_people']) for r in runs)} unseen people, "
         f"{sum(r['test_hours'] for r in runs):.1f} h.", "",
         "| Rule | False alerts on alert videos | Drowsy videos with >= 1 alert (95% CI) | People alerted more when drowsy |", "|---|---|---|---|"]
    for key, name in (("shipped_baseline_on_test", "Previous rule (PERCLOS > 15% + microsleep)"),
                      ("frozen_rule_on_test", "Current rule (shipped since 2026-10-03)"),
                      ("exploratory_rule_on_test", "Exploratory: any closure >= 0.7 s")):
        fa = sum(n for r in runs for k, n in r[key]["per_video"].items() if k.endswith("/0"))
        det = sum(int(r[key]["drowsy_videos_with_alert"].split("/")[0]) for r in runs)
        n = sum(int(r[key]["drowsy_videos_with_alert"].split("/")[1]) for r in runs)
        within = sum(int(r[key]["people_alerted_more_when_drowsy"].split("/")[0]) for r in runs)
        lo, hi = wilson(det, n)
        L.append(f"| {name} | {fa} alerts | {det}/{n} ({lo:.0%}-{hi:.0%}) | {within}/{n} |")
    md = "\n".join(L)
    (ROOT / "eval" / "results" / "rldd_heldout_pooled.md").write_text(md)
    print(md)


def to_markdown(r: dict) -> str:
    rule = r["preregistration"]["frozen_rule"]
    det = {"mediapipe_blink": "MediaPipe eyeBlink score", "trained_gbm": "trained eye model"}[rule["detector"]]
    L = [
        f"# Held-out test: frozen drowsiness rule on new people (UTA-RLDD {r.get('test_fold', TEST)})", "",
        f"Rule frozen on {r['preregistration']['date']} using only {r['preregistration']['development_data']} "
        f"(see `preregistration.json`, committed before this test ran): alert when the {det} shows "
        f">= {rule['k_per_60s']} eye closures of >= {rule['min_closure_s']} s within {rule['window_s']:.0f} s, "
        f"or one closure >= {rule['microsleep_s']} s; at most one alert per {rule['cooldown_s']:.0f} s.", "",
        f"Test: {len(r['test_people'])} new people ({', '.join(r['test_people'])}), {r['test_hours']:.1f} h.", "",
        "| Rule | False alerts/h (alert videos) | Drowsy videos with >= 1 alert | People alerted more when drowsy | Alerts/h: alert / low vigilant / drowsy |",
        "|---|---|---|---|---|",
    ]
    ex = r["preregistration"]["exploratory_rule"]
    for name, s in (("Shipped (PERCLOS > 15% + microsleep)", r["shipped_baseline_on_test"]), ("**New frozen rule (primary)**", r["frozen_rule_on_test"]),
                    (f"Exploratory: any closure >= {ex['min_closure_s']} s x{ex['k_per_60s']} ({ex['detector']})", r["exploratory_rule_on_test"])):
        a = s["alerts_per_h_by_state"]
        L.append(f"| {name} | {s['false_alerts_per_h_on_alert_videos']:.1f} | {s['drowsy_videos_with_alert']} | "
                 f"{s['people_alerted_more_when_drowsy']} | {a['alert']:.1f} / {a['low vigilant']:.1f} / {a['drowsy']:.1f} |")
    dv = r["preregistration"]["dev_result_frozen_rule"]
    L += ["", f"On the development people the frozen rule scored: {dv['false_alerts_per_h_on_alert_videos']:.1f} false alerts/h, "
          f"{dv['drowsy_videos_with_alert']} drowsy videos alerted, {dv['people_alerted_more_when_drowsy']} people alerted more when drowsy. "
          f"Eyeblink8 (awake people) check: {r['preregistration']['eyeblink8_awake_false_alerts_per_h']:.1f} alerts/h.", "",
          "## Drowsiness classifier, tested on the new people", "",
          "RLDD paper on its own test folds: HM-LSTM 65.2%, human judges 57.8%, chance 33%.", "",
          "| Trained on | Video accuracy (3-class) | Window ROC-AUC alert vs drowsy |", "|---|---|---|"]
    for name, c in r.get("classifiers_on_test", {"6 development people": r["classifier_on_test"]}).items():
        L.append(f"| {name} | {c['video_accuracy_3class']} | {c['window_auc_alert_vs_drowsy']:.2f} |")
    L += ["",
          "| Participant | State | Predicted | Source fps | Alerts (new rule) | Alerts (shipped) | Alerts (exploratory) |", "|---|---|---|---|---|---|---|"]
    names = {0: "alert", 5: "low vigilant", 10: "drowsy"}
    for p, lab, vote in r["classifier_on_test"]["video_predictions"]:
        key = f"{p}/{lab}"
        L.append(f"| {p} | {names[lab]} | {names[vote]} | {r['test_source_fps'].get(key, '?')} | "
                 f"{r['frozen_rule_on_test']['per_video'].get(key, '?')} | {r['shipped_baseline_on_test']['per_video'].get(key, '?')} | "
                 f"{r['exploratory_rule_on_test']['per_video'].get(key, '?')} |")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "test", "pooled"])
    ap.add_argument("--test-fold", default=TEST)
    ap.add_argument("--extra-train", nargs="*", default=[], help="already-tested folds to add to classifier training")
    a = ap.parse_args()
    {"select": select, "pooled": pooled}.get(a.mode, lambda: test(a.test_fold, tuple(a.extra_train)))()
