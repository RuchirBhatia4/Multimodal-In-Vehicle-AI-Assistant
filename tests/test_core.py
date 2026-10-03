"""Fast unit tests for the deterministic parts (no model weights needed except YOLO's
tiny tracker state). Run: .venv/bin/python -m pytest -q"""

import numpy as np
import pytest

from drivemind.brain.local_brain import _extract_json
from drivemind.brain.safety import AttentionManager
from drivemind.brain.tools import CarState, execute_vehicle_tool
from drivemind.perception.driver import LEFT_EYE, _ear


def test_ttc_from_looming_matches_ground_truth():
    """Constant closing speed: distance Z = v (T0 - t), image scale s = f W / Z, so
    s(t) = c / (T0 - t) and true TTC(t) = T0 - t. Recover it from scale alone."""
    from drivemind.perception.road import RoadPerception

    rp = RoadPerception.__new__(RoadPerception)  # skip loading YOLO; only test the math
    rp.histories, rp.ttc_window_s = {}, 0.8
    T0, fps = 4.0, 15
    for i in range(int(2.0 * fps)):
        t = i / fps
        est = rp._ttc(1, t, 100.0 / (T0 - t))
        truth = T0 - t
        if t >= 0.5:  # once the window is full
            assert est is not None
            assert abs(est - truth) < 0.15, (t, est, truth)


def test_static_object_has_no_ttc():
    from drivemind.perception.road import RoadPerception

    rp = RoadPerception.__new__(RoadPerception)
    rp.histories, rp.ttc_window_s = {}, 0.8
    rng = np.random.default_rng(0)
    out = [rp._ttc(1, i / 15, 100 + rng.normal(0, 0.5)) for i in range(20)]
    assert out[-1] is None


def test_ear_open_vs_closed():
    pts = np.zeros((478, 2))
    # p1..p6 of an open eye: width 30, openings 10
    p = {0: (0, 0), 1: (10, -5), 2: (20, -5), 3: (30, 0), 4: (20, 5), 5: (10, 5)}
    for j, idx in enumerate(LEFT_EYE):
        pts[idx] = p[j]
    open_ear = _ear(pts, LEFT_EYE)
    for j in (1, 2):
        pts[LEFT_EYE[j]][1] = -0.5
    for j in (4, 5):
        pts[LEFT_EYE[j]][1] = 0.5
    closed_ear = _ear(pts, LEFT_EYE)
    assert open_ear == pytest.approx(10 / 30, rel=1e-3)
    assert closed_ear < 0.05


def test_safety_envelope_clamps_and_blocks():
    car = CarState(speed_mph=70)
    assert "outside" in execute_vehicle_tool(car, "set_temperature", {"temperature_f": 200})
    assert car.temperature_f == 85
    msg = execute_vehicle_tool(car, "set_window", {"window": "all", "position": "open"})
    assert "blocked" in msg and set(car.windows.values()) == {"vent"}
    assert execute_vehicle_tool(car, "launch_rockets", {}).startswith("Error")
    assert execute_vehicle_tool(car, "set_fan", {}).startswith("Error")


def test_attention_defers_replies_during_critical_and_releases_after():
    am = AttentionManager()
    road_crit = {"fcw": "critical", "tracks": []}
    assert am.workload(road_crit, None) == "critical"
    assert am.gate_reply({"type": "reply", "text": "Hi."}, "critical") is None
    assert am.release_deferred("critical") == []
    released = am.release_deferred("low")
    assert len(released) == 1 and released[0]["deferred"]


def test_attention_shortens_under_high_workload():
    am = AttentionManager()
    out = am.gate_reply({"text": "There are two cars. One is red and one is blue."}, "high")
    assert out["text"] == "There are two cars." and out["shortened"]


def test_alert_cooldown():
    am = AttentionManager()
    road = {"fcw": "critical", "tracks": [{}]}
    assert len(am.proactive_alerts(road, None)) == 1
    assert am.proactive_alerts(road, None) == []  # rate-limited


@pytest.mark.parametrize(
    "raw,ok",
    [
        ('{"tool_calls": [], "say": "hi"}', True),
        ('```json\n{"tool_calls": [], "say": "hi"}\n```', True),
        ('Sure! {"tool_calls": [], "say": "hi"} hope that helps', True),
        ("I think it is a bus.", False),
    ],
)
def test_defensive_json_extraction(raw, ok):
    assert (_extract_json(raw) is not None) == ok


class CentreEarModel:
    """Stand-in classifier: "probability" = 1 if the centre frame's calibrated EAR < 0.5."""

    def predict_proba(self, X):
        from drivemind.perception.eye_state import HALF

        p = (X[:, HALF] < 0.5).astype(float)
        return np.column_stack([1 - p, p])


def test_eye_state_streaming_alignment(tmp_path):
    """The streaming model must decide about the frame HALF steps ago, forward-fill frames
    without a face, and report no-face centre frames as open."""
    import joblib

    from drivemind.perception.eye_state import HALF, RAW, EyeStateModel

    path = tmp_path / "m.joblib"
    joblib.dump({"model": CentreEarModel(), "threshold": 0.5, "half_window": HALF}, path)
    m = EyeStateModel(path)
    ears = [0.3] * 30
    ears[10] = 0.05  # one closed frame
    out = {}
    for i, e in enumerate(ears):
        raw = None if i == 15 else {k: 0.0 for k in RAW} | {"ear": e, "ear_l": e, "ear_r": e}
        r = m.push(raw, base=0.3)
        if r is not None:
            out[i - HALF] = r[0]
    assert out[10] is True
    assert not any(v for k, v in out.items() if k != 10)
    assert out[15] is False  # no-face frame is never "closed"
