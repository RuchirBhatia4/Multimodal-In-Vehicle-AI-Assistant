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


def _feed(rule, t0, closed_s, open_s, fps=30):
    """Feed one closure of closed_s seconds followed by open_s seconds open."""
    t, out = t0, None
    for _ in range(round(closed_s * fps)):
        out = rule.update(t, 0.9)
        t += 1 / fps
    for _ in range(round(open_s * fps)):
        out = rule.update(t, 0.1)
        t += 1 / fps
    return t, out


def test_drowsiness_rule_counts_long_closures():
    from drivemind.perception.drowsiness import DrowsinessRule

    r, t = DrowsinessRule(), 0.0
    for _ in range(3):
        t, out = _feed(r, t, 0.6, 5.0)
    assert out["long_closures_60s"] == 3 and not out["drowsy"]
    t, out = _feed(r, t, 0.6, 5.0)
    assert out["drowsy"]
    t, out = _feed(r, t, 0.0, 61.0)  # old closures expire after 60 s
    assert out["long_closures_60s"] == 0 and not out["drowsy"]


def test_drowsiness_rule_ignores_normal_blinks_and_flicker():
    from drivemind.perception.drowsiness import DrowsinessRule

    r, t = DrowsinessRule(), 0.0
    for _ in range(20):  # 20 ordinary 0.2 s blinks: no long closures
        t, out = _feed(r, t, 0.2, 2.0)
    assert out["long_closures_60s"] == 0
    # a 0.6 s closure interrupted by a 1-frame flicker is still one long closure
    t, _ = _feed(r, t, 0.3, 1 / 30)
    t, out = _feed(r, t, 0.3, 1.0)
    assert out["long_closures_60s"] == 1


def test_drowsiness_rule_microsleep_during_closure():
    from drivemind.perception.drowsiness import DrowsinessRule

    r, t = DrowsinessRule(), 0.0
    out = None
    for i in range(40):  # 1.33 s closed at 30 fps
        out = r.update(t + i / 30, 0.9)
        if i == 20:
            assert not out["microsleep"]  # 0.7 s in
    assert out["microsleep"]  # fires while the eyes are still closed
    assert not r.update(t + 2.0, None)["microsleep"]  # face lost ends the closure


def test_in_path_uses_distance_invariant_lateral_offset():
    """Pinhole camera, f = 500 px, 640x360 frame. A car parked at the curb (3 m to the side,
    1.8 m wide) is never 'in path' however close it gets; a car in our lane is, once its box
    is big enough to measure."""
    from drivemind.perception.road import is_in_path

    f, w, h = 500.0, 640, 360

    def box(lateral_m, width_m, dist_m):
        cx, bw = w / 2 + f * lateral_m / dist_m, f * width_m / dist_m
        return cx - bw / 2, 200.0, cx + bw / 2, 300.0  # bottom in the lower part of the frame

    for z in (40, 20, 10, 6):
        assert not is_in_path(*box(3.0, 1.8, z), w, h), z  # parked at the curb
    assert is_in_path(*box(0.4, 1.8, 15), w, h)  # ahead in our lane
    assert not is_in_path(*box(0.4, 1.8, 60), w, h)  # same car far away: box too small to judge


def test_ttc_ignores_identity_swaps_and_box_jumps():
    """A sudden size jump (tracker swapped objects at a scene cut, or a car emerging from
    behind another) must not produce a near-zero time-to-collision."""
    from drivemind.perception.road import RoadPerception

    rp = RoadPerception.__new__(RoadPerception)
    rp.histories, rp.ttc_window_s = {}, 0.8
    for i in range(10):  # small, steady box for 0.6 s
        rp._ttc(1, i / 15, 40.0)
    out = [rp._ttc(1, (10 + i) / 15, 160.0) for i in range(4)]  # then 4x bigger, instantly
    assert all(o is None for o in out)


# ----------------------------------------------------------------- visual-memory routing
class _FakeMemory:
    def __init__(self):
        self.searches = []

    def search(self, query, k=1):
        self.searches.append(query)
        import cv2

        jpeg = cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))[1].tobytes()
        return [{"jpeg": jpeg, "seconds_ago": 20.0, "similarity": 0.27, "summary": "sign"}][:k]


def _ctx(query):
    import cv2

    from drivemind.brain.common import BrainContext
    from drivemind.brain.tools import CarState

    jpeg = cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))[1].tobytes()
    return BrainContext(query, jpeg, "Detected: 2 cars.", "Driver state: alert.", CarState(), "low")


def _local_brain(outputs):
    """A LocalBrain whose VLM returns canned outputs in order (no weights loaded)."""
    from drivemind.brain.local_brain import LocalBrain

    brain = LocalBrain.__new__(LocalBrain)
    brain.prompts = []

    def fake_generate(system, user, image, max_tokens):
        brain.prompts.append(user)
        return outputs.pop(0), {}

    brain._generate = fake_generate
    return brain


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Which car is on my left?", False),  # the question from the demo recording
        ("Is that light green?", False),
        ("How many cars are ahead of us?", False),
        ("What did you just say?", False),  # past, but not about anything visible
        ("Did I set the temperature?", False),
        ("Which car was on my left a moment ago?", True),
        ("What did that sign say?", True),
        ("Remind me what the last sign said.", True),
        ("What did we just drive past?", True),
    ],
)
def test_memory_intent(query, expected):
    from drivemind.brain.common import memory_intent

    assert memory_intent(query) is expected


def test_present_question_answers_from_live_view_without_tools():
    mem = _FakeMemory()
    brain = _local_brain(["The car on your left is red."])
    res = brain.answer(_ctx("Which car is on my left?"), memory=mem)
    assert mem.searches == [] and res.retrieved == [] and res.text == "The car on your left is red."
    assert "Available tools" not in brain.prompts[0]  # no tool list or examples to copy
    assert res.detail["route"] == "look_and_answer"


def test_command_ignores_memory_tool_even_if_model_asks():
    mem = _FakeMemory()
    brain = _local_brain(['{"tool_calls": [{"name": "recall_scene", "args": {"query": "x"}}, '
                          '{"name": "set_defrost", "args": {"on": true}}], "say": "Defroster on."}'])
    res = brain.answer(_ctx("The windshield is fogging up, turn on the defroster"), memory=mem)
    assert mem.searches == [] and [c["name"] for c in res.tool_calls] == ["set_defrost"]


def test_past_question_always_uses_memory():
    mem = _FakeMemory()
    brain = _local_brain(["The car on your left was a white van."])
    res = brain.answer(_ctx("Which car was on my left a moment ago?"), memory=mem)
    assert len(mem.searches) == 1 and res.retrieved and "20 seconds ago" in brain.prompts[0]


def test_broken_json_is_never_spoken():
    brain = _local_brain(['{"tool_calls": [{"name": "say", "args": ["Turn the heat u', "Turning the heat up."])
    res = brain.answer(_ctx("Turn the heat up a bit"), memory=_FakeMemory())
    assert res.text == "Turning the heat up."


def test_safety_question_never_says_safe():
    brain = _local_brain(["The light is green, it's safe to go."])
    res = brain.answer(_ctx("Is it safe to go?"), memory=_FakeMemory())
    assert "safe to go" not in res.text and "check for yourself" in res.text


def test_claude_memory_tool_refused_for_present_question():
    from types import SimpleNamespace as NS

    from drivemind.brain.claude_brain import ClaudeBrain

    sent = []

    class FakeMessages:
        def create(self, **kw):
            sent.append(kw["messages"])
            usage = NS(input_tokens=1, output_tokens=1)
            if len(sent) == 1:
                tu = NS(type="tool_use", id="t1", name="recall_scene", input={"query": "car on left"})
                return NS(stop_reason="tool_use", content=[tu], usage=usage)
            return NS(stop_reason="end_turn", content=[NS(type="text", text="A red sedan.")], usage=usage)

    brain = ClaudeBrain.__new__(ClaudeBrain)
    brain.client = NS(beta=NS(messages=FakeMessages()))
    mem = _FakeMemory()
    res = brain.answer(_ctx("Which car is on my left?"), memory=mem)
    tool_result = sent[1][-1]["content"][0]
    assert mem.searches == [] and "Not searched" in tool_result["content"] and res.text == "A red sedan."
