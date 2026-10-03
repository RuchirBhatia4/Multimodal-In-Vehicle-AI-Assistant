# Held-out test: frozen drowsiness rule on new people (UTA-RLDD Fold5_part1)

Rule frozen on 2026-10-03 using only UTA-RLDD Fold3_part2 (participants ['31', '32', '33', '34', '35', '36']) (see `preregistration.json`, committed before this test ran): alert when the MediaPipe eyeBlink score shows >= 4 eye closures of >= 0.5 s within 60 s, or one closure >= 1.0 s; at most one alert per 60 s.

Test: 6 new people (49, 50, 51, 52, 53, 54), 3.4 h.

| Rule | False alerts/h (alert videos) | Drowsy videos with >= 1 alert | People alerted more when drowsy | Alerts/h: alert / low vigilant / drowsy |
|---|---|---|---|---|
| Shipped (PERCLOS > 15% + microsleep) | 0.0 | 5/6 | 5/6 | 0.0 / 0.9 / 62.1 |
| **New frozen rule (primary)** | 0.0 | 5/6 | 5/6 | 0.0 / 2.8 / 20.2 |
| Exploratory: any closure >= 0.7 s x1 (mediapipe_blink) | 2.0 | 6/6 | 6/6 | 2.0 / 5.6 / 22.5 |

On the development people the frozen rule scored: 0.0 false alerts/h, 4/6 drowsy videos alerted, 4/6 people alerted more when drowsy. Eyeblink8 (awake people) check: 0.0 alerts/h.

## Drowsiness classifier, tested on the new people

RLDD paper on its own test folds: HM-LSTM 65.2%, human judges 57.8%, chance 33%.

| Trained on | Video accuracy (3-class) | Window ROC-AUC alert vs drowsy |
|---|---|---|
| 6 development people | 10/18 = 55.6% (95% CI 34%-75%) | 0.80 |

| Participant | State | Predicted | Source fps | Alerts (new rule) | Alerts (shipped) | Alerts (exploratory) |
|---|---|---|---|---|---|---|
| 49 | alert | drowsy | 29.7 | 0 | 0 | 1 |
| 49 | low vigilant | low vigilant | 29.7 | 0 | 0 | 0 |
| 49 | drowsy | drowsy | 29.7 | 18 | 73 | 17 |
| 50 | alert | alert | 30.0 | 0 | 0 | 0 |
| 50 | low vigilant | alert | 30.0 | 0 | 0 | 0 |
| 50 | drowsy | alert | 30.0 | 1 | 1 | 1 |
| 51 | alert | alert | 30.0 | 0 | 0 | 0 |
| 51 | low vigilant | low vigilant | 30.0 | 0 | 0 | 1 |
| 51 | drowsy | drowsy | 30.0 | 3 | 1 | 3 |
| 52 | alert | alert | 30.0 | 0 | 0 | 0 |
| 52 | low vigilant | drowsy | 30.0 | 2 | 1 | 3 |
| 52 | drowsy | drowsy | 30.0 | 1 | 1 | 2 |
| 53 | alert | alert | 30.0 | 0 | 0 | 0 |
| 53 | low vigilant | drowsy | 30.0 | 1 | 0 | 2 |
| 53 | drowsy | drowsy | 30.0 | 3 | 4 | 5 |
| 54 | alert | low vigilant | 29.9 | 0 | 0 | 1 |
| 54 | low vigilant | alert | 16.0 | 0 | 0 | 0 |
| 54 | drowsy | low vigilant | 29.9 | 0 | 0 | 1 |