# Held-out test: frozen drowsiness rule on new people (UTA-RLDD Fold2_part1)

Rule frozen on 2026-10-03 using only UTA-RLDD Fold3_part2 (participants ['31', '32', '33', '34', '35', '36']) (see `preregistration.json`, committed before this test ran): alert when the MediaPipe eyeBlink score shows >= 4 eye closures of >= 0.5 s within 60 s, or one closure >= 1.0 s; at most one alert per 60 s.

Test: 6 new people (13, 14, 15, 16, 17, 18), 3.1 h.

| Rule | False alerts/h (alert videos) | Drowsy videos with >= 1 alert | People alerted more when drowsy | Alerts/h: alert / low vigilant / drowsy |
|---|---|---|---|---|
| Shipped (PERCLOS > 15% + microsleep) | 4.1 | 4/6 | 3/6 | 4.1 / 17.1 / 45.6 |
| **New frozen rule (primary)** | 1.0 | 5/6 | 5/6 | 1.0 / 15.2 / 21.3 |
| Exploratory: any closure >= 0.7 s x1 (mediapipe_blink) | 3.0 | 5/6 | 4/6 | 3.0 / 18.1 / 24.3 |

On the development people the frozen rule scored: 0.0 false alerts/h, 4/6 drowsy videos alerted, 4/6 people alerted more when drowsy. Eyeblink8 (awake people) check: 0.0 alerts/h.

## Drowsiness classifier, tested on the new people

RLDD paper on its own test folds: HM-LSTM 65.2%, human judges 57.8%, chance 33%.

| Trained on | Video accuracy (3-class) | Window ROC-AUC alert vs drowsy |
|---|---|---|
| 6 development people | 8/18 = 44.4% (95% CI 25%-66%) | 0.83 |
| 12 people (development + earlier test) | 10/18 = 55.6% (95% CI 34%-75%) | 0.89 |

| Participant | State | Predicted | Source fps | Alerts (new rule) | Alerts (shipped) | Alerts (exploratory) |
|---|---|---|---|---|---|---|
| 13 | alert | alert | 29.8 | 0 | 0 | 0 |
| 13 | low vigilant | alert | 29.8 | 1 | 3 | 3 |
| 13 | drowsy | drowsy | 29.8 | 6 | 6 | 7 |
| 14 | alert | alert | 30.0 | 0 | 0 | 0 |
| 14 | low vigilant | drowsy | 29.8 | 7 | 0 | 9 |
| 14 | drowsy | low vigilant | 30.0 | 1 | 0 | 3 |
| 15 | alert | low vigilant | 30.0 | 1 | 4 | 3 |
| 15 | low vigilant | low vigilant | 30.0 | 0 | 4 | 0 |
| 15 | drowsy | low vigilant | 30.0 | 2 | 2 | 2 |
| 16 | alert | alert | 24.0 | 0 | 0 | 0 |
| 16 | low vigilant | alert | 24.0 | 0 | 0 | 0 |
| 16 | drowsy | drowsy | 24.0 | 0 | 0 | 0 |
| 17 | alert | low vigilant | 30.0 | 0 | 0 | 0 |
| 17 | low vigilant | low vigilant | 30.0 | 1 | 1 | 1 |
| 17 | drowsy | low vigilant | 29.9 | 4 | 6 | 4 |
| 18 | alert | low vigilant | 19.4 | 0 | 0 | 0 |
| 18 | low vigilant | drowsy | 14.9 | 7 | 10 | 6 |
| 18 | drowsy | drowsy | 12.0 | 9 | 33 | 9 |