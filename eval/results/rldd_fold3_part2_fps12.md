# UTA-RLDD drowsiness evaluation (fold 3, part 2: 6 participants), every video degraded to 12 fps (frame-rate control)

Frame-rate confound check: source fps alone separates alert vs drowsy videos with ROC-AUC 0.62 (0.5 = no confound).

18 videos, 3.3 h, source frame rates 12-30 fps, face found 99%-100% of frames.

## A. Shipped logic, no RLDD training (means per self-reported state)

| Eye detector | State | PERCLOS | Blinks/min | Mean blink (s) | Long closures/min | Microsleep alerts/h | Drowsy alerts/h |
|---|---|---|---|---|---|---|---|
| Original calibrated-EAR rule | alert | 7.4% | 15.8 | 0.27 | 1.33 | 4.7 | 3.8 |
| Original calibrated-EAR rule | low vigilant | 12.6% | 19.0 | 0.37 | 2.18 | 45.4 | 18.2 |
| Original calibrated-EAR rule | drowsy | 29.0% | 17.0 | 1.04 | 4.01 | 154.4 | 15.5 |
| MediaPipe eyeBlink > 0.5 | alert | 2.4% | 12.6 | 0.14 | 0.07 | 0.0 | 1.8 |
| MediaPipe eyeBlink > 0.5 | low vigilant | 6.7% | 17.9 | 0.21 | 1.00 | 25.7 | 4.6 |
| MediaPipe eyeBlink > 0.5 | drowsy | 22.1% | 18.0 | 0.66 | 3.36 | 117.0 | 18.2 |
| Trained eye model (shipped) | alert | 1.3% | 9.5 | 0.11 | 0.02 | 0.0 | 0.9 |
| Trained eye model (shipped) | low vigilant | 4.2% | 14.2 | 0.14 | 0.56 | 16.8 | 3.0 |
| Trained eye model (shipped) | drowsy | 8.7% | 14.4 | 0.29 | 2.25 | 65.8 | 15.4 |

## B. Alert vs. drowsy separability (per video; 6 vs 6 videos, so treat as indicative)

| Eye detector | Metric | ROC-AUC | Within-person |
|---|---|---|---|
| Original calibrated-EAR rule | perclos | 0.78 | 5/6 people higher when drowsy |
| Original calibrated-EAR rule | mean_blink_s | 0.83 | 5/6 people higher when drowsy |
| Original calibrated-EAR rule | long_closures_per_min | 0.81 | 5/6 people higher when drowsy |
| Original calibrated-EAR rule | microsleep_per_h | 0.79 | 5/6 people higher when drowsy |
| Original calibrated-EAR rule | drowsy_alerts_per_h | 0.81 | 4/6 people higher when drowsy |
| MediaPipe eyeBlink > 0.5 | perclos | 0.86 | 6/6 people higher when drowsy |
| MediaPipe eyeBlink > 0.5 | mean_blink_s | 0.94 | 5/6 people higher when drowsy |
| MediaPipe eyeBlink > 0.5 | long_closures_per_min | 1.00 | 6/6 people higher when drowsy |
| MediaPipe eyeBlink > 0.5 | microsleep_per_h | 0.83 | 4/6 people higher when drowsy |
| MediaPipe eyeBlink > 0.5 | drowsy_alerts_per_h | 0.90 | 5/6 people higher when drowsy |
| Trained eye model (shipped) | perclos | 0.83 | 5/6 people higher when drowsy |
| Trained eye model (shipped) | mean_blink_s | 0.94 | 5/6 people higher when drowsy |
| Trained eye model (shipped) | long_closures_per_min | 0.94 | 5/6 people higher when drowsy |
| Trained eye model (shipped) | microsleep_per_h | 0.75 | 3/6 people higher when drowsy |
| Trained eye model (shipped) | drowsy_alerts_per_h | 0.81 | 4/6 people higher when drowsy |

## C. Trained drowsiness classifier (1-min windows, leave-one-participant-out)

191 windows. Paper baselines (video accuracy, 3-class): HM-LSTM 65.2% overall / 70% on fold 3; human judges 57.8% / 60%; chance 33%.

| Model | Window acc. (3-class) | Video acc. (3-class) | Window AUC alert vs drowsy |
|---|---|---|---|
| logreg_global | 62.3% | 13/18 = 72.2% (95% CI 49%-88%) | 0.95 |
| gbm_global | 58.1% | 11/18 = 61.1% (95% CI 39%-80%) | 0.95 |
| logreg_per_driver | 55.5% | 11/18 = 61.1% (95% CI 39%-80%) | 0.87 |
| gbm_per_driver | 51.3% | 12/18 = 66.7% (95% CI 44%-84%) | 0.90 |

## Per-video detail (trained eye model)

| Participant | State | Minutes | Source fps | Face found | PERCLOS | Mean blink (s) | Microsleep/h | Drowsy alerts/h | Yawns/h |
|---|---|---|---|---|---|---|---|---|---|
| 31 | alert | 11.3 | 25.0 | 100% | 4.4% | 0.08 | 0.0 | 5.3 | 0.0 |
| 31 | low vigilant | 10.1 | 25.0 | 100% | 16.4% | 0.23 | 89.0 | 11.9 | 0.0 |
| 31 | drowsy | 10.8 | 25.0 | 100% | 26.0% | 0.58 | 178.5 | 39.0 | 16.7 |
| 32 | alert | 15.0 | 12.0 | 100% | 0.5% | 0.16 | 0.0 | 0.0 | 0.0 |
| 32 | low vigilant | 15.0 | 15.3 | 100% | 0.8% | 0.14 | 0.0 | 0.0 | 0.0 |
| 32 | drowsy | 10.1 | 20.7 | 100% | 0.7% | 0.15 | 0.0 | 0.0 | 0.0 |
| 33 | alert | 9.8 | 30.0 | 100% | 0.3% | 0.11 | 0.0 | 0.0 | 0.0 |
| 33 | low vigilant | 10.1 | 15.0 | 100% | 2.2% | 0.16 | 6.0 | 0.0 | 0.0 |
| 33 | drowsy | 9.3 | 30.0 | 99% | 16.1% | 0.51 | 205.5 | 19.3 | 6.4 |
| 34 | alert | 12.6 | 15.4 | 100% | 0.3% | 0.12 | 0.0 | 0.0 | 0.0 |
| 34 | low vigilant | 11.7 | 29.6 | 100% | 2.6% | 0.12 | 0.0 | 0.0 | 0.0 |
| 34 | drowsy | 10.2 | 30.0 | 100% | 4.5% | 0.12 | 0.0 | 23.5 | 11.7 |
| 35 | alert | 10.3 | 29.9 | 100% | 0.7% | 0.10 | 0.0 | 0.0 | 0.0 |
| 35 | low vigilant | 10.2 | 29.9 | 100% | 1.8% | 0.12 | 5.9 | 5.9 | 0.0 |
| 35 | drowsy | 11.1 | 30.0 | 100% | 3.4% | 0.19 | 10.8 | 10.8 | 10.8 |
| 36 | alert | 10.4 | 30.0 | 100% | 1.9% | 0.09 | 0.0 | 0.0 | 0.0 |
| 36 | low vigilant | 10.5 | 29.8 | 100% | 1.3% | 0.10 | 0.0 | 0.0 | 0.0 |
| 36 | drowsy | 10.5 | 29.8 | 100% | 1.3% | 0.17 | 0.0 | 0.0 | 0.0 |