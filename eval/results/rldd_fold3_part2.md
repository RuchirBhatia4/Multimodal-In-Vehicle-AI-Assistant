# UTA-RLDD drowsiness evaluation (fold 3, part 2: 6 participants)

Frame-rate confound check: source fps alone separates alert vs drowsy videos with ROC-AUC 0.62 (0.5 = no confound).

18 videos, 3.3 h, source frame rates 12-30 fps, face found 99%-100% of frames.

## A. Shipped logic, no RLDD training (means per self-reported state)

| Eye detector | State | PERCLOS | Blinks/min | Mean blink (s) | Long closures/min | Microsleep alerts/h | Drowsy alerts/h |
|---|---|---|---|---|---|---|---|
| Original calibrated-EAR rule | alert | 7.2% | 16.0 | 0.27 | 1.18 | 4.7 | 4.7 |
| Original calibrated-EAR rule | low vigilant | 12.4% | 19.4 | 0.37 | 2.17 | 44.4 | 17.2 |
| Original calibrated-EAR rule | drowsy | 28.8% | 17.3 | 1.02 | 3.85 | 156.4 | 14.5 |
| MediaPipe eyeBlink > 0.5 | alert | 2.8% | 13.8 | 0.15 | 0.07 | 0.0 | 1.8 |
| MediaPipe eyeBlink > 0.5 | low vigilant | 7.1% | 19.4 | 0.21 | 0.94 | 25.7 | 4.6 |
| MediaPipe eyeBlink > 0.5 | drowsy | 22.3% | 18.7 | 0.63 | 3.31 | 119.1 | 18.3 |
| Trained eye model (shipped) | alert | 1.7% | 11.4 | 0.11 | 0.02 | 0.0 | 0.9 |
| Trained eye model (shipped) | low vigilant | 4.7% | 16.8 | 0.14 | 0.59 | 15.8 | 3.0 |
| Trained eye model (shipped) | drowsy | 9.0% | 15.6 | 0.28 | 2.29 | 65.8 | 16.4 |

## B. Alert vs. drowsy separability (per video; 6 vs 6 videos, so treat as indicative)

| Eye detector | Metric | ROC-AUC | Within-person |
|---|---|---|---|
| Original calibrated-EAR rule | perclos | 0.81 | 5/6 people higher when drowsy |
| Original calibrated-EAR rule | mean_blink_s | 0.72 | 4/6 people higher when drowsy |
| Original calibrated-EAR rule | long_closures_per_min | 0.81 | 5/6 people higher when drowsy |
| Original calibrated-EAR rule | microsleep_per_h | 0.79 | 5/6 people higher when drowsy |
| Original calibrated-EAR rule | drowsy_alerts_per_h | 0.78 | 3/6 people higher when drowsy |
| MediaPipe eyeBlink > 0.5 | perclos | 0.86 | 6/6 people higher when drowsy |
| MediaPipe eyeBlink > 0.5 | mean_blink_s | 0.97 | 6/6 people higher when drowsy |
| MediaPipe eyeBlink > 0.5 | long_closures_per_min | 0.94 | 6/6 people higher when drowsy |
| MediaPipe eyeBlink > 0.5 | microsleep_per_h | 0.83 | 4/6 people higher when drowsy |
| MediaPipe eyeBlink > 0.5 | drowsy_alerts_per_h | 0.90 | 5/6 people higher when drowsy |
| Trained eye model (shipped) | perclos | 0.81 | 5/6 people higher when drowsy |
| Trained eye model (shipped) | mean_blink_s | 0.92 | 5/6 people higher when drowsy |
| Trained eye model (shipped) | long_closures_per_min | 0.94 | 5/6 people higher when drowsy |
| Trained eye model (shipped) | microsleep_per_h | 0.75 | 3/6 people higher when drowsy |
| Trained eye model (shipped) | drowsy_alerts_per_h | 0.81 | 4/6 people higher when drowsy |

## C. Trained drowsiness classifier (1-min windows, leave-one-participant-out)

191 windows. Paper baselines (video accuracy, 3-class): HM-LSTM 65.2% overall / 70% on fold 3; human judges 57.8% / 60%; chance 33%.

| Model | Window acc. (3-class) | Video acc. (3-class) | Window AUC alert vs drowsy |
|---|---|---|---|
| logreg_global | 62.3% | 13/18 = 72.2% (95% CI 49%-88%) | 0.95 |
| gbm_global | 53.9% | 12/18 = 66.7% (95% CI 44%-84%) | 0.94 |
| logreg_per_driver | 53.4% | 11/18 = 61.1% (95% CI 39%-80%) | 0.86 |
| gbm_per_driver | 47.6% | 11/18 = 61.1% (95% CI 39%-80%) | 0.89 |

## Per-video detail (trained eye model)

| Participant | State | Minutes | Source fps | Face found | PERCLOS | Mean blink (s) | Microsleep/h | Drowsy alerts/h | Yawns/h |
|---|---|---|---|---|---|---|---|---|---|
| 31 | alert | 11.3 | 25.0 | 100% | 6.0% | 0.08 | 0.0 | 5.3 | 0.0 |
| 31 | low vigilant | 10.1 | 25.0 | 100% | 17.9% | 0.21 | 83.1 | 11.9 | 0.0 |
| 31 | drowsy | 10.8 | 25.0 | 100% | 26.2% | 0.57 | 178.4 | 39.0 | 16.7 |
| 32 | alert | 15.0 | 12.0 | 100% | 0.5% | 0.16 | 0.0 | 0.0 | 0.0 |
| 32 | low vigilant | 15.0 | 15.3 | 100% | 0.9% | 0.15 | 0.0 | 0.0 | 0.0 |
| 32 | drowsy | 10.1 | 20.7 | 100% | 0.7% | 0.15 | 0.0 | 0.0 | 0.0 |
| 33 | alert | 9.8 | 30.0 | 100% | 0.3% | 0.10 | 0.0 | 0.0 | 0.0 |
| 33 | low vigilant | 10.1 | 15.0 | 100% | 2.3% | 0.15 | 6.0 | 0.0 | 0.0 |
| 33 | drowsy | 9.3 | 30.0 | 99% | 16.2% | 0.50 | 205.5 | 19.3 | 6.4 |
| 34 | alert | 12.6 | 15.4 | 100% | 0.3% | 0.12 | 0.0 | 0.0 | 0.0 |
| 34 | low vigilant | 11.7 | 29.6 | 100% | 3.1% | 0.13 | 0.0 | 0.0 | 0.0 |
| 34 | drowsy | 10.2 | 30.0 | 100% | 5.2% | 0.14 | 0.0 | 29.4 | 11.7 |
| 35 | alert | 10.3 | 29.9 | 100% | 0.9% | 0.10 | 0.0 | 0.0 | 0.0 |
| 35 | low vigilant | 10.2 | 29.9 | 100% | 2.3% | 0.12 | 5.9 | 5.9 | 0.0 |
| 35 | drowsy | 11.1 | 30.0 | 100% | 4.0% | 0.16 | 10.8 | 10.8 | 10.8 |
| 36 | alert | 10.4 | 30.0 | 100% | 2.4% | 0.11 | 0.0 | 0.0 | 0.0 |
| 36 | low vigilant | 10.5 | 29.8 | 100% | 1.8% | 0.10 | 0.0 | 0.0 | 0.0 |
| 36 | drowsy | 10.5 | 29.8 | 100% | 1.5% | 0.16 | 0.0 | 0.0 | 0.0 |