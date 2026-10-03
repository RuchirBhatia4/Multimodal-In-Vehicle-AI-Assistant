# Eyeblink8 evaluation

0.66 h of video, 4 people, leave-one-person-out. Closed-eye frames are 2.51% of all frames.

| Method | Blink P | Blink R | Blink F1 | Closed-frame F1 | PR-AUC | PERCLOS MAE (pts) | False microsleep /h | False drowsy /h |
|---|---|---|---|---|---|---|---|---|
| Fixed EAR < 0.2 (textbook rule) | 0.820 | 0.963 | **0.886** | 0.416 | 0.893 | 7.91 | 57.2 | 19.6 |
| Deployed rule (calibrated EAR, before this work) | 0.886 | 0.929 | **0.907** | 0.497 | 0.911 | 4.48 | 33.1 | 16.6 |
| MediaPipe eyeBlink blendshape > 0.5 | 0.975 | 0.855 | **0.911** | 0.843 | 0.900 | 0.41 | 0.0 | 1.5 |
| EAR window + logistic regression (trained) | 0.937 | 0.944 | **0.940** | 0.863 | 0.929 | 0.65 | 0.0 | 1.5 |
| Gradient-boosted trees, multi-feature (trained) | 0.946 | 0.951 | **0.949** | 0.867 | 0.913 | 0.56 | 0.0 | 1.5 |

Per-person blink F1:

| Method | A | B | C | D |
|---|---|---|---|---|
| Fixed EAR < 0.2 (textbook rule) | 0.897 | 0.903 | 0.922 | 0.842 |
| Deployed rule (calibrated EAR, before this work) | 0.898 | 0.883 | 0.933 | 0.917 |
| MediaPipe eyeBlink blendshape > 0.5 | 0.920 | 0.828 | 0.957 | 0.936 |
| EAR window + logistic regression (trained) | 0.939 | 0.875 | 0.953 | 0.983 |
| Gradient-boosted trees, multi-feature (trained) | 0.943 | 0.901 | 0.966 | 0.982 |

Threshold sensitivity of the trained models (held-out scores; reporting only):

| Method | threshold | Blink F1 | PERCLOS MAE (pts) | False microsleep /h |
|---|---|---|---|---|
| EAR window + logistic regression (trained) | 0.1 | 0.872 | 2.79 | 16.6 |
| EAR window + logistic regression (trained) | 0.3 | 0.910 | 1.66 | 4.5 |
| EAR window + logistic regression (trained) | 0.5 | 0.923 | 1.18 | 1.5 |
| EAR window + logistic regression (trained) | 0.7 | 0.941 | 0.85 | 0.0 |
| EAR window + logistic regression (trained) | 0.9 | 0.947 | 0.54 | 0.0 |
| Gradient-boosted trees, multi-feature (trained) | 0.1 | 0.947 | 0.83 | 0.0 |
| Gradient-boosted trees, multi-feature (trained) | 0.3 | 0.951 | 0.69 | 0.0 |
| Gradient-boosted trees, multi-feature (trained) | 0.5 | 0.946 | 0.64 | 0.0 |
| Gradient-boosted trees, multi-feature (trained) | 0.7 | 0.947 | 0.66 | 0.0 |
| Gradient-boosted trees, multi-feature (trained) | 0.9 | 0.949 | 0.67 | 0.0 |

Ground-truth labels themselves would trigger: {'microsleep': 0, 'drowsy': 1}

## Protocol

Evaluate eye-closure / blink detection on Eyeblink8, and train a better detector.

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