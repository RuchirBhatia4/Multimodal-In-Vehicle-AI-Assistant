# Frozen drowsiness rules, pooled over all held-out tests

2 test sets, 12 unseen people, 6.4 h.

| Rule | False alerts on alert videos | Drowsy videos with >= 1 alert (95% CI) | People alerted more when drowsy |
|---|---|---|---|
| Previous rule (PERCLOS > 15% + microsleep) | 4 alerts | 9/12 (47%-91%) | 8/12 |
| Current rule (shipped since 2026-10-03) | 1 alerts | 10/12 (55%-95%) | 10/12 |
| Exploratory: any closure >= 0.7 s | 5 alerts | 11/12 (65%-99%) | 10/12 |