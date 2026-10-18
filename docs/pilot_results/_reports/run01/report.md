# pilot-joint — measured results

Complete: True; numerical admission: False. Human review required.
Budget per learner/seed: 200 same-instance batch epochs × 8 complete episodes = 1600 episodes. Events are measured, not budgeted.
Preserved failure evidence files: 0 (batch and supervisor records may describe the same attempt; see attempt-failures.csv).

| Method | Policy | Split/scale | TWT | Complete/failed/missing | Mean instance p95 ms |
|---|---|---|---:|---|---:|
| full | latest | validation/S | 4292.769 | 15/0/0 | 29.678 |
| full | latest | validation/M | 27445.622 | 15/0/0 | 69.892 |
| mlp | latest | validation/S | 5008.533 | 15/0/0 | 13.279 |
| mlp | latest | validation/M | 30799.932 | 15/0/0 | 54.958 |
| shared_relation | latest | validation/S | 4262.617 | 15/0/0 | 28.928 |
| shared_relation | latest | validation/M | 27896.575 | 15/0/0 | 72.407 |
| periodic_gate | latest | validation/S | 4447.146 | 15/0/0 | 28.686 |
| periodic_gate | latest | validation/M | 25129.604 | 15/0/0 | 71.091 |
| no_rule_prior | latest | validation/S | 4766.492 | 15/0/0 | 27.902 |
| no_rule_prior | latest | validation/M | 32601.504 | 15/0/0 | 68.433 |
| gat | latest | validation/S | 4146.732 | 15/0/0 | 27.090 |
| gat | latest | validation/M | 23308.077 | 15/0/0 | 60.703 |
| dan | latest | validation/S | 4176.302 | 15/0/0 | 9.724 |
| dan | latest | validation/M | 23185.609 | 15/0/0 | 31.100 |
| Fixed-EDD | fixed | validation/S | 6625.233 | 5/0/0 | 0.301 |
| Fixed-EDD | fixed | validation/M | 36587.757 | 5/0/0 | 0.856 |
| Fixed-ATC | fixed | validation/S | 4217.544 | 5/0/0 | 0.498 |
| Fixed-ATC | fixed | validation/M | 23617.054 | 5/0/0 | 3.492 |
| Periodic-ATC | fixed | validation/S | 3947.786 | 5/0/0 | 2.495 |
| Periodic-ATC | fixed | validation/M | 22096.280 | 5/0/0 | 4.858 |
| RH-MILP | fixed | validation/S | 5150.033 | 5/0/0 | 5.012 |
| RH-MILP | fixed | validation/M | 31986.928 | 5/0/0 | 10.458 |
| RH-ALNS | fixed | validation/S | 3992.250 | 15/0/0 | 7.520 |
| RH-ALNS | fixed | validation/M | 23704.968 | 15/0/0 | 24.465 |
| Opportunity-ATC | fixed | validation/S | 5418.569 | 5/0/0 | 2.153 |
| Opportunity-ATC | fixed | validation/M | 39921.720 | 5/0/0 | 24.851 |

No-graph and MLP-PPO share one result. Per-case/seed epoch and complete-episode budgets, measured events, runtime, failures and paired comparisons are in the adjacent CSV/JSON files.
No source in these tables is automatically a proven optimum. Training tail statistics are separate from final-policy test results.
