# Benchmark Summary

## Matrix

- scenes: 4
- planners: rrt, rrt_connect, rrt_star, cci_bridge_rrt
- ablations: baseline, cci_only, bridge_only, cci_bridge
- seeds: 20

## Comparison Figures

![success_rate_bar](figures/success_rate_bar.png)

![median_time_bar](figures/median_time_bar.png)

![complex_runtime_boxplot](figures/complex_runtime_boxplot.png)

![complex_pathlen_boxplot](figures/complex_pathlen_boxplot.png)

## Planner/Ablation Metrics

| planner | ablation | runs | success_rate | median_time_ms | median_path_len |
|---|---|---:|---:|---:|---:|
| rrt | baseline | 80 | 0.725 | 302.49 | 1.8590 |
| rrt | cci_only | 80 | 0.725 | 343.04 | 1.8590 |
| rrt | bridge_only | 80 | 0.725 | 328.61 | 1.8590 |
| rrt | cci_bridge | 80 | 0.725 | 357.71 | 1.8590 |
| rrt_connect | baseline | 80 | 0.775 | 360.73 | 1.8853 |
| rrt_connect | cci_only | 80 | 0.775 | 369.89 | 1.8853 |
| rrt_connect | bridge_only | 80 | 0.775 | 369.38 | 1.8853 |
| rrt_connect | cci_bridge | 80 | 0.775 | 345.36 | 1.8853 |
| rrt_star | baseline | 80 | 0.725 | 669.11 | 1.6058 |
| rrt_star | cci_only | 80 | 0.725 | 657.62 | 1.6058 |
| rrt_star | bridge_only | 80 | 0.713 | 1078.97 | 1.6032 |
| rrt_star | cci_bridge | 80 | 0.713 | 1097.87 | 1.6032 |
| cci_bridge_rrt | baseline | 80 | 0.725 | 1147.79 | 1.6058 |
| cci_bridge_rrt | cci_only | 80 | 1.000 | 294.25 | 1.4234 |
| cci_bridge_rrt | bridge_only | 80 | 0.700 | 1202.65 | 1.6058 |
| cci_bridge_rrt | cci_bridge | 160 | 0.994 | 459.81 | 1.4905 |

## Failure Counts

- timeout: 288
- collision_stuck: 6
- unstable: 3

## Tuning History

- round 1: {'bridge_bias': 0.25, 'step_size': 0.07200000000000001}
- round 2: {'no_change': 1.0}
