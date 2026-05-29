| run | steps | loss_moved | adapter_hash | value_accuracy | strict_accuracy | claim_status | public_claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multiturn-sql-100 | 100 | 1.637 -> 0.425 | 134183882799 | 0.630 | 0.530 | supported_proxy | local proxy result |
| multiturn-sql-semantic-50 | 50 | 2.186 -> 0.389 | c0399d7b13d3 | 0.640 | 0.420 | supported_proxy | local proxy result |
| multiturn-sql-schema-pruned-100 | 100 | 2.599 -> 0.257 | 2a72f88b6585 | 0.890 | 0.820 | diagnostic_upper_bound | oracle diagnostic ceiling |
| fresh 1-step GPU smoke | 1 | train_loss 1.684; 27.23s | ea12b9f4f1fe |  |  | completed_smoke | GPU path verified only; not a benchmark result |
| GPU visibility check |  | torch 2.10.0+cu128; cuda_available True; device_count 1; device_name NVIDIA GeForce RTX 5090; capability (12, 0); cuda matmul ok | not applicable |  |  | gpu_visible_outside_sandbox | NVIDIA GeForce RTX 5090; driver 591.86; CUDA 13.1; 32607 MiB |
