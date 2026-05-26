| run | prompt_variant | strict_accuracy | value_accuracy | syntax_accuracy | mean_latency_ms |
| --- | --- | --- | --- | --- | --- |
| Base Qwen 3.5 9B | default | 0.370 | 0.590 | 1.000 | 292.8 |
| 50-step LoRA | default | 0.420 | 0.610 | 0.980 | 360.1 |
| 100-step LoRA | default | 0.530 | 0.630 | 1.000 | 392.9 |
| Semantic 50-step LoRA | default | 0.420 | 0.630 | 1.000 | 2285.8 |
| Semantic 50-step LoRA + semantic_grounding | semantic_grounding | 0.440 | 0.630 | 1.000 | 542.2 |
| Semantic 50-step LoRA + minimal_executable | minimal_executable | 0.420 | 0.640 | 1.000 | 560.7 |
