# Frozen artifact manifest

**Status:** Frozen / Portfolio Complete

**Audit date:** 2026-08-20

**Scope:** existing releases, external artifacts, committed evaluation evidence, and licenses only. No model inference, training, quantization, or scientific claim was rerun for this audit.

## GitHub release

- Final formal release: [`v1.1.0`](https://github.com/kuotunyu/local-llm-lifecycle/releases/tag/v1.1.0), published 2026-07-31 18:14:18 UTC.
- Annotated tag object: `8ac3b4c7f6bdb203612b381b584dec3337cf56e7`.
- Release commit: `9c80434c1af28fde7a6dcf07c5d0ea99bfee301d`.
- The release intentionally has no uploaded binary assets. Its source archive contains the compact committed prediction/evidence files consumed by `scripts/57_verify_published_numbers.py`.
- No version bump was made during the freeze audit.

## Hugging Face assets

The links below are pinned to the audited repository revisions. Licenses are the values published in each Hugging Face card's metadata.

| Asset | Frozen revision | License |
|---|---|---|
| [LoRA adapter](https://huggingface.co/steven0226/Qwen3-8B-DRCD-zhTW-QA-LoRA/tree/4a95e2c1aa047dee0c45220a1a77e956972159a4) | `4a95e2c1aa047dee0c45220a1a77e956972159a4` | Apache-2.0 |
| [GGUF models](https://huggingface.co/steven0226/Qwen3-8B-DRCD-zhTW-QA-GGUF/tree/7eeec96fc05779165ee76bb3edf5603ae20f4b11) | `7eeec96fc05779165ee76bb3edf5603ae20f4b11` | Apache-2.0 |
| [SFT dataset](https://huggingface.co/datasets/steven0226/drcd-zhtw-extractive-qa-sft/tree/38478c7359c17dc5c46378348a5cd95858f30ab6) | `38478c7359c17dc5c46378348a5cd95858f30ab6` | CC BY-SA 4.0 |

Critical LFS objects reported by the Hugging Face API:

| Repository file | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3-8B-DRCD-zhTW-QA-LoRA/adapter_model.safetensors` | 174,655,536 | `8dc5ff6d45efaf5d079308d9d85361e8179804dc973eb4853f41fe848603a8d9` |
| `Qwen3-8B-DRCD-zhTW-QA-GGUF/qwen3-8b-drcd-qa-Q4_K_M.gguf` | 5,027,784,064 | `aae607ad07c3b5c92d7dc0ef6ffa84003725d595a2970a96c816cccfee7a33a2` |
| `Qwen3-8B-DRCD-zhTW-QA-GGUF/qwen3-8b-drcd-qa-Q8_0.gguf` | 8,709,518,720 | `54513f8c012a0e507575f3453cdcc862a8662b4672abb56a15bcca5c25f466af` |
| `drcd-zhtw-extractive-qa-sft/train_sft.jsonl` | 20,149,935 | `4ad3ebfc33a05c6be0799cac98af1fa46ec1c75483c384342435943d4e9b6218` |

## Committed evaluation evidence

The hashes below identify the files validated by the offline verifier. They are evidence records, not newly generated experiment output.

### Quantization evidence

| File | Records | SHA-256 |
|---|---:|---|
| `results/eval_raw/base_zeroshot.jsonl` | 4,699 | `1c2b7bd4a98ed60c9707d0fc19b4e5f5345793414ac39af9dd37b8092ea4df8e` |
| `results/eval_raw/base_fewshot.jsonl` | 4,699 | `acfc5f0bb018be61b736a1b97530d06d3b006647cfa158a4d7f6953e25340458` |
| `results/eval_raw/ft_unquantized.jsonl` | 4,699 | `d5a90a4aa2d3074b069af4bc83c92b7de60603de41605f6753279dc1786678fb` |
| `results/eval_raw/ft_q8.jsonl` | 4,699 | `06b59f84e0393af791bcc5d932a1d1143a92aba1ded9c87825d7701caae09b62` |
| `results/eval_raw/ft_q4.jsonl` | 4,699 | `893b75bb33a749c5596eb2fde43a77fc0ab0aef98a3c56a3f964ce55b4c71305` |
| `results/eval_summary.json` | summary | `ed8818441e769c10601c193e081abb67dccbbf37ef4c505e199934ac6e25279a` |
| `results/drcd_paired_stats.json` | paired statistics | `872dc89159448009a1e326bb2686032e2e5e186bf05fd8e7020419e170b246a0` |

### Catastrophic-forgetting correction evidence

| File | Records | SHA-256 |
|---|---:|---|
| `results/eval_raw/tmmlu_base.jsonl` | 20,118 | `744553df2fe11168e64ad06f50981cc21dfef08a949c3656a0669c14c3bd8850` |
| `results/eval_raw/tmmlu_ft.jsonl` | 20,118 | `77fd992aaca88dd5fdc1dcaba1b5b2f2271a457144e542e4603875ee29134913` |
| `results/tmmlu_summary.json` | summary | `761832aaf224811b7b98475796e5900d22af3e28b9be7aaf1b6b653e64f6b2ff` |
| `results/tmmlu_paired_stats.json` | paired statistics | `ef758c8c58439813f8572220119ad2c9cd911dedaf8ef760d9604d1381c11bc9` |

### Option-permutation evidence

| File | Records | SHA-256 |
|---|---:|---|
| `results/eval_perm/permutation_predictions.jsonl` | 20,110 compact item rows | `25ffb37e014cbbb8ff031f40ddaf3a2ec75b8ee225784173011fc75b70dcec3c` |
| `results/tmmlu_option_permutation.json` | summary | `a33793afad70af253d3042ef2ed6ff2cae294bf2e7b55a5fb4fa33e47cc8486f` |

### Batch-sensitivity evidence

| File | Records | SHA-256 |
|---|---:|---|
| `results/eval_perm/batch6144_predictions.jsonl` | 20,118 compact item rows | `9ec9d7a48bbe52296c3cf7e71e5912980a5ba542c86bfd08ae608eb29a8e9534` |
| `results/tmmlu_batch_sensitivity.json` | summary | `6efe6396cb8f2239af05090b6b1b58031d5ae41322178933484667cb3fbd2b3d` |

`scripts/54_test_option_permutation.py` validates the aggregation oracle without model inference. `scripts/57_verify_published_numbers.py` recomputes the published summaries from these committed records and checks the README claims against the resulting JSON.

## License boundary

- Repository source and documentation: [MIT](LICENSE).
- LoRA and GGUF model artifacts: Apache-2.0 as published in their Hugging Face metadata; base model `unsloth/Qwen3-8B` is also published as Apache-2.0.
- Derived SFT dataset: CC BY-SA 4.0. Its card attributes DRCD / Delta Research Center, identifies the upstream CC BY-SA 3.0 terms, and describes the adaptation.
