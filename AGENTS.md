# Chapter-4 artifact rules

1. Every training, evaluation, checkpoint-selection, smoke, preflight, and
   audit result must be written below `data/chapter4_final`.
2. `scripts/` contains BAT entry points only. Python orchestration belongs in
   `tools/` or `tools/_internal/`.
3. Do not create the retired top-level artifact roots `eval_ch4_rbe`,
   `eval_ch4_smoke`, `log_ch4_rbe`, or `models_ch4_rbe`.
4. Operational scripts must not hard-code retired artifact roots or historical
   fallback roots. Construct outputs with `registry.ch4_artifact_layout`.
5. Resolve every input artifact, including paths read from historical JSON,
   through `registry.ch4_artifact_layout.resolve_artifact_path()`.
6. Every experiment entry point must explicitly declare `family`, `stage`,
   `experiment_id`, `run_name`, `seed`, and `protocol`.
7. Historical model, CSV, JSON, Markdown, and manifest results are immutable
   during layout migration. Compatibility is provided by
   `relocation_manifest.json`; historical files are never rewritten.
8. Artifact migration is plan-first. Never run
   `scripts\organize_ch4_artifacts_v1.bat apply` implicitly.
