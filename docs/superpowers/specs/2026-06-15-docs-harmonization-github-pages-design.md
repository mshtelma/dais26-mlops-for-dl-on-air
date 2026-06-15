# Spec — Docs harmonization + GitHub Pages site

**Date:** 2026-06-15
**Status:** Approved ("push it")
**Owner:** Michael Shtelma

## Problem

The repo has 7 strong but overlapping docs (`README`, `docs/README`, `docs/ARCHITECTURE`,
`docs/RUNBOOK`, `docs/HPO`, `docs/BENCHMARKS`, `docs/TALK`, `docs/DECK_BRIEF_DL_MLOPS`,
`air/README`). They duplicate concepts (two-phase deploy, named config, @challenger→@champion,
troubleshooting) across files, contain a few broken anchors, and are not navigable as one site.
We want an ultra-detailed GitHub Pages site that documents **every lane** (DAB bundle + `air` CLI)
and the **whole MLOps lifecycle**, and we want the existing docs harmonized into it.

## Decisions (locked)

- **Tooling:** MkDocs + Material. `docs_dir: docs`.
- **Existing docs:** rewrite/harmonize in place; single source of truth per concept.
- **Deploy:** GitHub Actions Pages workflow on push to `main`; `mkdocs build --strict` as the
  broken-link gate. Pages source = "GitHub Actions".
- Site URL: `https://mshtelma.github.io/dais26-mlops-for-dl-on-air/`.

## Hard constraint — preserve code-referenced anchors

Source/notebooks/scripts/job-YAML cite docs by path + anchor. These MUST keep working:

- `docs/RUNBOOK.md#hf-cache-race` (barrier_dance.py, builder.py)
- `docs/RUNBOOK.md#hf-transfer-fuse-incompat` (hf_env.py, dentex_loader.py, notebook 02)
- `docs/RUNBOOK.md#ddp-trainable-only` (trainer_config.py)
- `docs/RUNBOOK.md#deployment-job` (constants.py)
- `docs/ARCHITECTURE.md#models-from-code`, `#pip-requirements-rationale` (referenced cross-doc)
- `docs/HPO.md` section titles referenced in prose by source/config/YAML: "DINOv3 A/B",
  "Round 3 returns", "Round 4 returns", "DINOv3 plateau", "DINOv3 ceiling", "Push to 0.60",
  "Multi-layer fusion"

Add explicit `{#anchor}` (attr_list) so anchors survive heading edits. **Fix** the already-broken
`#dinov2-fallback` and `#e3e4-latency-benchmark-protocol`.

Keep canonical files at their current paths: `docs/{ARCHITECTURE,RUNBOOK,HPO,BENCHMARKS,TALK,
DECK_BRIEF_DL_MLOPS}.md`. No `src/` edits (anchors preserved → comments stay valid).

## Information architecture (Material tabs)

- **Home** — `docs/index.md`
- **Get Started** — overview · prerequisites · install-auth · quickstart-dab · quickstart-air
- **The Two Lanes** — parity · dab · air · named-configuration
- **MLOps Lifecycle** — overview · setup-data · train-challenger · hpo-sweep ·
  eval-approve-promote · serve · embeddings-vs-drift · rollback (DAB ∥ air content tabs per stage)
- **Scenarios** — switch-backbone · dinov2-fallback · production-deploy · serving-smoke-test ·
  vector-search-query · drift-demo · cicd · env-overrides
- **Architecture** — harmonized `docs/ARCHITECTURE.md` (engineering-rationale anchors live here)
- **Reference** — notebooks · jobs · scripts · configuration · uc-resources · troubleshooting · cli
- **Project** — HPO log · Benchmarks · Operations/Runbook · The Talk · Deck Brief

## Conventions

- DAB vs air via Material content tabs (`=== "DAB"` / `=== "air CLI"`).
- Admonitions for war stories/gotchas (letterbox skew, broken-DINOv3 registration, ai_gateway
  placement, FUSE/hf_transfer, NCCL dead-rank).
- Consolidate: two-phase deploy → Architecture; named config → Named Configuration;
  troubleshooting → one Reference table; quickstarts → Get Started (root README trimmed to pointer).
- Preserve every number, run name, version pin, and rationale verbatim in meaning.

## Verification

- `mkdocs build --strict` (CI + local) — fails on any broken internal link/anchor.
- Grep check that the code-referenced anchors resolve.
- `make test` once — confirm zero code changes.
- `mkdocs serve` local visual pass.

## Files

- Add: `mkdocs.yml`, `docs/requirements-docs.txt`, `.github/workflows/docs.yml`, `docs/index.md`,
  `docs/{getting-started,lanes,lifecycle,scenarios,reference}/*.md`.
- Rewrite: the 7 docs + root `README.md` + `air/README.md` + `docs/README.md`.
- No `src/` changes.

## Risks

- Fact loss during rewrite → treat existing docs as authoritative source; fact-preservation review
  before commit.
- Broken anchors → `--strict` + explicit `{#…}`.
