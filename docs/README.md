# DAIS26 documentation

📖 **The full documentation is published as a website:
<https://mshtelma.github.io/dais26-mlops-for-dl-on-air/>** (built from this `docs/` folder with
MkDocs + Material).

> This `README.md` is a folder index for browsing on GitHub. The quickstart that used to live here
> now lives in the site's **Get Started** section (it was split into prerequisites, install, and
> per-lane quickstarts). The Markdown sources below render as the corresponding site sections.

## Build / preview the site locally

```bash
pip install -r ../docs-requirements.txt
mkdocs serve            # http://127.0.0.1:8000
mkdocs build --strict   # the CI gate (fails on broken internal links/anchors)
```

## What's in this folder

| Source | Site section |
|--------|--------------|
| `index.md` | Home |
| `getting-started/` | Get Started (overview, prerequisites, install, quickstarts) |
| `lanes/` | The Two Lanes (DAB / air / named configuration) |
| `lifecycle/` | MLOps Lifecycle (setup → train → sweep → eval → promote → serve → embeddings/VS/drift → rollback) |
| `scenarios/` | Scenarios cookbook (backbone switch, DINOv2 fallback, prod deploy, smoke test, VS query, drift, CI/CD, env overrides) |
| `ARCHITECTURE.md` | Architecture (+ engineering rationale lives in `RUNBOOK.md`) |
| `reference/` | Reference (notebooks, jobs, scripts, configuration, UC map, troubleshooting, CLI) |
| `HPO.md` / `BENCHMARKS.md` / `RUNBOOK.md` / `TALK.md` / `DECK_BRIEF_DL_MLOPS.md` | Project (HPO log, benchmarks, runbook, talk, deck brief) |

Start at the [site home](index.md) or [Overview & mental model](getting-started/overview.md).
