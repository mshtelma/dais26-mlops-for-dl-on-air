"""SweepRunner — the one side-effecting sweep brain behind both launch lanes.

Owns everything `notebooks/02b_hpo_sweep.py` used to do inline: the parent
MLflow run, the sequential trial loop, child-run nesting, winner selection,
the schedule retrains, and the `@challenger` best-in-experiment gate. The two
launch surfaces differ ONLY in the `launch` callable they inject:

* notebook lane (02b): `launch` dispatches one `serverless_gpu.@distributed`
  job per trial and returns the rank-0 run_id. The runner executes on the
  single-process notebook driver, where `broadcast_object` degrades to
  identity — the command loop collapses to a plain for-loop.
* sgcli/torchrun lane (`train.sweep_cli`): every rank executes the runner.
  `launch` constructs `Trainer(cfg, manage_process_group=False)` in-process.
  Rank 0 is the sole decision-maker; every coordinated step starts with one
  `broadcast_object` command so workers never diverge (and a rank-0 failure
  broadcasts `abort` instead of leaving peers hanging in the next collective
  until the NCCL timeout).

Command protocol (rank0 -> all): ``("trial", trial_id, cfg_kwargs)``,
``("retrain", epochs, cfg_kwargs)``, ``("done", outcome_dict)``,
``("abort", message)``.

The pure parts (trial enumeration, winner selection, the gate comparison)
stay in `train.sweep`; MLflow client calls go through the injected `client`
so unit tests run on a plain fake.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from dais26_dentex.config.campaigns import CampaignStage
from dais26_dentex.config.constants import ALIAS_CANDIDATE
from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.distributed import broadcast_object, is_rank0
from dais26_dentex.train.sweep import (
    TrialResult,
    beats_experiment_best,
    iter_trials,
    select_best,
)

logger = logging.getLogger(__name__)

# A launch callable receives fully-merged TrainerConfig kwargs and returns the
# MLflow run_id on the deciding process (rank 0 / notebook driver), None
# elsewhere. Kwargs (not a TrainerConfig) so the notebook lane can pickle them
# into the @distributed closure exactly as 02b always did.
LaunchFn = Callable[[dict[str, Any]], str | None]


class SweepAbortedError(RuntimeError):
    """Raised on every rank when rank 0 aborted the sweep mid-flight."""


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """Everything that defines one sweep execution (stage or legacy)."""

    stage_name: str
    backbone: str
    pinned: Mapping[str, Any]
    search_space: Mapping[str, Any]
    trial_epochs: int
    schedule_epochs: tuple[int, ...]
    max_trials: int
    register_winner: bool
    # Retrain the winner at the schedule epochs even when not registering
    # (measure-only campaign stages need the full-length metric for their
    # gate). The legacy SWEEP_* path skipped the retrain entirely when
    # SWEEP_REGISTER_WINNER was off — pass retrain_winner=register_winner there.
    retrain_winner: bool = True
    strategy: str = "random"
    seed: int = 42
    primary_metric: str = "val/best_mAP_50"

    @classmethod
    def from_stage(
        cls,
        name: str,
        stage: CampaignStage,
        *,
        strategy: str = "random",
        seed: int = 42,
        primary_metric: str = "val/best_mAP_50",
    ) -> SweepSpec:
        return cls(
            stage_name=name,
            backbone=stage.backbone,
            pinned=dict(stage.pinned),
            search_space=dict(stage.search_space),
            trial_epochs=stage.trial_epochs,
            schedule_epochs=tuple(stage.schedule_epochs),
            max_trials=stage.max_trials,
            register_winner=stage.register_winner,
            retrain_winner=True,
            strategy=strategy,
            seed=seed,
            primary_metric=primary_metric,
        )


@dataclass(frozen=True, slots=True)
class SweepOutcome:
    """What a sweep produced; identical on every rank."""

    parent_run_id: str | None = None
    trials: tuple[TrialResult, ...] = field(default_factory=tuple)
    winner: TrialResult | None = None
    retrain_run_id: str | None = None
    retrain_metric: float | None = None
    registered_version: str | None = None
    challenger_set: bool = False


def _version_model_id(mv: Any) -> str | None:
    """Resolve a model version's MLflow 3 LoggedModel id.

    UC does NOT populate `ModelVersion.model_id`; the LoggedModel link lives in
    `source` as `models:/<model_id>` for versions registered the MLflow 3 way.
    Classic versions (registered from a run artifact) have a `dbfs:/...` source
    and no link. Prefer an explicit `model_id` if a future client sets it.
    """
    mid = getattr(mv, "model_id", None)
    if mid:
        return mid
    src = getattr(mv, "source", "") or ""
    return src.split("models:/", 1)[1] if src.startswith("models:/") else None


class SweepRunner:
    """Drives one sweep: trials -> winner -> schedule retrains -> gate."""

    def __init__(
        self,
        spec: SweepSpec,
        *,
        base_config_kwargs: Mapping[str, Any],
        launch: LaunchFn,
        client: Any,
        model_fqn: str,
        candidate_alias: str = ALIAS_CANDIDATE,
        resolve_overrides: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        broadcast: Callable[..., Any] = broadcast_object,
    ) -> None:
        """Args:
        spec: the sweep definition (`SweepSpec.from_stage` for campaign stages).
        base_config_kwargs: environment-side TrainerConfig kwargs (catalog,
            schema, volume_path, cache_dir, experiment_name, model_name,
            backbone_revision, batch_size defaults, ...). The runner merges
            `spec.pinned` and each trial's override on top.
        launch: lane-specific trial executor (see `LaunchFn`).
        client: an `MlflowClient`-shaped object; only the methods used here are
            required (create_run, log_param, log_metric, set_tag, get_run,
            set_terminated, search_model_versions, get_logged_model,
            set_registered_model_alias, get_experiment_by_name,
            create_experiment).
        model_fqn: catalog.schema.model the winner registers into.
        resolve_overrides: optional hook mapping a trial's sampled params onto
            concrete TrainerConfig kwargs (e.g. anchor calibration).
        broadcast: injected for tests; production uses `broadcast_object`.
        """
        self.spec = spec
        self.base = dict(base_config_kwargs)
        self.launch = launch
        self.client = client
        self.model_fqn = model_fqn
        self.candidate_alias = candidate_alias
        self.resolve_overrides = resolve_overrides or (lambda params: dict(params))
        self.broadcast = broadcast

    # ------------------------------------------------------------------
    # Config assembly
    # ------------------------------------------------------------------

    def _trial_config_kwargs(self, override: Mapping[str, Any], *, epochs: int, register: bool) -> dict[str, Any]:
        """base kwargs + spec.pinned + trial override (override wins), with the
        schedule + registration flags applied. Validated before any GPU is
        touched so a typo'd field fails in milliseconds, not after dispatch."""
        merged: dict[str, Any] = {
            **self.base,
            "backbone_name": self.spec.backbone,
            "epochs": epochs,
            "register_model": register,
            "set_candidate_alias": register,
            **self.spec.pinned,
            **override,
        }
        # pinned/override may legitimately re-pin epochs for degenerate
        # 1-trial stages; the explicit schedule argument wins.
        merged["epochs"] = epochs
        merged["register_model"] = register
        merged["set_candidate_alias"] = register
        TrainerConfig.from_dict(merged).validate()
        return merged

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self) -> SweepOutcome:
        """Execute the sweep. Returns the same `SweepOutcome` on every rank."""
        if is_rank0():
            try:
                outcome = self._run_rank0()
            except SweepAbortedError:
                raise
            except BaseException as e:
                # Tell the workers before unwinding so nobody is left blocked
                # in the next collective until the NCCL timeout.
                try:
                    self.broadcast(("abort", repr(e)))
                except Exception:
                    logger.exception("Failed to broadcast sweep abort")
                raise
            self.broadcast(("done", asdict(outcome)))
            return outcome
        return self._run_worker()

    def _run_worker(self) -> SweepOutcome:
        """Non-rank0 torchrun ranks: obey rank 0's command stream."""
        while True:
            cmd = self.broadcast(None)
            kind = cmd[0]
            if kind in ("trial", "retrain"):
                self.launch(cmd[2])
            elif kind == "done":
                payload = dict(cmd[1])
                trials = tuple(TrialResult(**t) for t in payload.pop("trials", ()))
                winner = payload.pop("winner", None)
                return SweepOutcome(
                    trials=trials,
                    winner=TrialResult(**winner) if winner else None,
                    **payload,
                )
            elif kind == "abort":
                raise SweepAbortedError(f"rank 0 aborted the sweep: {cmd[1]}")
            else:  # defensive: protocol drift between package versions
                raise SweepAbortedError(f"unknown sweep command {kind!r}")

    def _dispatch(self, kind: str, cfg_kwargs: dict[str, Any], tag: Any) -> str | None:
        """Broadcast one launch command, then run it locally (rank 0 trains too)."""
        self.broadcast((kind, tag, cfg_kwargs))
        return self.launch(cfg_kwargs)

    # ------------------------------------------------------------------
    # Rank-0 orchestration (ported from notebooks/02b_hpo_sweep.py)
    # ------------------------------------------------------------------

    def _resolve_experiment_id(self) -> str:
        name = self.base.get("experiment_name") or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if not name:
            raise ValueError(
                "SweepRunner needs an MLflow experiment: set experiment_name in "
                "base_config_kwargs or export MLFLOW_EXPERIMENT_NAME."
            )
        exp = self.client.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        return self.client.create_experiment(name)

    def _run_rank0(self) -> SweepOutcome:
        spec = self.spec
        experiment_id = self._resolve_experiment_id()
        # Client API only — a fluent `mlflow.start_run()` parent would collide
        # with the fluent run `Trainer.run()` opens in this same process on
        # the torchrun lane.
        parent = self.client.create_run(
            experiment_id,
            run_name=f"hpo-sweep-{spec.backbone}",
            tags={"sweep_stage": spec.stage_name},
        )
        parent_run_id = parent.info.run_id
        for key, value in {
            "sweep_stage": spec.stage_name,
            "sweep_strategy": spec.strategy,
            "sweep_max_trials": spec.max_trials,
            "sweep_trial_epochs": spec.trial_epochs,
            "sweep_primary_metric": spec.primary_metric,
            "sweep_backbone": spec.backbone,
        }.items():
            self.client.log_param(parent_run_id, key, value)

        try:
            outcome = self._trials_and_retrain(parent_run_id)
            self.client.set_terminated(parent_run_id, "FINISHED")
        except BaseException:
            try:
                self.client.set_terminated(parent_run_id, "FAILED")
            except Exception:
                logger.exception("Could not mark parent run %s FAILED", parent_run_id)
            raise
        return outcome

    def _trials_and_retrain(self, parent_run_id: str) -> SweepOutcome:
        spec = self.spec
        trials = list(
            iter_trials(
                dict(spec.search_space),
                strategy=spec.strategy,
                max_trials=spec.max_trials,
                seed=spec.seed,
            )
        )
        logger.info(
            "Sweep %s: %d planned trials (%s, seed=%d)",
            spec.stage_name, len(trials), spec.strategy, spec.seed,
        )

        results: list[TrialResult] = []
        for trial in trials:
            override = self.resolve_overrides(dict(trial.params))
            cfg_kwargs = self._trial_config_kwargs(override, epochs=spec.trial_epochs, register=False)
            logger.info("Trial %d/%d: %s", trial.trial_id, len(trials) - 1, override)
            run_id = self._dispatch("trial", cfg_kwargs, trial.trial_id)

            metric = None
            if run_id:
                # Nest the trial's training run under the parent for the UI,
                # and read back the best metric the Trainer logged.
                self.client.set_tag(run_id, "mlflow.parentRunId", parent_run_id)
                self.client.set_tag(run_id, "sweep_trial_id", str(trial.trial_id))
                metric = self.client.get_run(run_id).data.metrics.get(spec.primary_metric)
            results.append(TrialResult(trial_id=trial.trial_id, params=override, metric=metric, run_id=run_id))
            logger.info("Trial %d: run_id=%s %s=%s", trial.trial_id, run_id, spec.primary_metric, metric)

        best = select_best(results, higher_is_better=True)
        if best is None:
            logger.warning("No trial produced a metric — nothing to retrain or register.")
            return SweepOutcome(parent_run_id=parent_run_id, trials=tuple(results))

        self.client.log_param(parent_run_id, "winner_trial_id", best.trial_id)
        self.client.log_param(parent_run_id, "winner_run_id", best.run_id or "")
        logger.info(
            "Winner: trial %d (%s=%s) params=%s",
            best.trial_id, spec.primary_metric, best.metric, best.params,
        )

        if not spec.retrain_winner:
            logger.info("retrain_winner=False — sweep ends at trial selection.")
            return SweepOutcome(parent_run_id=parent_run_id, trials=tuple(results), winner=best)

        winner_rid, winner_metric, winner_version = self._retrain_schedules(best)
        self.client.log_metric(
            parent_run_id,
            f"winner_{spec.primary_metric.replace('/', '_')}",
            winner_metric if winner_metric is not None else 0.0,
        )

        challenger_set = False
        if spec.register_winner and winner_version is not None:
            challenger_set = self._challenger_gate(winner_metric, winner_version, winner_rid)
        elif spec.register_winner:
            logger.warning("No registered_version on the winning run (%s); alias left as-is.", winner_rid)
        else:
            logger.info(
                "register_winner=False — measured full-length metric only; @%s unchanged.",
                self.candidate_alias,
            )
        return SweepOutcome(
            parent_run_id=parent_run_id,
            trials=tuple(results),
            winner=best,
            retrain_run_id=winner_rid,
            retrain_metric=winner_metric,
            registered_version=winner_version,
            challenger_set=challenger_set,
        )

    def _retrain_schedules(self, best: TrialResult) -> tuple[str | None, float | None, str | None]:
        """Retrain the winner at every schedule, return the better run's
        (run_id, metric, registered_version). Measure-only stages retrain with
        register=False purely for the full-length metric."""
        spec = self.spec
        schedule_runs: list[tuple[str | None, float | None, str | None]] = []
        for epochs in spec.schedule_epochs:
            verb = "registering as" if spec.register_winner else "measuring (no register)"
            logger.info("Retraining winner at %d epochs, %s %s...", epochs, verb, self.model_fqn)
            kwargs = self._trial_config_kwargs(best.params, epochs=epochs, register=spec.register_winner)
            rid = self._dispatch("retrain", kwargs, epochs)
            metric = version = None
            if rid:
                run_data = self.client.get_run(rid).data
                metric = run_data.metrics.get(spec.primary_metric)
                version = run_data.params.get("registered_version")
            logger.info("  epochs=%d: run_id=%s %s=%s version=%s", epochs, rid, spec.primary_metric, metric, version)
            schedule_runs.append((rid, metric, version))
        # None-safe pick: trained-with-metric beats no-metric, then higher wins.
        return max(schedule_runs, key=lambda rmv: (rmv[1] is not None, rmv[1] or -1.0))

    def _logged_model_metric(self, model_id: str | None, key: str) -> float | None:
        """Read `key` off a version's MLflow 3 LoggedModel, or None.

        The Trainer links best-epoch val metrics to the LoggedModel
        (`Trainer._log_metrics_to_logged_model`), so the gate compares versions
        straight off the Models tab — independent of whether the source run is
        still queryable.
        """
        if not model_id:
            return None
        try:
            lm = self.client.get_logged_model(model_id)
        except Exception:
            return None
        for metric in getattr(lm, "metrics", None) or []:
            if metric.key == key:
                return float(metric.value)
        return None

    def _challenger_gate(self, winner_metric: float | None, winner_version: str, winner_rid: str | None) -> bool:
        """Best-in-experiment gate: keep @challenger on the new version only if
        it strictly beats every prior version's metric; otherwise restore the
        alias to the prior best so a regression never triggers the deployment
        job. Returns True when the new version holds the alias."""
        spec = self.spec
        prior: list[tuple[str, float]] = []
        for mv in self.client.search_model_versions(f"name='{self.model_fqn}'"):
            if str(mv.version) == str(winner_version):
                continue
            # Prefer the LoggedModel metric (MLflow 3); fall back to the run
            # metric for legacy versions without a linked LoggedModel metric.
            m = self._logged_model_metric(_version_model_id(mv), spec.primary_metric)
            if m is None and mv.run_id:
                try:
                    m = self.client.get_run(mv.run_id).data.metrics.get(spec.primary_metric)
                except Exception:
                    m = None
            if m is not None:
                prior.append((str(mv.version), float(m)))

        if beats_experiment_best(winner_metric, [m for _, m in prior], higher_is_better=True):
            self.client.set_registered_model_alias(self.model_fqn, self.candidate_alias, winner_version)
            bar = max((m for _, m in prior), default=None)
            logger.info(
                "@%s -> version %s (run %s); %s=%s beats prior best %s.",
                self.candidate_alias, winner_version, winner_rid, spec.primary_metric, winner_metric, bar,
            )
            return True
        if prior:
            best_prior_version = max(prior, key=lambda vm: vm[1])[0]
            self.client.set_registered_model_alias(self.model_fqn, self.candidate_alias, best_prior_version)
            logger.warning(
                "Gate NOT passed: winner %s=%s does not beat prior best %s; restored @%s -> version %s. "
                "New version %s registered but not challenger.",
                spec.primary_metric, winner_metric, max(m for _, m in prior),
                self.candidate_alias, best_prior_version, winner_version,
            )
            return False
        self.client.set_registered_model_alias(self.model_fqn, self.candidate_alias, winner_version)
        logger.info(
            "@%s -> version %s (run %s); first measurable version.",
            self.candidate_alias, winner_version, winner_rid,
        )
        return True


__all__ = [
    "LaunchFn",
    "SweepAbortedError",
    "SweepOutcome",
    "SweepRunner",
    "SweepSpec",
]
