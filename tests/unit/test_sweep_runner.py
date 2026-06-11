"""Tests for `train.sweep_runner.SweepRunner` — the lane-agnostic sweep brain.

Everything runs single-process with a fake MLflow client, a fake launch
callable, and a recording broadcast, covering both the rank-0 orchestration
path and the worker command loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from dais26_dentex.train.sweep_runner import (
    SweepAbortedError,
    SweepOutcome,
    SweepRunner,
    SweepSpec,
)

MODEL = "ml_dev.dais26_vfm.cradio_detector"
METRIC = "val/best_mAP_50"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _RunInfo:
    run_id: str


@dataclass
class _Run:
    info: _RunInfo
    data: Any


@dataclass
class _RunData:
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class _Experiment:
    experiment_id: str


@dataclass
class _ModelVersion:
    version: str
    run_id: str | None = None
    source: str = ""


@dataclass
class _LoggedModelMetric:
    key: str
    value: float


@dataclass
class _LoggedModel:
    metrics: list[_LoggedModelMetric] = field(default_factory=list)


class FakeMlflowClient:
    """Just enough MlflowClient surface for SweepRunner."""

    def __init__(self) -> None:
        self.runs: dict[str, _Run] = {}
        self.experiments: dict[str, str] = {"existing-exp": "exp-1"}
        self.versions: list[_ModelVersion] = []
        self.logged_models: dict[str, _LoggedModel] = {}
        self.aliases: dict[str, str] = {}  # alias -> version
        self.tags: list[tuple[str, str, str]] = []
        self.parent_params: dict[str, Any] = {}
        self.parent_metrics: dict[str, float] = {}
        self.terminated: dict[str, str] = {}
        self._counter = 0

    # -- experiments ----------------------------------------------------
    def get_experiment_by_name(self, name: str) -> _Experiment | None:
        if name in self.experiments:
            return _Experiment(self.experiments[name])
        return None

    def create_experiment(self, name: str) -> str:
        exp_id = f"exp-{len(self.experiments) + 1}"
        self.experiments[name] = exp_id
        return exp_id

    # -- runs -------------------------------------------------------------
    def create_run(self, experiment_id: str, run_name: str | None = None, tags: dict | None = None) -> _Run:
        self._counter += 1
        run = _Run(info=_RunInfo(run_id=f"parent-{self._counter}"), data=_RunData())
        self.runs[run.info.run_id] = run
        return run

    def get_run(self, run_id: str) -> _Run:
        return self.runs[run_id]

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        self.parent_params[key] = value

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        self.parent_metrics[key] = value

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags.append((run_id, key, value))

    def set_terminated(self, run_id: str, status: str) -> None:
        self.terminated[run_id] = status

    # -- registry ---------------------------------------------------------
    def search_model_versions(self, _filter: str) -> list[_ModelVersion]:
        return list(self.versions)

    def get_logged_model(self, model_id: str) -> _LoggedModel:
        return self.logged_models[model_id]

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = str(version)


class FakeLaunch:
    """Returns a fresh run_id per call and seeds the fake client with the
    metrics/params the real Trainer would have logged."""

    def __init__(
        self,
        client: FakeMlflowClient,
        metric_by_call: list[float | None],
        register_version_from: int = 100,
    ) -> None:
        self.client = client
        self.metric_by_call = list(metric_by_call)
        self.calls: list[dict[str, Any]] = []
        self._next_version = register_version_from

    def __call__(self, cfg_kwargs: dict[str, Any]) -> str | None:
        self.calls.append(dict(cfg_kwargs))
        i = len(self.calls) - 1
        metric = self.metric_by_call[i] if i < len(self.metric_by_call) else None
        run_id = f"run-{i}"
        data = _RunData(metrics={} if metric is None else {METRIC: metric})
        if cfg_kwargs.get("register_model"):
            self._next_version += 1
            data.params["registered_version"] = str(self._next_version)
            self.client.versions.append(
                _ModelVersion(version=str(self._next_version), run_id=run_id)
            )
        self.client.runs[run_id] = _Run(info=_RunInfo(run_id=run_id), data=data)
        return run_id


class RecordingBroadcast:
    """Identity broadcast that records the command stream."""

    def __init__(self) -> None:
        self.commands: list[Any] = []

    def __call__(self, obj: Any, src: int = 0) -> Any:
        self.commands.append(obj)
        return obj


def _spec(**overrides: Any) -> SweepSpec:
    defaults: dict[str, Any] = dict(
        stage_name="test_stage",
        backbone="cradio_v4_so400m",
        pinned={"anchor_layout": "per_level", "nms_per_class": True, "lr": 2e-4},
        search_space={"focal_gamma": [2.0, 2.5]},
        trial_epochs=2,
        schedule_epochs=(5, 8),
        max_trials=2,
        register_winner=True,
        strategy="grid",
        seed=42,
    )
    defaults.update(overrides)
    return SweepSpec(**defaults)


def _runner(
    spec: SweepSpec,
    client: FakeMlflowClient,
    launch: FakeLaunch,
    broadcast: RecordingBroadcast | None = None,
) -> SweepRunner:
    return SweepRunner(
        spec,
        base_config_kwargs={
            "catalog": "ml_dev",
            "schema": "dais26_vfm",
            "experiment_name": "existing-exp",
            "model_name": "cradio_detector",
        },
        launch=launch,
        client=client,
        model_fqn=MODEL,
        broadcast=broadcast or RecordingBroadcast(),
    )


# ---------------------------------------------------------------------------
# Rank-0 orchestration
# ---------------------------------------------------------------------------


def test_happy_path_trials_retrains_and_registers() -> None:
    client = FakeMlflowClient()
    # 2 trials (0.30, 0.40), 2 schedule retrains (0.45, 0.50).
    launch = FakeLaunch(client, [0.30, 0.40, 0.45, 0.50])
    broadcast = RecordingBroadcast()
    outcome = _runner(_spec(), client, launch, broadcast).run()

    # 4 launches: every trial at trial_epochs/register=False, retrains registered.
    assert len(launch.calls) == 4
    assert [c["epochs"] for c in launch.calls] == [2, 2, 5, 8]
    assert [c["register_model"] for c in launch.calls] == [False, False, True, True]
    # pinned values flow into every config; trial override applied.
    assert all(c["anchor_layout"] == "per_level" for c in launch.calls)
    assert {launch.calls[0]["focal_gamma"], launch.calls[1]["focal_gamma"]} == {2.0, 2.5}

    # winner = trial 1 (0.40); best retrain = epochs 8 run (0.50), version 102.
    assert outcome.winner is not None and outcome.winner.trial_id == 1
    assert outcome.retrain_metric == pytest.approx(0.50)
    assert outcome.registered_version == "102"
    assert outcome.challenger_set is True
    assert client.aliases == {"challenger": "102"}

    # parent run lifecycle + nesting tags.
    assert client.terminated[outcome.parent_run_id] == "FINISHED"
    parent_tags = [(t[1], t[2]) for t in client.tags if t[0] == "run-0"]
    assert ("mlflow.parentRunId", outcome.parent_run_id) in parent_tags
    assert client.parent_params["winner_trial_id"] == 1
    assert client.parent_metrics["winner_val_best_mAP_50"] == pytest.approx(0.50)


def test_every_launch_is_preceded_by_a_matching_broadcast() -> None:
    client = FakeMlflowClient()
    launch = FakeLaunch(client, [0.30, 0.40, 0.45, 0.50])
    broadcast = RecordingBroadcast()
    _runner(_spec(), client, launch, broadcast).run()

    launch_cmds = [c for c in broadcast.commands if c[0] in ("trial", "retrain")]
    assert len(launch_cmds) == len(launch.calls)
    for cmd, call in zip(launch_cmds, launch.calls, strict=True):
        assert cmd[2] == call
    # final command closes the loop for the workers.
    assert broadcast.commands[-1][0] == "done"


def test_gate_failure_restores_prior_best_alias() -> None:
    client = FakeMlflowClient()
    # A prior version with a HIGHER LoggedModel metric than the new winner.
    client.versions.append(_ModelVersion(version="7", run_id="old-run", source="models:/lm-7"))
    client.logged_models["lm-7"] = _LoggedModel(metrics=[_LoggedModelMetric(METRIC, 0.90)])
    launch = FakeLaunch(client, [0.30, 0.40, 0.45, 0.50])
    outcome = _runner(_spec(), client, launch).run()

    assert outcome.challenger_set is False
    assert client.aliases == {"challenger": "7"}  # restored, not the new 102
    assert outcome.registered_version == "102"  # still registered, just not challenger


def test_gate_reads_run_metric_fallback_for_classic_versions() -> None:
    client = FakeMlflowClient()
    # Classic version: dbfs source (no LoggedModel link), metric only on the run.
    client.runs["old-run"] = _Run(
        info=_RunInfo(run_id="old-run"), data=_RunData(metrics={METRIC: 0.95})
    )
    client.versions.append(_ModelVersion(version="3", run_id="old-run", source="dbfs:/x/y"))
    launch = FakeLaunch(client, [0.30, 0.40, 0.45, 0.50])
    outcome = _runner(_spec(), client, launch).run()

    assert outcome.challenger_set is False
    assert client.aliases == {"challenger": "3"}


def test_measure_only_stage_retrains_without_registering() -> None:
    client = FakeMlflowClient()
    launch = FakeLaunch(client, [0.30, 0.40, 0.45])
    spec = _spec(register_winner=False, schedule_epochs=(5,))
    outcome = _runner(spec, client, launch).run()

    assert [c["register_model"] for c in launch.calls] == [False, False, False]
    assert outcome.retrain_metric == pytest.approx(0.45)
    assert outcome.registered_version is None
    assert outcome.challenger_set is False
    assert client.aliases == {}  # alias untouched


def test_legacy_no_register_skips_retrain_entirely() -> None:
    client = FakeMlflowClient()
    launch = FakeLaunch(client, [0.30, 0.40])
    spec = _spec(register_winner=False, retrain_winner=False)
    outcome = _runner(spec, client, launch).run()

    assert len(launch.calls) == 2  # trials only
    assert outcome.winner is not None and outcome.winner.trial_id == 1
    assert outcome.retrain_run_id is None


def test_no_metrics_means_no_retrain_and_no_winner() -> None:
    client = FakeMlflowClient()
    launch = FakeLaunch(client, [None, None])
    outcome = _runner(_spec(), client, launch).run()

    assert len(launch.calls) == 2
    assert outcome.winner is None
    assert outcome.registered_version is None
    assert client.terminated[outcome.parent_run_id] == "FINISHED"


def test_invalid_pinned_config_fails_before_any_launch() -> None:
    client = FakeMlflowClient()
    launch = FakeLaunch(client, [])
    spec = _spec(pinned={"lr": -1.0})  # validate() must reject
    broadcast = RecordingBroadcast()
    with pytest.raises(ValueError, match="lr"):
        _runner(spec, client, launch, broadcast).run()

    assert launch.calls == []  # nothing dispatched to GPUs
    assert broadcast.commands and broadcast.commands[-1][0] == "abort"
    parent_id = next(iter(client.terminated))
    assert client.terminated[parent_id] == "FAILED"


def test_rank0_launch_failure_broadcasts_abort_and_fails_parent() -> None:
    client = FakeMlflowClient()

    def exploding_launch(cfg_kwargs: dict[str, Any]) -> str | None:
        raise RuntimeError("CUDA OOM")

    broadcast = RecordingBroadcast()
    runner = SweepRunner(
        _spec(),
        base_config_kwargs={"catalog": "c", "schema": "s", "experiment_name": "existing-exp"},
        launch=exploding_launch,
        client=client,
        model_fqn=MODEL,
        broadcast=broadcast,
    )
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        runner.run()
    assert broadcast.commands[-1] == ("abort", "RuntimeError('CUDA OOM')")
    assert list(client.terminated.values()) == ["FAILED"]


def test_experiment_created_when_missing() -> None:
    client = FakeMlflowClient()
    launch = FakeLaunch(client, [0.3, 0.4, 0.5, 0.6])
    runner = SweepRunner(
        _spec(),
        base_config_kwargs={"catalog": "c", "schema": "s", "experiment_name": "brand-new-exp"},
        launch=launch,
        client=client,
        model_fqn=MODEL,
        broadcast=RecordingBroadcast(),
    )
    runner.run()
    assert "brand-new-exp" in client.experiments


def test_missing_experiment_name_raises_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    client = FakeMlflowClient()
    runner = SweepRunner(
        _spec(),
        base_config_kwargs={"catalog": "c", "schema": "s"},
        launch=FakeLaunch(client, []),
        client=client,
        model_fqn=MODEL,
        broadcast=RecordingBroadcast(),
    )
    with pytest.raises(ValueError, match="MLFLOW_EXPERIMENT_NAME"):
        runner.run()


# ---------------------------------------------------------------------------
# Worker command loop (simulated non-rank0)
# ---------------------------------------------------------------------------


class ScriptedBroadcast:
    """Feeds a worker a pre-scripted command stream."""

    def __init__(self, commands: list[Any]) -> None:
        self.commands = list(commands)

    def __call__(self, obj: Any, src: int = 0) -> Any:
        return self.commands.pop(0)


def _worker_runner(commands: list[Any], launch: Any) -> SweepRunner:
    return SweepRunner(
        _spec(),
        base_config_kwargs={"catalog": "c", "schema": "s"},
        launch=launch,
        client=FakeMlflowClient(),
        model_fqn=MODEL,
        broadcast=ScriptedBroadcast(commands),
    )


def test_worker_executes_commands_and_returns_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dais26_dentex.train.sweep_runner.is_rank0", lambda: False)
    launched: list[dict[str, Any]] = []

    import dataclasses

    done_payload = dataclasses.asdict(
        SweepOutcome(parent_run_id="parent-1", registered_version="9", challenger_set=True)
    )
    commands = [
        ("trial", 0, {"epochs": 2}),
        ("retrain", 5, {"epochs": 5}),
        ("done", done_payload),
    ]
    outcome = _worker_runner(commands, lambda kw: launched.append(kw)).run()

    assert launched == [{"epochs": 2}, {"epochs": 5}]
    assert outcome.parent_run_id == "parent-1"
    assert outcome.registered_version == "9"
    assert outcome.challenger_set is True


def test_worker_raises_on_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dais26_dentex.train.sweep_runner.is_rank0", lambda: False)
    with pytest.raises(SweepAbortedError, match="boom"):
        _worker_runner([("abort", "boom")], lambda kw: None).run()


def test_worker_raises_on_protocol_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dais26_dentex.train.sweep_runner.is_rank0", lambda: False)
    with pytest.raises(SweepAbortedError, match="unknown sweep command"):
        _worker_runner([("dance", None)], lambda kw: None).run()


# ---------------------------------------------------------------------------
# SweepSpec.from_stage
# ---------------------------------------------------------------------------


def test_from_stage_maps_campaign_fields() -> None:
    from dais26_dentex.config.campaigns import CAMPAIGN_STAGES

    spec = SweepSpec.from_stage("dinov3_s1", CAMPAIGN_STAGES["dinov3_s1"], seed=7)
    assert spec.stage_name == "dinov3_s1"
    assert spec.backbone == "dinov3_vitl16"
    assert spec.schedule_epochs == (50, 75)
    assert spec.max_trials == 6
    assert spec.register_winner is False
    assert spec.retrain_winner is True  # measure-only stages still retrain
    assert spec.seed == 7
