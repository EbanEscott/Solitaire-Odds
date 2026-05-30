"""Experiment spec parsing and validation for the experiments driver."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .architectures import (
    ARCHIVED_EPISODE_LOGS_DATASET_KIND,
    RUN_COLLECTION_EPISODE_LOGS_DATASET_KIND,
    SUPPORTED_DATASET_KINDS,
    SUPPORTED_TRAINING_KINDS,
    get_adapter_for_family,
    get_adapter_for_training_kind,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
SUPPORTED_API_VERSION = "v1"
SUPPORTED_TASK_KINDS = {
    "noop",
    "command",
    "endgame_collect_shard",
    "policy_value_train",
    "alpha_level_eval_shard",
    "evaluation_report",
}
SUPPORTED_COLLECTION_KINDS = {"endgame_level_dataset"}
SUPPORTED_EVALUATION_KINDS = {"alpha_level_suite"}
DEFAULT_ENDGAME_COLLECTION_TEST = (
    "ai.games.training.EndgameTrainingDataGenerator.testGenerateEndgameDataset"
)
DEFAULT_ALPHA_EVALUATION_TEST = "ai.games.training.AlphaSolitaireLevelTest.testOpponent"


class SpecValidationError(ValueError):
    """Raised when an experiment spec is structurally invalid."""


@dataclass(frozen=True)
class WorkflowTaskSpec:
    """Task definition loaded from a spec or derived from the default workflow."""

    name: str
    kind: str
    payload_key: str | None
    payload_overrides: Mapping[str, Any] | None
    command: tuple[str, ...]
    working_directory: str | None


@dataclass(frozen=True)
class ExperimentSpec:
    """Validated experiment configuration consumed by the experiments driver."""

    spec_path: Path
    spec_hash: str
    api_version: str
    experiment_id: str
    description: str
    architecture: Mapping[str, Any]
    collection: Mapping[str, Any]
    dataset: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    workflow_tasks: tuple[WorkflowTaskSpec, ...]
    raw: Mapping[str, Any]

    def to_json(self) -> str:
        """Return a canonical JSON representation for persistence and hashing."""
        return json.dumps(self.raw, indent=2, sort_keys=True)


def _as_dict(name: str, value: Any) -> Dict[str, Any]:
    """Normalize optional spec sections to dictionaries for downstream consumers."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SpecValidationError(f"'{name}' must be an object")
    return dict(value)


def _resolve_repo_path(path_text: str) -> Path:
    """Resolve a spec path relative to the repository root unless it is already absolute."""

    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _validate_sources(dataset: Mapping[str, Any]) -> None:
    """Validate dataset source paths early so invalid specs fail before a run is created."""

    sources = dataset.get("sources", [])
    if not isinstance(sources, list):
        raise SpecValidationError("'dataset.sources' must be an array")
    for source in sources:
        if not isinstance(source, str):
            raise SpecValidationError("'dataset.sources' values must be strings")
        if not _resolve_repo_path(source).exists():
            raise SpecValidationError(f"dataset source does not exist: {source}")


def _validate_dataset(
    dataset: Mapping[str, Any],
    collection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate dataset configuration and return a normalized representation."""

    if not dataset:
        return {}

    kind = dataset.get("kind")
    if not isinstance(kind, str) or kind not in SUPPORTED_DATASET_KINDS:
        raise SpecValidationError(
            f"'dataset.kind' must be one of {sorted(SUPPORTED_DATASET_KINDS)}"
        )

    if kind == ARCHIVED_EPISODE_LOGS_DATASET_KIND:
        _validate_sources(dataset)
        return {
            "kind": kind,
            "sources": list(dataset.get("sources", [])),
        }

    if dataset.get("sources") not in (None, []):
        raise SpecValidationError(
            f"'dataset.kind={RUN_COLLECTION_EPISODE_LOGS_DATASET_KIND}' does not accept explicit 'dataset.sources'; it resolves successful collection artifacts from the same run"
        )
    if not collection:
        raise SpecValidationError(
            f"'dataset.kind={RUN_COLLECTION_EPISODE_LOGS_DATASET_KIND}' requires a 'collection' section in the same spec"
        )

    return {
        "kind": kind,
        "sources": [],
    }


def _require_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    """Validate and normalize integer spec fields with optional lower bounds."""

    if not isinstance(value, int):
        raise SpecValidationError(f"'{name}' must be an integer")
    if minimum is not None and value < minimum:
        raise SpecValidationError(f"'{name}' must be >= {minimum}")
    return value


def _require_bool(name: str, value: Any) -> bool:
    """Validate and normalize boolean spec fields."""

    if not isinstance(value, bool):
        raise SpecValidationError(f"'{name}' must be a boolean")
    return value


def _validate_collection(collection: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the collection section and return normalized values."""

    if not collection:
        return {}

    kind = collection.get("kind")
    if not isinstance(kind, str) or kind not in SUPPORTED_COLLECTION_KINDS:
        raise SpecValidationError(
            f"'collection.kind' must be one of {sorted(SUPPORTED_COLLECTION_KINDS)}"
        )

    level = _require_int("collection.level", collection.get("level"), minimum=1)
    shard_count = _require_int("collection.shard_count", collection.get("shard_count"), minimum=1)
    games_per_shard = _require_int(
        "collection.games_per_shard",
        collection.get("games_per_shard"),
        minimum=1,
    )
    randomise = _require_bool("collection.randomise", collection.get("randomise", False))

    if shard_count > 1 and not randomise:
        raise SpecValidationError(
            "multi-shard endgame collection requires 'collection.randomise=true' so shards do not duplicate deterministic data"
        )

    seed_base = collection.get("seed_base")
    if seed_base is None:
        seed_base = 0
    else:
        seed_base = _require_int("collection.seed_base", seed_base)

    engine_test = collection.get("engine_test", DEFAULT_ENDGAME_COLLECTION_TEST)
    if not isinstance(engine_test, str) or not engine_test:
        raise SpecValidationError("'collection.engine_test' must be a non-empty string")

    return {
        "kind": kind,
        "level": level,
        "shard_count": shard_count,
        "games_per_shard": games_per_shard,
        "randomise": randomise,
        "seed_base": seed_base,
        "engine_test": engine_test,
    }


def _validate_architecture(architecture: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the architecture section and normalize family-specific defaults."""

    if not architecture:
        return {}

    family = architecture.get("family")
    if not isinstance(family, str) or not family:
        raise SpecValidationError("'architecture.family' must be provided")

    try:
        return get_adapter_for_family(family).validate_architecture(architecture)
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc


def _validate_training(
    training: Mapping[str, Any],
    architecture: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the training section and normalize its defaults."""

    if not training:
        return {}

    kind = training.get("kind")
    if kind is None:
        return dict(training)
    if not isinstance(kind, str) or kind not in SUPPORTED_TRAINING_KINDS:
        raise SpecValidationError(
            f"'training.kind' must be one of {sorted(SUPPORTED_TRAINING_KINDS)}"
        )

    family = architecture.get("family")
    if not isinstance(family, str) or not family:
        raise SpecValidationError("'training.kind' requires 'architecture.family'")

    try:
        adapter = get_adapter_for_family(family)
        training_adapter = get_adapter_for_training_kind(kind)
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc

    if training_adapter.family != adapter.family:
        raise SpecValidationError(
            f"'training.kind={kind}' requires 'architecture.family={training_adapter.family}'"
        )

    try:
        normalized_training = training_adapter.validate_training(training, dataset)
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc

    resume_from = normalized_training.get("resume_from")
    if resume_from is not None and not _resolve_repo_path(str(resume_from)).exists():
        raise SpecValidationError(f"training resume checkpoint does not exist: {resume_from}")
    return normalized_training


def _validate_evaluation(
    evaluation: Mapping[str, Any],
    architecture: Mapping[str, Any],
    training: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the evaluation section and normalize its defaults."""

    if not evaluation:
        return {}

    kind = evaluation.get("kind")
    if kind is None:
        return dict(evaluation)
    if not isinstance(kind, str) or kind not in SUPPORTED_EVALUATION_KINDS:
        raise SpecValidationError(
            f"'evaluation.kind' must be one of {sorted(SUPPORTED_EVALUATION_KINDS)}"
        )

    family = architecture.get("family")
    if not isinstance(family, str) or not family:
        raise SpecValidationError("'evaluation.kind' requires 'architecture.family'")

    try:
        adapter = get_adapter_for_family(family)
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc

    explicit_checkpoint = evaluation.get("checkpoint")
    if explicit_checkpoint is None and training.get("kind") != adapter.training_kind:
        raise SpecValidationError(
            f"'evaluation.kind=alpha_level_suite' requires either 'evaluation.checkpoint' or a preceding 'training.kind={adapter.training_kind}' section"
        )
    if explicit_checkpoint is not None:
        if not isinstance(explicit_checkpoint, str) or not explicit_checkpoint:
            raise SpecValidationError("'evaluation.checkpoint' must be a non-empty string when provided")
        if not _resolve_repo_path(explicit_checkpoint).exists():
            raise SpecValidationError(f"evaluation checkpoint does not exist: {explicit_checkpoint}")

    level = _require_int("evaluation.level", evaluation.get("level"), minimum=1)
    games_per_shard = _require_int(
        "evaluation.games_per_shard",
        evaluation.get("games_per_shard"),
        minimum=1,
    )
    seed_start = _require_int("evaluation.seed_start", evaluation.get("seed_start", 0), minimum=0)
    seed_end = _require_int("evaluation.seed_end", evaluation.get("seed_end"), minimum=1)
    if seed_end <= seed_start:
        raise SpecValidationError("'evaluation.seed_end' must be greater than 'evaluation.seed_start'")

    service_host = evaluation.get("service_host", "127.0.0.1")
    if not isinstance(service_host, str) or not service_host:
        raise SpecValidationError("'evaluation.service_host' must be a non-empty string")

    service_port = _require_int(
        "evaluation.service_port",
        evaluation.get("service_port", 8000),
        minimum=1,
    )
    mcts_simulations = _require_int(
        "evaluation.mcts_simulations",
        evaluation.get("mcts_simulations", 256),
        minimum=1,
    )

    mcts_max_depth = evaluation.get("mcts_max_depth")
    if mcts_max_depth is not None:
        mcts_max_depth = _require_int("evaluation.mcts_max_depth", mcts_max_depth, minimum=1)

    mcts_cpuct = evaluation.get("mcts_cpuct")
    if mcts_cpuct is not None:
        if not isinstance(mcts_cpuct, (int, float)) or mcts_cpuct <= 0:
            raise SpecValidationError("'evaluation.mcts_cpuct' must be a positive number when provided")
        mcts_cpuct = float(mcts_cpuct)

    engine_test = evaluation.get("engine_test", DEFAULT_ALPHA_EVALUATION_TEST)
    if not isinstance(engine_test, str) or not engine_test:
        raise SpecValidationError("'evaluation.engine_test' must be a non-empty string")

    shards_parquet_filename = evaluation.get("shards_parquet_filename", "evaluation_shards.parquet")
    if not isinstance(shards_parquet_filename, str) or not shards_parquet_filename:
        raise SpecValidationError("'evaluation.shards_parquet_filename' must be a non-empty string")

    rollups_parquet_filename = evaluation.get("rollups_parquet_filename", "evaluation_rollups.parquet")
    if not isinstance(rollups_parquet_filename, str) or not rollups_parquet_filename:
        raise SpecValidationError("'evaluation.rollups_parquet_filename' must be a non-empty string")

    report_markdown_filename = evaluation.get("report_markdown_filename", "evaluation_report.md")
    if not isinstance(report_markdown_filename, str) or not report_markdown_filename:
        raise SpecValidationError("'evaluation.report_markdown_filename' must be a non-empty string")

    duckdb_filename = evaluation.get("duckdb_filename", "evaluation.duckdb")
    if not isinstance(duckdb_filename, str) or not duckdb_filename:
        raise SpecValidationError("'evaluation.duckdb_filename' must be a non-empty string")

    queries_filename = evaluation.get("queries_filename", "evaluation_queries.sql")
    if not isinstance(queries_filename, str) or not queries_filename:
        raise SpecValidationError("'evaluation.queries_filename' must be a non-empty string")

    return {
        "kind": kind,
        "level": level,
        "games_per_shard": games_per_shard,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "checkpoint": explicit_checkpoint,
        "service_host": service_host,
        "service_port": service_port,
        "mcts_simulations": mcts_simulations,
        "mcts_max_depth": mcts_max_depth,
        "mcts_cpuct": mcts_cpuct,
        "engine_test": engine_test,
        "shards_parquet_filename": shards_parquet_filename,
        "rollups_parquet_filename": rollups_parquet_filename,
        "report_markdown_filename": report_markdown_filename,
        "duckdb_filename": duckdb_filename,
        "queries_filename": queries_filename,
    }


def _build_collection_workflow(raw: Mapping[str, Any]) -> List[WorkflowTaskSpec]:
    """Expand the collection section into concrete shard tasks."""

    collection = _validate_collection(_as_dict("collection", raw.get("collection")))
    if not collection:
        return []

    tasks: List[WorkflowTaskSpec] = []
    shard_count = int(collection["shard_count"])
    games_per_shard = int(collection["games_per_shard"])
    randomise = bool(collection["randomise"])
    seed_base = int(collection["seed_base"])

    for shard_index in range(shard_count):
        shard_seed = seed_base + shard_index if randomise else None
        tasks.append(
            WorkflowTaskSpec(
                name=f"collect-shard-{shard_index + 1:03d}",
                kind="endgame_collect_shard",
                payload_key="collection",
                payload_overrides={
                    "requested_games": games_per_shard,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "shard_seed": shard_seed,
                },
                command=(),
                working_directory="engine",
            )
        )

    return tasks


def _build_training_workflow(raw: Mapping[str, Any]) -> List[WorkflowTaskSpec]:
    """Expand the training section into a concrete training task when supported."""

    architecture = _validate_architecture(_as_dict("architecture", raw.get("architecture")))
    collection = _validate_collection(_as_dict("collection", raw.get("collection")))
    dataset = _validate_dataset(_as_dict("dataset", raw.get("dataset")), collection)
    training = _validate_training(_as_dict("training", raw.get("training")), architecture, dataset)
    if not training:
        return []

    adapter = get_adapter_for_family(str(architecture.get("family")))
    if training.get("kind") != adapter.training_kind:
        return []

    return [
        WorkflowTaskSpec(
            name="train",
            kind="policy_value_train",
            payload_key="training",
            payload_overrides={
                "architecture_family": architecture.get("family"),
                "architecture_params": architecture.get("params", {}),
                "dataset_kind": dataset.get("kind"),
                "dataset_sources": dataset.get("sources", []),
            },
            command=(),
            working_directory="neural-network",
        )
    ]


def _build_evaluation_workflow(raw: Mapping[str, Any]) -> List[WorkflowTaskSpec]:
    """Expand the evaluation section into concrete shard and report tasks when supported."""

    architecture = _validate_architecture(_as_dict("architecture", raw.get("architecture")))
    collection = _validate_collection(_as_dict("collection", raw.get("collection")))
    dataset = _validate_dataset(_as_dict("dataset", raw.get("dataset")), collection)
    training = _validate_training(_as_dict("training", raw.get("training")), architecture, dataset)
    evaluation = _validate_evaluation(_as_dict("evaluation", raw.get("evaluation")), architecture, training)
    if not evaluation or evaluation.get("kind") != "alpha_level_suite":
        return []

    tasks: List[WorkflowTaskSpec] = []
    seed_start = int(evaluation["seed_start"])
    seed_end = int(evaluation["seed_end"])
    games_per_shard = int(evaluation["games_per_shard"])
    shard_starts = list(range(seed_start, seed_end, games_per_shard))

    for shard_index, game_start_index in enumerate(shard_starts):
        requested_games = min(games_per_shard, seed_end - game_start_index)
        tasks.append(
            WorkflowTaskSpec(
                name=f"evaluate-shard-{shard_index + 1:03d}",
                kind="alpha_level_eval_shard",
                payload_key="evaluation",
                payload_overrides={
                    "requested_games": requested_games,
                    "game_start_index": game_start_index,
                    "game_end_index_exclusive": game_start_index + requested_games,
                    "shard_index": shard_index,
                    "shard_count": len(shard_starts),
                    "architecture_family": architecture.get("family"),
                    "architecture_params": architecture.get("params", {}),
                    "checkpoint_prefix": training.get("checkpoint_prefix"),
                    "training_kind": training.get("kind"),
                },
                command=(),
                working_directory="engine",
            )
        )

    tasks.append(
        WorkflowTaskSpec(
            name="report-evaluation",
            kind="evaluation_report",
            payload_key="evaluation",
            payload_overrides={
                "architecture_family": architecture.get("family"),
                "architecture_params": architecture.get("params", {}),
                "checkpoint_prefix": training.get("checkpoint_prefix"),
                "training_kind": training.get("kind"),
            },
            command=(),
            working_directory=None,
        )
    )
    return tasks


def _build_default_workflow(raw: Mapping[str, Any]) -> List[WorkflowTaskSpec]:
    """Derive a minimal workflow from top-level sections when no explicit workflow is present."""

    tasks: List[WorkflowTaskSpec] = []
    collection_tasks = _build_collection_workflow(raw)
    training_tasks = _build_training_workflow(raw)
    evaluation_tasks = _build_evaluation_workflow(raw)
    if collection_tasks:
        tasks.extend(collection_tasks)
    elif raw.get("dataset") and not training_tasks:
        # Dataset-backed collection falls back to a noop task when the spec only declares input
        # data. That keeps the workflow structure, registry state, and artifact layout consistent.
        tasks.append(WorkflowTaskSpec(
            name="collect",
            kind="noop",
            payload_key="dataset",
            payload_overrides=None,
            command=(),
            working_directory=None,
        ))
    if training_tasks:
        tasks.extend(training_tasks)
    elif raw.get("training"):
        tasks.append(WorkflowTaskSpec(
            name="train",
            kind="noop",
            payload_key="training",
            payload_overrides=None,
            command=(),
            working_directory=None,
        ))
    if evaluation_tasks:
        tasks.extend(evaluation_tasks)
    elif raw.get("evaluation"):
        tasks.append(WorkflowTaskSpec(
            name="evaluate",
            kind="noop",
            payload_key="evaluation",
            payload_overrides=None,
            command=(),
            working_directory=None,
        ))
    return tasks


def _load_workflow(raw: Mapping[str, Any]) -> tuple[WorkflowTaskSpec, ...]:
    """Load the workflow section or synthesize the default task sequence."""

    workflow = _as_dict("workflow", raw.get("workflow"))
    task_defs = workflow.get("tasks")
    if task_defs is None:
        task_defs = _build_default_workflow(raw)
        if not task_defs:
            raise SpecValidationError("spec must define at least one workflow task or one runnable section")
        return tuple(task_defs)

    if not isinstance(task_defs, list) or not task_defs:
        raise SpecValidationError("'workflow.tasks' must be a non-empty array")

    names: set[str] = set()
    tasks: List[WorkflowTaskSpec] = []
    for index, task_def in enumerate(task_defs, start=1):
        if not isinstance(task_def, dict):
            raise SpecValidationError(f"workflow task #{index} must be an object")

        name = task_def.get("name")
        if not isinstance(name, str) or not name:
            raise SpecValidationError(f"workflow task #{index} must define a non-empty 'name'")
        if name in names:
            raise SpecValidationError(f"workflow task names must be unique: {name}")
        names.add(name)

        kind = str(task_def.get("kind", "noop"))
        if kind not in SUPPORTED_TASK_KINDS:
            raise SpecValidationError(
                f"workflow task '{name}' kind must be one of {sorted(SUPPORTED_TASK_KINDS)}"
            )

        payload_key = task_def.get("payload_key")
        if payload_key is not None and not isinstance(payload_key, str):
            raise SpecValidationError(f"workflow task '{name}' payload_key must be a string")

        payload_overrides = task_def.get("payload")
        if payload_overrides is not None and not isinstance(payload_overrides, dict):
            raise SpecValidationError(f"workflow task '{name}' payload must be an object")

        raw_command = task_def.get("command", [])
        command: tuple[str, ...] = ()
        if kind == "command":
            # Command tasks are the future bridge to the engine and trainer, so validate the
            # payload strictly now and keep the task record deterministic in the registry.
            if not isinstance(raw_command, list) or not raw_command:
                raise SpecValidationError(f"workflow task '{name}' must define a non-empty command array")
            if not all(isinstance(item, str) and item for item in raw_command):
                raise SpecValidationError(f"workflow task '{name}' command items must be strings")
            command = tuple(raw_command)

        working_directory = task_def.get("working_directory")
        if working_directory is not None:
            if not isinstance(working_directory, str):
                raise SpecValidationError(
                    f"workflow task '{name}' working_directory must be a string"
                )
            if not _resolve_repo_path(working_directory).exists():
                raise SpecValidationError(
                    f"workflow task '{name}' working_directory does not exist: {working_directory}"
                )

        tasks.append(
            WorkflowTaskSpec(
                name=name,
                kind=kind,
                payload_key=payload_key,
                payload_overrides=dict(payload_overrides) if payload_overrides is not None else None,
                command=command,
                working_directory=working_directory,
            )
        )

    return tuple(tasks)


def flatten_run_parameters(raw: Mapping[str, Any]) -> Dict[str, str]:
    """Flatten key experiment dimensions for registry-level querying."""

    parameters: Dict[str, str] = {}

    def store(key: str, value: Any) -> None:
        # Store everything as JSON text so the registry has one consistent representation for
        # scalar and structured values without introducing a per-field schema in the registry.
        if isinstance(value, (str, int, float, bool)) or value is None:
            parameters[key] = json.dumps(value)
        else:
            parameters[key] = json.dumps(value, sort_keys=True)

    architecture = _validate_architecture(_as_dict("architecture", raw.get("architecture")))
    collection = _validate_collection(_as_dict("collection", raw.get("collection")))
    dataset = _validate_dataset(_as_dict("dataset", raw.get("dataset")), collection)
    training = _validate_training(_as_dict("training", raw.get("training")), architecture, dataset)
    evaluation = _validate_evaluation(_as_dict("evaluation", raw.get("evaluation")), architecture, training)
    store("architecture.family", architecture.get("family"))
    store("architecture.params", architecture.get("params", {}))
    store("collection", collection)
    store("dataset.kind", dataset.get("kind"))
    store("dataset.sources", dataset.get("sources", []))
    store("training", training)
    store("evaluation", evaluation)
    return parameters


def load_experiment_spec(spec_path: str | Path) -> ExperimentSpec:
    """Load, validate, and normalize an experiment spec from JSON."""

    path = Path(spec_path).resolve()
    if not path.exists():
        raise SpecValidationError(f"spec file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise SpecValidationError("spec root must be a JSON object")

    api_version = str(raw.get("api_version", ""))
    if api_version != SUPPORTED_API_VERSION:
        raise SpecValidationError(
            f"unsupported api_version '{api_version}'; expected '{SUPPORTED_API_VERSION}'"
        )

    experiment_id = raw.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise SpecValidationError("'experiment_id' must be a non-empty string")
    if not SPEC_ID_PATTERN.match(experiment_id):
        raise SpecValidationError(
            "'experiment_id' must match the pattern [A-Za-z0-9._-]+"
        )

    description = str(raw.get("description", ""))

    architecture = _validate_architecture(_as_dict("architecture", raw.get("architecture")))
    if not architecture:
        raise SpecValidationError("'architecture.family' must be provided")

    collection = _validate_collection(_as_dict("collection", raw.get("collection")))
    dataset = _validate_dataset(_as_dict("dataset", raw.get("dataset")), collection)
    training = _validate_training(_as_dict("training", raw.get("training")), architecture, dataset)
    evaluation = _validate_evaluation(_as_dict("evaluation", raw.get("evaluation")), architecture, training)
    workflow_tasks = _load_workflow(raw)

    # The spec hash is the registry's guardrail against accidentally resuming a run under a
    # different configuration while reusing the same run identifier.
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    spec_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return ExperimentSpec(
        spec_path=path,
        spec_hash=spec_hash,
        api_version=api_version,
        experiment_id=experiment_id,
        description=description,
        architecture=architecture,
        collection=collection,
        dataset=dataset,
        training=training,
        evaluation=evaluation,
        workflow_tasks=workflow_tasks,
        raw=raw,
    )


def task_payload(
    spec: ExperimentSpec,
    payload_key: str | None,
    payload_overrides: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return the section payload associated with a workflow task."""

    payload: Dict[str, Any] = {}
    if payload_key is None:
        payload = {}
    else:
        normalized_sections: Dict[str, Mapping[str, Any]] = {
            "architecture": spec.architecture,
            "collection": spec.collection,
            "dataset": spec.dataset,
            "training": spec.training,
            "evaluation": spec.evaluation,
        }
        resolved_payload = normalized_sections.get(payload_key, spec.raw.get(payload_key))
        if resolved_payload is None:
            payload = {}
        else:
            if not isinstance(resolved_payload, dict):
                raise SpecValidationError(f"task payload '{payload_key}' must resolve to an object")
            payload = dict(resolved_payload)

    if payload_overrides is not None:
        payload.update(dict(payload_overrides))

    return payload


def resolve_working_directory(path_text: str | None) -> Path:
    """Resolve a task working directory relative to the repository root."""

    # Defaulting to the repo root keeps command tasks predictable and avoids hidden dependence
    # on whichever directory the operator happened to be in when launching the driver.
    if path_text is None:
        return REPO_ROOT
    return _resolve_repo_path(path_text)
