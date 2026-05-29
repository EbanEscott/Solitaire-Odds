"""Architecture adapters for experiment validation and command construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping


SUPPORTED_DATASET_KIND = "archived_episode_logs"


def _require_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    """Validate one integer architecture parameter."""

    if not isinstance(value, int):
        raise ValueError(f"'{name}' must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"'{name}' must be >= {minimum}")
    return value


def _require_bool(name: str, value: Any) -> bool:
    """Validate one boolean architecture parameter."""

    if not isinstance(value, bool):
        raise ValueError(f"'{name}' must be a boolean")
    return value


def _reject_unknown_params(section_name: str, params: Mapping[str, Any], allowed: set[str]) -> None:
    """Reject misspelled or unsupported architecture parameters early."""

    unknown = sorted(key for key in params if key not in allowed)
    if unknown:
        raise ValueError(
            f"'{section_name}' contains unsupported keys {unknown}; expected only {sorted(allowed)}"
        )


def _normalize_common_training_fields(training: Mapping[str, Any], training_kind: str) -> Dict[str, Any]:
    """Validate the training fields shared by all supported model families."""

    epochs = _require_int("training.epochs", training.get("epochs"), minimum=1)
    batch_size = _require_int("training.batch_size", training.get("batch_size"), minimum=1)
    checkpoint_every_epochs = _require_int(
        "training.checkpoint_every_epochs",
        training.get("checkpoint_every_epochs", 1),
        minimum=1,
    )
    simulate_interrupt_after_epoch = training.get("simulate_interrupt_after_epoch")
    if simulate_interrupt_after_epoch is not None:
        simulate_interrupt_after_epoch = _require_int(
            "training.simulate_interrupt_after_epoch",
            simulate_interrupt_after_epoch,
            minimum=1,
        )

    learning_rate = training.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("'training.learning_rate' must be a positive number")

    checkpoint_prefix = training.get("checkpoint_prefix", "policy_value")
    if not isinstance(checkpoint_prefix, str) or not checkpoint_prefix:
        raise ValueError("'training.checkpoint_prefix' must be a non-empty string")

    metrics_filename = training.get("metrics_filename", "epoch_metrics.jsonl")
    if not isinstance(metrics_filename, str) or not metrics_filename:
        raise ValueError("'training.metrics_filename' must be a non-empty string")

    resume_from = training.get("resume_from")
    if resume_from is not None and (not isinstance(resume_from, str) or not resume_from):
        raise ValueError("'training.resume_from' must be a non-empty string when provided")

    return {
        "kind": training_kind,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "checkpoint_prefix": checkpoint_prefix,
        "metrics_filename": metrics_filename,
        "resume_from": resume_from,
        "simulate_interrupt_after_epoch": simulate_interrupt_after_epoch,
    }


class ArchitectureAdapter(ABC):
    """Contract for family-specific validation and trainer command generation."""

    family: str
    training_kind: str

    def validate_architecture(self, architecture: Mapping[str, Any]) -> Dict[str, Any]:
        """Validate the architecture section and normalize its parameter block."""

        raw_params = architecture.get("params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise ValueError("'architecture.params' must be an object")
        return {
            "family": self.family,
            "params": self._normalize_architecture_params(raw_params),
        }

    def validate_training(
        self,
        training: Mapping[str, Any],
        dataset: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Validate a family-specific training section."""

        if dataset.get("kind") != SUPPORTED_DATASET_KIND or not dataset.get("sources"):
            raise ValueError(
                f"'training.kind={self.training_kind}' requires 'dataset.kind={SUPPORTED_DATASET_KIND}' with one or more sources"
            )
        return _normalize_common_training_fields(training, self.training_kind)

    @abstractmethod
    def _normalize_architecture_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Normalize the family-specific architecture parameters."""

    @abstractmethod
    def build_training_command_args(self, architecture_params: Mapping[str, Any]) -> list[str]:
        """Build trainer CLI arguments for this family."""


class MlpArchitectureAdapter(ArchitectureAdapter):
    """Adapter for the flat board-state MLP model family."""

    family = "mlp"
    training_kind = "mlp_policy_value"

    def _normalize_architecture_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        allowed = {"hidden_dim", "num_layers", "batch_norm", "residual"}
        _reject_unknown_params("architecture.params", params, allowed)
        return {
            "hidden_dim": _require_int("architecture.params.hidden_dim", params.get("hidden_dim", 256), minimum=1),
            "num_layers": _require_int("architecture.params.num_layers", params.get("num_layers", 2), minimum=1),
            "batch_norm": _require_bool("architecture.params.batch_norm", params.get("batch_norm", False)),
            "residual": _require_bool("architecture.params.residual", params.get("residual", False)),
        }

    def build_training_command_args(self, architecture_params: Mapping[str, Any]) -> list[str]:
        command = [
            "--model-family",
            self.family,
            "--hidden-dim",
            str(architecture_params.get("hidden_dim", 256)),
            "--num-layers",
            str(architecture_params.get("num_layers", 2)),
        ]
        if architecture_params.get("batch_norm"):
            command.append("--batch-norm")
        if architecture_params.get("residual"):
            command.append("--residual")
        return command


class GnnArchitectureAdapter(ArchitectureAdapter):
    """Adapter for the legal-move graph policy-value model family."""

    family = "gnn"
    training_kind = "gnn_policy_value"

    def _normalize_architecture_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        allowed = {"hidden_dim", "num_layers", "action_embedding_dim", "message_passing_steps", "dropout"}
        _reject_unknown_params("architecture.params", params, allowed)

        dropout = params.get("dropout", 0.0)
        if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("'architecture.params.dropout' must be a number in the range [0.0, 1.0)")

        return {
            "hidden_dim": _require_int("architecture.params.hidden_dim", params.get("hidden_dim", 256), minimum=1),
            "num_layers": _require_int("architecture.params.num_layers", params.get("num_layers", 2), minimum=1),
            "action_embedding_dim": _require_int(
                "architecture.params.action_embedding_dim",
                params.get("action_embedding_dim", 128),
                minimum=1,
            ),
            "message_passing_steps": _require_int(
                "architecture.params.message_passing_steps",
                params.get("message_passing_steps", 2),
                minimum=1,
            ),
            "dropout": float(dropout),
        }

    def build_training_command_args(self, architecture_params: Mapping[str, Any]) -> list[str]:
        return [
            "--model-family",
            self.family,
            "--hidden-dim",
            str(architecture_params.get("hidden_dim", 256)),
            "--num-layers",
            str(architecture_params.get("num_layers", 2)),
            "--action-embedding-dim",
            str(architecture_params.get("action_embedding_dim", 128)),
            "--message-passing-steps",
            str(architecture_params.get("message_passing_steps", 2)),
            "--dropout",
            str(architecture_params.get("dropout", 0.0)),
        ]


_ADAPTERS: tuple[ArchitectureAdapter, ...] = (
    MlpArchitectureAdapter(),
    GnnArchitectureAdapter(),
)
_ADAPTERS_BY_FAMILY = {adapter.family: adapter for adapter in _ADAPTERS}
_ADAPTERS_BY_TRAINING_KIND = {adapter.training_kind: adapter for adapter in _ADAPTERS}

SUPPORTED_ARCHITECTURE_FAMILIES = frozenset(_ADAPTERS_BY_FAMILY)
SUPPORTED_TRAINING_KINDS = frozenset(_ADAPTERS_BY_TRAINING_KIND)


def get_adapter_for_family(family: str) -> ArchitectureAdapter:
    """Return the adapter for one architecture family."""

    normalized = str(family).strip().lower()
    adapter = _ADAPTERS_BY_FAMILY.get(normalized)
    if adapter is None:
        raise ValueError(
            f"'architecture.family' must be one of {sorted(SUPPORTED_ARCHITECTURE_FAMILIES)}"
        )
    return adapter


def get_adapter_for_training_kind(training_kind: str) -> ArchitectureAdapter:
    """Return the adapter that owns one training kind."""

    normalized = str(training_kind).strip()
    adapter = _ADAPTERS_BY_TRAINING_KIND.get(normalized)
    if adapter is None:
        raise ValueError(f"'training.kind' must be one of {sorted(SUPPORTED_TRAINING_KINDS)}")
    return adapter


__all__ = [
    "ArchitectureAdapter",
    "SUPPORTED_ARCHITECTURE_FAMILIES",
    "SUPPORTED_TRAINING_KINDS",
    "get_adapter_for_family",
    "get_adapter_for_training_kind",
]