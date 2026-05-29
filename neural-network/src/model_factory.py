from __future__ import annotations

from typing import Any, Dict, Mapping

from .graph_model import GraphPolicyValueNet
from .model import PolicyValueNet


SUPPORTED_MODEL_FAMILIES = {"mlp", "gnn"}


def normalize_model_family(model_family: str | None) -> str:
    """Normalize a model family identifier and validate that it is supported."""

    if model_family is None:
        return "mlp"

    normalized = str(model_family).strip().lower()
    if not normalized:
        return "mlp"
    if normalized not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"unsupported model family '{model_family}'; expected one of {sorted(SUPPORTED_MODEL_FAMILIES)}"
        )
    return normalized


def checkpoint_model_family(checkpoint: Mapping[str, Any]) -> str:
    """Infer the model family stored in a checkpoint, defaulting old checkpoints to MLP."""

    top_level_family = checkpoint.get("model_family")
    if isinstance(top_level_family, str) and top_level_family.strip():
        return normalize_model_family(top_level_family)

    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict):
        architecture = metadata.get("architecture")
        if isinstance(architecture, dict):
            family = architecture.get("family")
            if isinstance(family, str) and family.strip():
                return normalize_model_family(family)

    return "mlp"


def checkpoint_architecture_params(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract architecture parameters from a checkpoint in a backward-compatible format."""

    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict):
        architecture = metadata.get("architecture")
        if isinstance(architecture, dict):
            params = architecture.get("params")
            if isinstance(params, dict):
                return dict(params)

            # Older checkpoints stored the MLP parameters directly under metadata.architecture.
            fallback_keys = {
                "hidden_dim",
                "num_layers",
                "batch_norm",
                "residual",
                "action_embedding_dim",
                "message_passing_steps",
                "dropout",
            }
            extracted = {key: architecture[key] for key in fallback_keys if key in architecture}
            if extracted:
                return extracted

    return {}


def build_model(
    *,
    model_family: str,
    state_dim: int,
    num_actions: int,
    architecture_params: Mapping[str, Any] | None = None,
):
    """Build a policy-value model for the requested architecture family."""

    family = normalize_model_family(model_family)
    params = dict(architecture_params or {})

    if family == "mlp":
        return PolicyValueNet(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dim=int(params.get("hidden_dim", 256)),
            num_layers=int(params.get("num_layers", 2)),
            use_batch_norm=bool(params.get("batch_norm", False)),
            use_residual=bool(params.get("residual", False)),
        )

    if family == "gnn":
        return GraphPolicyValueNet(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dim=int(params.get("hidden_dim", 256)),
            num_layers=int(params.get("num_layers", 2)),
            action_embedding_dim=int(params.get("action_embedding_dim", 128)),
            message_passing_steps=int(params.get("message_passing_steps", 2)),
            dropout=float(params.get("dropout", 0.0)),
        )

    raise ValueError(f"unsupported model family: {family}")


__all__ = [
    "SUPPORTED_MODEL_FAMILIES",
    "build_model",
    "checkpoint_architecture_params",
    "checkpoint_model_family",
    "normalize_model_family",
]