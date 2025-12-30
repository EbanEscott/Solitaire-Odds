from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset

from .action_encoding import ActionSpace, encode_action
from .log_loader import Episode, load_episodes_from_log
from .state_encoding import encode_state


@dataclass
class SampleIndex:
    """Index into a (episode, step) pair for efficient dataset sampling."""
    episode_idx: int
    step_idx: int


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory-aware training."""
    use_trajectory_value: bool = True  # Use full game outcome as value target
    use_bootstrapped_value: bool = False  # Use V(next_state) as bootstrapped target (for self-play)
    discount_factor: float = 0.99  # Discount factor for bootstrapped value
    n_step: int = 1  # For n-step returns (currently 1-step, can extend for multi-step)


class SolitaireStateDataset(Dataset):
    """
    PyTorch Dataset for Solitaire game episodes.

    Loads episodes from one or more JSON log files produced by the Java engine
    (with -Dlog.episodes=true enabled). Each episode contains multiple steps,
    and each step is a separate training sample (state, policy label, value label).

    Memory model:
    - All episodes are loaded into memory at initialization (one-time cost).
    - Individual samples are encoded to tensors on-demand via __getitem__.
    - Supports efficient batching via DataLoader.
    - Full trajectory context available: can use game outcome or bootstrapped targets.

    Attributes:
        _episodes: List of Episode objects parsed from log files.
        _indices: Pre-computed mapping of sample index → (episode_idx, step_idx).
        action_space: ActionSpace mapping move commands to indices.
        trajectory_config: Configuration for trajectory-aware training.
    """

    def __init__(
        self,
        log_paths: Sequence[str | Path],
        trajectory_config: TrajectoryConfig = None,
    ) -> None:
        super().__init__()
        self._episodes: List[Episode] = []
        self.trajectory_config = trajectory_config or TrajectoryConfig()

        # Load all episodes from log files
        for i, path in enumerate(log_paths, 1):
            print(f"  [{i}/{len(log_paths)}] Loading {Path(path).name}...", flush=True)
            episodes = load_episodes_from_log(path)
            self._episodes.extend(episodes)
            print(f"    → {len(episodes)} episode(s), {sum(len(e.steps) for e in episodes)} step(s)", flush=True)

        print(f"Total: {len(self._episodes)} episode(s) loaded", flush=True)

        # Build action space from all loaded episodes
        print("Building action space...", flush=True)
        self.action_space: ActionSpace = ActionSpace.from_episodes(self._episodes)
        print(f"  → {self.action_space.size} unique actions", flush=True)

        # Pre-compute (episode_idx, step_idx) mapping for fast O(1) sample lookup
        print("Indexing samples...", flush=True)
        self._indices: List[SampleIndex] = []
        for epi_idx, episode in enumerate(self._episodes):
            for step_idx, _ in enumerate(episode.steps):
                self._indices.append(SampleIndex(episode_idx=epi_idx, step_idx=step_idx))
        print(f"  → {len(self._indices)} sample(s) indexed", flush=True)
        
        # Print trajectory config
        print(
            f"Trajectory config: use_trajectory_value={self.trajectory_config.use_trajectory_value}, "
            f"use_bootstrapped_value={self.trajectory_config.use_bootstrapped_value}",
            flush=True,
        )

    def __len__(self) -> int:  # type: ignore[override]
        """Return total number of training samples (across all steps in all episodes)."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """
        Get a single training sample with trajectory-aware targets.

        For self-play training loops, the value target can be:
        - trajectory_value: The actual game outcome (1.0 = won, 0.0 = lost)
        - bootstrapped_value: V(next_state) * gamma + immediate_reward (for RL)

        Args:
            idx: Index into the flattened sample space.

        Returns:
            Dict with keys:
            - 'state': (296,) float tensor of encoded game state
            - 'policy': (num_actions,) one-hot tensor (1.0 at chosen action, 0.0 elsewhere)
            - 'value': scalar tensor (value target: game outcome or bootstrapped)
            - 'foundation_move': scalar tensor (1.0 or 0.0)
            - 'revealed_facedown': scalar tensor (1.0 or 0.0)
            - 'talon_move': scalar tensor (1.0 or 0.0)
            - 'is_cascading_move': scalar tensor (1.0 or 0.0)
            - 'step_index': step index in trajectory (for optional n-step returns)
        """
        ref = self._indices[idx]
        episode = self._episodes[ref.episode_idx]
        step = episode.steps[ref.step_idx]

        # Encode the game state to a feature vector
        state = encode_state(step)

        # Create one-hot policy label from the chosen command
        action_idx = encode_action(self.action_space, step.chosen_command)
        policy = torch.zeros(self.action_space.size, dtype=torch.float32)
        if 0 <= action_idx < self.action_space.size:
            policy[action_idx] = 1.0

        # Value target: trajectory outcome (full game outcome) or bootstrapped
        game_won = episode.summary.won if episode.summary else False
        trajectory_value = 1.0 if game_won else 0.0
        
        # If using bootstrapped value (for self-play refinement), compute discounted future return
        if self.trajectory_config.use_bootstrapped_value and ref.step_idx + 1 < len(episode.steps):
            # Could implement n-step bootstrapped value here
            # For now, use trajectory value but prepare infrastructure for RL
            value = torch.tensor(trajectory_value, dtype=torch.float32)
        else:
            # Standard supervised learning: label each step with final game outcome
            value = torch.tensor(trajectory_value, dtype=torch.float32)

        # Extract Tier 1 metrics (convert booleans to float tensors)
        foundation_move = torch.tensor(1.0 if step.foundation_move else 0.0, dtype=torch.float32)
        revealed_facedown = torch.tensor(1.0 if step.revealed_facedown else 0.0, dtype=torch.float32)
        talon_move = torch.tensor(1.0 if step.talon_move else 0.0, dtype=torch.float32)
        is_cascading_move = torch.tensor(1.0 if step.is_cascading_move else 0.0, dtype=torch.float32)

        return {
            'state': state,
            'policy': policy,
            'value': value,
            'foundation_move': foundation_move,
            'revealed_facedown': revealed_facedown,
            'talon_move': talon_move,
            'is_cascading_move': is_cascading_move,
        }


def build_dataset_from_logs(log_dir: str | Path, pattern: str = "*.log") -> SolitaireStateDataset:
    root = Path(log_dir)
    paths = sorted(root.glob(pattern))
    return SolitaireStateDataset(paths)


__all__ = ["SolitaireStateDataset", "build_dataset_from_logs"]

