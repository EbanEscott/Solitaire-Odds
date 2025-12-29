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

    Attributes:
        _episodes: List of Episode objects parsed from log files.
        _indices: Pre-computed mapping of sample index → (episode_idx, step_idx).
        action_space: ActionSpace mapping move commands to indices.
    """

    def __init__(self, log_paths: Sequence[str | Path]) -> None:
        super().__init__()
        self._episodes: List[Episode] = []

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

    def __len__(self) -> int:  # type: ignore[override]
        """Return total number of training samples (across all steps in all episodes)."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """
        Get a single training sample with all targets.

        Args:
            idx: Index into the flattened sample space.

        Returns:
            Dict with keys:
            - 'state': (296,) float tensor of encoded game state
            - 'policy': (num_actions,) one-hot tensor (1.0 at chosen action, 0.0 elsewhere)
            - 'value': scalar tensor (1.0 if episode won, 0.0 if lost)
            - 'foundation_move': scalar tensor (1.0 or 0.0)
            - 'revealed_facedown': scalar tensor (1.0 or 0.0)
            - 'talon_move': scalar tensor (1.0 or 0.0)
            - 'is_cascading_move': scalar tensor (1.0 or 0.0)
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

        # Create value label based on game progress heuristic.
        # Combines two key indicators:
        # 1. Foundation progress: (cards_in_foundation / 52) × 0.4
        # 2. Facedown revelation: (facedown_revealed / initial_facedown) × 0.6
        #
        # Expert insight: Revealing all facedown cards is the strongest predictor of win,
        # so we weight it more heavily (60%) than foundation progress (40%).
        #
        # This heuristic applies to ALL games (winning and losing):
        # - Good moves that made progress are rewarded (high value)
        # - Moves made in stuck positions get low value
        # - The network learns to recognize "promising states" regardless of final outcome
        # This preserves 100% of the training data while giving meaningful per-step signals.
        
        foundation = step.foundation  # List of 4 lists (one per suit)
        num_foundation_cards = sum(len(suit_cards) for suit_cards in foundation)
        foundation_progress = num_foundation_cards / 52.0
        
        # Calculate facedown progress: how many have been revealed since start of episode
        current_facedown = step.tableau_face_down  # List of counts per column
        num_current_facedown = sum(current_facedown)
        
        # Get initial facedown count from first step of episode
        initial_step = episode.steps[0]
        initial_facedown = initial_step.tableau_face_down
        num_initial_facedown = sum(initial_facedown)
        
        # Avoid division by zero (though Solitaire starts with 20 facedown cards)
        facedown_progress = 0.0
        if num_initial_facedown > 0:
            num_revealed = num_initial_facedown - num_current_facedown
            facedown_progress = num_revealed / num_initial_facedown
        
        # Combined value: weight facedown revelation (0.6) more than foundation (0.4)
        # This applies to all games—even a losing game gets high value if it made good progress
        value = torch.tensor(
            (foundation_progress * 0.4) + (facedown_progress * 0.6),
            dtype=torch.float32
        )

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

