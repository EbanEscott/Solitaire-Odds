#!/usr/bin/env python3
"""
Analyse Solitaire episode logs to validate value heuristic assumptions.

Shows foundation and facedown card progression for chosen games,
computes the heuristic value (40% foundation + 60% facedown revelation),
and helps understand what the network is learning to predict.

Usage:
    python tools/log_analyser.py ../engine/logs/episode.log
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


class GameAnalyser:
    def __init__(self):
        self.games = defaultdict(lambda: {'steps': [], 'won': None, 'initial_facedown': None})

    def load_log(self, log_path):
        """Load all games from episode log file."""
        print(f"Loading log file: {log_path}")
        
        with open(log_path, 'r') as f:
            for line in f:
                if line.startswith('EPISODE_STEP '):
                    data = json.loads(line[len('EPISODE_STEP '):])
                    game_idx = data['game_index']
                    
                    # Extract metrics
                    foundation_cards = sum(len(suit) for suit in data['foundation'])
                    current_facedown = sum(data['tableau_face_down'])
                    
                    # Get initial facedown from first step of game
                    if self.games[game_idx]['initial_facedown'] is None:
                        self.games[game_idx]['initial_facedown'] = current_facedown
                    
                    initial_facedown = self.games[game_idx]['initial_facedown']
                    
                    # Compute heuristic value
                    foundation_progress = foundation_cards / 52.0
                    facedown_progress = 0.0
                    if initial_facedown > 0:
                        num_revealed = initial_facedown - current_facedown
                        facedown_progress = num_revealed / initial_facedown
                    
                    heuristic_value = (foundation_progress * 0.4) + (facedown_progress * 0.6)
                    
                    step = {
                        'step_idx': data['step_index'],
                        'foundation': foundation_cards,
                        'facedown': current_facedown,
                        'foundation_progress': foundation_progress,
                        'facedown_progress': facedown_progress,
                        'value': heuristic_value,
                        'command': data['chosen_command'],
                    }
                    self.games[game_idx]['steps'].append(step)
                    
                elif line.startswith('EPISODE_SUMMARY '):
                    data = json.loads(line[len('EPISODE_SUMMARY '):])
                    game_idx = data['game_index']
                    self.games[game_idx]['won'] = data['won']

    def print_summary(self):
        """Print summary of all games."""
        print(f"\n=== Episode Log Summary ===")
        print(f"Total games: {len(self.games)}")
        
        wins = sum(1 for g in self.games.values() if g['won'])
        print(f"Games won: {wins} ({wins * 100.0 / len(self.games):.2f}%)")
        
        total_steps = sum(len(g['steps']) for g in self.games.values())
        print(f"Total steps: {total_steps}")

    def analyse_game(self, game_idx):
        """Display detailed analysis of a single game."""
        if game_idx not in self.games:
            print(f"Game {game_idx} not found.")
            return
        
        game = self.games[game_idx]
        steps = game['steps']
        
        if not steps:
            print(f"Game {game_idx} has no steps.")
            return
        
        print(f"\n=== Game {game_idx} Analysis ===")
        print(f"Outcome: {'WON ✓' if game['won'] else 'LOST ✗'}")
        print(f"Total steps: {len(steps)}")
        print(f"Initial facedown cards: {game['initial_facedown']}\n")
        
        # Header
        print("Step | Found | FaceDown | Found% | Facedn% | Value  | Command")
        print("-----|-------|----------|--------|---------|--------|------------------")
        
        # Show steps (sample if too many)
        display_freq = max(1, len(steps) // 50)
        
        for i, step in enumerate(steps):
            if i % display_freq != 0 and i != len(steps) - 1:
                continue
            
            print(f"{step['step_idx']:4d} | {step['foundation']:2d}/52 | "
                  f"{step['facedown']:2d}/{game['initial_facedown']:2d} | "
                  f"{step['foundation_progress']*100:5.1f}% | "
                  f"{step['facedown_progress']*100:5.1f}% | "
                  f"{step['value']:.3f} | "
                  f"{step['command'][:16]}")
        
        # Statistics
        print(f"\n=== Value Progression Summary ===")
        first = steps[0]
        last = steps[-1]
        values = [s['value'] for s in steps]
        avg = sum(values) / len(values)
        
        print(f"First step value: {first['value']:.3f}")
        print(f"Last step value:  {last['value']:.3f}")
        print(f"Average value:    {avg:.3f}")
        print(f"Min value:        {min(values):.3f}")
        print(f"Max value:        {max(values):.3f}")
        
        # Check monotonicity
        monotonic = all(values[i] >= values[i-1] for i in range(1, len(values)))
        print(f"Monotonic:        {'YES ✓' if monotonic else 'NO ✗'}")
        
        # Analysis
        print(f"\n=== Analysis ===")
        if not monotonic:
            print("⚠ Value is NOT monotonically increasing.")
            print("  Some steps made progress while others didn't.")
            print("  This is normal for losing games (they get stuck).")
        
        if game['won'] and last['value'] < 0.95:
            print(f"⚠ Game won but final value only {last['value']:.3f} (expected ~1.0)")
            print("  This suggests the heuristic doesn't capture full game completion.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python log_analyzer.py <log_file>")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)
    
    analyser = GameAnalyser()
    analyser.load_log(log_path)
    analyser.print_summary()
    
    # Interactive mode
    print("\nEnter 'quit' to exit.\n")
    while True:
        try:
            game_idx = input("Enter game index to analyze: ").strip()
            if game_idx.lower() == 'quit':
                break
            analyser.analyse_game(int(game_idx))
        except ValueError:
            print("Invalid input. Enter a valid game index.")
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


if __name__ == '__main__':
    main()
