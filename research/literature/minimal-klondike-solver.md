# Minimal Klondike Solver

- **Citation key:** `shootmeMinimalKlondike`
- **Author:** ShootMe (GitHub account)
- **Publication:** Software repository; no companion PDF
- **Source:** [Repository](https://github.com/ShootMe/MinimalKlondike)
- **Version tested:** `8983a1375aa15c5ca7f8c3df054aef37218f85c8`
- **Checked:** 2026-09-29

## Summary

C# solver for minimum-length solutions in Thoughtful Klondike. Its test suite provides explicit deals, rule settings and expected solution lengths.

## Local reproduction — 29 September 2026

Built successfully. The complete test suite returned **16 passes and 1 failure**. The failing test found its expected valid 98-move win but exhausted its 300,000-node budget before proving minimality. Calling the unchanged solver with a 600,000-node cap returned `Minimal` after **339,121 states**, with the same 98 moves and all 52 cards in the foundations; a 1,200,000-node cap confirmed that result.

The CLI example with Greenfelt seed 123, default draw-one, returned a minimum solution of **107 moves** in about seven seconds. Commands, the external budget probe and evidence are in the [reproduction record](../../../Solitaire-Repos/literature-reproduction/001-literature-reproduction/README.md#minimalklondike).

## Conditions and limitations

The project targets .NET 7; local execution used .NET 8.0.21 via `DOTNET_ROLL_FORWARD=Major`. The unchanged suite still has one failing assertion. Tests vary foundation-backtracking and stock-round settings; these results do not establish a general win rate. `Solved` means a winning sequence was found, whereas `Minimal` additionally reports completion of the minimum-length search under configured limits.

## Relevance to Solitaire Odds

Useful minimum-length fixtures and a practical example of separating a found solution from a completed optimality search. Record search budgets and rules with every comparison.
