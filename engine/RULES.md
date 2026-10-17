# Klondike rules

This is the agreed contract for configurable rules. Configuration, draw count, and pass limits are implemented. Knowledge filtering and the distinction between training undo and search rollback are pending; the current engine does not yet enforce every rule below. Track implementation in the [checklist](../experiments/klondike_rules_checklist.md).

## Configuration

Each game has an immutable configuration. Copies and planning boards inherit it. The settings are independent; all 12 combinations are supported by this contract. Reject unsupported values rather than silently substituting defaults.

| Setting | Accepted values | Default |
| --- | --- | --- |
| `drawCount` | Integer `1` or `3` | `3` |
| `stockVisibility` | `topOnly` or `allDrawn` | `topOnly` |
| `maxStockPasses` | Integer `1`, integer `3`, or `null` (unlimited) | `null` |

An omitted setting uses its default. Explicit `null` is valid only for `maxStockPasses`. JVM properties use the `game.` prefix, for example `-Dgame.drawCount=1 -Dgame.maxStockPasses=3`. Omission uses the defaults above; the literal property value `null` means unlimited passes. Explicit Java configuration uses the immutable `GameRules` record.

```json
{
  "drawCount": 3,
  "stockVisibility": "topOnly",
  "maxStockPasses": null
}
```

## Standard play

- Use one shuffled 52-card deck without jokers. Deal seven tableau piles of one through seven cards, with only each pile's top card face-up. The remaining 24 cards form the stock; waste and foundations start empty.
- Build tableau sequences downward by rank in alternating colours. Move a face-up card and its valid face-up sequence together. Only a king or a sequence beginning with a king may fill an empty tableau pile.
- When a move uncovers a face-down tableau top, turn it face-up automatically and record its identity as observed.
- Build each foundation upward in one suit, from ace to king. Move only one exposed top card at a time onto a foundation.
- Only the top waste card is playable, to a legal tableau or foundation destination. Covered waste cards cannot be moved directly.
- Foundation-to-tableau moves (worry-back) remain enabled and follow ordinary tableau placement rules. Only the foundation top may move.
- Stock cards enter play through a stock turn, never a direct card move.
- Win when all 52 cards are on the foundations. Stock exhaustion alone does not end the game. Move caps and player decisions to quit are separate from the rules configuration.

## Drawing and observation

A stock turn moves `min(drawCount, remaining stock size)` cards onto the waste, one at a time in stock order. The last card moved is the new waste top.

With `topOnly`, only that final card becomes newly observed. Intermediate cards are not exposed to the player during the action. With `allDrawn`, every card moved becomes observed immediately. Only the final card is playable in either case. With draw-one, both visibility settings have the same effect.

Playing the waste top exposes and records the next waste card, if any. Covering a card with another draw never erases earlier knowledge. A previously observed card stays known even when it is covered or recycled into the stock.

For example, suppose the next stock cards in draw order are A♣, 7♥, K♠:

| Action | `topOnly` newly observed | `allDrawn` newly observed | Playable waste card |
| --- | --- | --- | --- |
| Draw three | K♠ | A♣, 7♥, K♠ | K♠ |
| Play K♠ to a legal destination | 7♥ | None | 7♥ |
| Play 7♥ to a legal destination | A♣ | None | A♣ |

If only A♣ and 7♥ remain in stock, a draw-three action moves both; `topOnly` reveals 7♥, while `allDrawn` reveals both. A complete stock traversal with `allDrawn` reveals every card traversed. With `topOnly`, it need not do so.

## Stock passes and recycling

The initial traversal is pass 1. Drawing the last stock card exhausts the current pass. The pass number increases only when a subsequent turn recycles the waste and starts another pass; ordinary draws and waste moves do not increase it.

- `1`: no recycling.
- `3`: at most two recycles, allowing three traversals including the initial one.
- `null`: no limit on recycling.

When stock is empty and waste is nonempty, `turn` recycles and immediately draws in the same action if another pass is allowed. There is no separate recycle action. A turn is unavailable when both piles are empty, or when stock is empty and the pass allowance is exhausted. Rejected turns change neither state nor history. Other legal moves remain available after the final pass.

Recycling turns the waste over without shuffling. Waste listed bottom-to-top as `[A♣, 7♥, K♠]` becomes a stock whose next draw order is A♣, 7♥, K♠. If 7♥ and K♠ were played away, only A♣ returns. Removing waste cards can therefore change later draw groupings and which cards become accessible.

## True state and player knowledge

The true game state retains every card identity and location. Player knowledge contains only observed identities and the positions/order available from public actions and observation history. Initial face-up tableau cards are observed; initial face-down tableau and stock cards are not.

All player-facing views must preserve known identities and mask unobserved identities, including covered waste cards. Pile sizes, pass allowance, and rules are public. Knowledge of a covered card does not make it playable. Remembered identities do not authorize inspection of other hidden cards.

Planning must use that knowledge boundary. Hypothetical assignments and sampled worlds belong only to the simulation and never become actual observations. Copies preserve rules, pass state, and knowledge independently. Search keys and position comparisons must distinguish relevant knowledge and finite remaining pass allowances. Unlimited pass counts must not defeat cycle detection by making otherwise identical positions different on every traversal.

## Undo and rollback

Player-facing training undo restores card positions, face-up status, and pass allowance, but retains all identities actually observed during the game. This includes cards revealed by an undone draw or tableau move. A new game resets knowledge. Training undo is an assistance feature, separate from the three rules settings.

Search rollback restores the exact earlier simulated state, including its knowledge, assumptions, and pass allowance. Information revealed only in a discarded hypothetical branch must not leak into another branch or real play. Implementation must distinguish these two operations even if they share action records internally.

## Results and reproducibility

Record all three resolved settings with each new episode and benchmark run. Historical results must retain their original conditions; the new defaults do not retroactively establish which information historical players received. See the [historical audit](../experiments/klondike_rules_checklist.md#historical-visibility-audit).
