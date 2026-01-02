package ai.games.player.moves;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import java.util.List;

/**
 * Abstract base class for computing legal moves in Solitaire.
 * <p>
 * Two implementations exist:
 * <ul>
 *   <li><b>GameMovesHelper:</b> For GAME mode (real game with known cards)
 *   <li><b>PlanningMovesHelper:</b> For PLAN mode (lookahead with UNKNOWN cards)
 * </ul>
 * <p>
 * The base class defines the interface and shares common validation logic
 * (checking rules for tableau and foundation placement).
 */
public abstract class MovesHelper {
    /** The Solitaire game state being analyzed. */
    protected Solitaire solitaire;

    /**
     * Computes all legal moves for the given Solitaire state.
     * <p>
     * Delegates to mode-specific implementations via abstract methods.
     *
     * @param solitaire the game state; must not be null
     * @return a list of legal move commands (including "turn" and "quit")
     */
    public abstract List<String> listLegalMoves(Solitaire solitaire);

    /**
     * Generates all legal moves from tableau piles to foundation piles.
     * <p>
     * Tableau-to-foundation moves are relatively simple: only the top visible
     * card of each tableau pile can move, and only if it matches the foundation
     * (Ace for empty, matching suit and one rank higher otherwise).
     *
     * @param out the list to append generated moves to
     */
    protected abstract void addTableauToFoundation(List<String> out);

    /**
     * Generates all legal moves between tableau piles (including moves to UNKNOWN positions).
     * <p>
     * Tableau-to-tableau moves are complex: any visible card can be the start of a stack,
     * and the destination must follow tableau rules (alternating colors, descending ranks).
     * In PLAN mode, this also includes moves to UNKNOWN positions.
     *
     * @param out the list to append generated moves to
     */
    protected abstract void addTableauToTableau(List<String> out);

    /**
     * Generates all legal moves from talon (waste pile) to foundation piles.
     * <p>
     * Only the top card of the talon is playable. In both GAME and PLAN modes,
     * the talon only contains real (not UNKNOWN) cards, so this implementation is identical.
     *
     * @param out the list to append generated moves to
     */
    protected abstract void addTalonToFoundation(List<String> out);

    /**
     * Generates all legal moves from talon (waste pile) to tableau piles.
     * <p>
     * Only the top card of the talon is playable. In both GAME and PLAN modes,
     * the talon only contains real (not UNKNOWN) cards, so this implementation is identical.
     *
     * @param out the list to append generated moves to
     */
    protected abstract void addTalonToTableau(List<String> out);

    /**
     * Generates all legal moves from foundation piles back to tableau piles.
     * <p>
     * Foundations can only move their top card (usually a strategic repositioning move).
     * Aces and Twos are excluded as they're almost always strategically bad to move down.
     * In both GAME and PLAN modes, this implementation is identical.
     *
     * @param out the list to append generated moves to
     */
    protected abstract void addFoundationToTableau(List<String> out);

    /**
     * Validates whether a card can be placed on a tableau pile.
     * <p>
     * Tableau rules:
     * <ul>
     *   <li><b>Empty pile:</b> Only Kings can start a new cascade
     *   <li><b>Non-empty pile:</b> Card must be opposite color and exactly one rank lower than the top card
     * </ul>
     * <p>
     * In PLAN mode, this method may be overridden to handle UNKNOWN target cards
     * (checking if any possibility could match).
     *
     * @param moving the card being moved; must not be null
     * @param toPile the destination tableau pile; must not be null
     * @return {@code true} if the move follows tableau placement rules; {@code false} otherwise
     */
    protected boolean canPlaceOnTableau(Card moving, List<Card> toPile) {
        if (moving == null || toPile.isEmpty()) {
            if (moving != null) {
                return moving.getRank() == Rank.KING;
            }
            return false;
        }
        
        Card target = toPile.get(toPile.size() - 1);
        
        // UNKNOWN cards are never valid targets for normal placement
        // (PlanningMovesHelper overrides this for special handling)
        if (target.getRank() == Rank.UNKNOWN) {
            return false;
        }
        
        boolean alternatingColor = moving.getSuit().isRed() != target.getSuit().isRed();
        boolean oneLower = moving.getRank().getValue() == target.getRank().getValue() - 1;
        return alternatingColor && oneLower;
    }

    /**
     * Validates whether a card can be placed on a foundation pile.
     * <p>
     * Foundation rules:
     * <ul>
     *   <li><b>Empty pile:</b> Only Aces can start a foundation
     *   <li><b>Non-empty pile:</b> Card must match suit and be exactly one rank higher than the top card
     * </ul>
     * <p>
     * This method is identical in both GAME and PLAN modes (foundation never receives UNKNOWN).
     *
     * @param moving the card being moved; must not be null
     * @param toPile the destination foundation pile; must not be null
     * @return {@code true} if the move follows foundation placement rules; {@code false} otherwise
     */
    protected boolean canPlaceOnFoundation(Card moving, List<Card> toPile) {
        if (moving == null) {
            return false;
        }
        
        if (toPile.isEmpty()) {
            return moving.getRank() == Rank.ACE;
        }
        
        Card target = toPile.get(toPile.size() - 1);
        boolean sameSuit = moving.getSuit() == target.getSuit();
        boolean oneHigher = moving.getRank().getValue() == target.getRank().getValue() + 1;
        return sameSuit && oneHigher;
    }
}
