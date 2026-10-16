package ai.games.game.moves;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import java.util.ArrayList;
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
    private static final Move.PileRef[] TABLEAU_PILES = pileRefs(Move.PileType.TABLEAU, 7);
    private static final Move.PileRef[] FOUNDATION_PILES = pileRefs(Move.PileType.FOUNDATION, 4);
    protected static final Move.PileRef WASTE = new Move.PileRef(Move.PileType.WASTE, -1);

    /** The Solitaire game state being analyzed. */
    protected Solitaire solitaire;

    /**
     * Computes all legal moves for the given Solitaire state.
     * <p>
     * Delegates to mode-specific implementations via abstract methods. The visible
     * tableau, face-up counts, foundations, and talon are read once per call and
     * passed to the move categories that use them.
     *
     * @param solitaire the game state; must not be null
     * @return a list of structured legal moves (including turn and quit)
     */
    public abstract List<Move> listMoves(Solitaire solitaire);

    /**
     * Generates all legal moves from tableau piles to foundation piles.
     * <p>
     * Tableau-to-foundation moves are relatively simple: only the top visible
     * card of each tableau pile can move, and only if it matches the foundation
     * (Ace for empty, matching suit and one rank higher otherwise).
     *
     * @param out the list to append generated moves to
     */
    protected abstract void addTableauToFoundation(List<Move> out, List<List<Card>> tableau,
            List<Integer> faceUps, List<List<Card>> foundations);

    /**
     * Generates legal moves between tableau piles with concrete card identities.
     * <p>
     * Tableau-to-tableau moves are complex: any visible card can be the start of a stack,
     * and the destination must follow tableau rules (alternating colors, descending ranks).
     * PLAN boards use the same rules after revealed UNKNOWN positions have been assigned.
     *
     * @param out the list to append generated moves to
     */
    protected abstract void addTableauToTableau(List<Move> out, List<List<Card>> tableau, List<Integer> faceUps);

    /**
     * Generates all legal moves from talon (waste pile) to foundation piles.
     * <p>
     * Only the top card of the talon is playable. Planning players assign identities to
     * revealed UNKNOWN waste cards before expansion; unresolved UNKNOWN sources are excluded.
     *
     * @param out the list to append generated moves to
     */
    protected abstract void addTalonToFoundation(List<Move> out, List<Card> talon, List<List<Card>> foundations);

    /**
     * Generates all legal moves from talon (waste pile) to tableau piles.
     * <p>
     * Only the top card of the talon is playable. Planning players assign identities to
     * revealed UNKNOWN waste cards before expansion; unresolved UNKNOWN sources are excluded.
     *
     * @param out the list to append generated moves to
     */
    protected abstract void addTalonToTableau(List<Move> out, List<Card> talon);

    /**
     * Generates all legal moves from foundation piles back to tableau piles.
     * <p>
     * Foundations can only move their top card (usually a strategic repositioning move).
     * Aces and Twos are excluded as they're almost always strategically bad to move down.
     * In both GAME and PLAN modes, this implementation is identical.
     *
     * @param out the list to append generated moves to
     */
    protected abstract void addFoundationToTableau(List<Move> out, List<List<Card>> tableau, List<List<Card>> foundations);

    /**
     * Compatibility adapter for players and displays that consume text commands.
     * Preserve the generator's card spelling, including UNKNOWN cards ("??").
     */
    public List<String> listLegalMoves(Solitaire solitaire) {
        List<Move> moves = listMoves(solitaire);
        List<String> commands = new ArrayList<>(moves.size());
        for (Move move : moves) {
            if (move.card() == null) {
                commands.add(move.toCommandString());
            } else {
                Move.CardRef card = move.card();
                commands.add("move " + move.from().toCode() + " "
                        + card.rank().toString() + card.suit().getSymbol() + " " + move.to().toCode());
            }
        }
        return commands;
    }

    private static Move.PileRef[] pileRefs(Move.PileType type, int count) {
        Move.PileRef[] refs = new Move.PileRef[count];
        for (int i = 0; i < count; i++) {
            refs[i] = new Move.PileRef(type, i);
        }
        return refs;
    }

    protected static Move.PileRef tableauPile(int index) {
        return TABLEAU_PILES[index];
    }

    protected static Move.PileRef foundationPile(int index) {
        return FOUNDATION_PILES[index];
    }

    protected static Move.CardRef cardRef(Card card) {
        return new Move.CardRef(card.getRank(), card.getSuit());
    }

    /**
     * Validates whether a card can be placed on a tableau pile.
     * <p>
     * Tableau rules:
     * <ul>
     *   <li><b>Empty pile:</b> Only Kings can start a new cascade
     *   <li><b>Non-empty pile:</b> Card must be opposite color and exactly one rank lower than the top card
     * </ul>
     * <p>
     * Both GAME and PLAN boards use these rules. UNKNOWN sources and destinations are
     * rejected until the planner assigns concrete identities.
     *
     * @param moving the card being moved; must not be null
     * @param toPile the destination tableau pile; must not be null
     * @return {@code true} if the move follows tableau placement rules; {@code false} otherwise
     */
    protected boolean canPlaceOnTableau(Card moving, List<Card> toPile) {
        if (moving == null || moving.getRank() == Rank.UNKNOWN) return false;
        if (toPile.isEmpty()) {
            return moving.getRank() == Rank.KING;
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
        if (moving == null || moving.getRank() == Rank.UNKNOWN) {
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

    /**
     * Placement requirements for one generation call. Computing these once avoids
     * reading the same destination pile and top card for every visible source card.
     * UNKNOWN destinations are rejected here; planning players assign their identities before expansion.
     */
    protected static final class TableauDestinations {
        private final int[] requiredRanks;
        private final boolean[] requiredRed;
        private final boolean[] empty;

        TableauDestinations(List<List<Card>> tableau) {
            requiredRanks = new int[tableau.size()];
            requiredRed = new boolean[tableau.size()];
            empty = new boolean[tableau.size()];
            for (int to = 0; to < tableau.size(); to++) {
                List<Card> pile = tableau.get(to);
                if (pile.isEmpty()) {
                    requiredRanks[to] = Rank.KING.getValue();
                    empty[to] = true;
                } else {
                    Card top = pile.get(pile.size() - 1);
                    requiredRanks[to] = top.getRank() == Rank.UNKNOWN
                            ? -1 : top.getRank().getValue() - 1;
                    requiredRed[to] = !top.getSuit().isRed();
                }
            }
        }

        boolean accepts(int destination, int movingRank, boolean movingIsRed) {
            return movingRank > 0 && movingRank == requiredRanks[destination]
                    && (empty[destination] || movingIsRed == requiredRed[destination]);
        }
    }
}
