package ai.games.game.moves;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.game.UnknownCardGuess;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Computes legal moves for PLAN mode (lookahead with UNKNOWN cards).
 * <p>
 * In PLAN mode, face-down cards are masked with UNKNOWN placeholders to prevent
 * information leaks during AI search. This implementation extends the base logic to:
 * <ul>
 *   <li>Generate moves to UNKNOWN card positions (gambling on hidden cards)
 *   <li>Validate moves using UnknownCardGuess plausibility checking
 *   <li>Handle mixed visible/UNKNOWN tableau stacks correctly
 * </ul>
 * <p>
 * The key insight: when we move a card to an UNKNOWN, we're making an assumption
 * about what that card is. The UnknownCardGuess mechanism tracks those assumptions
 * across the lookahead tree.
 */
public class PlanningMovesHelper extends MovesHelper {

    @Override
    public List<Move> listMoves(Solitaire solitaire) {
        if (solitaire == null) {
            return new ArrayList<>();
        }
        
        this.solitaire = solitaire;
        List<Move> moves = new ArrayList<>();
        
        // These lists describe the same board throughout this generation call.
        // Read them once so the move categories share snapshots and views.
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        List<List<Card>> foundations = solitaire.getFoundation();
        List<Card> talon = solitaire.getTalon();

        addTableauToFoundation(moves, tableau, faceUps, foundations);
        addTableauToTableau(moves, tableau, faceUps);
        addTalonToFoundation(moves, talon, foundations);
        addTalonToTableau(moves, talon);
        addFoundationToTableau(moves, tableau, foundations);
        
        // "turn" is always legal if stock or talon still has cards to cycle.
        if (!solitaire.getStockpile().isEmpty() || !talon.isEmpty()) {
            moves.add(Move.turn());
        }
        
        // "quit" is always a legal command from the engine's perspective.
        moves.add(Move.quit());
        
        return moves;
    }

    @Override
    protected void addTableauToFoundation(List<Move> out, List<List<Card>> tableau,
            List<Integer> faceUps, List<List<Card>> foundations) {
        // Only top face-up cards can move to foundation.
        for (int t = 0; t < tableau.size(); t++) {
            List<Card> pile = tableau.get(t);
            int faceUp = faceUps.get(t);
            if (pile.isEmpty() || faceUp <= 0) {
                continue;
            }
            Card top = pile.get(pile.size() - 1);
            for (int f = 0; f < foundations.size(); f++) {
                if (canPlaceOnFoundation(top, foundations.get(f))) {
                    out.add(Move.move(tableauPile(t), cardRef(top), foundationPile(f)));
                }
            }
        }
    }

    @Override
    protected void addTableauToTableau(List<Move> out, List<List<Card>> tableau, List<Integer> faceUps) {
        TableauDestinations destinations = new TableauDestinations(tableau);
        List<List<Card>> fullTableau = solitaire.getTableau();  // Need full piles to find UNKNOWN cards
        
        // Any visible card can move as the start of a stack to another tableau pile.
        for (int from = 0; from < tableau.size(); from++) {
            List<Card> fromPile = tableau.get(from);
            int faceUp = faceUps.get(from);
            if (fromPile.isEmpty() || faceUp <= 0) {
                continue;
            }
            int start = Math.max(0, fromPile.size() - faceUp);
            for (int idx = start; idx < fromPile.size(); idx++) {
                Card moving = fromPile.get(idx);
                if (moving == null) {
                    continue;
                }
                int movingRank = moving.getRank().getValue();
                boolean movingIsRed = moving.getSuit().isRed();
                
                // Move to visible cards (original logic)
                for (int to = 0; to < tableau.size(); to++) {
                    if (to == from) {
                        continue;
                    }
                    if (destinations.accepts(to, movingRank, movingIsRed)) {
                        out.add(Move.move(tableauPile(from), cardRef(moving), tableauPile(to)));
                    }
                }
                
                // Also try moves to UNKNOWN cards (new PLAN mode logic)
                for (int to = 0; to < fullTableau.size(); to++) {
                    if (to == from) {
                        continue;
                    }
                    List<Card> toPile = fullTableau.get(to);
                    if (!toPile.isEmpty()) {
                        Card topCard = toPile.get(toPile.size() - 1);
                        // Check if top card is UNKNOWN and if we can plausibly place on it
                        if (topCard.getRank() == Rank.UNKNOWN && canPlaceOnUnknown(moving, topCard)) {
                            out.add(Move.move(tableauPile(from), cardRef(moving), tableauPile(to)));
                        }
                    }
                }
            }
        }
    }

    @Override
    protected void addTalonToFoundation(List<Move> out, List<Card> talon, List<List<Card>> foundations) {
        if (talon.isEmpty()) {
            return;
        }
        // Only top of talon is playable.
        Card top = talon.get(talon.size() - 1);
        for (int f = 0; f < foundations.size(); f++) {
            if (canPlaceOnFoundation(top, foundations.get(f))) {
                out.add(Move.move(WASTE, null, foundationPile(f)));
            }
        }
    }

    @Override
    protected void addTalonToTableau(List<Move> out, List<Card> talon) {
        if (talon.isEmpty()) {
            return;
        }
        // Only top of talon is playable.
        Card top = talon.get(talon.size() - 1);
        List<List<Card>> tableau = solitaire.getTableau();
        for (int t = 0; t < tableau.size(); t++) {
            if (canPlaceOnTableau(top, tableau.get(t))) {
                out.add(Move.move(WASTE, null, tableauPile(t)));
            }
        }
    }

    @Override
    protected void addFoundationToTableau(List<Move> out, List<List<Card>> tableau, List<List<Card>> foundations) {
        // Foundations can only move their top card back to tableau.
        for (int f = 0; f < foundations.size(); f++) {
            List<Card> pile = foundations.get(f);
            if (pile.isEmpty()) {
                continue;
            }
            Card top = pile.get(pile.size() - 1);
            // Do not suggest moving Aces or Twos down from foundations; that is almost
            // always strategically bad and creates noisy moves for the AI.
            if (top.getRank() == Rank.ACE || top.getRank() == Rank.TWO) {
                continue;
            }
            for (int t = 0; t < tableau.size(); t++) {
                if (canPlaceOnTableau(top, tableau.get(t))) {
                    out.add(Move.move(foundationPile(f), null, tableauPile(t)));
                }
            }
        }
    }

    /**
     * Validates whether a card can plausibly be placed on an UNKNOWN card.
     * <p>
     * Conservative approach: checks if any of the possibilities in the UnknownCardGuess
     * for this UNKNOWN could accept the moving card according to tableau rules.
     * If no guess exists yet for this UNKNOWN, creates one with plausible possibilities.
     *
     * @param moving the card being moved; must not be null
     * @param unknownTarget the UNKNOWN card at the top of the destination pile; must not be null
     * @return {@code true} if at least one plausible possibility could accept this card; {@code false} otherwise
     */
    protected boolean canPlaceOnUnknown(Card moving, Card unknownTarget) {
        if (moving == null || unknownTarget == null) {
            return false;
        }
        
        if (moving.getRank() == Rank.UNKNOWN) {
            // UNKNOWN cards cannot move to foundation (no rank/suit to validate)
            // and moving UNKNOWN to UNKNOWN is not useful for planning
            return false;
        }
        
        // Get existing guess for this UNKNOWN, or create one with plausible possibilities
        Map<Card, UnknownCardGuess> guesses = solitaire.getUnknownCardGuesses();
        UnknownCardGuess guess = guesses.get(unknownTarget);
        
        if (guess == null) {
            // No guess yet; create one with plausible candidates (both suits of the rank below moving card)
            guess = createPlausibleGuess(moving, unknownTarget);
            if (guess == null) {
                return false;
            }
            // Store for future moves in this path
            guesses.put(unknownTarget, guess);
        }
        
        // Check if any possibility in the guess would allow this move
        for (Card possibility : guess.getPossibilities()) {
            if (canPlaceOnTableau(moving, createTempPile(possibility))) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * Creates a plausible guess for an UNKNOWN card based on a moving card.
     * <p>
     * For a moving card, the target must be one rank higher and opposite color.
     * If two such cards exist (different suits of same rank), both are possibilities.
     * If only one exists (other already visible), just that one.
     * If none exist, the move is impossible.
     *
     * @param moving the card trying to move onto the UNKNOWN
     * @param unknownTarget the UNKNOWN card (for tracking)
     * @return a UnknownCardGuess with plausible candidates, or null if impossible
     */
    private UnknownCardGuess createPlausibleGuess(Card moving, Card unknownTarget) {
        // What rank must the UNKNOWN be? One higher than moving card.
        int targetRank = moving.getRank().getValue() + 1;
        if (targetRank > 13) {
            return null;  // Can't place on King
        }
        
        // What color must it be? Opposite of moving card.
        boolean movingIsRed = moving.getSuit().isRed();
        
        // Find the correct rank
        Rank targetRankEnum = null;
        for (Rank rank : Rank.values()) {
            if (rank.getValue() == targetRank) {
                targetRankEnum = rank;
                break;
            }
        }
        
        if (targetRankEnum == null) {
            return null;
        }
        
        // Create possibilities with both suits of opposite color
        List<Card> possibilities = new ArrayList<>();
        Suit suit1 = findOppositeSuit(movingIsRed, 0);
        Suit suit2 = findOppositeSuit(movingIsRed, 1);
        possibilities.add(new Card(targetRankEnum, suit1));
        possibilities.add(new Card(targetRankEnum, suit2));
        
        return new UnknownCardGuess(unknownTarget, possibilities);
    }

    /**
     * Helper to find a suit of opposite color.
     * Red suits: Diamonds, Hearts
     * Black suits: Clubs, Spades
     *
     * @param isRed whether we want opposite of red (i.e., black)
     * @param index which suit to return (0 or 1)
     * @return a Suit enum value
     */
    private Suit findOppositeSuit(boolean isRed, int index) {
        if (!isRed) {  // moving is black, need red
            return index == 0 ? Suit.DIAMONDS : Suit.HEARTS;
        } else {  // moving is red, need black
            return index == 0 ? Suit.CLUBS : Suit.SPADES;
        }
    }

    /**
     * Creates a temporary single-card pile for validation purposes.
     *
     * @param card the card to place in the pile
     * @return a list with just that card
     */
    private List<Card> createTempPile(Card card) {
        List<Card> pile = new ArrayList<>();
        pile.add(card);
        return pile;
    }
}
