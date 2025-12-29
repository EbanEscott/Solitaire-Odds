package ai.games.player.ai.alpha;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import ai.games.player.LegalMovesHelper;
import java.util.ArrayList;
import java.util.List;

/**
 * DTO used to send the current Solitaire state and legal moves to the Python
 * AlphaSolitaire model service.
 *
 * The JSON produced by this class is intentionally aligned with the request
 * shape expected by the Python /evaluate endpoint.
 *
 * This request encodes the complete game state needed for the neural network
 * to evaluate move probabilities and estimate win probability:
 * - Visible tableau cards and face-down counts
 * - Foundation piles (completed suits)
 * - Talon (waste) pile
 * - Stockpile size (hidden card order not exposed)
 * - List of legal moves available
 */
public class AlphaSolitaireRequest {

    /**
     * Visible cards in each tableau column (7 columns).
     * Each inner list contains card short names (e.g., "A♥", "K♠") in display order.
     */
    private final List<List<String>> tableauVisible;

    /**
     * Count of face-down cards in each tableau column (7 columns).
     * Indicates how many hidden cards remain under the visible cards.
     */
    private final List<Integer> tableauFaceDown;

    /**
     * Cards in each foundation pile (4 suits).
     * Each inner list contains completed cards in the suit, in order.
     * Empty list means no cards placed on that foundation yet.
     */
    private final List<List<String>> foundation;

    /**
     * Cards in the talon (waste) pile.
     * Shows cards drawn from the stockpile but not yet played.
     * Listed in display order (most recently drawn first).
     */
    private final List<String> talon;

    /**
     * Number of face-down cards remaining in the stockpile.
     * Hidden card order is not exposed to the model.
     */
    private final int stockSize;

    /**
     * List of legal moves available in the current position.
     * Each move is a command string (e.g., "move T1 A♥ F1", "turn", "quit").
     */
    private final List<String> legalMoves;

    /**
     * Constructs a request with the full game state.
     *
     * @param tableauVisible visible tableau cards (7 columns)
     * @param tableauFaceDown face-down counts per column (7 columns)
     * @param foundation foundation piles (4 suits)
     * @param talon cards in the waste/talon pile
     * @param stockSize count of cards remaining in stockpile
     * @param legalMoves list of legal move commands
     */
    public AlphaSolitaireRequest(
            List<List<String>> tableauVisible,
            List<Integer> tableauFaceDown,
            List<List<String>> foundation,
            List<String> talon,
            int stockSize,
            List<String> legalMoves) {

        this.tableauVisible = tableauVisible;
        this.tableauFaceDown = tableauFaceDown;
        this.foundation = foundation;
        this.talon = talon;
        this.stockSize = stockSize;
        this.legalMoves = legalMoves;
    }

    /**
     * Get the visible tableau cards.
     *
     * @return list of 7 columns, each containing visible card short names
     */
    public List<List<String>> getTableauVisible() {
        return tableauVisible;
    }

    /**
     * Get the face-down card counts per tableau column.
     *
     * @return list of 7 integers representing hidden card counts
     */
    public List<Integer> getTableauFaceDown() {
        return tableauFaceDown;
    }

    /**
     * Get the foundation piles.
     *
     * @return list of 4 foundation piles, each containing completed cards in suit order
     */
    public List<List<String>> getFoundation() {
        return foundation;
    }

    /**
     * Get the talon (waste) pile.
     *
     * @return list of card short names in the talon
     */
    public List<String> getTalon() {
        return talon;
    }

    /**
     * Get the stockpile size.
     *
     * @return number of face-down cards remaining in the stockpile
     */
    public int getStockSize() {
        return stockSize;
    }

    /**
     * Get the legal moves available.
     *
     * @return list of legal move command strings
     */
    public List<String> getLegalMoves() {
        return legalMoves;
    }

    /**
     * Build a request from the current Solitaire game state.
     *
     * Extracts all necessary state information from the Solitaire model
     * and constructs an AlphaSolitaireRequest suitable for sending to the
     * Python /evaluate endpoint. All cards are encoded as short names
     * (rank + suit symbol, e.g., "A♥", "K♠").
     *
     * @param solitaire the current game state
     * @return a new AlphaSolitaireRequest with the encoded game state
     */
    public static AlphaSolitaireRequest fromSolitaire(Solitaire solitaire) {
        List<List<String>> tableauVisible = new ArrayList<>();
        for (List<Card> pile : solitaire.getVisibleTableau()) {
            List<String> pileShortNames = new ArrayList<>(pile.size());
            for (Card c : pile) {
                pileShortNames.add(c.getRank().toString() + c.getSuit().getSymbol());
            }
            tableauVisible.add(pileShortNames);
        }

        List<Integer> tableauFaceDown = solitaire.getTableauFaceDownCounts();

        List<List<String>> foundation = new ArrayList<>();
        for (List<Card> pile : solitaire.getFoundation()) {
            List<String> pileShortNames = new ArrayList<>(pile.size());
            for (Card c : pile) {
                pileShortNames.add(c.getRank().toString() + c.getSuit().getSymbol());
            }
            foundation.add(pileShortNames);
        }

        List<String> talon = new ArrayList<>();
        for (Card c : solitaire.getTalon()) {
            talon.add(c.getRank().toString() + c.getSuit().getSymbol());
        }

        int stockSize = solitaire.getStockpile().size();

        List<String> legalMoves = LegalMovesHelper.listLegalMoves(solitaire);

        return new AlphaSolitaireRequest(
                tableauVisible,
                tableauFaceDown,
                foundation,
                talon,
                stockSize,
                legalMoves);
    }
}

