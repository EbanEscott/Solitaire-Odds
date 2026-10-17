package ai.games.player.ai.alpha;

import ai.games.game.Card;
import ai.games.game.GameRules;
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
     * Listed bottom-to-top, with unknown covered cards represented by placeholders.
     */
    private final List<String> talon;

    /**
     * Number of face-down cards remaining in the stockpile.
     * Unobserved card order is masked; known order is carried separately in knownStock.
     */
    private final int stockSize;

    /**
     * List of legal moves available in the current position.
     * Each move is a command string (e.g., "move T1 A♥ F1", "turn", "quit").
     */
    private final List<String> legalMoves;

    /** Rules attached to the observation; absent on legacy manually constructed requests. */
    private GameRules rules;

    /** Current stock traversal, including the initial pass. */
    private long stockPass;

    /** Possible unseen identities derived only from observations, never hidden locations. */
    private List<String> unknownCards;

    /**
     * Returns the unseen identity inventory used by the existing neural feature encoder.
     *
     * @return possible card identities with observed and assigned cards removed
     */
    public List<String> getUnknownCards() {
        return unknownCards;
    }

    /** Remembered stock identities in next-draw order, with unknown slots masked. */
    private List<String> knownStock;

    /** Known tableau identities, including thoughtful knowledge and cards covered by training undo. */
    private List<List<String>> knownTableau;

    /**
     * Returns the rules governing the observed game.
     *
     * @return the rules, or null for a legacy manually constructed request
     */
    public GameRules getRules() {
        return rules;
    }

    /**
     * Returns the current stock traversal.
     *
     * @return the one-based pass number, or zero for a legacy request
     */
    public long getStockPass() {
        return stockPass;
    }

    /**
     * Returns stock knowledge without exposing unobserved identities.
     *
     * @return known cards and UNKNOWN placeholders in next-draw order
     */
    public List<String> getKnownStock() {
        return knownStock;
    }

    /**
     * Returns tableau knowledge independently of card playability.
     *
     * @return masked columns in bottom-to-top order
     */
    public List<List<String>> getKnownTableau() {
        return knownTableau;
    }

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
        solitaire = solitaire.planningCopy();
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

        AlphaSolitaireRequest request = new AlphaSolitaireRequest(
                tableauVisible,
                tableauFaceDown,
                foundation,
                talon,
                stockSize,
                legalMoves);
        request.unknownCards = solitaire.getUnknownCards().stream().map(Card::shortName).toList();
        request.rules = solitaire.getRules();
        request.stockPass = solitaire.getStockPass();
        request.knownStock = solitaire.getStockpile().reversed().stream().map(Card::shortName).toList();
        request.knownTableau = solitaire.getTableau().stream()
                .map(pile -> pile.stream().map(Card::shortName).toList()).toList();
        return request;
    }
}

