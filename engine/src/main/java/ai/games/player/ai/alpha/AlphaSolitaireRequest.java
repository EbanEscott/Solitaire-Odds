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
 */
public class AlphaSolitaireRequest {

    private final List<List<String>> tableauVisible;
    private final List<Integer> tableauFaceDown;
    private final List<List<String>> foundation;
    private final List<String> talon;
    private final int stockSize;
    private final List<String> legalMoves;

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

    public List<List<String>> getTableauVisible() {
        return tableauVisible;
    }

    public List<Integer> getTableauFaceDown() {
        return tableauFaceDown;
    }

    public List<List<String>> getFoundation() {
        return foundation;
    }

    public List<String> getTalon() {
        return talon;
    }

    public int getStockSize() {
        return stockSize;
    }

    public List<String> getLegalMoves() {
        return legalMoves;
    }

    /**
     * Build a request from the current Solitaire state using the existing
     * helpers on the game model.
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

