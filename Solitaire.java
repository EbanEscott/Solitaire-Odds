import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Minimal model of a Solitaire/Klondike layout: tableau, foundation, stockpile, and talon.
 */
public class Solitaire {
    // Tableau: seven piles where most play occurs.
    private final List<List<Card>> tableau = new ArrayList<>();
    // Foundation: four suit piles built up from Ace to King.
    private final List<List<Card>> foundation = new ArrayList<>();
    // Stockpile: undealt cards remaining after the tableau is dealt.
    private final List<Card> stockpile = new ArrayList<>();
    // Talon: revealed cards from the stockpile.
    private final List<Card> talon = new ArrayList<>();

    public Solitaire(Deck deck) {
        Objects.requireNonNull(deck, "deck");
        initializeFoundation();
        dealTableau(deck);
        moveRemainingToStockpile(deck);
    }

    public List<List<Card>> getTableau() {
        return unmodifiablePiles(tableau);
    }

    public List<List<Card>> getFoundation() {
        return unmodifiablePiles(foundation);
    }

    public List<Card> getStockpile() {
        return Collections.unmodifiableList(stockpile);
    }

    public List<Card> getTalon() {
        return Collections.unmodifiableList(talon);
    }

    /**
     * Turn three cards from the stockpile onto the talon, preserving order of draw.
     */
    public void turnThree() {
        for (int i = 0; i < 3 && !stockpile.isEmpty(); i++) {
            talon.add(stockpile.remove(stockpile.size() - 1));
        }
    }

    /**
     * Attempt to move the top card from one pile to another (e.g., "T6" -> "F3").
     * Returns true on success, false if the move is illegal or piles are invalid/empty.
     */
    public boolean moveCard(String from, String to) {
        List<Card> fromPile = resolvePile(from);
        List<Card> toPile = resolvePile(to);
        if (fromPile == null || toPile == null || fromPile.isEmpty()) {
            return false;
        }

        Card moving = fromPile.get(fromPile.size() - 1);
        if (!isLegalMove(moving, to, toPile)) {
            return false;
        }

        fromPile.remove(fromPile.size() - 1);
        toPile.add(moving);
        return true;
    }

    private void initializeFoundation() {
        for (int i = 0; i < 4; i++) {
            foundation.add(new ArrayList<>());
        }
    }

    private void dealTableau(Deck deck) {
        for (int pileIndex = 0; pileIndex < 7; pileIndex++) {
            List<Card> pile = new ArrayList<>();
            for (int cardCount = 0; cardCount <= pileIndex && !deck.isEmpty(); cardCount++) {
                pile.add(deck.draw());
            }
            tableau.add(pile);
        }
    }

    private void moveRemainingToStockpile(Deck deck) {
        while (!deck.isEmpty()) {
            stockpile.add(deck.draw());
        }
    }

    private List<List<Card>> unmodifiablePiles(List<List<Card>> piles) {
        List<List<Card>> snapshot = new ArrayList<>(piles.size());
        for (List<Card> pile : piles) {
            snapshot.add(Collections.unmodifiableList(pile));
        }
        return Collections.unmodifiableList(snapshot);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        appendPiles(sb, "Tableau", tableau);
        appendPiles(sb, "Foundation", foundation);
        sb.append("Stockpile: ").append(stockpile.size()).append(" cards").append('\n');
        sb.append("Talon: ");
        appendPileCards(sb, talon);
        return sb.toString();
    }

    private void appendPiles(StringBuilder sb, String title, List<List<Card>> piles) {
        sb.append(title).append(":\n");
        for (int i = 0; i < piles.size(); i++) {
            sb.append("  ").append(i + 1).append(": ");
            appendPileCards(sb, piles.get(i));
        }
    }

    private void appendPileCards(StringBuilder sb, List<Card> pile) {
        for (int i = 0; i < pile.size(); i++) {
            sb.append(pile.get(i));
            if (i < pile.size() - 1) {
                sb.append(' ');
            }
        }
        sb.append('\n');
    }

    private List<Card> resolvePile(String code) {
        if (code == null || code.isEmpty()) {
            return null;
        }
        String normalized = code.trim().toUpperCase();
        char type = normalized.charAt(0);
        int index = 0;
        if (normalized.length() > 1) {
            try {
                index = Integer.parseInt(normalized.substring(1)) - 1;
            } catch (NumberFormatException e) {
                return null;
            }
        }

        switch (type) {
            case 'T':
                return (index >= 0 && index < tableau.size()) ? tableau.get(index) : null;
            case 'F':
                return (index >= 0 && index < foundation.size()) ? foundation.get(index) : null;
            case 'W': // Waste / talon
                return talon;
            case 'S': // Stockpile
                return stockpile;
            default:
                return null;
        }
    }

    /**
     * Solitaire move rules:
     * - To a tableau pile (Tn): must alternate color and be one rank lower than the destination top; empty tableau accepts only a King.
     * - To a foundation pile (Fn): must match suit and be one rank higher than the destination top; empty foundation accepts only an Ace.
     */
    private boolean isLegalMove(Card moving, String toCode, List<Card> toPile) {
        if (moving == null || toCode == null) {
            return false;
        }
        String normalized = toCode.trim().toUpperCase();
        if (normalized.isEmpty()) {
            return false;
        }

        char type = normalized.charAt(0);
        switch (type) {
            case 'T':
                return canPlaceOnTableau(moving, toPile);
            case 'F':
                return canPlaceOnFoundation(moving, toPile);
            default:
                return false;
        }
    }

    private boolean canPlaceOnTableau(Card moving, List<Card> toPile) {
        if (toPile.isEmpty()) {
            return moving.getRank() == Rank.KING;
        }
        Card target = toPile.get(toPile.size() - 1);
        boolean alternatingColor = moving.getSuit().isRed() != target.getSuit().isRed();
        boolean oneLower = moving.getRank().getValue() == target.getRank().getValue() - 1;
        return alternatingColor && oneLower;
    }

    private boolean canPlaceOnFoundation(Card moving, List<Card> toPile) {
        if (toPile.isEmpty()) {
            return moving.getRank() == Rank.ACE;
        }
        Card target = toPile.get(toPile.size() - 1);
        boolean sameSuit = moving.getSuit() == target.getSuit();
        boolean oneHigher = moving.getRank().getValue() == target.getRank().getValue() + 1;
        return sameSuit && oneHigher;
    }
}
