import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Minimal model of a Solitaire/Klondike layout: tableau, foundation, stockpile, and talon.
 */
public class Solitaire {
    private static final int CELL_WIDTH = 8;

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
        appendFoundationSection(sb);
        appendTableauSection(sb);
        appendStockAndTalon(sb);
        return sb.toString();
    }

    private void appendFoundationSection(StringBuilder sb) {
        sb.append("FOUNDATION\n");
        List<String> labels = new ArrayList<>();
        List<String> values = new ArrayList<>();
        for (int i = 0; i < foundation.size(); i++) {
            labels.add("F" + (i + 1));
            List<Card> pile = foundation.get(i);
            values.add(pile.isEmpty() ? "--" : pile.get(pile.size() - 1).toString());
        }
        appendBoxRow(sb, labels, values, "    ");
        sb.append('\n');
    }

    private void appendTableauSection(StringBuilder sb) {
        sb.append("TABLEAU\n");
        List<String> labels = new ArrayList<>();
        List<String> values = new ArrayList<>();
        for (int i = 0; i < tableau.size(); i++) {
            labels.add("T" + (i + 1));
            List<Card> pile = tableau.get(i);
            if (pile.isEmpty()) {
                values.add("(empty)");
            } else {
                int hidden = pile.size() - 1;
                String top = pile.get(pile.size() - 1).toString();
                values.add((hidden > 0 ? "[" + hidden + "] " : "") + top);
            }
        }
        appendBoxRow(sb, labels, values, "  ");
        sb.append('\n');
    }

    private void appendCardsInline(StringBuilder sb, List<Card> pile) {
        for (int i = 0; i < pile.size(); i++) {
            sb.append(pile.get(i));
            if (i < pile.size() - 1) {
                sb.append(' ');
            }
        }
    }

    private void appendBoxRow(StringBuilder sb, List<String> labels, List<String> contents, String indent) {
        String top = buildBorder(labels.size(), indent);
        String labelLine = buildRow(labels, indent);
        String contentLine = buildRow(contents, indent);
        sb.append(top).append('\n').append(labelLine).append('\n').append(contentLine).append('\n').append(top);
    }

    private String buildBorder(int count, String indent) {
        StringBuilder line = new StringBuilder(indent);
        for (int i = 0; i < count; i++) {
            line.append("+").append("-".repeat(CELL_WIDTH + 2)).append("+");
            if (i < count - 1) {
                line.append("  ");
            }
        }
        return line.toString();
    }

    private String buildRow(List<String> cells, String indent) {
        StringBuilder line = new StringBuilder(indent);
        for (int i = 0; i < cells.size(); i++) {
            String content = cells.get(i);
            line.append("| ").append(padCell(content, CELL_WIDTH)).append(" |");
            if (i < cells.size() - 1) {
                line.append("  ");
            }
        }
        return line.toString();
    }

    private String padCell(String value, int width) {
        int visible = visibleLength(value);
        if (visible >= width) {
            return value;
        }
        int totalPad = width - visible;
        int left = totalPad / 2;
        int right = totalPad - left;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < left; i++) {
            sb.append(' ');
        }
        sb.append(value);
        for (int i = 0; i < right; i++) {
            sb.append(' ');
        }
        return sb.toString();
    }

    private int visibleLength(String value) {
        // Strip ANSI color codes when calculating padding.
        String stripped = value.replaceAll("\\u001B\\[[;\\d]*m", "");
        return stripped.length();
    }

    private void appendStockAndTalon(StringBuilder sb) {
        sb.append("STOCKPILE & TALON\n");
        List<String> labels = new ArrayList<>();
        List<String> values = new ArrayList<>();

        labels.add("STOCK");
        labels.add("TALON");

        values.add(stockpile.isEmpty() ? "empty" : stockpile.size() + " down");
        if (talon.isEmpty()) {
            values.add("--");
        } else {
            Card top = talon.get(talon.size() - 1);
            values.add(top + " (" + talon.size() + ")");
        }

        appendBoxRow(sb, labels, values, "      ");
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
