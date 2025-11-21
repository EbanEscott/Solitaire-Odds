package ai.games.game;

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
    private final List<Integer> tableauFaceUp = new ArrayList<>(); // face-up card counts per tableau pile
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
        String fromNormalized = normalizeCode(from);
        String toNormalized = normalizeCode(to);
        if (fromNormalized == null || toNormalized == null) {
            return false;
        }

        char fromType = fromNormalized.charAt(0);
        char toType = toNormalized.charAt(0);
        int fromIndex = parseIndex(fromNormalized);
        int toIndex = parseIndex(toNormalized);

        List<Card> fromPile = resolvePile(fromNormalized);
        List<Card> toPile = resolvePile(toNormalized);
        if (fromPile == null || toPile == null || fromPile.isEmpty()) {
            return false;
        }

        Card moving = peekMovableCard(fromType, fromIndex, fromPile);
        if (moving == null) {
            return false;
        }

        if (!isLegalMove(moving, toNormalized, toPile)) {
            return false;
        }

        fromPile.remove(fromPile.size() - 1);
        if (fromType == 'T') {
            decrementFaceUp(fromIndex, fromPile);
        }

        toPile.add(moving);
        if (toType == 'T' && toIndex >= 0 && toIndex < tableauFaceUp.size()) {
            tableauFaceUp.set(toIndex, tableauFaceUp.get(toIndex) + 1);
        }
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
            tableauFaceUp.add(pile.isEmpty() ? 0 : 1); // only the top card is face up initially
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
        int tableauCellWidth = computeTableauCellWidth();
        String tableauBorder = buildBorder(tableau.size(), "  ", tableauCellWidth);
        sb.append("-".repeat(tableauBorder.length())).append('\n');
        appendFoundationSection(sb);
        appendTableauSection(sb, tableauCellWidth);
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
        int width = Math.max(CELL_WIDTH, Math.max(maxVisibleLength(labels), maxVisibleLength(values)));
        appendBoxRow(sb, labels, values, "    ", width);
        sb.append('\n');
    }

    private void appendTableauSection(StringBuilder sb, int cellWidth) {
        sb.append("TABLEAU\n");
        TableauDisplay display = buildTableauDisplay();
        int width = Math.max(cellWidth, Math.max(maxVisibleLength(display.labels),
                maxVisibleLengthColumns(display.columns)));

        String indent = "  ";
        String border = buildBorder(display.labels.size(), indent, width);
        sb.append(border).append('\n');
        sb.append(buildRow(display.labels, indent, width)).append('\n');

        int maxRows = 0;
        for (List<String> col : display.columns) {
            maxRows = Math.max(maxRows, col.size());
        }
        for (int row = 0; row < maxRows; row++) {
            List<String> rowCells = new ArrayList<>();
            for (List<String> col : display.columns) {
                rowCells.add(row < col.size() ? col.get(row) : "");
            }
            sb.append(buildRow(rowCells, indent, width)).append('\n');
        }
        sb.append(border).append('\n');
    }

    private void appendCardsInline(StringBuilder sb, List<Card> pile) {
        for (int i = 0; i < pile.size(); i++) {
            sb.append(pile.get(i));
            if (i < pile.size() - 1) {
                sb.append(' ');
            }
        }
    }

    private void appendBoxRow(StringBuilder sb, List<String> labels, List<String> contents, String indent, int width) {
        String top = buildBorder(labels.size(), indent, width);
        String labelLine = buildRow(labels, indent, width);
        String contentLine = buildRow(contents, indent, width);
        sb.append(top).append('\n').append(labelLine).append('\n').append(contentLine).append('\n').append(top);
    }

    private String buildBorder(int count, String indent, int cellWidth) {
        StringBuilder line = new StringBuilder(indent);
        for (int i = 0; i < count; i++) {
            line.append("+").append("-".repeat(cellWidth + 2)).append("+");
            if (i < count - 1) {
                line.append("  ");
            }
        }
        return line.toString();
    }

    private String buildRow(List<String> cells, String indent, int cellWidth) {
        StringBuilder line = new StringBuilder(indent);
        for (int i = 0; i < cells.size(); i++) {
            String content = cells.get(i);
            line.append("| ").append(padCell(content, cellWidth)).append(" |");
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

    private int maxVisibleLength(List<String> items) {
        int max = 0;
        for (String item : items) {
            max = Math.max(max, visibleLength(item));
        }
        return max;
    }

    private int maxVisibleLengthColumns(List<List<String>> columns) {
        int max = 0;
        for (List<String> col : columns) {
            for (String item : col) {
                max = Math.max(max, visibleLength(item));
            }
        }
        return max;
    }

    private int computeTableauCellWidth() {
        TableauDisplay display = buildTableauDisplay();
        int maxContent = Math.max(maxVisibleLength(display.labels), maxVisibleLengthColumns(display.columns));
        return Math.max(CELL_WIDTH, maxContent);
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

        int width = Math.max(CELL_WIDTH, Math.max(maxVisibleLength(labels), maxVisibleLength(values)));
        appendBoxRow(sb, labels, values, "      ", width);
    }

    private TableauDisplay buildTableauDisplay() {
        List<String> labels = new ArrayList<>();
        List<List<String>> columns = new ArrayList<>();
        for (int i = 0; i < tableau.size(); i++) {
            List<Card> pile = tableau.get(i);
            int faceUp = tableauFaceUp.get(i);
            int faceDown = Math.max(0, pile.size() - faceUp);
            labels.add("T" + (i + 1) + " [" + faceDown + "]");

            List<String> col = new ArrayList<>();
            if (pile.isEmpty()) {
                col.add("(empty)");
            } else {
                int start = Math.max(0, pile.size() - faceUp);
                for (int j = start; j < pile.size(); j++) {
                    col.add(pile.get(j).toString());
                }
            }
            columns.add(col);
        }
        return new TableauDisplay(labels, columns);
    }

    private List<Card> resolvePile(String code) {
        if (code == null || code.isEmpty()) {
            return null;
        }
        String normalized = code;
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

    private String normalizeCode(String code) {
        if (code == null || code.trim().isEmpty()) {
            return null;
        }
        return code.trim().toUpperCase();
    }

    private int parseIndex(String normalized) {
        if (normalized == null || normalized.length() <= 1) {
            return 0;
        }
        try {
            return Integer.parseInt(normalized.substring(1)) - 1;
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    private Card peekMovableCard(char fromType, int fromIndex, List<Card> fromPile) {
        if (fromPile.isEmpty()) {
            return null;
        }
        switch (fromType) {
            case 'T':
                if (fromIndex < 0 || fromIndex >= tableauFaceUp.size()) {
                    return null;
                }
                if (tableauFaceUp.get(fromIndex) <= 0) {
                    return null;
                }
                return fromPile.get(fromPile.size() - 1);
            case 'W':
            case 'F':
            case 'S':
                return fromPile.get(fromPile.size() - 1);
            default:
                return null;
        }
    }

    private void decrementFaceUp(int fromIndex, List<Card> fromPile) {
        int current = tableauFaceUp.get(fromIndex);
        current = Math.max(0, current - 1);
        if (current == 0 && !fromPile.isEmpty()) {
            current = 1; // flip next card
        }
        tableauFaceUp.set(fromIndex, current);
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

    private static class TableauDisplay {
        final List<String> labels;
        final List<List<String>> columns;

        TableauDisplay(List<String> labels, List<List<String>> columns) {
            this.labels = labels;
            this.columns = columns;
        }
    }
}
