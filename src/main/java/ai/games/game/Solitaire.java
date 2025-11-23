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
    private static final long[] STATE_ZOBRIST = initStateZobrist();

    // Tableau: seven piles where most play occurs.
    private final List<List<Card>> tableau = new ArrayList<>();
    // face-up card counts per tableau pile; visible cards are the last N cards in the pile.
    private final List<Integer> tableauFaceUp = new ArrayList<>();
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

    /**
     * Returns full tableau piles including facedown cards.
     * Engine/UI only; AIs should prefer {@link #getVisibleTableau()}.
     */
    @Deprecated
    public List<List<Card>> getTableau() {
        return unmodifiablePiles(tableau);
    }

    /**
     * Returns a visibility-safe view of the tableau where each pile contains only
     * the face-up (visible) suffix. Hidden cards are excluded and the returned
     * lists are immutable snapshots.
     */
    public List<List<Card>> getVisibleTableau() {
        List<List<Card>> visible = new ArrayList<>(tableau.size());
        for (int i = 0; i < tableau.size(); i++) {
            List<Card> pile = tableau.get(i);
            int faceUp = i < tableauFaceUp.size() ? tableauFaceUp.get(i) : 0;
            if (pile.isEmpty() || faceUp <= 0) {
                visible.add(Collections.emptyList());
                continue;
            }
            int start = Math.max(0, pile.size() - faceUp);
            List<Card> suffix = new ArrayList<>(pile.subList(start, pile.size()));
            visible.add(Collections.unmodifiableList(suffix));
        }
        return Collections.unmodifiableList(visible);
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
     * Returns the count of face-up (visible) cards for each tableau pile.
     * Visible cards are the last N cards in the corresponding internal pile.
     */
    public List<Integer> getTableauFaceUpCounts() {
        return Collections.unmodifiableList(new ArrayList<>(tableauFaceUp));
    }

    /**
     * Returns the count of facedown (hidden) cards for each tableau pile,
     * derived as total size minus the face-up count.
     */
    public List<Integer> getTableauFaceDownCounts() {
        List<Integer> faceDown = new ArrayList<>(tableau.size());
        for (int i = 0; i < tableau.size(); i++) {
            List<Card> pile = tableau.get(i);
            int faceUp = i < tableauFaceUp.size() ? tableauFaceUp.get(i) : 0;
            int count = Math.max(0, pile.size() - faceUp);
            faceDown.add(count);
        }
        return Collections.unmodifiableList(faceDown);
    }

    /**
     * Turn three cards from the stockpile onto the talon, preserving order of draw.
     * If the stockpile is empty, recycle the talon back into the stockpile (face-down)
     * before turning.
     */
    public void turnThree() {
        if (stockpile.isEmpty() && !talon.isEmpty()) {
            // Recycle talon back to stockpile: top of talon becomes bottom of new stock.
            for (int i = talon.size() - 1; i >= 0; i--) {
                stockpile.add(talon.get(i));
            }
            talon.clear();
        }

        for (int i = 0; i < 3 && !stockpile.isEmpty(); i++) {
            talon.add(stockpile.remove(stockpile.size() - 1));
        }
    }

    /**
     * Attempt to move a card (and any cards above it) from one pile to another (e.g., "move T6 Q♣ F3").
     * If cardCode is null, uses the top visible card for tableau, or the top card for other piles.
     * Returns true on success, false if the move is illegal or piles are invalid/empty.
     */
    public boolean moveCard(String from, String cardCode, String to) {
        MoveResult result = attemptMove(from, cardCode, to);
        return result.success;
    }

    /**
     * Attempt to move a card and return a descriptive result for error reporting.
     */
    public MoveResult attemptMove(String from, String cardCode, String to) {
        String fromNormalized = normalizeCode(from);
        String toNormalized = normalizeCode(to);
        if (fromNormalized == null) {
            return MoveResult.failure("Invalid source pile code: " + from);
        }
        if (toNormalized == null) {
            return MoveResult.failure("Invalid destination pile code: " + to);
        }

        char fromType = fromNormalized.charAt(0);
        char toType = toNormalized.charAt(0);
        int fromIndex = parseIndex(fromNormalized);
        int toIndex = parseIndex(toNormalized);

        List<Card> fromPile = resolvePile(fromNormalized);
        List<Card> toPile = resolvePile(toNormalized);
        if (fromPile == null) {
            return MoveResult.failure("Unknown source pile: " + fromNormalized);
        }
        if (toPile == null) {
            return MoveResult.failure("Unknown destination pile: " + toNormalized);
        }
        if (fromPile.isEmpty()) {
            return MoveResult.failure("Source pile " + fromNormalized + " is empty.");
        }

        int faceUp = fromType == 'T' ? tableauFaceUp.get(fromIndex) : 1;
        if (fromType == 'T' && faceUp <= 0) {
            return MoveResult.failure("No face-up cards available in " + fromNormalized + ".");
        }
        int startFaceUpIdx = fromType == 'T' ? Math.max(0, fromPile.size() - faceUp) : fromPile.size() - 1;
        int movingIdx = fromPile.size() - 1; // default top card
        if (cardCode != null) {
            movingIdx = -1;
            for (int i = startFaceUpIdx; i < fromPile.size(); i++) {
                if (fromPile.get(i).matchesShortName(cardCode)) {
                    movingIdx = i;
                    break;
                }
            }
            if (movingIdx == -1) {
                return MoveResult.failure("Card " + cardCode + " is not visible in " + fromNormalized + ".");
            }
        }
        // Non-tableau sources must move the top card only.
        if (fromType != 'T' && movingIdx != fromPile.size() - 1) {
            return MoveResult.failure("Can only move the top card from " + fromNormalized + ".");
        }

        Card moving = fromPile.get(movingIdx);
        if (!isLegalMove(moving, toNormalized, toPile)) {
            String reason = toType == 'T'
                    ? "Tableau requires alternating color and one rank lower."
                    : "Foundation requires same suit and one rank higher starting with Ace.";
            return MoveResult.failure("Cannot place " + moving.shortName() + " on " + toNormalized + ". " + reason);
        }

        // Foundation accepts only a single top card.
        if (toType == 'F' && movingIdx != fromPile.size() - 1) {
            return MoveResult.failure("Foundation only accepts the top card, not a stack.");
        }

        int movedCount = fromPile.size() - movingIdx;
        List<Card> segment = new ArrayList<>(fromPile.subList(movingIdx, fromPile.size()));
        for (int i = fromPile.size() - 1; i >= movingIdx; i--) {
            fromPile.remove(i);
        }
        toPile.addAll(segment);

        if (fromType == 'T') {
            int remainingFaceUp = Math.max(0, tableauFaceUp.get(fromIndex) - movedCount);
            if (remainingFaceUp == 0 && !fromPile.isEmpty()) {
                remainingFaceUp = 1; // flip next card
            }
            tableauFaceUp.set(fromIndex, remainingFaceUp);
        }

        if (toType == 'T' && toIndex >= 0 && toIndex < tableauFaceUp.size()) {
            tableauFaceUp.set(toIndex, tableauFaceUp.get(toIndex) + movedCount);
        }
        return MoveResult.success("Moved " + moving.shortName() + " from " + fromNormalized + " to " + toNormalized + ".");
    }

    // Backward-compatible signature: move top/visible card.
    public boolean moveCard(String from, String to) {
        return moveCard(from, null, to);
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

    /**
     * Returns an opaque 64-bit key representing the full internal game state.
     * This is intended for AI cycle detection and does not expose hidden cards directly.
     */
    public long getStateKey() {
        long h = 0L;
        int idx = 0;

        // Tableau piles (include hidden + visible).
        for (List<Card> pile : tableau) {
            for (Card c : pile) {
                h ^= cardSlotKey(idx++, c);
            }
        }

        // Foundation piles.
        for (List<Card> pile : foundation) {
            for (Card c : pile) {
                h ^= cardSlotKey(idx++, c);
            }
        }

        // Stockpile.
        for (Card c : stockpile) {
            h ^= cardSlotKey(idx++, c);
        }

        // Talon.
        for (Card c : talon) {
            h ^= cardSlotKey(idx++, c);
        }

        // Encode stock vs talon presence as a simple extra bit of state.
        if (stockpile.isEmpty() && !talon.isEmpty()) {
            h ^= STATE_ZOBRIST[STATE_ZOBRIST.length - 1];
        }

        return h;
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

    public static class MoveResult {
        public final boolean success;
        public final String message;

        private MoveResult(boolean success, String message) {
            this.success = success;
            this.message = message;
        }

        public static MoveResult success(String message) {
            return new MoveResult(true, message);
        }

        public static MoveResult failure(String message) {
            return new MoveResult(false, message);
        }
    }

    private static class TableauDisplay {
        final List<String> labels;
        final List<List<String>> columns;

        TableauDisplay(List<String> labels, List<List<String>> columns) {
            this.labels = labels;
            this.columns = columns;
        }
    }

    private static long[] initStateZobrist() {
        // Enough entries for all possible card slots plus one extra flag.
        int size = 64 * 64 + 1;
        long[] table = new long[size];
        long seed = 0x9E3779B97F4A7C15L;
        for (int i = 0; i < size; i++) {
            seed ^= (seed << 7);
            seed ^= (seed >>> 9);
            seed ^= (seed << 8);
            table[i] = seed;
        }
        return table;
    }

    private long cardSlotKey(int slotIndex, Card card) {
        int ordinal = card.getSuit().ordinal() * Rank.values().length + card.getRank().ordinal();
        int index = (slotIndex * 64 + ordinal) % (STATE_ZOBRIST.length - 1);
        return STATE_ZOBRIST[index];
    }
}
