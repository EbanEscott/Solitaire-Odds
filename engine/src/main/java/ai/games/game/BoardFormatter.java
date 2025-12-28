package ai.games.game;

import java.util.ArrayList;
import java.util.List;

/**
 * Handles formatting and rendering of the Solitaire board for console display.
 * <p>
 * Encapsulates all display logic including borders, padding, ANSI colour handling, and layout.
 * Responsible for converting game state into a human-readable board representation with coloured
 * cards (red/black suits), padded cells, and bordered sections for foundation, tableau, and stock.
 * <p>
 * All length calculations account for ANSI colour escape sequences, ensuring proper alignment
 * regardless of whether colour codes are present in the output.
 * <p>
 * <strong>TODO: Full Talon Display:</strong>
 * The talon section currently displays only the top card plus a count (e.g., "Q♣ (3)") to avoid
 * cluttering the console. Per official Klondike rules, all three drawn cards are face-up and
 * visible to the player. Consider implementing a new gameplay command (e.g., `show talon` or `show W`)
 * that displays the full sequence of talon cards to improve player awareness. This would be especially
 * useful during debugging and AI analysis, as the underlying {@link Solitaire#getTalon()} method
 * already provides full access to all cards.
 */
public class BoardFormatter {
    /** Default cell width (in characters) used when content is narrower than this value. */
    private static final int CELL_WIDTH = 8;

    /** Reference to the Solitaire game state being formatted. */
    private final Solitaire solitaire;

    /**
     * Constructs a new BoardFormatter for the given Solitaire game state.
     *
     * @param solitaire the Solitaire game state to format; must not be null
     * @throws NullPointerException if solitaire is null
     */
    public BoardFormatter(Solitaire solitaire) {
        this.solitaire = solitaire;
    }

    /**
     * Renders the complete board state as a formatted string with all sections.
     * <p>
     * Generates a multi-section display including:
     * <ul>
     *   <li>A top border line for visual separation</li>
     *   <li>Foundation piles (F1–F4) showing the top card of each suit</li>
     *   <li>Tableau piles (T1–T7) with all visible cards in vertical layout and face-down counts</li>
     *   <li>Stockpile and talon showing remaining cards and current waste</li>
     * </ul>
     * All cells are padded and bordered for readability, and ANSI colour codes are preserved.
     *
     * @return a formatted board representation as a multi-line string
     */
    public String format() {
        StringBuilder sb = new StringBuilder();
        int tableauCellWidth = computeTableauCellWidth();
        String tableauBorder = buildBorder(solitaire.getVisibleTableau().size(), "  ", tableauCellWidth);
        sb.append("-".repeat(tableauBorder.length())).append('\n');
        appendFoundationSection(sb);
        appendTableauSection(sb, tableauCellWidth);
        appendStockAndTalon(sb);
        return sb.toString();
    }

    /**
     * Appends the foundation piles section to the board display.
     * <p>
     * Displays all four foundation piles (F1–F4) in a boxed row format, showing either the
     * top card of each pile (coloured) or \"--\" for empty piles.
     *
     * @param sb the StringBuilder to append to; must not be null
     */
    private void appendFoundationSection(StringBuilder sb) {
        sb.append("FOUNDATION\n");
        List<String> labels = new ArrayList<>();
        List<String> values = new ArrayList<>();
        for (int i = 0; i < solitaire.getFoundation().size(); i++) {
            labels.add("F" + (i + 1));
            List<Card> pile = solitaire.getFoundation().get(i);
            values.add(pile.isEmpty() ? "--" : pile.get(pile.size() - 1).toString());
        }
        int width = Math.max(CELL_WIDTH, Math.max(maxVisibleLength(labels), maxVisibleLength(values)));
        appendBoxRow(sb, labels, values, "    ", width);
        sb.append('\n');
    }

    /**
     * Appends the tableau piles section to the board display.
     * <p>
     * Displays all seven tableau piles (T1–T7) with headers showing face-down card counts
     * and all visible cards in vertical layout. Piles are separated by borders and can have
     * varying heights based on the number of visible cards in each pile.
     *
     * @param sb the StringBuilder to append to; must not be null
     * @param cellWidth the optimal cell width for layout (from computeTableauCellWidth)
     */
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

    /**
     * Appends a boxed row (two rows: labels and contents) to the display.
     * <p>
     * Creates a bordered box with a header row (labels) and content row (values), both centred
     * within cells of the given width. Used for foundation and stockpile/talon sections.
     *
     * @param sb the StringBuilder to append to; must not be null
     * @param labels the header labels (e.g., \"F1\", \"F2\", etc.); must not be null
     * @param contents the values to display (e.g., card names or \"--\"); must not be null
     * @param indent the indentation prefix (spaces); must not be null
     * @param width the cell width in characters
     */
    private void appendBoxRow(StringBuilder sb, List<String> labels, List<String> contents, String indent, int width) {
        String top = buildBorder(labels.size(), indent, width);
        String labelLine = buildRow(labels, indent, width);
        String contentLine = buildRow(contents, indent, width);
        sb.append(top).append('\n').append(labelLine).append('\n').append(contentLine).append('\n').append(top);
    }

    /**
     * Constructs a horizontal border line with cells separated by spaces.
     * <p>
     * Generates a line of the form: \"{indent}+---+  +---+  +---+\" based on the number
     * of cells and their width. Cells are joined by \"  \" (two spaces) for readability.
     *
     * @param count the number of cells in the border
     * @param indent the indentation prefix (spaces); must not be null
     * @param cellWidth the width of each cell (content area between + signs)
     * @return the constructed border line
     */
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

    /**
     * Constructs a data row with cells separated by borders and indentation.
     * <p>
     * Generates a line of the form: \"{indent}| cell1 | | cell2 |\" with each cell
     * padded to the specified width. Cells are separated by \"  \" (two spaces).
     *
     * @param cells the content for each cell; must not be null
     * @param indent the indentation prefix (spaces); must not be null
     * @param cellWidth the width of each cell (for padding and layout)
     * @return the constructed data row
     */
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

    /**
     * Pads a string value to centre it within the given width.
     * <p>
     * If the value (accounting for ANSI colour codes) is shorter than the width, spaces
     * are added equally on both sides (or one more on the right if padding is odd).
     * If the value is already wider than or equal to the width, it is returned unchanged.
     *
     * @param value the string to pad (may contain ANSI colour codes); must not be null
     * @param width the target width in visible characters
     * @return the centred string, padded as necessary
     */
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

    /**
     * Calculates the visible length of a string, excluding ANSI colour escape sequences.
     * <p>
     * ANSI colour codes (e.g., \"\\u001B[31m\" for red) are invisible on screen and must be
     * excluded from width calculations to ensure proper cell alignment.
     *
     * @param value the string to measure (may contain ANSI colour codes); must not be null
     * @return the number of visible characters (excluding escape sequences)
     */
    private int visibleLength(String value) {
        String stripped = value.replaceAll("\\u001B\\[[;\\d]*m", "");
        return stripped.length();
    }

    /**
     * Finds the maximum visible length among a list of strings.
     * <p>
     * Accounts for ANSI colour codes by using {@link #visibleLength(String)}.
     *
     * @param items the strings to measure; must not be null
     * @return the maximum visible length among all items, or 0 if the list is empty
     */
    private int maxVisibleLength(List<String> items) {
        int max = 0;
        for (String item : items) {
            max = Math.max(max, visibleLength(item));
        }
        return max;
    }

    /**
     * Finds the maximum visible length among all strings in a 2D column structure.
     * <p>
     * Iterates through all columns and all items within columns to find the longest
     * visible string. Accounts for ANSI colour codes.
     *
     * @param columns a list of columns, where each column is a list of strings; must not be null
     * @return the maximum visible length across all items in all columns, or 0 if empty
     */
    private int maxVisibleLengthColumns(List<List<String>> columns) {
        int max = 0;
        for (List<String> col : columns) {
            for (String item : col) {
                max = Math.max(max, visibleLength(item));
            }
        }
        return max;
    }

    /**
     * Computes the optimal cell width for tableau display based on content.
     * <p>
     * Examines all tableau labels and card names to determine the minimum width needed
     * to display them without truncation. Returns the maximum of this computed width and
     * the default {@link #CELL_WIDTH}.
     *
     * @return the optimal cell width in characters, guaranteed to be at least CELL_WIDTH
     */
    private int computeTableauCellWidth() {
        TableauDisplay display = buildTableauDisplay();
        int maxContent = Math.max(maxVisibleLength(display.labels), maxVisibleLengthColumns(display.columns));
        return Math.max(CELL_WIDTH, maxContent);
    }

    /**
     * Appends the stockpile and talon section to the board display.
     * <p>
     * Shows the stockpile state (either \"empty\" or the number of face-down cards) and the
     * talon state (either \"--\" for empty or the top card with count in parentheses).
     * Displays as a boxed row similar to the foundation section.
     *
     * @param sb the StringBuilder to append to; must not be null
     */
    private void appendStockAndTalon(StringBuilder sb) {
        sb.append("STOCKPILE & TALON\n");
        List<String> labels = new ArrayList<>();
        List<String> values = new ArrayList<>();

        labels.add("STOCK");
        labels.add("TALON");

        values.add(solitaire.getStockpile().isEmpty() ? "empty" : solitaire.getStockpile().size() + " down");
        if (solitaire.getTalon().isEmpty()) {
            values.add("--");
        } else {
            Card top = solitaire.getTalon().get(solitaire.getTalon().size() - 1);
            values.add(top + " (" + solitaire.getTalon().size() + ")");
        }

        int width = Math.max(CELL_WIDTH, Math.max(maxVisibleLength(labels), maxVisibleLength(values)));
        appendBoxRow(sb, labels, values, "      ", width);
    }

    /**
     * Constructs a TableauDisplay object containing labels and visible cards for all tableau piles.
     * <p>
     * For each pile, creates a label showing the pile number and face-down card count
     * (e.g., \"T1 [2]\"), then collects all visible cards in vertical order. Empty piles
     * are marked with \"(empty)\".
     *
     * @return a TableauDisplay containing pile labels and card columns
     */
    private TableauDisplay buildTableauDisplay() {
        List<String> labels = new ArrayList<>();
        List<List<String>> columns = new ArrayList<>();
        List<List<Card>> visibleTableau = solitaire.getVisibleTableau();
        List<Integer> faceDownCounts = solitaire.getTableauFaceDownCounts();
        
        for (int i = 0; i < visibleTableau.size(); i++) {
            List<Card> pile = visibleTableau.get(i);
            int faceDown = faceDownCounts.get(i);
            labels.add("T" + (i + 1) + " [" + faceDown + "]");

            List<String> col = new ArrayList<>();
            if (pile.isEmpty()) {
                col.add("(empty)");
            } else {
                for (Card card : pile) {
                    col.add(card.toString());
                }
            }
            columns.add(col);
        }
        return new TableauDisplay(labels, columns);
    }

    /**
     * Helper class for organising tableau display data.
     * <p>
     * Packages the seven tableau pile labels and their corresponding visible card columns
     * for efficient layout computation and rendering.
     */
    private static class TableauDisplay {
        /** Labels for each tableau pile, e.g., [\"T1 [0]\", \"T2 [1]\", ...]. */
        final List<String> labels;
        
        /** Visible card columns: each element is a list of cards for one pile. */
        final List<List<String>> columns;

        /**
         * Constructs a TableauDisplay with the given labels and columns.
         *
         * @param labels the pile labels; must not be null and have exactly 7 elements
         * @param columns the visible card columns; must not be null and have exactly 7 elements
         */
        TableauDisplay(List<String> labels, List<List<String>> columns) {
            this.labels = labels;
            this.columns = columns;
        }
    }
}
