package ai.games.player.ai.tree;

/**
 * Represents a move signature: the source, destination, and card moved.
 * Used to detect inverse moves (ping-pong) across different search strategies.
 * 
 * <p>A move signature captures just enough information to determine if two moves
 * are inversions of each other (e.g., moving from T1 to T2, then back from T2 to T1).
 * This is useful for pruning obviously-wasteful two-move loops during search and playouts.
 */
public final class MoveSignature {
    final String from;
    final String to;
    final String card;

    private MoveSignature(String from, String to, String card) {
        this.from = from;
        this.to = to;
        this.card = card;
    }

    /**
     * Parses a move command string into a MoveSignature.
     *
     * <p>Expects moves in the format:
     * <ul>
     *   <li>"move [from] [card] [to]" - standard tableau/foundation moves
     *   <li>"move [from] [to]" - simplified two-arg format (card is empty string)
     * </ul>
     *
     * @param command the move command string
     * @return a MoveSignature if parsing succeeds, null otherwise
     */
    public static MoveSignature tryParse(String command) {
        if (command == null) {
            return null;
        }
        String[] parts = command.trim().split("\\s+");
        if (parts.length < 3) {
            return null;
        }
        if (!parts[0].equalsIgnoreCase("move")) {
            return null;
        }
        String from = parts[1].toUpperCase();
        String card = parts[2].toUpperCase();
        String to;
        if (parts.length == 4) {
            to = parts[3].toUpperCase();
        } else if (parts.length == 3) {
            to = "";
        } else {
            return null;
        }
        return new MoveSignature(from, to, card);
    }

    /**
     * Checks if this move is the inverse of another (ping-pong detection).
     *
     * <p>Two moves are inverses if they move the same card but swap source and destination:
     * <ul>
     *   <li>"move T1 AS T2" is the inverse of "move T2 AS T1"
     *   <li>"move T1 AS F" is NOT the inverse of anything (F cannot move back)
     * </ul>
     *
     * @param other the other move signature to compare against
     * @return true if this move is the inverse of other, false otherwise
     */
    public boolean isInverseOf(MoveSignature other) {
        if (other == null) {
            return false;
        }
        if (!this.card.equals(other.card)) {
            return false;
        }
        return this.from.equals(other.to) && this.to.equals(other.from);
    }
}
