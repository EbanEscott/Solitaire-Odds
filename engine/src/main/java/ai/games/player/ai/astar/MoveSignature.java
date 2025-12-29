package ai.games.player.ai.astar;

/**
 * Represents a move signature: the source, destination, and card moved.
 * Used to detect inverse moves (ping-pong) within A* search.
 */
final class MoveSignature {
    final String from;
    final String to;
    final String card;

    private MoveSignature(String from, String to, String card) {
        this.from = from;
        this.to = to;
        this.card = card;
    }

    static MoveSignature tryParse(String command) {
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

    boolean isInverseOf(MoveSignature other) {
        if (other == null) {
            return false;
        }
        if (!this.card.equals(other.card)) {
            return false;
        }
        return this.from.equals(other.to) && this.to.equals(other.from);
    }
}
