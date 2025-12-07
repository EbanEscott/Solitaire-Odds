package ai.games.player.ai.alpha;

import java.util.List;

/**
 * DTO representing the response from the Python AlphaSolitaire service.
 *
 * Matches the JSON shape returned by /evaluate.
 */
public class AlphaSolitaireResponse {

    public static class MoveScore {
        private String command;
        private double probability;

        public MoveScore() {
            // Default constructor for JSON binding.
        }

        public String getCommand() {
            return command;
        }

        public void setCommand(String command) {
            this.command = command;
        }

        public double getProbability() {
            return probability;
        }

        public void setProbability(double probability) {
            this.probability = probability;
        }
    }

    private String chosenCommand;
    private double winProbability;
    private List<MoveScore> legalMoves;

    public AlphaSolitaireResponse() {
        // Default constructor for JSON binding.
    }

    public String getChosenCommand() {
        return chosenCommand;
    }

    public void setChosenCommand(String chosenCommand) {
        this.chosenCommand = chosenCommand;
    }

    public double getWinProbability() {
        return winProbability;
    }

    public void setWinProbability(double winProbability) {
        this.winProbability = winProbability;
    }

    public List<MoveScore> getLegalMoves() {
        return legalMoves;
    }

    public void setLegalMoves(List<MoveScore> legalMoves) {
        this.legalMoves = legalMoves;
    }
}

