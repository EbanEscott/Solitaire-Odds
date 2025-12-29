package ai.games.player.ai.alpha;

import java.util.List;

/**
 * DTO representing the response from the Python AlphaSolitaire service.
 *
 * The service returns the output of a policy–value neural network:
 * - Policy head: probabilities for each legal move
 * - Value head: scalar estimate of win probability for the current position
 *
 * Matches the JSON shape returned by /evaluate.
 */
public class AlphaSolitaireResponse {

    /**
     * The move command chosen by the service (for reference; typically used for logging).
     * This is derived from the policy head's highest-probability move.
     */
    private String chosenCommand;

    /**
     * Win probability estimate from the value head.
     * Represents the neural network's prediction of how likely the current
     * board position is to result in a win (0.0 = certain loss, 1.0 = certain win).
     */
    private double winProbability;
    
    /**
     * Policy head outputs: probabilities for each legal move.
     * Used by MCTS as prior probabilities to guide exploration.
     */
    private List<MoveScore> legalMoves;

    /**
     * Default constructor for JSON deserialization.
     */
    public AlphaSolitaireResponse() {
        // Default constructor for JSON binding.
    }

    /**
     * Get the move chosen by the service (for reference; typically used for logging).
     *
     * @return the chosen command as a string (e.g., "move T1 A♥ F1")
     */
    public String getChosenCommand() {
        return chosenCommand;
    }

    /**
     * Set the chosen command.
     *
     * @param chosenCommand the command string
     */
    public void setChosenCommand(String chosenCommand) {
        this.chosenCommand = chosenCommand;
    }

    /**
     * Get the estimated win probability for the current position (value head output).
     *
     * @return probability between 0.0 (certain loss) and 1.0 (certain win)
     */
    public double getWinProbability() {
        return winProbability;
    }

    /**
     * Set the win probability.
     *
     * @param winProbability the estimated win probability
     */
    public void setWinProbability(double winProbability) {
        this.winProbability = winProbability;
    }

    /**
     * Get the list of legal moves with their policy probabilities.
     *
     * @return list of MoveScore objects (move command + policy probability)
     */
    public List<MoveScore> getLegalMoves() {
        return legalMoves;
    }

    /**
     * Set the list of legal moves.
     *
     * @param legalMoves list of MoveScore objects from the policy head
     */
    public void setLegalMoves(List<MoveScore> legalMoves) {
        this.legalMoves = legalMoves;
    }

    /**
     * Represents a single move and its policy probability from the neural network.
     * Used to communicate what moves are available and how likely each is to be winning.
     */
    public static class MoveScore {
        private String command;
        private double probability;

        /**
         * Default constructor for JSON deserialization.
         */
        public MoveScore() {
            // Default constructor for JSON binding.
        }

        /**
         * Get the move command (e.g., "move T1 A♥ F1", "turn", "quit").
         *
         * @return the move command string
         */
        public String getCommand() {
            return command;
        }

        /**
         * Set the move command.
         *
         * @param command the move command string
         */
        public void setCommand(String command) {
            this.command = command;
        }

        /**
         * Get the probability of this move according to the policy head.
         * Higher values indicate the neural network predicts this move is stronger.
         *
         * @return probability between 0.0 and 1.0
         */
        public double getProbability() {
            return probability;
        }

        /**
         * Set the policy probability for this move.
         *
         * @param probability the move probability from the policy head
         */
        public void setProbability(double probability) {
            this.probability = probability;
        }
    }
}

