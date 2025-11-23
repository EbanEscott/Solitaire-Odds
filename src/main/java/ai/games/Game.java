package ai.games;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.WebApplicationType;

@SpringBootApplication
public class Game implements CommandLineRunner {
    private static final Logger log = LoggerFactory.getLogger(Game.class);
    private final Player player;

    public Game(Player player) {
        this.player = player;
    }

    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(Game.class);
        app.setWebApplicationType(WebApplicationType.NONE);
        app.run(args);
    }

    @Override
    public void run(String... args) {
        Deck deck = new Deck();
        Solitaire solitaire = new Solitaire(deck);
        boolean aiMode = player instanceof AIPlayer;
        String feedback = "";
        String illegalFeedback = "";
        java.util.Map<String, Guidance> persistentGuidance = new java.util.HashMap<>();
        String lastCommand = null;
        String secondLastCommand = null;
        int sameCommandCount = 0;
        int pingPongCount = 0;
        int turnsSinceLastMove = 0;
        int previousStockSize = solitaire.getStockpile().size();
        int iterations = 0;
        int stockEmptyStrikes = 0;          // full stock passes with zero successful moves
        int movesSinceLastStockEmpty = 0;   // successful moves since last time STOCK hit 0
        
        // Cap iterations at ~8× a typical winning game (≈120–135 moves incl. stock turns). Anything beyond this
        // is overwhelmingly likely to be looping or non-productive searching, so we bail out to keep runs finite.
        final int maxIterations = 10000;

        while (true) {
            if (log.isDebugEnabled()) {
                log.debug("Current board:\n{}", stripAnsi(solitaire.toString()));
            }
            System.out.println(solitaire);
            if (!feedback.isBlank() && aiMode) {
                System.out.println("Feedback: " + feedback);
            }
            if (isWon(solitaire)) {
                System.out.println("🎉🤗🎉 Congrats, you moved every card to the foundations! 🎉🤗🎉");
                if (log.isDebugEnabled()) {
                    log.debug("Game won by {}", player.getClass().getSimpleName());
                }
                break;
            }

            if (!aiMode) {
                System.out.print("Enter command (turn | move FROM TO | quit): ");
            }

            String input = player.nextCommand(solitaire, feedback);
            if (input == null) {
                System.out.println("Input closed. Exiting.");
                if (log.isDebugEnabled()) {
                    log.debug("Input closed for player {}", player.getClass().getSimpleName());
                }
                break;
            }
            input = input.trim();
            // Track simple repetition and ping-pong patterns (A,B,A,B,...).
            if (lastCommand != null && input.equalsIgnoreCase(lastCommand)) {
                sameCommandCount++;
            } else {
                sameCommandCount = 1;
            }
            if (secondLastCommand != null
                    && input.equalsIgnoreCase(secondLastCommand)
                    && !input.equalsIgnoreCase(lastCommand)) {
                // e.g. history ... A,B and we see A again -> potential ping-pong.
                pingPongCount++;
            } else if (!input.equalsIgnoreCase(lastCommand)) {
                pingPongCount = 1;
            }
            secondLastCommand = lastCommand;
            lastCommand = input;
            if (log.isDebugEnabled()) {
                log.debug("Received command from {}: {}", player.getClass().getSimpleName(), input);
            }
            if (aiMode) {
                System.out.println("AI command: " + input);
            }
            iterations++;
            if (iterations > maxIterations) {
                if (log.isDebugEnabled()) {
                    log.debug("Max iterations ({}) reached, stopping game loop to avoid runaway execution.", maxIterations);
                }
                break;
            }
            java.util.List<String> suggestionLines = new java.util.ArrayList<>();
            int stockBefore = solitaire.getStockpile().size();
            if (input.equalsIgnoreCase("quit")) {
                feedback = "Player chose to quit.";
                illegalFeedback = "";
                break;
            } else if (input.equalsIgnoreCase("turn")) {
                solitaire.turnThree();
                illegalFeedback = "";
                turnsSinceLastMove++;
            } else if (input.toLowerCase().startsWith("move")) {
                String[] parts = input.split("\\s+");
                if (parts.length == 4) {
                    Solitaire.MoveResult result = solitaire.attemptMove(parts[1], parts[2], parts[3]);
                    if (!result.success) {
                        String reason = result.message == null ? "Illegal move." : result.message;
                        illegalFeedback = "Your last command was illegal:\n"
                                + "- " + input + "\n"
                                + "- Reason: " + reason + "\n"
                                + "- Do NOT repeat this exact command.";
                        System.out.println("Illegal move: " + reason);
                        if (log.isDebugEnabled()) {
                            log.debug("Illegal move command: {} ({})", input, reason);
                        }
                    } else {
                        if (log.isDebugEnabled()) {
                            log.debug("Applied move command: {}", input);
                        }
                        illegalFeedback = "";
                        turnsSinceLastMove = 0;
                        movesSinceLastStockEmpty++;
                        persistentGuidance.remove("quit");
                        stockEmptyStrikes = 0;
                    }
                } else if (parts.length == 3) {
                    Solitaire.MoveResult result = solitaire.attemptMove(parts[1], null, parts[2]);
                    if (!result.success) {
                        String reason = result.message == null ? "Illegal move." : result.message;
                        illegalFeedback = "Your last command was illegal:\n"
                                + "- " + input + "\n"
                                + "- Reason: " + reason + "\n"
                                + "- Do NOT repeat this exact command.";
                        System.out.println("Illegal move: " + reason);
                        if (log.isDebugEnabled()) {
                            log.debug("Illegal move command: {} ({})", input, reason);
                        }
                    } else {
                        if (log.isDebugEnabled()) {
                            log.debug("Applied move command: {}", input);
                        }
                        illegalFeedback = "";
                        turnsSinceLastMove = 0;
                        movesSinceLastStockEmpty++;
                        persistentGuidance.remove("quit");
                        stockEmptyStrikes = 0;
                    }
                } else {
                    illegalFeedback = "Usage error:\n"
                            + "- Usage: move FROM [CARD] TO (e.g., move W T1 or move T7 Q♣ F1)";
                    System.out.println("Usage: move FROM [CARD] TO (e.g., move W T1 or move T7 Q♣ F1)");
                    if (log.isDebugEnabled()) {
                        log.debug("Invalid move format from {}: {}", player.getClass().getSimpleName(), input);
                    }
                }
            } else {
                illegalFeedback = "Unknown command:\n"
                        + "- \"" + input + "\" is not recognised.\n"
                        + "- Use 'turn', 'move FROM TO', or 'quit'.";
                System.out.println("Unknown command. Use 'turn', 'move FROM TO', or 'quit'.");
                if (log.isDebugEnabled()) {
                    log.debug("Unknown command from {}: {}", player.getClass().getSimpleName(), input);
                }
            }

            // Suggestion 1: repeated identical command (same move over and over).
            if (sameCommandCount >= 4 && !input.equalsIgnoreCase("turn")) {
                String reason = "you have chosen this many times without making progress.";
                Guidance g = persistentGuidance.get(input);
                if (g == null) {
                    persistentGuidance.put(input, new Guidance(reason, 1, 8, iterations));
                } else {
                    g.refresh(iterations, 3);
                }
            }

            // Suggestion 1b: ping-pong between two commands (A,B,A,B,...).
            // Use current legal moves to tune TTL: if a move is no longer legal, let it decay faster.
            java.util.List<String> legalMovesForPingPong = LegalMovesHelper.listLegalMoves(solitaire);
            if (pingPongCount >= 3 && lastCommand != null && secondLastCommand != null) {
                String reason = "you are ping-ponging between two moves without improving the board.";
                for (String cmd : new String[]{ lastCommand, secondLastCommand }) {
                    int ttlBoost = legalMovesForPingPong.contains(cmd) ? 4 : 1;
                    Guidance g = persistentGuidance.get(cmd);
                    if (g == null) {
                        persistentGuidance.put(cmd, new Guidance(reason, 1, 8, iterations));
                    } else {
                        g.refresh(iterations, ttlBoost);
                    }
                }
            }

            // Suggestion 2: STOCK emptied without any successful moves since the last empty.
            int stockAfter = solitaire.getStockpile().size();
            if (input.equalsIgnoreCase("turn")
                    && stockBefore > 0
                    && stockAfter == 0) {

                if (movesSinceLastStockEmpty == 0) {
                    stockEmptyStrikes++;
                } else {
                    stockEmptyStrikes = 0;
                }
                movesSinceLastStockEmpty = 0;

                // Only start warning to quit after repeated unproductive passes.
                if (stockEmptyStrikes >= 2) {
                    String reason = "you have turned through the entire stockpile "
                            + stockEmptyStrikes + " times without making any moves.";
                    Guidance g = persistentGuidance.get("quit");
                    if (g == null) {
                        persistentGuidance.put("quit", new Guidance(reason, 1, 16, iterations));
                    } else {
                        g.refresh(iterations, 6);
                    }
                }
            }

            // Recalculate legal moves for the *next* turn, after applying the current command.
            java.util.List<String> legalMoves = LegalMovesHelper.listLegalMoves(solitaire);

            // Build guidance for this turn from persistent guidance that is still relevant.
            java.util.Iterator<java.util.Map.Entry<String, Guidance>> it = persistentGuidance.entrySet().iterator();
            while (it.hasNext()) {
                var entry = it.next();
                String cmd = entry.getKey();
                Guidance g = entry.getValue();

                // Expire old guidance only after its TTL.
                if (!g.stillAlive(iterations)) {
                    it.remove();
                    continue;
                }

                // Show only when relevant to the current legal move set,
                // but keep it alive even when temporarily illegal.
                if (!legalMoves.contains(cmd)) {
                    continue;
                }

                if ("turn".equalsIgnoreCase(cmd)) {
                    continue;
                }

                if ("quit".equalsIgnoreCase(cmd)) {
                    suggestionLines.add("quit (" + g.reason + ")");
                } else {
                    suggestionLines.add("don't " + cmd + " (" + g.reason + ")");
                }
            }

            String suggestions = "";
            if (!suggestionLines.isEmpty()) {
                suggestions = "Guidance for this turn:\n- " + String.join("\n- ", suggestionLines);
            }

            // Build feedback for next turn from illegal feedback + suggestions.
            if (!illegalFeedback.isBlank() && !suggestions.isBlank()) {
                feedback = illegalFeedback + "\n\n" + suggestions;
            } else if (!illegalFeedback.isBlank()) {
                feedback = illegalFeedback;
            } else if (!suggestions.isBlank()) {
                feedback = suggestions;
            } else {
                feedback = "";
            }
        }
    }

    private static final class Guidance {
        final String reason;
        int strikes;
        int ttlTurns;
        int lastSeenIteration;

        Guidance(String reason, int strikes, int ttlTurns, int lastSeenIteration) {
            this.reason = reason;
            this.strikes = strikes;
            this.ttlTurns = ttlTurns;
            this.lastSeenIteration = lastSeenIteration;
        }

        void refresh(int iteration, int ttlBoost) {
            this.strikes++;
            this.ttlTurns = Math.min(this.ttlTurns + ttlBoost, 30);
            this.lastSeenIteration = iteration;
        }

        boolean stillAlive(int iteration) {
            return (iteration - lastSeenIteration) <= ttlTurns;
        }
    }

    private static boolean isWon(Solitaire solitaire) {
        int total = 0;
        for (java.util.List<Card> pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total == 52;
    }

    private static String stripAnsi(String input) {
        return input.replaceAll("\\u001B\\[[;\\d]*m", "");
    }
}
