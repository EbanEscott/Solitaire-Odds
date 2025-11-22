package ai.games;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Game implements CommandLineRunner {
    private static final Logger log = LoggerFactory.getLogger(Game.class);
    private final Player player;

    public Game(Player player) {
        this.player = player;
    }

    public static void main(String[] args) {
        SpringApplication.run(Game.class, args);
    }

    @Override
    public void run(String... args) {
        Deck deck = new Deck();
        Solitaire solitaire = new Solitaire(deck);
        boolean aiMode = player instanceof AIPlayer;
        int iterations = 0;
        final int maxIterations = 500;

        while (true) {
            if (log.isDebugEnabled()) {
                log.debug("Current board:\n{}", stripAnsi(solitaire.toString()));
            }
            System.out.println(solitaire);
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

            String input = player.nextCommand(solitaire);
            if (input == null) {
                System.out.println("Input closed. Exiting.");
                if (log.isDebugEnabled()) {
                    log.debug("Input closed for player {}", player.getClass().getSimpleName());
                }
                break;
            }
            input = input.trim();
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
            if (input.equalsIgnoreCase("quit")) {
                break;
            } else if (input.equalsIgnoreCase("turn")) {
                solitaire.turnThree();
            } else if (input.toLowerCase().startsWith("move")) {
                String[] parts = input.split("\\s+");
                if (parts.length == 3) {
                    boolean moved = solitaire.moveCard(parts[1], parts[2]);
                    if (!moved) {
                        System.out.println("Illegal move. Try again.");
                        if (log.isDebugEnabled()) {
                            log.debug("Illegal move command: {}", input);
                        }
                    } else if (log.isDebugEnabled()) {
                        log.debug("Applied move command: {}", input);
                    }
                } else {
                    System.out.println("Usage: move FROM TO (e.g., move W T1 or move T7 F1)");
                    if (log.isDebugEnabled()) {
                        log.debug("Invalid move format from {}: {}", player.getClass().getSimpleName(), input);
                    }
                }
            } else {
                System.out.println("Unknown command. Use 'turn', 'move FROM TO', or 'quit'.");
                if (log.isDebugEnabled()) {
                    log.debug("Unknown command from {}: {}", player.getClass().getSimpleName(), input);
                }
            }
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
