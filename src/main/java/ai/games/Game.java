package ai.games;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import ai.games.player.Player;
import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.game.Card;

@SpringBootApplication
public class Game implements CommandLineRunner {
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

        while (true) {
            System.out.println(solitaire);
            if (isWon(solitaire)) {
                System.out.println("🎉🤗🎉 Congrats, you moved every card to the foundations! 🎉🤗🎉");
                break;
            }

            String input = player.nextCommand(solitaire);
            if (input == null) {
                System.out.println("Input closed. Exiting.");
                break;
            }
            input = input.trim();
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
                    }
                } else {
                    System.out.println("Usage: move FROM TO (e.g., move W T1 or move T7 F1)");
                }
            } else {
                System.out.println("Unknown command. Use 'turn', 'move FROM TO', or 'quit'.");
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
}
