package ai.games;

import java.util.Scanner;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class HelloWorld implements CommandLineRunner {

    public static void main(String[] args) {
        SpringApplication.run(HelloWorld.class, args);
    }

    @Override
    public void run(String... args) {
        Deck deck = new Deck();
        Solitaire solitaire = new Solitaire(deck);
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println(solitaire);
            if (isWon(solitaire)) {
                System.out.println("🎉🤗🎉 Congrats, you moved every card to the foundations! 🎉🤗🎉");
                break;
            }

            System.out.print("Enter command (turn | move FROM TO | quit): ");
            if (!scanner.hasNextLine()) {
                System.out.println("Input closed. Exiting.");
                break;
            }
            String input = scanner.nextLine().trim();
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
