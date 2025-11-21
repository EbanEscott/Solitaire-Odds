public class HelloWorld {
    public static void main(String[] args) {
        Deck deck = new Deck();
        Solitaire solitaire = new Solitaire(deck);
        java.util.Scanner scanner = new java.util.Scanner(System.in);

        while (true) {
            System.out.println(solitaire);
            if (isWon(solitaire)) {
                System.out.println("🎉🤗🎉 Congrats, you moved every card to the foundations! 🎉🤗🎉");
                break;
            }

            System.out.print("Enter command (turn | move FROM TO | quit): ");
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
