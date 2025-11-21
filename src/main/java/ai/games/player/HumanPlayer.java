package ai.games;

import java.util.Scanner;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

/**
 * Human player that reads commands from stdin (CLI).
 */
@Component
@Primary
public class HumanPlayer implements Player {
    private final Scanner scanner = new Scanner(System.in);

    @Override
    public String nextCommand(Solitaire solitaire) {
        System.out.print("Enter command (turn | move FROM TO | quit): ");
        if (!scanner.hasNextLine()) {
            return null;
        }
        return scanner.nextLine();
    }
}
