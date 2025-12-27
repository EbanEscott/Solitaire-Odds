package ai.games.player;

import ai.games.config.TrainingModeProperties;
import ai.games.game.Solitaire;
import java.util.Scanner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Human player that reads commands from stdin (CLI).
 */
@Component
@Profile("ai-human")
public class HumanPlayer implements Player {
    private final Scanner scanner = new Scanner(System.in);
    private final TrainingModeProperties trainingMode;

    public HumanPlayer(TrainingModeProperties trainingMode) {
        this.trainingMode = trainingMode;
    }

    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        String prompt = buildPrompt(solitaire);
        System.out.print(prompt);
        if (!scanner.hasNextLine()) {
            return null;
        }
        return scanner.nextLine();
    }

    /**
     * Builds the command prompt, optionally including 'undo' if training mode
     * is enabled and there is at least one move to undo.
     *
     * @param solitaire the current game state
     * @return the prompt string
     */
    private String buildPrompt(Solitaire solitaire) {
        if (!trainingMode.isMode()) {
            return "Enter command (turn | move FROM TO | quit): ";
        }
        
        // In training mode, show undo only if there are moves to undo
        if (solitaire.canUndo()) {
            return "Enter command (turn | move FROM TO | undo | quit): ";
        } else {
            return "Enter command (turn | move FROM TO | quit): ";
        }
    }
}

