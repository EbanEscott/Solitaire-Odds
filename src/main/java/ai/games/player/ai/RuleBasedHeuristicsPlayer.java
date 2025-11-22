package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import java.util.List;
import org.springframework.context.annotation.Primary;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Rule-based, deterministic heuristics: tries simple priorities without search.
 */
@Component
@Primary
@Profile("ai-rule")
public class RuleBasedHeuristicsPlayer extends AIPlayer implements Player {

    @Override
    public String nextCommand(Solitaire solitaire) {
        // 1) Talon to foundation if possible
        String move = talonToFoundation(solitaire);
        if (move != null) {
            return move;
        }

        // 2) Tableau to foundation if possible
        move = tableauToFoundation(solitaire);
        if (move != null) {
            return move;
        }

        // 3) Talon to tableau
        move = talonToTableau(solitaire);
        if (move != null) {
            return move;
        }

        // 4) Tableau to tableau
        move = tableauToTableau(solitaire);
        if (move != null) {
            return move;
        }

        // 5) Turn if stock is not empty
        if (!solitaire.getStockpile().isEmpty()) {
            return "turn";
        }

        // 6) Otherwise stop
        return "quit";
    }

    private String talonToFoundation(Solitaire solitaire) {
        List<Card> talon = solitaire.getTalon();
        Card moving = top(talon);
        if (moving == null) {
            return null;
        }
        List<List<Card>> foundations = solitaire.getFoundation();
        for (int i = 0; i < foundations.size(); i++) {
            if (canMoveToFoundation(moving, foundations.get(i))) {
                return "move W F" + (i + 1);
            }
        }
        return null;
    }

    private String tableauToFoundation(Solitaire solitaire) {
        List<List<Card>> tableau = solitaire.getTableau();
        List<List<Card>> foundations = solitaire.getFoundation();
        for (int t = 0; t < tableau.size(); t++) {
            Card moving = top(tableau.get(t));
            if (moving == null) {
                continue;
            }
            for (int f = 0; f < foundations.size(); f++) {
                if (canMoveToFoundation(moving, foundations.get(f))) {
                    return "move T" + (t + 1) + " F" + (f + 1);
                }
            }
        }
        return null;
    }

    private String talonToTableau(Solitaire solitaire) {
        Card moving = top(solitaire.getTalon());
        if (moving == null) {
            return null;
        }
        List<List<Card>> tableau = solitaire.getTableau();
        for (int t = 0; t < tableau.size(); t++) {
            if (canMoveToTableau(moving, tableau.get(t))) {
                return "move W T" + (t + 1);
            }
        }
        return null;
    }

    private String tableauToTableau(Solitaire solitaire) {
        List<List<Card>> tableau = solitaire.getTableau();
        for (int from = 0; from < tableau.size(); from++) {
            Card moving = top(tableau.get(from));
            if (moving == null) {
                continue;
            }
            for (int to = 0; to < tableau.size(); to++) {
                if (from == to) {
                    continue;
                }
                if (canMoveToTableau(moving, tableau.get(to))) {
                    return "move T" + (from + 1) + " T" + (to + 1);
                }
            }
        }
        return null;
    }
}
