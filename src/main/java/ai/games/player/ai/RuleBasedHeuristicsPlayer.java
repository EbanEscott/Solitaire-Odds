package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import ai.games.game.Rank;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import java.util.List;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Rule-based, deterministic heuristics. Rules:
 * 1) First move: always turn three.
 * 2) Check tableau then talon for foundation moves; if found, perform and repeat step 2.
 * 3) Scan T7 -> T1 for tableau moves considering the bottom visible card only (earliest face-up). If it fits, move the entire visible pile (one move command) to foundation or tableau; then return to step 2.
 * 4) If no tableau move, try talon to foundation/tableau; if moved, return to step 2.
 * 5) If no moves and stock remains, turn three and return to step 2.
 * 6) If stock is exhausted and no talon move has occurred in this pass, quit (avoid endless turns).
 */
@Component
@Profile("ai-rule")
public class RuleBasedHeuristicsPlayer extends AIPlayer implements Player {
    private boolean hasTurnedOnce = false;
    private boolean talonMovedThisPass = false;
    private boolean sawEmptyStock = false;

    @Override
    public String nextCommand(Solitaire solitaire, String feedback) {
        // Step 1: first move is to turn three.
        if (!hasTurnedOnce) {
            hasTurnedOnce = true;
            return "turn";
        }

        // Step 2: foundation priority from tableau then talon.
        String foundationMove = foundationPriority(solitaire);
        if (foundationMove != null) {
            return foundationMove;
        }

        // Step 3: tableau scan T7 -> T1.
        String tableauMove = tableauScan(solitaire);
        if (tableauMove != null) {
            return tableauMove;
        }

        // Step 4: talon move.
        String talonMove = talonMove(solitaire);
        if (talonMove != null) {
            talonMovedThisPass = true;
            return talonMove;
        }

        // Step 5/6: stock handling with pass detection.
        if (!solitaire.getStockpile().isEmpty()) {
            sawEmptyStock = false;
            return "turn";
        } else {
            if (sawEmptyStock && !talonMovedThisPass) {
                // Stock was already empty once in this pass and no talon moves happened -> quit.
                return "quit";
            }
            // First time hitting empty stock in this pass: remember and keep trying (turn will do nothing).
            sawEmptyStock = true;
            talonMovedThisPass = false;
            return "turn";
        }
    }

    private String foundationPriority(Solitaire solitaire) {
        List<List<Card>> tableau = solitaire.getTableau();
        List<Integer> faceUpCounts = solitaire.getTableauFaceUpCounts();
        List<List<Card>> foundations = solitaire.getFoundation();
        // Tableau to foundation
        for (int t = tableau.size() - 1; t >= 0; t--) {
            // Only consider the top/active visible card for foundation moves.
            Card moving = topTableauCard(tableau.get(t), faceUpCounts.get(t));
            if (moving == null) {
                continue;
            }
            for (int f = 0; f < foundations.size(); f++) {
                if (canMoveToFoundation(moving, foundations.get(f))) {
                    return "move T" + (t + 1) + " F" + (f + 1);
                }
            }
        }
        // Talon to foundation
        Card moving = top(solitaire.getTalon());
        if (moving != null) {
            for (int f = 0; f < foundations.size(); f++) {
                if (canMoveToFoundation(moving, foundations.get(f))) {
                    talonMovedThisPass = true;
                    return "move W F" + (f + 1);
                }
            }
        }
        return null;
    }

    private String tableauScan(Solitaire solitaire) {
        List<List<Card>> tableau = solitaire.getTableau();
        List<List<Card>> foundations = solitaire.getFoundation();
        List<Integer> faceUpCounts = solitaire.getTableauFaceUpCounts();
        // T7 down to T1
        for (int from = tableau.size() - 1; from >= 0; from--) {
            Card moving = bottomVisibleCard(tableau.get(from), faceUpCounts.get(from));
            if (moving == null) {
                continue;
            }
            // Try foundation first
            for (int f = 0; f < foundations.size(); f++) {
                if (canMoveToFoundation(moving, foundations.get(f))) {
                    return "move T" + (from + 1) + " " + moving.shortName() + " F" + (f + 1);
                }
            }
            // Then try tableau targets
            for (int to = tableau.size() - 1; to >= 0; to--) {
                if (to == from) {
                    continue;
                }
                // Skip ping-ponging a lone king: only consider kings if the source pile has more than one card (would reveal something).
                if (moving.getRank() == Rank.KING && tableau.get(from).size() == faceUpCounts.get(from)) {
                    continue;
                }
                if (canMoveToTableau(moving, tableau.get(to))) {
                    return "move T" + (from + 1) + " " + moving.shortName() + " T" + (to + 1);
                }
            }
        }
        return null;
    }

    private String talonMove(Solitaire solitaire) {
        Card moving = top(solitaire.getTalon());
        if (moving == null) {
            return null;
        }
        // Try foundation
        List<List<Card>> foundations = solitaire.getFoundation();
        for (int f = 0; f < foundations.size(); f++) {
            if (canMoveToFoundation(moving, foundations.get(f))) {
                    return "move W " + moving.shortName() + " F" + (f + 1);
            }
        }
        // Try tableau T7 -> T1
        List<List<Card>> tableau = solitaire.getTableau();
        for (int t = tableau.size() - 1; t >= 0; t--) {
            if (canMoveToTableau(moving, tableau.get(t))) {
                return "move W " + moving.shortName() + " T" + (t + 1);
            }
        }
        return null;
    }

    private Card bottomVisibleCard(List<Card> pile, int faceUpCount) {
        if (pile == null || pile.isEmpty() || faceUpCount <= 0) {
            return null;
        }
        int start = Math.max(0, pile.size() - faceUpCount);
        return pile.get(start);
    }
}
