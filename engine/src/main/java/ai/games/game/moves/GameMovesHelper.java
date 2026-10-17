package ai.games.game.moves;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import java.util.ArrayList;
import java.util.List;

/**
 * Computes legal moves for GAME mode (real game with known cards).
 * <p>
 * In GAME mode, all cards are known (no UNKNOWN placeholders), so the move generation
 * logic is straightforward: standard Solitaire rules apply without special handling
 * for ambiguous or hidden cards.
 * <p>
 * This class contains the original logic from {@link ai.games.player.LegalMovesHelper}
 * with shared board reads for each generation call.
 */
public class GameMovesHelper extends MovesHelper {

    @Override
    public List<Move> listMoves(Solitaire solitaire) {
        if (solitaire == null) {
            return new ArrayList<>();
        }
        
        this.solitaire = solitaire;
        List<Move> moves = new ArrayList<>();
        
        // These lists describe the same board throughout this generation call.
        // Read them once so the move categories share snapshots and views.
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        List<List<Card>> foundations = solitaire.getFoundation();
        List<Card> talon = solitaire.getTalon();

        addTableauToFoundation(moves, tableau, faceUps, foundations);
        addTableauToTableau(moves, tableau, faceUps);
        addTalonToFoundation(moves, talon, foundations);
        addTalonToTableau(moves, talon);
        addFoundationToTableau(moves, tableau, foundations);
        
        // Stock turns must respect the remaining pass allowance.
        if (solitaire.canTurnStock()) {
            moves.add(Move.turn());
        }
        
        // "quit" is always a legal command from the engine's perspective.
        moves.add(Move.quit());
        
        return moves;
    }

    @Override
    protected void addTableauToFoundation(List<Move> out, List<List<Card>> tableau,
            List<Integer> faceUps, List<List<Card>> foundations) {
        // Only top face-up cards can move to foundation.
        for (int t = 0; t < tableau.size(); t++) {
            List<Card> pile = tableau.get(t);
            int faceUp = faceUps.get(t);
            if (pile.isEmpty() || faceUp <= 0) {
                continue;
            }
            Card top = pile.get(pile.size() - 1);
            for (int f = 0; f < foundations.size(); f++) {
                if (canPlaceOnFoundation(top, foundations.get(f))) {
                    out.add(Move.move(tableauPile(t), cardRef(top), foundationPile(f)));
                }
            }
        }
    }

    @Override
    protected void addTableauToTableau(List<Move> out, List<List<Card>> tableau, List<Integer> faceUps) {
        TableauDestinations destinations = new TableauDestinations(tableau);
        // Any visible card can move as the start of a stack to another tableau pile.
        for (int from = 0; from < tableau.size(); from++) {
            List<Card> fromPile = tableau.get(from);
            int faceUp = faceUps.get(from);
            if (fromPile.isEmpty() || faceUp <= 0) {
                continue;
            }
            int start = Math.max(0, fromPile.size() - faceUp);
            for (int idx = start; idx < fromPile.size(); idx++) {
                Card moving = fromPile.get(idx);
                if (moving == null) {
                    continue;
                }
                int movingRank = moving.getRank().getValue();
                boolean movingIsRed = moving.getSuit().isRed();
                for (int to = 0; to < tableau.size(); to++) {
                    if (to == from) {
                        continue;
                    }
                    if (destinations.accepts(to, movingRank, movingIsRed)) {
                        out.add(Move.move(tableauPile(from), cardRef(moving), tableauPile(to)));
                    }
                }
            }
        }
    }

    @Override
    protected void addTalonToFoundation(List<Move> out, List<Card> talon, List<List<Card>> foundations) {
        if (talon.isEmpty()) {
            return;
        }
        // Only top of talon is playable.
        Card top = talon.get(talon.size() - 1);
        for (int f = 0; f < foundations.size(); f++) {
            if (canPlaceOnFoundation(top, foundations.get(f))) {
                out.add(Move.move(WASTE, null, foundationPile(f)));
            }
        }
    }

    @Override
    protected void addTalonToTableau(List<Move> out, List<Card> talon) {
        if (talon.isEmpty()) {
            return;
        }
        // Only top of talon is playable.
        Card top = talon.get(talon.size() - 1);
        List<List<Card>> tableau = solitaire.getTableau();
        for (int t = 0; t < tableau.size(); t++) {
            if (canPlaceOnTableau(top, tableau.get(t))) {
                out.add(Move.move(WASTE, null, tableauPile(t)));
            }
        }
    }

    @Override
    protected void addFoundationToTableau(List<Move> out, List<List<Card>> tableau, List<List<Card>> foundations) {
        // Foundations can only move their top card back to tableau.
        for (int f = 0; f < foundations.size(); f++) {
            List<Card> pile = foundations.get(f);
            if (pile.isEmpty()) {
                continue;
            }
            Card top = pile.get(pile.size() - 1);
            // Do not suggest moving Aces or Twos down from foundations; that is almost
            // always strategically bad and creates noisy moves for the AI.
            if (top.getRank() == Rank.ACE || top.getRank() == Rank.TWO) {
                continue;
            }
            for (int t = 0; t < tableau.size(); t++) {
                if (canPlaceOnTableau(top, tableau.get(t))) {
                    out.add(Move.move(foundationPile(f), null, tableauPile(t)));
                }
            }
        }
    }
}
