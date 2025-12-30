package ai.games.training;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/**
 * Applies reverse move commands to Solitaire game states without rule validation.
 * Used for reconstructing endgame training positions from reverse move sequences.
 * 
 * <p>This bypasses {@link Solitaire#moveCard(String, String, String)} which enforces
 * game rules (e.g., foundation cards can't move to talon). For training data generation,
 * we need to construct arbitrary game states from reverse moves, which may not all be
 * rule-compliant intermediate states.</p>
 */
public final class ReverseMovesApplier {
    private static final Logger log = LoggerFactory.getLogger(ReverseMovesApplier.class);
    
    private ReverseMovesApplier() {
    }

    /**
     * Apply a reverse move command to game state by directly manipulating internal collections.
     * 
     * @param solitaire the game state to modify
     * @param move command string (e.g., "move F1 K♣ T1", "move T1 T2", "turn")
     * @return true if move was applied, false if parse/validation failed
     */
    public static boolean applyReverseMove(Solitaire solitaire, String move) {
        if (move == null || move.isBlank()) {
            return false;
        }
        
        String trimmed = move.trim();
        
        // Handle "turn" (reverse cycle: talon -> stockpile)
        if (trimmed.equalsIgnoreCase("turn")) {
            return applyReverseTurn(solitaire);
        }
        
        // Parse "move <from> <card> <to>" or "move <from> <to>"
        String[] parts = trimmed.split("\\s+");
        if (parts.length < 3 || !parts[0].equalsIgnoreCase("move")) {
            if (log.isDebugEnabled()) {
                log.debug("Invalid move format: {}", move);
            }
            return false;
        }
        
        String from = parts[1];
        String to;
        String cardName = null;
        
        if (parts.length == 4) {
            // "move <from> <card> <to>"
            cardName = parts[2];
            to = parts[3];
        } else if (parts.length == 3) {
            // "move <from> <to>"
            to = parts[2];
        } else {
            if (log.isDebugEnabled()) {
                log.debug("Invalid move format: {}", move);
            }
            return false;
        }
        
        boolean success = applyMove(solitaire, from, cardName, to);
        if (success && log.isDebugEnabled()) {
            log.debug("Applied reverse move: {}", move);
        }
        return success;
    }

    /**
     * Apply a reverse move by directly manipulating collections.
     * 
     * @param solitaire the game state
     * @param fromStr location (e.g., "F1", "T2", "W")
     * @param cardName card identifier (e.g., "K♣") or null if moving stack
     * @param toStr destination (e.g., "F1", "T2", "W")
     * @return true if move was applied
     */
    private static boolean applyMove(Solitaire solitaire, String fromStr, String cardName, String toStr) {
        try {
            // Parse locations
            Location from = parseLocation(fromStr);
            Location to = parseLocation(toStr);
            
            if (from == null || to == null) {
                if (log.isDebugEnabled()) {
                    log.debug("Invalid locations: from={}, to={}", fromStr, toStr);
                }
                return false;
            }
            
            // Extract card(s) from source
            List<Card> cardsToMove = extractCards(solitaire, from, cardName);
            if (cardsToMove.isEmpty()) {
                if (log.isDebugEnabled()) {
                    log.debug("No cards found at source: {} {}", fromStr, cardName);
                }
                return false;
            }
            
            if (log.isDebugEnabled()) {
                log.debug("Moving {} card(s) from {} to {}", cardsToMove.size(), fromStr, toStr);
            }
            
            // Add cards to destination
            addCardsToLocation(solitaire, to, cardsToMove, from);
            
            if (log.isDebugEnabled()) {
                int foundCount = 0;
                for (List<Card> pile : solitaire.getFoundation()) {
                    foundCount += pile.size();
                }
                log.debug("After move: {} cards on foundations", foundCount);
            }
            
            return true;
        } catch (Exception e) {
            if (log.isDebugEnabled()) {
                log.debug("Failed to apply move: from={}, card={}, to={}", fromStr, cardName, toStr, e);
            }
            return false;
        }
    }

    /**
     * Extract cards from a location. For tableau, extracts from the named card onwards.
     * For foundation/talon, extracts the top card.
     */
    private static List<Card> extractCards(Solitaire solitaire, Location from, String cardName) {
        List<Card> result = new ArrayList<>();
        
        try {
            switch (from.type) {
                case FOUNDATION:
                    List<List<Card>> foundationPiles = getInternalList(solitaire, "foundation");
                    List<Card> foundationPile = foundationPiles.get(from.index);
                    if (!foundationPile.isEmpty()) {
                        Card top = foundationPile.remove(foundationPile.size() - 1);
                        result.add(top);
                    }
                    break;
                    
                case TABLEAU:
                    List<List<Card>> tableauPiles = getInternalList(solitaire, "tableau");
                    List<Card> tableauPile = tableauPiles.get(from.index);
                    if (!tableauPile.isEmpty()) {
                        if (cardName != null) {
                            // Find card by name and extract from that point onwards
                            int foundIdx = -1;
                            for (int i = 0; i < tableauPile.size(); i++) {
                                if (tableauPile.get(i).shortName().equals(cardName)) {
                                    foundIdx = i;
                                    break;
                                }
                            }
                            if (foundIdx >= 0) {
                                // Extract from foundIdx to end (in correct order)
                                while (tableauPile.size() > foundIdx) {
                                    result.add(tableauPile.remove(foundIdx));
                                }
                            }
                        } else {
                            // Extract just the top card
                            result.add(tableauPile.remove(tableauPile.size() - 1));
                        }
                    }
                    break;
                    
                case TALON:
                    List<Card> talon = getInternalList(solitaire, "talon");
                    if (!talon.isEmpty()) {
                        Card top = talon.remove(talon.size() - 1);
                        result.add(top);
                    }
                    break;
                    
                default:
                    break;
            }
        } catch (Exception e) {
            if (log.isDebugEnabled()) {
                log.debug("Failed to extract cards from {}", from.type, e);
            }
        }
        
        return result;
    }
    
    /**
     * Get the internal mutable list from a Solitaire object by reflection.
     */
    @SuppressWarnings("unchecked")
    private static List getInternalList(Solitaire solitaire, String fieldName) throws Exception {
        Field field = Solitaire.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        return (List) field.get(solitaire);
    }

    /**
     * Add cards to a destination location, updating face-up counts for tableau.
     */
    private static void addCardsToLocation(Solitaire solitaire, Location to, List<Card> cards, Location from) {
        if (cards.isEmpty()) {
            return;
        }
        
        try {
            switch (to.type) {
                case FOUNDATION:
                    List<List<Card>> foundationPiles = getInternalList(solitaire, "foundation");
                    foundationPiles.get(to.index).addAll(cards);
                    break;
                    
                case TABLEAU:
                    List<List<Card>> tableauPiles = getInternalList(solitaire, "tableau");
                    List<Card> tableauPile = tableauPiles.get(to.index);
                    tableauPile.addAll(cards);
                    // Update face-up count: cards coming from anywhere are now face-up
                    List<Integer> faceUpCounts = getInternalList(solitaire, "tableauFaceUp");
                    faceUpCounts.set(to.index, tableauPile.size());
                    break;
                    
                case TALON:
                    List<Card> talon = getInternalList(solitaire, "talon");
                    talon.addAll(cards);
                    break;
                    
                default:
                    break;
            }
        } catch (Exception e) {
            if (log.isDebugEnabled()) {
                log.debug("Failed to add cards to {}", to.type, e);
            }
        }
    }

    /**
     * Apply a reverse "turn" operation: move cards from talon back to stockpile.
     * This reverses the forward turnThree() operation which moves up to 3 cards from stockpile to talon.
     */
    private static boolean applyReverseTurn(Solitaire solitaire) {
        try {
            // For training purposes, reverse a "turn" operation by moving cards from talon to stockpile
            List<Card> stockpile = getInternalList(solitaire, "stockpile");
            List<Card> talon = getInternalList(solitaire, "talon");
            
            if (!talon.isEmpty()) {
                stockpile.add(talon.remove(talon.size() - 1));
                return true;
            }
        } catch (Exception e) {
            if (log.isDebugEnabled()) {
                log.debug("Failed to apply reverse turn", e);
            }
        }
        
        return false;
    }

    /**
     * Parse a location string (e.g., "F1", "T2", "W") into a Location object.
     */
    private static Location parseLocation(String locStr) {
        if (locStr == null || locStr.isBlank()) {
            return null;
        }
        
        String upper = locStr.toUpperCase();
        
        if (upper.equals("W")) {
            return new Location(LocationType.TALON, -1);
        }
        
        if (upper.startsWith("F")) {
            try {
                int idx = Integer.parseInt(upper.substring(1)) - 1;
                if (idx >= 0 && idx < 4) {
                    return new Location(LocationType.FOUNDATION, idx);
                }
            } catch (NumberFormatException e) {
                // Fall through
            }
        }
        
        if (upper.startsWith("T")) {
            try {
                int idx = Integer.parseInt(upper.substring(1)) - 1;
                if (idx >= 0 && idx < 7) {
                    return new Location(LocationType.TABLEAU, idx);
                }
            } catch (NumberFormatException e) {
                // Fall through
            }
        }
        
        return null;
    }

    /**
     * Internal class representing a location in the game (foundation, tableau, or talon).
     */
    private static class Location {
        LocationType type;
        int index;  // -1 for talon
        
        Location(LocationType type, int index) {
            this.type = type;
            this.index = index;
        }
    }

    /**
     * Enum for location types.
     */
    private enum LocationType {
        FOUNDATION, TABLEAU, TALON
    }
}
