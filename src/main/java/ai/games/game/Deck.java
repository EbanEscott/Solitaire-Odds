package ai.games.game;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Deck {
    private final List<Card> cards = new ArrayList<>();

    public Deck() {
        reset();
    }

    public void shuffle() {
        Collections.shuffle(cards);
    }

    public Card draw() {
        if (cards.isEmpty()) {
            return null;
        }
        return cards.remove(cards.size() - 1);
    }

    public int size() {
        return cards.size();
    }

    public boolean isEmpty() {
        return cards.isEmpty();
    }

    public List<Card> asUnmodifiableList() {
        return Collections.unmodifiableList(cards);
    }

    public final void reset() {
        cards.clear();
        for (Suit suit : Suit.values()) {
            for (Rank rank : Rank.values()) {
                cards.add(new Card(rank, suit));
            }
        }
        shuffle(); // Shuffle by default after creating the full deck.
    }

    @Override
    public String toString() {
        return "Deck(size=" + cards.size() + ")";
    }
}
