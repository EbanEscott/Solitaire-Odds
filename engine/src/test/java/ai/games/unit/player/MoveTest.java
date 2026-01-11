package ai.games.unit.player;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Rank;
import ai.games.game.Suit;
import ai.games.player.ai.tree.Move;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("Move")
class MoveTest {

    @Nested
    @DisplayName("Parsing")
    class ParsingTests {

        @Test
        void tryParseNullReturnsNull() {
            assertNull(Move.tryParse(null));
        }

        @Test
        void tryParseBlankReturnsNull() {
            assertNull(Move.tryParse("   ")); 
        }

        @Test
        void parsesQuit() {
            Move m = Move.parse("quit");
            assertTrue(m.isQuit());
            assertEquals(Move.Type.QUIT, m.type());
            assertEquals("quit", m.toCommandString());
            assertEquals(Move.quit(), m);
        }

        @Test
        void parsesTurnAndNormalises() {
            assertEquals(Move.turn(), Move.parse("turn"));
            assertEquals(Move.turn(), Move.parse("TURN"));
            assertEquals(Move.turn(), Move.parse("turn 3"));
            assertEquals("turn", Move.parse("turn 3").toCommandString());
        }

        @Test
        void parsesMoveWithCardTokenUnicodeSuit() {
            Move m = Move.parse("move T6 A♠ F2");
            assertTrue(m.isMove());
            assertEquals(Move.Type.MOVE, m.type());
            assertNotNull(m.from());
            assertNotNull(m.to());
            assertNotNull(m.card());

            assertEquals(Move.PileType.TABLEAU, m.from().type());
            assertEquals(5, m.from().index());

            assertEquals(Move.PileType.FOUNDATION, m.to().type());
            assertEquals(1, m.to().index());

            assertEquals(Rank.ACE, m.card().rank());
            assertEquals(Suit.SPADES, m.card().suit());

            assertEquals("move T6 A♠ F2", m.toCommandString());
        }

        @Test
        void parsesMoveWithCardTokenLetterSuitAndNormalisesToUnicode() {
            Move m = Move.parse("move t6 as f2");
            assertTrue(m.isMove());
            assertEquals(Rank.ACE, m.card().rank());
            assertEquals(Suit.SPADES, m.card().suit());
            assertEquals("move T6 A♠ F2", m.toCommandString());
        }

        @Test
        void parsesMoveWithoutCardToken() {
            Move m = Move.parse("move W F1");
            assertTrue(m.isMove());
            assertNotNull(m.from());
            assertNotNull(m.to());
            assertNull(m.card());
            assertEquals(Move.PileType.WASTE, m.from().type());
            assertEquals(-1, m.from().index());
            assertEquals(Move.PileType.FOUNDATION, m.to().type());
            assertEquals(0, m.to().index());
            assertEquals("move W F1", m.toCommandString());
        }

        @Test
        void parseRejectsUnknownCommand() {
            assertThrows(IllegalArgumentException.class, () -> Move.parse("hello"));
        }

        @Test
        void parseRejectsMalformedMoveCommand() {
            assertThrows(IllegalArgumentException.class, () -> Move.parse("move"));
            assertThrows(IllegalArgumentException.class, () -> Move.parse("move T1"));
            assertThrows(IllegalArgumentException.class, () -> Move.parse("move T1 X"));
            assertThrows(IllegalArgumentException.class, () -> Move.parse("move T1 A♠"));
        }
    }

    @Nested
    @DisplayName("Equality and keys")
    class EqualityAndKeysTests {

        @Test
        void equalityIsWhitespaceAndCaseInsensitive() {
            Move a = Move.parse("move T6 A♠ F2");
            Move b = Move.parse("  MOVE   t6   a♠   f2  ");
            assertEquals(a, b);
            assertEquals(a.hashCode(), b.hashCode());
            assertEquals(a.key(), b.key());
        }

        @Test
        void differentMovesHaveDifferentKeys() {
            Move a = Move.parse("move T6 A♠ F2");
            Move b = Move.parse("move T6 A♠ F3");
            assertNotEquals(a, b);
            assertNotEquals(a.key(), b.key());
        }

        @Test
        void explicitCardVsImplicitCardAreDifferentMoves() {
            // Even if engine typically emits 3-token waste moves, the parser supports 4-token forms.
            Move implicit = Move.parse("move W F1");
            Move explicit = Move.parse("move W A♠ F1");
            assertNotEquals(implicit, explicit);
            assertNotEquals(implicit.key(), explicit.key());
        }
    }

    @Nested
    @DisplayName("Value constraints")
    class ValueConstraintTests {

        @Test
        void pileRefRejectsInvalidIndices() {
            assertThrows(IllegalArgumentException.class, () -> new Move.PileRef(Move.PileType.TABLEAU, 7));
            assertThrows(IllegalArgumentException.class, () -> new Move.PileRef(Move.PileType.FOUNDATION, 4));
            assertThrows(IllegalArgumentException.class, () -> new Move.PileRef(Move.PileType.WASTE, 0));
            assertThrows(IllegalArgumentException.class, () -> new Move.PileRef(Move.PileType.STOCK, 1));
        }

        @Test
        void cardRefShortNameUnknownIsQuestionMark() {
            Move.CardRef unknown = new Move.CardRef(Rank.UNKNOWN, Suit.UNKNOWN);
            assertEquals("?", unknown.shortName());
        }

        @Test
        void cardRefShortNameNormalises() {
            Move.CardRef tenDiamonds = new Move.CardRef(Rank.TEN, Suit.DIAMONDS);
            assertEquals("10♦", tenDiamonds.shortName());
        }

        @Test
        void moveFactoryAcceptsNullCardForThreeTokenMoves() {
            Move m = Move.move(new Move.PileRef(Move.PileType.WASTE, -1), null, new Move.PileRef(Move.PileType.FOUNDATION, 0));
            assertTrue(m.isMove());
            assertNull(m.card());
            assertEquals("move W F1", m.toCommandString());
        }

        @Test
        void moveFactoryRejectsNullFromOrTo() {
            assertThrows(NullPointerException.class, () -> Move.move(null, null, new Move.PileRef(Move.PileType.FOUNDATION, 0)));
            assertThrows(NullPointerException.class, () -> Move.move(new Move.PileRef(Move.PileType.WASTE, -1), null, null));
        }

        @Test
        void tryParseCardTokenUnknownIsAccepted() {
            Move m = Move.parse("move T1 ? F1");
            assertNotNull(m.card());
            assertEquals(Rank.UNKNOWN, m.card().rank());
            assertEquals(Suit.UNKNOWN, m.card().suit());
        }

        @Test
        void tryParseRejectsCardWithoutSuit() {
            assertFalse(Move.tryParse("move T1 A F1") != null);
        }
    }
}
