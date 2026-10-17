package ai.games.game;

import java.util.Objects;
import java.util.Properties;

/**
 * Immutable configuration for a single Klondike game.
 * <p>
 * Games capture these settings at construction and retain them in ordinary and planning
 * copies. Draw count, observation policy, and stock pass allowance are independent.
 * Visibility controls observation independently of waste-card playability.
 *
 * @param drawCount the number of cards drawn per stock turn, either 1 or 3
 * @param stockVisibility the observation policy for drawn cards; must not be null
 * @param maxStockPasses the total permitted traversals, either 1, 3, or null for unlimited
 * @param thoughtful whether all card identities and positions are known from the start
 */
public record GameRules(int drawCount, StockVisibility stockVisibility, Integer maxStockPasses, boolean thoughtful) {
    /**
     * Constructs standard Klondike rules with discovery through play.
     *
     * @param drawCount the number of cards drawn per turn, either 1 or 3
     * @param stockVisibility the observation policy for drawn cards
     * @param maxStockPasses the total permitted traversals, or null for unlimited
     */
    public GameRules(int drawCount, StockVisibility stockVisibility, Integer maxStockPasses) {
        this(drawCount, stockVisibility, maxStockPasses, false);
    }

    /** Defines which drawn card identities the configured observation policy exposes. */
    public enum StockVisibility {
        /** Only the final drawn card is newly observed; previously observed cards stay known. */
        topOnly,
        /** Every card in the drawn group is observed, while only the waste top is playable. */
        allDrawn
    }

    /** Default contract: draw three, observe only the waste top, and allow unlimited passes. */
    public static final GameRules DEFAULT = new GameRules(3, StockVisibility.topOnly, null);

    /**
     * Validates the independent settings before a game can use them.
     *
     * @param drawCount the number of cards drawn per turn, either 1 or 3
     * @param stockVisibility the configured observation policy
     * @param maxStockPasses the total permitted traversals, or null for unlimited
     * @param thoughtful whether every identity and position is initially known
     * @throws IllegalArgumentException if draw count or pass allowance is unsupported
     * @throws NullPointerException if stockVisibility is null
     */
    public GameRules {
        if (drawCount != 1 && drawCount != 3) {
            throw new IllegalArgumentException("drawCount must be 1 or 3");
        }
        Objects.requireNonNull(stockVisibility, "stockVisibility");
        if (maxStockPasses != null && maxStockPasses != 1 && maxStockPasses != 3) {
            throw new IllegalArgumentException("maxStockPasses must be 1, 3, or null");
        }
    }

    /**
     * Resolves the current JVM properties when creating a game.
     * <p>
     * Copies retain their source's rules rather than reading runtime configuration again.
     *
     * @return validated rules with defaults for omitted properties
     * @throws IllegalArgumentException if a configured value is invalid
     */
    public static GameRules configured() {
        return fromProperties(System.getProperties());
    }

    /**
     * Reads game.drawCount, game.stockVisibility, game.maxStockPasses, and game.thoughtful.
     * <p>
     * Missing properties use the default contract. The literal pass value {@code null}
     * means unlimited; unsupported or malformed values are rejected.
     *
     * @param properties the source properties; must not be null
     * @return validated rules independent of subsequent property changes
     * @throws IllegalArgumentException if a configured value is invalid
     * @throws NullPointerException if properties is null
     */
    public static GameRules fromProperties(Properties properties) {
        String thoughtful = properties.getProperty("game.thoughtful", "false");
        if (!thoughtful.equals("true") && !thoughtful.equals("false")) {
            throw new IllegalArgumentException("game.thoughtful must be true or false");
        }
        String passes = properties.getProperty("game.maxStockPasses", "null");
        return new GameRules(
                Integer.parseInt(properties.getProperty("game.drawCount", "3")),
                StockVisibility.valueOf(properties.getProperty("game.stockVisibility", "topOnly")),
                passes.equals("null") ? null : Integer.valueOf(passes),
                Boolean.parseBoolean(thoughtful));
    }
}
