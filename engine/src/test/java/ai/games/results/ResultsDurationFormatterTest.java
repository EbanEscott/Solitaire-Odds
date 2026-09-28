package ai.games.results;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ResultsDurationFormatterTest {

    @Test
    void formatsDurationsUsingReadableAdaptiveUnits() {
        assertEquals("37ms", ResultsDurationFormatter.formatSeconds(0.037));
        assertEquals("4.009s", ResultsDurationFormatter.formatSeconds(4.009));
        assertEquals("7m 55s", ResultsDurationFormatter.formatSeconds(474.557));
        assertEquals("13h 11m", ResultsDurationFormatter.formatSeconds(47_455.715));
        assertEquals("1d 44m", ResultsDurationFormatter.formatSeconds(89_063.176));
        assertEquals("1d 3h 38m", ResultsDurationFormatter.formatSeconds(99_462.347));
    }

    @Test
    void rejectsInvalidDurations() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ResultsDurationFormatter.formatSeconds(-1.0));
        assertThrows(
                IllegalArgumentException.class,
                () -> ResultsDurationFormatter.formatSeconds(Double.NaN));
    }
}
