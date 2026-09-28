package ai.games.results;

import java.util.Locale;

/** Formats benchmark durations for direct use in the README result tables. */
final class ResultsDurationFormatter {
    private static final double SECONDS_PER_MINUTE = 60.0;
    private static final double SECONDS_PER_HOUR = 3_600.0;
    private static final double SECONDS_PER_DAY = 86_400.0;

    private ResultsDurationFormatter() {
    }

    /** Uses progressively larger units while preserving useful precision for short benchmarks. */
    static String formatSeconds(double seconds) {
        if (!Double.isFinite(seconds) || seconds < 0.0) {
            throw new IllegalArgumentException("Duration must be a finite non-negative value");
        }
        if (seconds < 1.0) {
            return Math.round(seconds * 1_000.0) + "ms";
        }
        if (seconds < SECONDS_PER_MINUTE) {
            return String.format(Locale.ROOT, "%.3fs", seconds);
        }
        if (seconds < SECONDS_PER_HOUR) {
            long totalSeconds = Math.round(seconds);
            return totalSeconds / 60 + "m " + totalSeconds % 60 + "s";
        }
        long totalMinutes = Math.round(seconds / SECONDS_PER_MINUTE);
        if (seconds < SECONDS_PER_DAY) {
            long hours = totalMinutes / 60;
            long minutes = totalMinutes % 60;
            return minutes == 0 ? hours + "h" : hours + "h " + minutes + "m";
        }
        long days = totalMinutes / (24 * 60);
        long remainingMinutes = totalMinutes % (24 * 60);
        long hours = remainingMinutes / 60;
        long minutes = remainingMinutes % 60;
        StringBuilder formatted = new StringBuilder(days + "d");
        if (hours > 0) {
            formatted.append(' ').append(hours).append('h');
        }
        if (minutes > 0) {
            formatted.append(' ').append(minutes).append('m');
        }
        return formatted.toString();
    }
}
