package com.yishape.lab.math.plot;

import java.util.List;

/**
 * Legend position constants and auto-placement logic (seaborn-style "best" detection).
 * 图例位置常量与 seaborn 风格自动放置逻辑。
 *
 * @author lteb2
 */
public final class LegendPositions {

    /** Inside plot, upper-right corner (seaborn default). */
    public static final String TOP_RIGHT = "top-right";
    /** Inside plot, upper-left corner. */
    public static final String TOP_LEFT = "top-left";
    /** Inside plot, lower-right corner. */
    public static final String BOTTOM_RIGHT = "bottom-right";
    /** Inside plot, lower-left corner. */
    public static final String BOTTOM_LEFT = "bottom-left";
    /** Outside plot, right side. */
    public static final String RIGHT = "right";
    /** Outside plot, left side. */
    public static final String LEFT = "left";
    /** Outside plot, above. */
    public static final String TOP = "top";
    /** Outside plot, below. */
    public static final String BOTTOM = "bottom";
    /** Auto-detect best position (same as no-arg {@code legend()}). */
    public static final String BEST = "best";

    private LegendPositions() {}

    /**
     * Auto-detect the best legend corner by picking the quadrant with the fewest data points.
     * Falls back to {@link #TOP_RIGHT} when no data is available.
     *
     * @param points list of {x, y} arrays (may be null or empty)
     * @param xMin  plot area x minimum
     * @param xMax  plot area x maximum
     * @param yMin  plot area y minimum
     * @param yMax  plot area y maximum
     * @return position string (one of the TOP_RIGHT / TOP_LEFT / BOTTOM_RIGHT / BOTTOM_LEFT)
     */
    public static String best(List<double[]> points, double xMin, double xMax, double yMin, double yMax) {
        if (points == null || points.isEmpty() || xMin >= xMax || yMin >= yMax) {
            return TOP_RIGHT;
        }
        double xMid = (xMin + xMax) / 2.0;
        double yMid = (yMin + yMax) / 2.0;
        int[] counts = new int[4]; // TR=0, TL=1, BR=2, BL=3
        int sampled = 0;
        int step = Math.max(1, points.size() / 1000);
        for (int i = 0; i < points.size(); i += step) {
            double[] p = points.get(i);
            if (p == null || p.length < 2) continue;
            double x = p[0], y = p[1];
            if (x < xMin || x > xMax || y < yMin || y > yMax) continue;
            boolean right = x >= xMid;
            boolean bottom = y >= yMid;
            if (right && !bottom) counts[0]++;       // top-right
            else if (!right && !bottom) counts[1]++; // top-left
            else if (right && bottom) counts[2]++;   // bottom-right
            else counts[3]++;                        // bottom-left
            sampled++;
        }
        if (sampled == 0) return TOP_RIGHT;
        int bestIdx = 0;
        for (int i = 1; i < 4; i++) {
            if (counts[i] < counts[bestIdx]) bestIdx = i;
        }
        return switch (bestIdx) {
            case 1 -> TOP_LEFT;
            case 2 -> BOTTOM_RIGHT;
            case 3 -> BOTTOM_LEFT;
            default -> TOP_RIGHT;
        };
    }

    /**
     * Validates a position string. Returns the input if valid, otherwise falls back to TOP_RIGHT.
     */
    public static String sanitize(String position) {
        if (position == null || position.isBlank()) return TOP_RIGHT;
        return switch (position.toLowerCase().trim()) {
            case TOP_RIGHT, TOP_LEFT, BOTTOM_RIGHT, BOTTOM_LEFT, RIGHT, LEFT, TOP, BOTTOM, BEST -> position.toLowerCase().trim();
            default -> TOP_RIGHT;
        };
    }
}
