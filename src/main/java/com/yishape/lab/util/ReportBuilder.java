package com.yishape.lab.util;

import java.util.Locale;

/**
 * 报表构建器 / Report Builder
 * <p>
 * 提供统一的格式化输出能力，包括对齐的键值对、表格、分隔线等。
 * Provides unified formatting output capabilities including aligned key-value pairs, tables, separators, etc.
 * </p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * String report = new ReportBuilder("My Model")
 *     .kv("Accuracy", 0.9523)
 *     .kv("Converged", true)
 *     .h2("Details")
 *     .tableHeader("Class", "Precision", "Recall")
 *     .tableRow("A", "0.95", "0.93")
 *     .build();
 * }</pre>
 *
 * @author RereMouse
 * @since 1.0
 */
public class ReportBuilder {

    private static final int DEFAULT_KEY_WIDTH = 24;
    private static final int DEFAULT_COL_WIDTH = 12;
    private static final int RULE_WIDTH = 60;

    private final StringBuilder sb;

    public ReportBuilder() {
        this.sb = new StringBuilder();
    }

    public ReportBuilder(String title) {
        this.sb = new StringBuilder();
        h1(title);
    }

    // ==================== 分隔线与空白 / Rules and spacing ====================

    public ReportBuilder rule() {
        sb.append("=".repeat(RULE_WIDTH)).append('\n');
        return this;
    }

    public ReportBuilder rule2() {
        sb.append("-".repeat(RULE_WIDTH)).append('\n');
        return this;
    }

    public ReportBuilder blank() {
        sb.append('\n');
        return this;
    }

    // ==================== 区块标题 / Section headers ====================

    public ReportBuilder h1(String title) {
        sb.append("=== ").append(title).append(" ===\n");
        return this;
    }

    public ReportBuilder h2(String title) {
        sb.append("--- ").append(title).append(" ---\n");
        return this;
    }

    // ==================== 键值对 / Key-value pairs ====================

    public ReportBuilder kv(String key, String value) {
        sb.append(String.format("%-" + DEFAULT_KEY_WIDTH + "s: %s\n", key, value != null ? value : "N/A"));
        return this;
    }

    public ReportBuilder kv(String key, int value) {
        sb.append(String.format("%-" + DEFAULT_KEY_WIDTH + "s: %d\n", key, value));
        return this;
    }

    public ReportBuilder kv(String key, long value) {
        sb.append(String.format("%-" + DEFAULT_KEY_WIDTH + "s: %d\n", key, value));
        return this;
    }

    public ReportBuilder kv(String key, double value) {
        sb.append(String.format("%-" + DEFAULT_KEY_WIDTH + "s: %.4f\n", key, value));
        return this;
    }

    public ReportBuilder kv(String key, boolean value) {
        sb.append(String.format("%-" + DEFAULT_KEY_WIDTH + "s: %s\n", key, value ? "Yes" : "No"));
        return this;
    }

    public ReportBuilder kvf(String key, String fmt, Object value) {
        String formatted;
        if (value instanceof Double || value instanceof Float) {
            formatted = String.format(Locale.ROOT, fmt, value);
        } else {
            formatted = String.format(fmt, value);
        }
        sb.append(String.format("%-" + DEFAULT_KEY_WIDTH + "s: %s\n", key, formatted));
        return this;
    }

    // ==================== 表格 / Table ====================

    public ReportBuilder tableHeader(String... columns) {
        for (int i = 0; i < columns.length; i++) {
            if (i > 0) sb.append("  ");
            sb.append(String.format("%-" + DEFAULT_COL_WIDTH + "s", columns[i]));
        }
        sb.append('\n');
        for (int i = 0; i < columns.length; i++) {
            if (i > 0) sb.append("  ");
            sb.append("-".repeat(DEFAULT_COL_WIDTH));
        }
        sb.append('\n');
        return this;
    }

    public ReportBuilder tableRow(String... values) {
        for (int i = 0; i < values.length; i++) {
            if (i > 0) sb.append("  ");
            sb.append(String.format("%-" + DEFAULT_COL_WIDTH + "s", values[i]));
        }
        sb.append('\n');
        return this;
    }

    public ReportBuilder tableSeparator(int nCols) {
        for (int i = 0; i < nCols; i++) {
            if (i > 0) sb.append("  ");
            sb.append("-".repeat(DEFAULT_COL_WIDTH));
        }
        sb.append('\n');
        return this;
    }

    // ==================== 自由文本 / Free text ====================

    public ReportBuilder append(String text) {
        sb.append(text);
        return this;
    }

    public ReportBuilder appendf(String fmt, Object... args) {
        sb.append(String.format(fmt, args));
        return this;
    }

    // ==================== 输出 / Output ====================

    public String build() {
        return sb.toString();
    }

    // ==================== 静态格式化工具 / Static formatting utilities ====================

    public static String pct(double value) {
        return String.format("%.2f%%", value * 100.0);
    }

    public static String num(double value) {
        return String.format("%.4f", value);
    }

    public static String sci(double value) {
        return String.format("%.6e", value);
    }

    public static String duration(long ms) {
        if (ms < 1000) {
            return ms + " ms";
        }
        if (ms < 60_000) {
            return String.format("%.2f s", ms / 1000.0);
        }
        long minutes = ms / 60_000;
        long seconds = (ms % 60_000) / 1000;
        return String.format("%d min %d s", minutes, seconds);
    }
}
