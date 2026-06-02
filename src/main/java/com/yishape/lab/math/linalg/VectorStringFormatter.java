package com.yishape.lab.math.linalg;

import java.util.Locale;

/**
 * 向量 {@code toString} 的共享格式：方括号列表、逗号分隔，长向量省略中间并标注长度。
 */
final class VectorStringFormatter {

    /** 不超过该长度时打印全部元素 */
    private static final int MAX_FULL_PRINT = 16;
    /** 超长时保留前段元素个数 */
    private static final int HEAD = 6;
    /** 超长时保留末段元素个数 */
    private static final int TAIL = 4;

    private VectorStringFormatter() {
    }

    static String formatDoubles(double[] data) {
        if (data == null) {
            return "null";
        }
        int n = data.length;
        if (n == 0) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder(Math.min(n, MAX_FULL_PRINT) * 14 + 24);
        sb.append('[');
        if (n <= MAX_FULL_PRINT) {
            appendDoubles(sb, data, 0, n);
        } else {
            appendDoubles(sb, data, 0, HEAD);
            sb.append(", ..., ");
            appendDoubles(sb, data, n - TAIL, n);
            sb.append("] (length=").append(n).append(')');
            return sb.toString();
        }
        sb.append(']');
        return sb.toString();
    }

    static String formatFloats(float[] data) {
        if (data == null) {
            return "null";
        }
        int n = data.length;
        if (n == 0) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder(Math.min(n, MAX_FULL_PRINT) * 14 + 24);
        sb.append('[');
        if (n <= MAX_FULL_PRINT) {
            appendFloats(sb, data, 0, n);
        } else {
            appendFloats(sb, data, 0, HEAD);
            sb.append(", ..., ");
            appendFloats(sb, data, n - TAIL, n);
            sb.append("] (length=").append(n).append(')');
            return sb.toString();
        }
        sb.append(']');
        return sb.toString();
    }

    private static void appendDoubles(StringBuilder sb, double[] data, int from, int toExclusive) {
        for (int i = from; i < toExclusive; i++) {
            if (i > from) {
                sb.append(", ");
            }
            sb.append(formatDouble(data[i]));
        }
    }

    private static void appendFloats(StringBuilder sb, float[] data, int from, int toExclusive) {
        for (int i = from; i < toExclusive; i++) {
            if (i > from) {
                sb.append(", ");
            }
            sb.append(formatFloat(data[i]));
        }
    }

    static String formatDouble(double v) {
        if (Double.isNaN(v)) {
            return "NaN";
        }
        if (Double.isInfinite(v)) {
            return v > 0 ? "Infinity" : "-Infinity";
        }
        double av = Math.abs(v);
        if (av != 0.0 && (av < 1e-5 || av >= 1e8)) {
            return trimPlainNumber(String.format(Locale.US, "%.6e", v));
        }
        return trimPlainNumber(String.format(Locale.US, "%.6g", v));
    }

    static String formatFloat(float v) {
        if (Float.isNaN(v)) {
            return "NaN";
        }
        if (Float.isInfinite(v)) {
            return v > 0 ? "Infinity" : "-Infinity";
        }
        double av = Math.abs(v);
        if (av != 0.0f && (av < 1e-4f || av >= 1e6f)) {
            return trimPlainNumber(String.format(Locale.US, "%.5e", (double) v));
        }
        return trimPlainNumber(String.format(Locale.US, "%.6g", (double) v));
    }

    /**
     * 去掉小数点后多余 0；不改变科学计数法串（含 {@code e}/{@code E}）以免破坏指数。
     */
    private static String trimPlainNumber(String s) {
        if (s == null || s.isEmpty()) {
            return s;
        }
        if (s.indexOf('e') >= 0 || s.indexOf('E') >= 0) {
            return s;
        }
        if (!s.contains(".")) {
            return s;
        }
        int end = s.length();
        while (end > 0 && s.charAt(end - 1) == '0') {
            end--;
        }
        if (end > 0 && s.charAt(end - 1) == '.') {
            end--;
        }
        if (end == 0) {
            return "0";
        }
        String t = s.substring(0, end);
        if ("-0".equals(t)) {
            return "0";
        }
        return t;
    }
}
