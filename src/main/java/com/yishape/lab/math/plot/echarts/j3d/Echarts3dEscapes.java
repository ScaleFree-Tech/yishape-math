package com.yishape.lab.math.plot.echarts.j3d;

/**
 * HTML / JS 字面量里的转义，避免重复实现。
 */
final class Echarts3dEscapes {

    private Echarts3dEscapes() {
    }

    static String html(String s) {
        if (s == null) return "";
        return s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;");
    }

    static String jsSingleQuoted(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r");
    }
}
