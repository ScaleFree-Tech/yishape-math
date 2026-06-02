package com.yishape.lab.util;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON serializer/deserializer for model persistence.
 * No external dependencies. Handles only the types needed by toParams()/fromParams().
 */
public final class JsonUtil {

    private JsonUtil() {}

    // ==================== Serialization ====================

    public static String toJson(Map<String, Object> map) {
        StringBuilder sb = new StringBuilder();
        writeMap(sb, map, 0);
        return sb.toString();
    }

    private static void writeValue(StringBuilder sb, Object value, int indent) {
        if (value == null) {
            sb.append("null");
        } else if (value instanceof String s) {
            writeString(sb, s);
        } else if (value instanceof Boolean b) {
            sb.append(b);
        } else if (value instanceof Integer i) {
            sb.append(i);
        } else if (value instanceof Long l) {
            sb.append(l);
        } else if (value instanceof Double d) {
            if (Double.isNaN(d) || Double.isInfinite(d)) {
                sb.append("null");
            } else {
                sb.append(d);
            }
        } else if (value instanceof Float f) {
            if (Float.isNaN(f) || Float.isInfinite(f)) {
                sb.append("null");
            } else {
                sb.append(f);
            }
        } else if (value instanceof double[] arr) {
            writeDoubleArray(sb, arr);
        } else if (value instanceof int[] arr) {
            writeIntArray(sb, arr);
        } else if (value instanceof String[] arr) {
            writeStringArray(sb, arr);
        } else if (value instanceof double[][] mat) {
            writeDoubleMatrix(sb, mat);
        } else if (value instanceof Map<?, ?> m) {
            @SuppressWarnings("unchecked")
            Map<String, Object> sm = (Map<String, Object>) m;
            writeMap(sb, sm, indent);
        } else if (value instanceof List<?> list) {
            writeList(sb, list, indent);
        } else {
            writeString(sb, value.toString());
        }
    }

    private static void writeString(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> sb.append(c);
            }
        }
        sb.append('"');
    }

    private static void writeDoubleArray(StringBuilder sb, double[] arr) {
        sb.append('[');
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            if (Double.isNaN(arr[i]) || Double.isInfinite(arr[i])) {
                sb.append("null");
            } else {
                sb.append(arr[i]);
            }
        }
        sb.append(']');
    }

    private static void writeIntArray(StringBuilder sb, int[] arr) {
        sb.append('[');
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(arr[i]);
        }
        sb.append(']');
    }

    private static void writeStringArray(StringBuilder sb, String[] arr) {
        sb.append('[');
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            writeString(sb, arr[i]);
        }
        sb.append(']');
    }

    private static void writeDoubleMatrix(StringBuilder sb, double[][] mat) {
        sb.append("[\n");
        for (int i = 0; i < mat.length; i++) {
            if (i > 0) sb.append(",\n");
            sb.append("  ");
            writeDoubleArray(sb, mat[i]);
        }
        sb.append("\n]");
    }

    private static void writeList(StringBuilder sb, List<?> list, int indent) {
        sb.append('[');
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) sb.append(", ");
            writeValue(sb, list.get(i), indent);
        }
        sb.append(']');
    }

    @SuppressWarnings("unchecked")
    private static void writeMap(StringBuilder sb, Map<String, Object> map, int indent) {
        String baseIndent = "  ".repeat(indent);
        String innerIndent = "  ".repeat(indent + 1);
        sb.append("{\n");
        int i = 0;
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            if (i > 0) sb.append(",\n");
            sb.append(innerIndent).append('"').append(entry.getKey()).append("\": ");
            writeValue(sb, entry.getValue(), indent + 1);
            i++;
        }
        sb.append('\n').append(baseIndent).append('}');
    }

    // ==================== Deserialization ====================

    public static Map<String, Object> fromJson(String json) {
        Parser p = new Parser(json);
        Object result = p.parseValue();
        if (result instanceof Map<?, ?> m) {
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) m;
            return map;
        }
        throw new IllegalArgumentException("JSON root must be an object, got: " + (result == null ? "null" : result.getClass().getSimpleName()));
    }

    private static class Parser {
        private final String json;
        private int pos;

        Parser(String json) {
            this.json = json;
            this.pos = 0;
        }

        void skipWhitespace() {
            while (pos < json.length() && Character.isWhitespace(json.charAt(pos))) {
                pos++;
            }
        }

        char peek() {
            skipWhitespace();
            return pos < json.length() ? json.charAt(pos) : '\0';
        }

        char next() {
            skipWhitespace();
            return pos < json.length() ? json.charAt(pos++) : '\0';
        }

        Object parseValue() {
            skipWhitespace();
            if (pos >= json.length()) return null;
            char c = json.charAt(pos);
            return switch (c) {
                case '{' -> parseMap();
                case '[' -> parseArray();
                case '"' -> parseString();
                case 'n' -> { pos += 4; yield null; } // null
                case 't' -> { pos += 4; yield true; }  // true
                case 'f' -> { pos += 5; yield false; } // false
                default -> parseNumber();
            };
        }

        Map<String, Object> parseMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            pos++; // skip '{'
            skipWhitespace();
            if (peek() == '}') { pos++; return map; }
            while (true) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                pos++; // skip ':'
                Object value = parseValue();
                map.put(key, value);
                skipWhitespace();
                if (peek() == '}') { pos++; return map; }
                pos++; // skip ','
            }
        }

        Object parseArray() {
            pos++; // skip '['
            skipWhitespace();
            if (peek() == ']') { pos++; return new double[0]; }

            // Peek first element to decide array type
            List<Object> list = new ArrayList<>();
            while (true) {
                Object element = parseValue();
                list.add(element);
                skipWhitespace();
                if (peek() == ']') { pos++; break; }
                pos++; // skip ','
            }

            // Coerce to typed array if homogeneous
            if (list.isEmpty()) return new double[0];
            Object first = list.get(0);
            if (first instanceof List<?>) {
                // double[][]
                double[][] mat = new double[list.size()][];
                for (int i = 0; i < list.size(); i++) {
                    @SuppressWarnings("unchecked")
                    List<Object> row = (List<Object>) list.get(i);
                    mat[i] = doublesFromList(row);
                }
                return mat;
            }
            if (first instanceof String) {
                return list.toArray(new String[0]);
            }
            if (first instanceof Number) {
                return doublesFromList(list);
            }
            // fallback: return list as-is
            return list;
        }

        String parseString() {
            StringBuilder sb = new StringBuilder();
            pos++; // skip opening "
            while (pos < json.length()) {
                char c = json.charAt(pos++);
                if (c == '"') return sb.toString();
                if (c == '\\' && pos < json.length()) {
                    char esc = json.charAt(pos++);
                    switch (esc) {
                        case '"' -> sb.append('"');
                        case '\\' -> sb.append('\\');
                        case 'n' -> sb.append('\n');
                        case 'r' -> sb.append('\r');
                        case 't' -> sb.append('\t');
                        default -> { sb.append('\\'); sb.append(esc); }
                    }
                } else {
                    sb.append(c);
                }
            }
            return sb.toString();
        }

        Object parseNumber() {
            int start = pos;
            while (pos < json.length()) {
                char c = json.charAt(pos);
                if (Character.isDigit(c) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
                    pos++;
                } else {
                    break;
                }
            }
            String numStr = json.substring(start, pos);
            if (numStr.contains(".") || numStr.contains("e") || numStr.contains("E")) {
                return Double.parseDouble(numStr);
            }
            try {
                return Integer.parseInt(numStr);
            } catch (NumberFormatException e) {
                return Double.parseDouble(numStr);
            }
        }

        static double[] doublesFromList(List<Object> list) {
            double[] arr = new double[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object o = list.get(i);
                if (o instanceof Number n) {
                    arr[i] = n.doubleValue();
                }
            }
            return arr;
        }
    }
}
