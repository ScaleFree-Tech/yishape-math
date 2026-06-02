package com.yishape.lab.math.plot;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 样式表达式解析器，支持类似matplotlib的样式字符串
 * 例如: "r-", "b--o", "g:^", "#FF0000-s"等
 * 
 * 支持的格式：
 * - 颜色: r,g,b,k,w,y,c,m 或 #RRGGBB 或颜色名称
 * - 线条样式: -, --, :, -.
 * - 标记: o,s,^,v,&lt;,&gt;,d,*,+,x,|,_ 
 * 
 * @author lteb2
 */
public class StyleExpression implements Serializable {
    
    // ========== 样式映射表 ==========
    
    // 线条样式映射
    private static final Map<String, String> LINE_STYLE_MAP = new HashMap<>();
    static {
        LINE_STYLE_MAP.put("-", "solid");      // 实线
        LINE_STYLE_MAP.put("--", "dashed");    // 虚线
        LINE_STYLE_MAP.put(":", "dotted");     // 点线
        LINE_STYLE_MAP.put("-.", "dashdot");   // 点划线
        LINE_STYLE_MAP.put("", "solid");       // 默认实线
    }
    
    // 标记样式映射
    private static final Map<String, String> MARKER_MAP = new HashMap<>();
    static {
        MARKER_MAP.put("o", "circle");         // 圆形
        MARKER_MAP.put("s", "rect");           // 正方形
        MARKER_MAP.put("^", "triangle");       // 上三角
        MARKER_MAP.put("v", "triangle");       // 下三角
        MARKER_MAP.put("<", "triangle");       // 左三角
        MARKER_MAP.put(">", "triangle");       // 右三角
        MARKER_MAP.put("d", "diamond");        // 菱形
        MARKER_MAP.put("*", "star");           // 星形
        MARKER_MAP.put("+", "plus");           // 加号
        MARKER_MAP.put("x", "cross");          // 叉号
        MARKER_MAP.put("|", "line");           // 竖线
        MARKER_MAP.put("_", "line");           // 横线
        MARKER_MAP.put(".", "circle");         // 小点（圆形）
        MARKER_MAP.put("", "circle");          // 默认圆形
    }
    
    // 正则表达式模式
    private static final Pattern STYLE_PATTERN = Pattern.compile(
        "^" +                                           // 开始
        "([rgbkwycm]|#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3}|[A-Za-z]+)?" + // 颜色部分（可选）
        "(-{1,2}|:|\\-\\.)?" +                         // 线条样式（可选）
        "([os^v<>d*+x|_\\.]?)" +                       // 标记（可选）
        "$"                                             // 结束
    );
    
    private static final Pattern EXTENDED_PATTERN = Pattern.compile(
        "^" +
        "([rgbkwycm]|C\\d+|#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3}|[A-Za-z]+)?" + // 扩展颜色
        "(-{1,2}|:|\\-\\.)?" +                         // 线条样式
        "([os^v<>d*+x|_\\.]?)" +                       // 标记
        "$"
    );
    
    /**
     * 解析样式字符串并返回PlotStyle对象
     * @param styleString 样式字符串，如 "r-", "b--o", "g:^"
     * @return 解析后的PlotStyle对象
     */
    public static PlotStyle parse(String styleString) {
        if (styleString == null || styleString.trim().isEmpty()) {
            return PlotStyle.defaultStyle();
        }
        
        String trimmed = styleString.trim();
        
        // 尝试使用扩展模式匹配
        Matcher matcher = EXTENDED_PATTERN.matcher(trimmed);
        if (!matcher.matches()) {
            // 如果扩展模式不匹配，尝试基本模式
            matcher = STYLE_PATTERN.matcher(trimmed);
            if (!matcher.matches()) {
                // 如果都不匹配，返回默认样式
                return PlotStyle.defaultStyle();
            }
        }
        
        PlotStyle style = new PlotStyle();
        
        // 解析颜色
        String colorPart = matcher.group(1);
        if (colorPart != null && !colorPart.isEmpty()) {
            String color = ColorPalette.parseColor(colorPart);
            style.setColor(color);
            style.setFaceColor(color);
            style.setMarkerColor(color);
        }
        
        // 解析线条样式
        String linePart = matcher.group(2);
        if (linePart != null && !linePart.isEmpty()) {
            String lineStyle = LINE_STYLE_MAP.getOrDefault(linePart, "solid");
            style.setLineStyle(lineStyle);
        }
        
        // 解析标记
        String markerPart = matcher.group(3);
        if (markerPart != null && !markerPart.isEmpty()) {
            String marker = MARKER_MAP.getOrDefault(markerPart, "circle");
            style.setMarker(marker);
            
            // 根据标记类型设置特殊样式
            if (markerPart.equals(".")) {
                style.setMarkerSize(2.0f); // 小点
            } else if (markerPart.equals("*")) {
                style.setMarkerSize(8.0f); // 星形稍大
            }
        } else {
            // 当没有指定标记时，使用默认的circle标记
            style.setMarker("circle");
        }
        
        return style;
    }
    
    /**
     * 解析复合样式字符串（支持多个样式组合）
     * 例如: "color='red', linestyle='--', marker='o'"
     * @param styleString 复合样式字符串
     * @return 解析后的PlotStyle对象
     */
    public static PlotStyle parseComplex(String styleString) {
        if (styleString == null || styleString.trim().isEmpty()) {
            return PlotStyle.defaultStyle();
        }
        
        PlotStyle style = new PlotStyle();
        
        // 移除空白字符
        String cleaned = styleString.replaceAll("\\s+", "");
        
        // 分割参数
        String[] parts = cleaned.split(",");
        
        for (String part : parts) {
            if (part.contains("=")) {
                String[] keyValue = part.split("=", 2);
                if (keyValue.length == 2) {
                    String key = keyValue[0].toLowerCase();
                    String value = keyValue[1].replaceAll("['\"]", ""); // 移除引号
                    
                    applyStyleProperty(style, key, value);
                }
            } else {
                // 尝试作为简单样式字符串解析
                PlotStyle parsedStyle = parse(part);
                mergeStyles(style, parsedStyle);
            }
        }
        
        return style;
    }
    
    /**
     * 解析高级样式字符串（支持更多配置选项）
     * 例如: "color='red', linewidth=2, shadow=true, emphasis='highlight'"
     * @param styleString 高级样式字符串
     * @return 解析后的PlotStyle对象
     */
    public static PlotStyle parseAdvanced(String styleString) {
        if (styleString == null || styleString.trim().isEmpty()) {
            return PlotStyle.defaultStyle();
        }
        
        PlotStyle style = new PlotStyle();
        
        // 移除空白字符
        String cleaned = styleString.replaceAll("\\s+", "");
        
        // 分割参数
        String[] parts = cleaned.split(",");
        
        for (String part : parts) {
            if (part.contains("=")) {
                String[] keyValue = part.split("=", 2);
                if (keyValue.length == 2) {
                    String key = keyValue[0].toLowerCase();
                    String value = keyValue[1].replaceAll("['\"]", ""); // 移除引号
                    
                    applyAdvancedStyleProperty(style, key, value);
                }
            } else {
                // 尝试作为简单样式字符串解析
                PlotStyle parsedStyle = parse(part);
                mergeStyles(style, parsedStyle);
            }
        }
        
        return style;
    }
    
    /**
     * 解析主题样式字符串
     * 例如: "theme='dark', palette='viridis', animation='bounce'"
     * @param styleString 主题样式字符串
     * @return 解析后的PlotStyle对象
     */
    public static PlotStyle parseTheme(String styleString) {
        if (styleString == null || styleString.trim().isEmpty()) {
            return PlotStyle.defaultStyle();
        }
        
        PlotStyle style = new PlotStyle();
        
        // 移除空白字符
        String cleaned = styleString.replaceAll("\\s+", "");
        
        // 分割参数
        String[] parts = cleaned.split(",");
        
        for (String part : parts) {
            if (part.contains("=")) {
                String[] keyValue = part.split("=", 2);
                if (keyValue.length == 2) {
                    String key = keyValue[0].toLowerCase();
                    String value = keyValue[1].replaceAll("['\"]", ""); // 移除引号
                    
                    applyThemeProperty(style, key, value);
                }
            }
        }
        
        return style;
    }
    
    /**
     * 应用样式属性
     * @param style 样式对象
     * @param key 属性键
     * @param value 属性值
     */
    private static void applyStyleProperty(PlotStyle style, String key, String value) {
        switch (key) {
            case "color":
            case "c":
                style.setColor(ColorPalette.parseColor(value));
                break;
            case "facecolor":
            case "fc":
                style.setFaceColor(ColorPalette.parseColor(value));
                break;
            case "edgecolor":
            case "ec":
                style.setEdgeColor(ColorPalette.parseColor(value));
                break;
            case "alpha":
                try {
                    style.setAlpha(Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
                break;
            case "linestyle":
            case "ls":
                style.setLineStyle(LINE_STYLE_MAP.getOrDefault(value, value));
                break;
            case "linewidth":
            case "lw":
                try {
                    style.setLineWidth(Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
                break;
            case "marker":
                style.setMarker(MARKER_MAP.getOrDefault(value, value));
                break;
            case "markersize":
            case "ms":
                try {
                    style.setMarkerSize(Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
                break;
            case "markercolor":
            case "mc":
                style.setMarkerColor(ColorPalette.parseColor(value));
                break;
            case "label":
                style.setLabel(value);
                break;
            case "fontsize":
                try {
                    style.setFontSize(Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
                break;
            case "animation":
                style.setEnableAnimation(Boolean.parseBoolean(value));
                break;
            default:
                // 将未知属性添加到扩展属性中
                style.setProperty(key, value);
                break;
        }
    }
    
    /**
     * 应用高级样式属性
     * @param style 样式对象
     * @param key 属性键
     * @param value 属性值
     */
    private static void applyAdvancedStyleProperty(PlotStyle style, String key, String value) {
        switch (key) {
            case "color", "c" -> style.setColor(ColorPalette.parseColor(value));
            case "facecolor", "fc" -> style.setFaceColor(ColorPalette.parseColor(value));
            case "edgecolor", "ec" -> style.setEdgeColor(ColorPalette.parseColor(value));
            case "alpha" -> {
                try {
                    style.setAlpha(Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
            }
            case "linestyle", "ls" -> style.setLineStyle(LINE_STYLE_MAP.getOrDefault(value, value));
            case "linewidth", "lw" -> {
                try {
                    style.setLineWidth(Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
            }
            case "marker" -> style.setMarker(MARKER_MAP.getOrDefault(value, value));
            case "markersize", "ms" -> {
                try {
                    style.setMarkerSize(Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
            }
            case "markercolor", "mc" -> style.setMarkerColor(ColorPalette.parseColor(value));
            case "label" -> style.setLabel(value);
            case "fontsize" -> {
                try {
                    style.setFontSize(Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
            }
            case "animation" -> style.setEnableAnimation(Boolean.parseBoolean(value));
            case "shadow" -> {
                if (Boolean.parseBoolean(value)) {
                    style.setProperty("shadowBlur", 10);
                    style.setProperty("shadowColor", "rgba(0, 0, 0, 0.3)");
                }
            }
            case "shadowblur" -> {
                try {
                    style.setProperty("shadowBlur", Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
            }
            case "shadowcolor" -> style.setProperty("shadowColor", value);
            case "shadowoffsetx" -> {
                try {
                    style.setProperty("shadowOffsetX", Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
            }
            case "shadowoffsety" -> {
                try {
                    style.setProperty("shadowOffsetY", Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
            }
            case "emphasis" -> {
                if ("highlight".equalsIgnoreCase(value)) {
                    style.setProperty("emphasis", "highlight");
                }
            }
            case "gradient" -> {
                if (Boolean.parseBoolean(value)) {
                    style.setProperty("gradient", true);
                }
            }
            case "gradientstart" -> style.setProperty("gradientStart", value);
            case "gradientend" -> style.setProperty("gradientEnd", value);
            default -> // 将未知属性添加到扩展属性中
                style.setProperty(key, value);
        }
        // 高级属性
            }
    
    /**
     * 应用主题属性
     * @param style 样式对象
     * @param key 属性键
     * @param value 属性值
     */
    private static void applyThemeProperty(PlotStyle style, String key, String value) {
        switch (key) {
            case "theme":
                style.setProperty("theme", value);
                break;
            case "palette":
                style.setProperty("palette", value);
                break;
            case "animation":
                if ("bounce".equalsIgnoreCase(value)) {
                    style.setProperty("animationEasing", "bounceOut");
                } else if ("elastic".equalsIgnoreCase(value)) {
                    style.setProperty("animationEasing", "elasticOut");
                } else if ("back".equalsIgnoreCase(value)) {
                    style.setProperty("animationEasing", "backOut");
                } else {
                    style.setProperty("animationEasing", value);
                }
                break;
            case "background":
                style.setProperty("backgroundColor", value);
                break;
            case "textcolor":
                style.setProperty("textColor", value);
                break;
            case "fontfamily":
                style.setProperty("fontFamily", value);
                break;
            case "fontsize":
                try {
                    style.setProperty("fontSize", Float.parseFloat(value));
                } catch (NumberFormatException e) {
                    // 忽略无效值
                }
                break;
            default:
                // 将未知属性添加到扩展属性中
                style.setProperty(key, value);
                break;
        }
    }
    
    /**
     * 合并两个样式对象
     * @param target 目标样式（会被修改）
     * @param source 源样式
     */
    private static void mergeStyles(PlotStyle target, PlotStyle source) {
        if (source.getColor() != null && !source.getColor().equals(target.getColor())) {
            target.setColor(source.getColor());
        }
        if (source.getLineStyle() != null && !source.getLineStyle().equals(target.getLineStyle())) {
            target.setLineStyle(source.getLineStyle());
        }
        if (source.getMarker() != null && !source.getMarker().equals(target.getMarker())) {
            target.setMarker(source.getMarker());
        }
        if (source.getMarkerSize() != target.getMarkerSize()) {
            target.setMarkerSize(source.getMarkerSize());
        }
    }
    
    /**
     * 验证样式字符串是否有效
     * @param styleString 样式字符串
     * @return 是否有效
     */
    public static boolean isValidStyleString(String styleString) {
        if (styleString == null || styleString.trim().isEmpty()) {
            return true; // 空字符串是有效的（使用默认样式）
        }
        
        String trimmed = styleString.trim();
        return EXTENDED_PATTERN.matcher(trimmed).matches() || 
               STYLE_PATTERN.matcher(trimmed).matches();
    }
    
    /**
     * 获取样式字符串的描述
     * @param styleString 样式字符串
     * @return 样式描述
     */
    public static String getStyleDescription(String styleString) {
        PlotStyle style = parse(styleString);
        return String.format("颜色: %s, 线条: %s, 标记: %s", 
                style.getColor(), style.getLineStyle(), style.getMarker());
    }
    
    /**
     * 将PlotStyle对象转换为样式字符串（简化版）
     * @param style 样式对象
     * @return 样式字符串
     */
    public static String toStyleString(PlotStyle style) {
        StringBuilder sb = new StringBuilder();
        
        // 添加颜色
        String color = style.getColor();
        if (color != null) {
            // 尝试找到颜色的简写形式
            for (Map.Entry<String, String> entry : ColorPalette.COLOR_NAME_MAP.entrySet()) {
                if (entry.getValue().equalsIgnoreCase(color)) {
                    sb.append(entry.getKey());
                    break;
                }
            }
            if (sb.length() == 0) {
                sb.append(color); // 使用完整颜色值
            }
        }
        
        // 添加线条样式
        String lineStyle = style.getLineStyle();
        if (lineStyle != null) {
            for (Map.Entry<String, String> entry : LINE_STYLE_MAP.entrySet()) {
                if (entry.getValue().equals(lineStyle)) {
                    sb.append(entry.getKey());
                    break;
                }
            }
        }
        
        // 添加标记
        String marker = style.getMarker();
        if (marker != null) {
            for (Map.Entry<String, String> entry : MARKER_MAP.entrySet()) {
                if (entry.getValue().equals(marker)) {
                    sb.append(entry.getKey());
                    break;
                }
            }
        }
        
        return sb.toString();
    }
    
    // ========== 预定义样式常量 ==========
    
    /** 红色实线 */
    public static final String RED_SOLID = "r-";
    
    /** 蓝色虚线 */
    public static final String BLUE_DASHED = "b--";
    
    /** 绿色点线 */
    public static final String GREEN_DOTTED = "g:";
    
    /** 黑色点标记 */
    public static final String BLACK_DOTS = "k.";
    
    /** 红色圆圈标记 */
    public static final String RED_CIRCLES = "ro";
    
    /** 蓝色方形标记 */
    public static final String BLUE_SQUARES = "bs";
    
    /** 绿色三角标记 */
    public static final String GREEN_TRIANGLES = "g^";
    
    /**
     * 创建预定义样式
     * @return 包含预定义样式的映射
     */
    public static Map<String, PlotStyle> createPresetStyles() {
        Map<String, PlotStyle> presets = new HashMap<>();
        
        presets.put("red_line", parse(RED_SOLID));
        presets.put("blue_dashed", parse(BLUE_DASHED));
        presets.put("green_dotted", parse(GREEN_DOTTED));
        presets.put("black_dots", parse(BLACK_DOTS));
        presets.put("red_circles", parse(RED_CIRCLES));
        presets.put("blue_squares", parse(BLUE_SQUARES));
        presets.put("green_triangles", parse(GREEN_TRIANGLES));
        
        return presets;
    }
}