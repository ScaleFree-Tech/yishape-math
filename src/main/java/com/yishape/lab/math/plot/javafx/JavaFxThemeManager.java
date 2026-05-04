package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.plot.ColorPalette;
import com.yishape.lab.math.plot.PlotStyle;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * JavaFX图表主题管理器
 * 提供多种主题和调色板，支持自定义主题
 * 
 * @author lteb2
 */
public class JavaFxThemeManager implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // ========== 内置主题常量 ==========
    public static final String THEME_DEFAULT = "default";
    public static final String THEME_LIGHT = "light";
    public static final String THEME_DARK = "dark";
    public static final String THEME_ACADEMIC = "academic";
    public static final String THEME_BUSINESS = "business";
    public static final String THEME_MINIMAL = "minimal";
    public static final String THEME_RAINBOW = "rainbow";
    public static final String THEME_VINTAGE = "vintage";
    public static final String THEME_FUTURISTIC = "futuristic";
    public static final String THEME_SEABORN = "seaborn";
    public static final String THEME_COLORBLIND = "colorblind";
    
    // ========== 主题注册表 ==========
    private static final Map<String, Theme> CUSTOM_THEMES = new HashMap<>();
    
    // ========== 主题配置 ==========
    private String currentTheme = THEME_DEFAULT;
    private Map<String, Object> themeConfig;
    
    public JavaFxThemeManager() {
        this.themeConfig = getBuiltinTheme(THEME_DEFAULT);
    }
    
    public JavaFxThemeManager(String theme) {
        setTheme(theme);
    }
    
    /**
     * 设置当前主题
     * @param themeName 主题名称
     */
    public void setTheme(String themeName) {
        this.currentTheme = themeName;
        if (CUSTOM_THEMES.containsKey(themeName)) {
            this.themeConfig = CUSTOM_THEMES.get(themeName).getConfig();
        } else {
            this.themeConfig = getBuiltinTheme(themeName);
        }
    }
    
    /**
     * 获取当前主题名称
     * @return 主题名称
     */
    public String getCurrentTheme() {
        return currentTheme;
    }
    
    /**
     * 获取当前主题配置
     * @return 主题配置
     */
    public Map<String, Object> getCurrentConfig() {
        return themeConfig;
    }
    
    /**
     * 获取背景色
     * @return 背景色
     */
    public Color getBackgroundColor() {
        if (themeConfig.containsKey("backgroundColor")) {
            return Color.web((String) themeConfig.get("backgroundColor"));
        }
        return Color.WHITE;
    }
    
    /**
     * 获取调色板
     * @return 颜色数组
     */
    public String[] getColorPalette() {
        if (themeConfig.containsKey("color")) {
            return (String[]) themeConfig.get("color");
        }
        return ColorPalette.PREMIUM_CHART_COLORS;
    }
    
    /**
     * 获取文本颜色
     * @return 文本颜色
     */
    public Color getTextColor() {
        if (themeConfig.containsKey("textColor")) {
            return Color.web((String) themeConfig.get("textColor"));
        }
        return Color.web("#0f172a");
    }

    /**
     * 副标题、坐标辅助文字等次要文本颜色
     */
    public Color getMutedTextColor() {
        if (themeConfig.containsKey("mutedTextColor")) {
            return Color.web((String) themeConfig.get("mutedTextColor"));
        }
        Color t = getTextColor();
        return new Color(t.getRed(), t.getGreen(), t.getBlue(), Math.min(1.0, t.getOpacity() * 0.68));
    }
    
    /**
     * 获取标题字体
     * @return 标题字体
     */
    public Font getTitleFont() {
        String fontFamily = "Arial";
        double fontSize = 18;
        if (themeConfig.containsKey("titleFontFamily")) {
            fontFamily = (String) themeConfig.get("titleFontFamily");
        }
        if (themeConfig.containsKey("titleFontSize")) {
            fontSize = ((Number) themeConfig.get("titleFontSize")).doubleValue();
        }
        return Font.font(fontFamily, FontWeight.BOLD, fontSize);
    }
    
    /**
     * 获取标签字体
     * @return 标签字体
     */
    public Font getLabelFont() {
        String fontFamily = "Arial";
        double fontSize = 12;
        if (themeConfig.containsKey("labelFontFamily")) {
            fontFamily = (String) themeConfig.get("labelFontFamily");
        }
        if (themeConfig.containsKey("labelFontSize")) {
            fontSize = ((Number) themeConfig.get("labelFontSize")).doubleValue();
        }
        return Font.font(fontFamily, fontSize);
    }
    
    /**
     * 应用主题到样式
     * @param style 原始样式
     * @return 应用主题后的样式
     */
    public PlotStyle applyThemeToStyle(PlotStyle style) {
        PlotStyle themedStyle = new PlotStyle(style);
        
        String[] colors = getColorPalette();
        if (colors.length > 0 && themedStyle.getColor().equals(PlotStyle.defaultStyle().getColor())) {
            themedStyle.setColor(colors[0]);
        }
        
        // 应用主题特定的样式调整
        applyThemeSpecificStyles(themedStyle, currentTheme);
        
        return themedStyle;
    }
    
    /**
     * 获取内置主题配置
     * @param themeName 主题名称
     * @return 主题配置
     */
    public static Map<String, Object> getBuiltinTheme(String themeName) {
        Map<String, Object> theme = new HashMap<>();
        
        switch (themeName.toLowerCase()) {
            case "dark":
                theme.put("backgroundColor", "#0f172a");
                theme.put("textColor", "#f1f5f9");
                theme.put("mutedTextColor", "#94a3b8");
                theme.put("color", new String[]{
                    "#60a5fa", "#34d399", "#fbbf24", "#fb7185", "#a78bfa",
                    "#22d3ee", "#f472b6", "#4ade80", "#fb923c", "#cbd5e1"
                });
                theme.put("titleFontFamily", "Segoe UI");
                theme.put("titleFontSize", 20);
                theme.put("labelFontFamily", "Segoe UI");
                theme.put("labelFontSize", 11);
                theme.put("gridColor", "#1e293b");
                theme.put("axisColor", "#475569");
                break;
                
            case "light":
                theme.put("backgroundColor", "#ffffff");
                theme.put("textColor", "#111827");
                theme.put("mutedTextColor", "#6b7280");
                theme.put("color", ColorPalette.PREMIUM_CHART_COLORS);
                theme.put("titleFontFamily", "Segoe UI");
                theme.put("titleFontSize", 20);
                theme.put("labelFontFamily", "Segoe UI");
                theme.put("labelFontSize", 11);
                theme.put("gridColor", "#f1f5f9");
                theme.put("axisColor", "#cbd5e1");
                break;
                
            case "academic":
                theme.put("backgroundColor", "#fafaf9");
                theme.put("textColor", "#1c1917");
                theme.put("mutedTextColor", "#78716c");
                theme.put("color", new String[]{
                    "#1e40af", "#047857", "#b45309", "#b91c1c", "#6d28d9",
                    "#0f766e", "#a16207", "#be185d", "#4338ca", "#57534e"
                });
                theme.put("titleFontFamily", "Georgia");
                theme.put("titleFontSize", 19);
                theme.put("labelFontFamily", "Georgia");
                theme.put("labelFontSize", 11);
                theme.put("gridColor", "#e7e5e4");
                theme.put("axisColor", "#a8a29e");
                theme.put("showGrid", true);
                break;
                
            case "business":
                theme.put("backgroundColor", "#f8fafc");
                theme.put("textColor", "#0f172a");
                theme.put("mutedTextColor", "#64748b");
                theme.put("color", new String[]{
                    "#0c4a6e", "#047857", "#b45309", "#a21caf", "#c2410c",
                    "#0369a1", "#15803d", "#7c2d12", "#5b21b6", "#9a3412"
                });
                theme.put("titleFontFamily", "Segoe UI");
                theme.put("titleFontSize", 20);
                theme.put("labelFontFamily", "Segoe UI");
                theme.put("labelFontSize", 11);
                theme.put("gridColor", "#e2e8f0");
                theme.put("axisColor", "#94a3b8");
                break;
                
            case "minimal":
                theme.put("backgroundColor", "#ffffff");
                theme.put("textColor", "#1f2937");
                theme.put("mutedTextColor", "#9ca3af");
                theme.put("color", new String[]{
                    "#374151", "#6b7280", "#2563eb", "#059669", "#d97706",
                    "#7c3aed", "#db2777", "#0d9488", "#4b5563", "#525252"
                });
                theme.put("titleFontFamily", "Segoe UI");
                theme.put("titleFontSize", 18);
                theme.put("labelFontFamily", "Segoe UI");
                theme.put("labelFontSize", 11);
                theme.put("showGrid", false);
                theme.put("axisColor", "#d1d5db");
                break;
                
            case "rainbow":
                theme.put("backgroundColor", "#fafafa");
                theme.put("textColor", "#262626");
                theme.put("mutedTextColor", "#737373");
                theme.put("color", new String[]{
                    "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
                    "#3b82f6", "#8b5cf6", "#ec4899", "#6366f1", "#06b6d4"
                });
                theme.put("titleFontFamily", "Segoe UI");
                theme.put("titleFontSize", 20);
                theme.put("labelFontFamily", "Segoe UI");
                theme.put("labelFontSize", 11);
                theme.put("gridColor", "#f5f5f5");
                theme.put("axisColor", "#d4d4d4");
                break;
                
            case "vintage":
                theme.put("backgroundColor", "#faf8f3");
                theme.put("textColor", "#3f372c");
                theme.put("mutedTextColor", "#7a6f63");
                theme.put("color", new String[]{
                    "#5c7c5c", "#c9a227", "#b8574e", "#4a5d78", "#8b7355",
                    "#6b9080", "#c17f59", "#7d6b91", "#a67c52", "#556270"
                });
                theme.put("titleFontFamily", "Georgia");
                theme.put("titleFontSize", 18);
                theme.put("labelFontFamily", "Georgia");
                theme.put("labelFontSize", 11);
                theme.put("gridColor", "#ebe4d7");
                theme.put("axisColor", "#c9b89a");
                break;
                
            case "futuristic":
                theme.put("backgroundColor", "#0b1020");
                theme.put("textColor", "#e2e8f0");
                theme.put("mutedTextColor", "#7ec8e3");
                theme.put("color", new String[]{
                    "#38bdf8", "#4ade80", "#c084fc", "#fb923c", "#f472b6",
                    "#22d3ee", "#a3e635", "#facc15", "#94a3b8", "#e11d48"
                });
                theme.put("titleFontFamily", "Consolas");
                theme.put("titleFontSize", 18);
                theme.put("labelFontFamily", "Consolas");
                theme.put("labelFontSize", 10);
                theme.put("gridColor", "#1e293b");
                theme.put("axisColor", "#475569");
                break;
                
            case "seaborn":
                theme.put("backgroundColor", "#f4f6f9");
                theme.put("textColor", "#1e293b");
                theme.put("mutedTextColor", "#64748b");
                theme.put("color", ColorPalette.SEABORN_MUTED);
                theme.put("titleFontFamily", "Segoe UI");
                theme.put("titleFontSize", 20);
                theme.put("labelFontFamily", "Segoe UI");
                theme.put("labelFontSize", 11);
                theme.put("gridColor", "#e2e8f0");
                theme.put("axisColor", "#cbd5e1");
                break;
                
            case "colorblind":
                theme.put("backgroundColor", "#ffffff");
                theme.put("textColor", "#0a0a0a");
                theme.put("mutedTextColor", "#525252");
                theme.put("color", ColorPalette.COLORBLIND_FRIENDLY);
                theme.put("titleFontFamily", "Segoe UI");
                theme.put("titleFontSize", 18);
                theme.put("labelFontFamily", "Segoe UI");
                theme.put("labelFontSize", 11);
                theme.put("gridColor", "#e5e5e5");
                theme.put("axisColor", "#a3a3a3");
                break;
                
            default: // default — 浅色报告风
                theme.put("backgroundColor", "#f8fafc");
                theme.put("textColor", "#0f172a");
                theme.put("mutedTextColor", "#64748b");
                theme.put("color", ColorPalette.PREMIUM_CHART_COLORS);
                theme.put("titleFontFamily", "Segoe UI");
                theme.put("titleFontSize", 20);
                theme.put("labelFontFamily", "Segoe UI");
                theme.put("labelFontSize", 11);
                theme.put("gridColor", "#e8edf3");
                theme.put("axisColor", "#94a3b8");
                break;
        }
        
        return theme;
    }
    
    /**
     * 应用主题特定样式
     * @param style 样式对象
     * @param themeName 主题名称
     */
    private void applyThemeSpecificStyles(PlotStyle style, String themeName) {
        switch (themeName.toLowerCase()) {
            case THEME_ACADEMIC:
                style.setLineWidth(1.5);
                style.setMarkerSize(4);
                break;
            case THEME_BUSINESS:
                style.setLineWidth(2.5);
                style.setMarkerSize(6);
                break;
            case THEME_MINIMAL:
                style.setLineWidth(1);
                style.setMarkerSize(3);
                style.setAlpha(0.9);
                break;
            case THEME_RAINBOW:
                style.setLineWidth(3);
                style.setMarkerSize(8);
                break;
            case THEME_VINTAGE:
                style.setLineWidth(2);
                style.setMarkerSize(5);
                style.setAlpha(0.85);
                break;
            case THEME_FUTURISTIC:
                style.setLineWidth(2);
                style.setMarkerSize(6);
                break;
        }
    }
    
    /**
     * 注册自定义主题
     * @param name 主题名称
     * @param theme 主题对象
     */
    public static void registerCustomTheme(String name, Theme theme) {
        CUSTOM_THEMES.put(name, theme);
    }

    /**
     * 内置主题名（与 {@link #getBuiltinTheme(String)} 中 {@code switch} 分支一致）。
     *
     * @return 内置主题标识列表
     */
    public static List<String> getBuiltinThemeNames() {
        List<String> names = new ArrayList<>();
        names.add(THEME_DEFAULT);
        names.add(THEME_LIGHT);
        names.add(THEME_DARK);
        names.add(THEME_ACADEMIC);
        names.add(THEME_BUSINESS);
        names.add(THEME_MINIMAL);
        names.add(THEME_RAINBOW);
        names.add(THEME_VINTAGE);
        names.add(THEME_FUTURISTIC);
        names.add(THEME_SEABORN);
        names.add(THEME_COLORBLIND);
        return names;
    }

    /**
     * 内置主题名以及通过 {@link #registerCustomTheme(String, Theme)} 注册的自定义主题名。
     *
     * @return 全部可用主题名（内置在前，注册的键紧随其后；不排除与内置重名的注册项）
     */
    public static List<String> getRegisteredThemeNames() {
        List<String> names = new ArrayList<>(getBuiltinThemeNames());
        names.addAll(CUSTOM_THEMES.keySet());
        return names;
    }
    
    /**
     * 创建渐变主题
     * @param name 主题名称
     * @param startColor 起始颜色
     * @param endColor 结束颜色
     * @param backgroundColor 背景颜色
     * @return 主题对象
     */
    public static Theme createGradientTheme(String name, String startColor, 
                                           String endColor, String backgroundColor) {
        String[] gradientColors = ColorPalette.generateAdvancedGradient(
            startColor, endColor, 10, "linear");
        
        Map<String, Object> config = new HashMap<>();
        config.put("backgroundColor", backgroundColor);
        config.put("textColor", getContrastColor(backgroundColor));
        config.put("color", gradientColors);
        
        return new Theme(name, config);
    }
    
    /**
     * 获取对比色
     * @param backgroundColor 背景颜色
     * @return 对比色
     */
    private static String getContrastColor(String backgroundColor) {
        return ColorPalette.getContrastColor(backgroundColor);
    }
    
    /**
     * 主题类
     */
    public static class Theme implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private final String name;
        private final Map<String, Object> config;
        
        public Theme(String name, Map<String, Object> config) {
            this.name = name;
            this.config = config;
        }
        
        public String getName() {
            return name;
        }
        
        public Map<String, Object> getConfig() {
            return config;
        }
    }
}
