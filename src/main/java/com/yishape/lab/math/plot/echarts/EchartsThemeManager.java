package com.yishape.lab.math.plot.echarts;

import com.yishape.lab.math.plot.ColorPalette;
import com.yishape.lab.math.plot.PlotStyle;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import org.icepear.echarts.Option;
import org.icepear.echarts.components.grid.Grid;

/**
 * 主题管理器，提供ECharts主题的创建、应用和管理功能
 * 支持内置主题和自定义主题，增强了与样式系统的集成
 * 
 * @author lteb2
 */
public class EchartsThemeManager implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // ========== 内置主题常量 ==========
    
    /** 默认主题 */
    public static final String THEME_DEFAULT = "default";
    
    /** 浅色主题 */
    public static final String THEME_LIGHT = "light";
    
    /** 深色主题 */
    public static final String THEME_DARK = "dark";
    
    /** 蓝色主题 */
    public static final String THEME_BLUE = "blue";
    
    /** 绿色主题 */
    public static final String THEME_GREEN = "green";
    
    /** 红色主题 */
    public static final String THEME_RED = "red";
    
    /** 紫色主题 */
    public static final String THEME_PURPLE = "purple";
    
    /** 橙色主题 */
    public static final String THEME_ORANGE = "orange";
    
    /** 学术风格主题 */
    public static final String THEME_ACADEMIC = "academic";
    
    /** 商务风格主题 */
    public static final String THEME_BUSINESS = "business";
    
    /** 简约风格主题 */
    public static final String THEME_MINIMAL = "minimal";
    
    /** 彩虹风格主题 */
    public static final String THEME_RAINBOW = "rainbow";
    
    /** 复古风格主题 */
    public static final String THEME_VINTAGE = "vintage";
    
    /** 未来风格主题 */
    public static final String THEME_FUTURISTIC = "futuristic";
    
    // ========== 主题注册表 ==========
    
    private static final Map<String, CustomTheme> CUSTOM_THEMES = new HashMap<>();
    
    // ========== 内置主题配置 ==========
    
    /**
     * 获取内置主题配置
     * @param themeName 主题名称
     * @return 主题配置
     */
    public static Map<String, Object> getBuiltinTheme(String themeName) {
        Map<String, Object> theme = new HashMap<>();
        
        switch (themeName.toLowerCase()) {
            case "light":
                theme.put("backgroundColor", "#ffffff");
                theme.put("textStyle", createTextStyle("#333333", 12, "Arial"));
                theme.put("color", new String[]{
                    "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
                    "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#8c8c8c"
                });
                break;
                
            case "dark":
                theme.put("backgroundColor", "#1e1e1e");
                theme.put("textStyle", createTextStyle("#ffffff", 12, "Arial"));
                theme.put("color", new String[]{
                    "#4992ff", "#7cffb2", "#fddd60", "#ff6e76", "#58d9f9",
                    "#05c091", "#ff8a45", "#8d4bbb", "#ff9f7f", "#9ca0a6"
                });
                break;
                
            case "blue":
                theme.put("backgroundColor", "#f0f8ff");
                theme.put("textStyle", createTextStyle("#1e3a8a", 12, "Arial"));
                theme.put("color", new String[]{
                    "#1e40af", "#3b82f6", "#60a5fa", "#93c5fd", "#dbeafe",
                    "#1e3a8a", "#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa"
                });
                break;
                
            case "green":
                theme.put("backgroundColor", "#f0fdf4");
                theme.put("textStyle", createTextStyle("#14532d", 12, "Arial"));
                theme.put("color", new String[]{
                    "#166534", "#16a34a", "#22c55e", "#4ade80", "#bbf7d0",
                    "#14532d", "#15803d", "#16a34a", "#22c55e", "#4ade80"
                });
                break;
                
            case "red":
                theme.put("backgroundColor", "#fef2f2");
                theme.put("textStyle", createTextStyle("#991b1b", 12, "Arial"));
                theme.put("color", new String[]{
                    "#dc2626", "#ef4444", "#f87171", "#fca5a5", "#fecaca",
                    "#991b1b", "#b91c1c", "#dc2626", "#ef4444", "#f87171"
                });
                break;
                
            case "purple":
                theme.put("backgroundColor", "#faf5ff");
                theme.put("textStyle", createTextStyle("#581c87", 12, "Arial"));
                theme.put("color", new String[]{
                    "#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe",
                    "#581c87", "#6b21a8", "#7c3aed", "#8b5cf6", "#a78bfa"
                });
                break;
                
            case "orange":
                theme.put("backgroundColor", "#fff7ed");
                theme.put("textStyle", createTextStyle("#9a3412", 12, "Arial"));
                theme.put("color", new String[]{
                    "#ea580c", "#f97316", "#fb923c", "#fdba74", "#fed7aa",
                    "#9a3412", "#c2410c", "#ea580c", "#f97316", "#fb923c"
                });
                break;
                
            case "academic":
                theme.put("backgroundColor", "#fafafa");
                theme.put("textStyle", createTextStyle("#2d3748", 11, "Georgia"));
                theme.put("color", new String[]{
                    "#2b6cb0", "#38a169", "#d69e2e", "#e53e3e", "#805ad5",
                    "#319795", "#dd6b20", "#9f7aea", "#ed64a6", "#4a5568"
                });
                theme.put("grid", createGridConfig(true, "#e2e8f0"));
                theme.put("axisTick", createAxisTickConfig(true));
                break;
                
            case "business":
                theme.put("backgroundColor", "#f8fafc");
                theme.put("textStyle", createTextStyle("#1a202c", 12, "Segoe UI"));
                theme.put("color", new String[]{
                    "#0369a1", "#059669", "#dc2626", "#7c3aed", "#ea580c",
                    "#0891b2", "#16a34a", "#be123c", "#9333ea", "#c2410c"
                });
                theme.put("grid", createGridConfig(true, "#e2e8f0"));
                theme.put("legend", createLegendConfig("top", "center"));
                break;
                
            case "minimal":
                theme.put("backgroundColor", "#ffffff");
                theme.put("textStyle", createTextStyle("#374151", 11, "SF Pro Display"));
                theme.put("color", new String[]{
                    "#6b7280", "#9ca3af", "#d1d5db", "#4b5563", "#374151",
                    "#111827", "#1f2937", "#6b7280", "#9ca3af", "#d1d5db"
                });
                theme.put("grid", createGridConfig(false, "transparent"));
                theme.put("axisLine", createAxisLineConfig(false));
                break;
                
            case "rainbow":
                theme.put("backgroundColor", "#fefefe");
                theme.put("textStyle", createTextStyle("#2d3748", 12, "Arial"));
                theme.put("color", new String[]{
                    "#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24", "#f0932b",
                    "#eb4d4b", "#6c5ce7", "#a29bfe", "#fd79a8", "#fdcb6e"
                });
                break;
                
            case "vintage":
                theme.put("backgroundColor", "#f4f1de");
                theme.put("textStyle", createTextStyle("#3d405b", 12, "Times New Roman"));
                theme.put("color", new String[]{
                    "#81b29a", "#f2cc8f", "#e07a5f", "#3d405b", "#f4f3ee",
                    "#a8dadc", "#457b9d", "#1d3557", "#f1faee", "#e63946"
                });
                theme.put("grid", createGridConfig(true, "#e9c46a"));
                break;
                
            case "futuristic":
                theme.put("backgroundColor", "#0f0f23");
                theme.put("textStyle", createTextStyle("#00ff41", 12, "Consolas"));
                theme.put("color", new String[]{
                    "#00ff41", "#00d4aa", "#0099cc", "#cc00ff", "#ff0080",
                    "#ffff00", "#ff6600", "#00ccff", "#ff3366", "#66ff00"
                });
                theme.put("grid", createGridConfig(true, "#1a1a3a"));
                break;
                
            default: // default
                theme.put("backgroundColor", "#ffffff");
                theme.put("textStyle", createTextStyle("#333333", 12, "Arial"));
                theme.put("color", new String[]{
                    "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
                    "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#8c8c8c"
                });
                break;
        }
        
        return theme;
    }
    
    /**
     * 创建文本样式配置
     * @param color 文本颜色
     * @param fontSize 字体大小
     * @param fontFamily 字体族
     * @return 文本样式配置
     */
    private static Map<String, Object> createTextStyle(String color, int fontSize, String fontFamily) {
        Map<String, Object> textStyle = new HashMap<>();
        textStyle.put("color", color);
        textStyle.put("fontSize", fontSize);
        textStyle.put("fontFamily", fontFamily);
        return textStyle;
    }
    
    /**
     * 创建网格配置
     * @param show 是否显示网格
     * @param color 网格颜色
     * @return 网格配置
     */
    private static Map<String, Object> createGridConfig(boolean show, String color) {
        Map<String, Object> grid = new HashMap<>();
        grid.put("show", show);
        if (show) {
            Map<String, Object> splitLine = new HashMap<>();
            splitLine.put("show", true);
            
            Map<String, Object> lineStyle = new HashMap<>();
            lineStyle.put("color", color);
            lineStyle.put("type", "solid");
            lineStyle.put("width", 1);
            splitLine.put("lineStyle", lineStyle);
            
            grid.put("splitLine", splitLine);
        }
        return grid;
    }
    
    /**
     * 创建轴刻度配置
     * @param show 是否显示刻度
     * @return 轴刻度配置
     */
    private static Map<String, Object> createAxisTickConfig(boolean show) {
        Map<String, Object> axisTick = new HashMap<>();
        axisTick.put("show", show);
        if (show) {
            axisTick.put("length", 5);
            axisTick.put("lineStyle", Map.of("color", "#cccccc", "width", 1));
        }
        return axisTick;
    }
    
    /**
     * 创建图例配置
     * @param orient 图例方向
     * @param align 图例对齐方式
     * @return 图例配置
     */
    private static Map<String, Object> createLegendConfig(String orient, String align) {
        Map<String, Object> legend = new HashMap<>();
        legend.put("orient", orient);
        legend.put("align", align);
        legend.put("itemGap", 10);
        legend.put("textStyle", Map.of("fontSize", 12));
        return legend;
    }
    
    /**
     * 创建轴线配置
     * @param show 是否显示轴线
     * @return 轴线配置
     */
    private static Map<String, Object> createAxisLineConfig(boolean show) {
        Map<String, Object> axisLine = new HashMap<>();
        axisLine.put("show", show);
        if (show) {
            axisLine.put("lineStyle", Map.of("color", "#cccccc", "width", 1));
        }
        return axisLine;
    }
    
    // ========== 主题应用方法 ==========
    
    /**
     * 应用主题到ECharts选项
     * @param option ECharts选项对象
     * @param themeName 主题名称
     */
    public static void applyTheme(Option option, String themeName) {
        if (option == null || themeName == null) return;
        
        // 检查是否为自定义主题
        if (CUSTOM_THEMES.containsKey(themeName)) {
            CustomTheme customTheme = CUSTOM_THEMES.get(themeName);
            customTheme.applyToOption(option);
        } else {
            // 应用内置主题
            Map<String, Object> themeConfig = getBuiltinTheme(themeName);
            applyThemeConfig(option, themeConfig);
        }
    }
    
    /**
     * 应用主题配置到选项
     * @param option ECharts选项对象
     * @param themeConfig 主题配置
     */
    private static void applyThemeConfig(Option option, Map<String, Object> themeConfig) {
        // 设置背景色
        if (themeConfig.containsKey("backgroundColor")) {
            option.setBackgroundColor((String) themeConfig.get("backgroundColor"));
        }
        
        // 设置文本样式
        if (themeConfig.containsKey("textStyle")) {
            option.setTextStyle((Map<String, Object>) themeConfig.get("textStyle"));
        }
        
        // 设置颜色调色板
        if (themeConfig.containsKey("color")) {
            option.setColor((String[]) themeConfig.get("color"));
        }
        
        // 网格配置在主题中用于轴线样式，不适用于Grid组件的定位
        // Grid config in themes is for axis styling, not suitable for Grid component positioning
        // The grid config will be handled by individual chart components when they need axis styling
    }
    
    /**
     * 从配置映射创建网格对象用于布局定位
     * @param left 左边距
     * @param top 上边距
     * @param right 右边距
     * @param bottom 下边距
     * @return Grid对象
     */
    public static Grid createGridLayout(Object left, Object top, Object right, Object bottom) {
        Grid grid = new Grid();
        
        if (left != null) {
            if (left instanceof Number) {
                grid.setLeft((Number) left);
            } else if (left instanceof String) {
                grid.setLeft((String) left);
            }
        }
        
        if (top != null) {
            if (top instanceof Number) {
                grid.setTop((Number) top);
            } else if (top instanceof String) {
                grid.setTop((String) top);
            }
        }
        
        if (right != null) {
            if (right instanceof Number) {
                grid.setRight((Number) right);
            } else if (right instanceof String) {
                grid.setRight((String) right);
            }
        }
        
        if (bottom != null) {
            if (bottom instanceof Number) {
                grid.setBottom((Number) bottom);
            } else if (bottom instanceof String) {
                grid.setBottom((String) bottom);
            }
        }
        
        return grid;
    }
    
    /**
     * 获取主题中的网格样式配置（用于轴线样式）
     * @param themeName 主题名称
     * @return 网格样式配置，如果不存在则返回null
     */
    public static Map<String, Object> getGridStyleFromTheme(String themeName) {
        Map<String, Object> themeConfig = getBuiltinTheme(themeName);
        return (Map<String, Object>) themeConfig.get("grid");
    }
    
    // ========== 自定义主题管理 ==========
    
    /**
     * 注册自定义主题
     * @param themeName 主题名称
     * @param theme 自定义主题对象
     */
    public static void registerCustomTheme(String themeName, CustomTheme theme) {
        CUSTOM_THEMES.put(themeName, theme);
    }
    
    /**
     * 获取自定义主题
     * @param themeName 主题名称
     * @return 自定义主题对象
     */
    public static CustomTheme getCustomTheme(String themeName) {
        return CUSTOM_THEMES.get(themeName);
    }
    
    /**
     * 移除自定义主题
     * @param themeName 主题名称
     * @return 被移除的主题对象
     */
    public static CustomTheme removeCustomTheme(String themeName) {
        return CUSTOM_THEMES.remove(themeName);
    }
    
    /**
     * 获取所有已注册的主题名称
     * @return 主题名称列表
     */
    public static List<String> getRegisteredThemeNames() {
        List<String> names = new ArrayList<>();
        names.addAll(getBuiltinThemeNames());
        names.addAll(CUSTOM_THEMES.keySet());
        return names;
    }
    
    /**
     * 获取内置主题名称列表
     * @return 内置主题名称列表
     */
    public static List<String> getBuiltinThemeNames() {
        List<String> names = new ArrayList<>();
        names.add(THEME_DEFAULT);
        names.add(THEME_LIGHT);
        names.add(THEME_DARK);
        names.add(THEME_BLUE);
        names.add(THEME_GREEN);
        names.add(THEME_RED);
        names.add(THEME_PURPLE);
        names.add(THEME_ORANGE);
        names.add(THEME_ACADEMIC);
        names.add(THEME_BUSINESS);
        names.add(THEME_MINIMAL);
        names.add(THEME_RAINBOW);
        names.add(THEME_VINTAGE);
        names.add(THEME_FUTURISTIC);
        return names;
    }
    
    /**
     * 创建季节性主题
     * @param season 季节："spring", "summer", "autumn", "winter"
     * @return 季节主题对象
     */
    public static CustomTheme createSeasonalTheme(String season) {
        return new CustomTheme("seasonal_" + season) {
            @Override
            public void applyToOption(Option option) {
                switch (season.toLowerCase()) {
                    case "spring":
                        option.setBackgroundColor("#f0f9ff");
                        option.setColor(new String[]{
                            "#10b981", "#34d399", "#6ee7b7", "#a7f3d0", "#d1fae5",
                            "#059669", "#047857", "#065f46", "#064e3b", "#022c22"
                        });
                        break;
                    case "summer":
                        option.setBackgroundColor("#fffbeb");
                        option.setColor(new String[]{
                            "#f59e0b", "#fbbf24", "#fcd34d", "#fde68a", "#fef3c7",
                            "#d97706", "#b45309", "#92400e", "#78350f", "#451a03"
                        });
                        break;
                    case "autumn":
                        option.setBackgroundColor("#fef2f2");
                        option.setColor(new String[]{
                            "#dc2626", "#ef4444", "#f87171", "#fca5a5", "#fecaca",
                            "#b91c1c", "#991b1b", "#7f1d1d", "#651919", "#450a0a"
                        });
                        break;
                    case "winter":
                        option.setBackgroundColor("#f8fafc");
                        option.setColor(new String[]{
                            "#1e40af", "#3b82f6", "#60a5fa", "#93c5fd", "#dbeafe",
                            "#1e3a8a", "#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa"
                        });
                        break;
                    default:
                        option.setBackgroundColor("#ffffff");
                        option.setColor(new String[]{
                            "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
                            "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#8c8c8c"
                        });
                        break;
                }
                
                // 设置文本样式
                Map<String, Object> textStyle = new HashMap<>();
                textStyle.put("color", "#374151");
                textStyle.put("fontSize", 12);
                textStyle.put("fontFamily", "Arial");
                option.setTextStyle(textStyle);
            }
        };
    }
    
    /**
     * 创建行业主题
     * @param industry 行业类型："finance", "healthcare", "technology", "education", "retail"
     * @return 行业主题对象
     */
    public static CustomTheme createIndustryTheme(String industry) {
        return new CustomTheme("industry_" + industry) {
            @Override
            public void applyToOption(Option option) {
                switch (industry.toLowerCase()) {
                    case "finance":
                        option.setBackgroundColor("#0f172a");
                        option.setColor(new String[]{
                            "#10b981", "#059669", "#047857", "#065f46", "#064e3b",
                            "#fbbf24", "#f59e0b", "#d97706", "#b45309", "#92400e"
                        });
                        break;
                    case "healthcare":
                        option.setBackgroundColor("#f0fdf4");
                        option.setColor(new String[]{
                            "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#dbeafe",
                            "#10b981", "#34d399", "#6ee7b7", "#a7f3d0", "#d1fae5"
                        });
                        break;
                    case "technology":
                        option.setBackgroundColor("#0c0c0c");
                        option.setColor(new String[]{
                            "#00ff41", "#00d4aa", "#0099cc", "#3366ff", "#6633ff",
                            "#ff3366", "#ff6633", "#ffff00", "#ccff00", "#00ffcc"
                        });
                        break;
                    case "education":
                        option.setBackgroundColor("#fefce8");
                        option.setColor(new String[]{
                            "#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe",
                            "#06b6d4", "#22d3ee", "#67e8f9", "#a5f3fc", "#cffafe"
                        });
                        break;
                    case "retail":
                        option.setBackgroundColor("#fdf2f8");
                        option.setColor(new String[]{
                            "#ec4899", "#f472b6", "#f9a8d4", "#fbcfe8", "#fce7f3",
                            "#f59e0b", "#fbbf24", "#fcd34d", "#fde68a", "#fef3c7"
                        });
                        break;
                    default:
                        option.setBackgroundColor("#ffffff");
                        option.setColor(new String[]{
                            "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
                            "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#8c8c8c"
                        });
                        break;
                }
                
                // 设置文本样式
                Map<String, Object> textStyle = new HashMap<>();
                textStyle.put("color", getContrastColor(option.getBackgroundColor()));
                textStyle.put("fontSize", 12);
                textStyle.put("fontFamily", "Arial");
                option.setTextStyle(textStyle);
            }
        };
    }
    
    /**
     * 创建无障碍访问主题
     * @return 无障碍访问主题对象
     */
    public static CustomTheme createAccessibilityTheme() {
        return new CustomTheme("accessibility") {
            @Override
            public void applyToOption(Option option) {
                // 使用高对比度和色盲友好的颜色
                option.setBackgroundColor("#ffffff");
                option.setColor(ColorPalette.COLORBLIND_FRIENDLY);
                
                // 设置更大的字体和更明显的样式
                Map<String, Object> textStyle = new HashMap<>();
                textStyle.put("color", "#000000");
                textStyle.put("fontSize", 14); // 更大的字体
                textStyle.put("fontFamily", "Arial");
                textStyle.put("fontWeight", "bold"); // 加粗字体
                option.setTextStyle(textStyle);
                
                // 增强网格线和轴线的可见性
                Grid grid = new Grid()
                    .setShow(true)
                    .setBorderWidth(2)
                    .setBorderColor("#000000");
                option.setGrid(grid);
            }
        };
    }
    
    // ========== 主题创建辅助方法 ==========
    
    /**
     * 创建渐变主题
     * @param name 主题名称
     * @param startColor 起始颜色
     * @param endColor 结束颜色
     * @param backgroundColor 背景颜色
     * @return 自定义主题对象
     */
    public static CustomTheme createGradientTheme(String name, String startColor, String endColor, String backgroundColor) {
        return new CustomTheme(name) {
            @Override
            public void applyToOption(Option option) {
                option.setBackgroundColor(backgroundColor);
                
                // 创建渐变颜色数组
                String[] gradientColors = createGradientColors(startColor, endColor, 10);
                option.setColor(gradientColors);
                
                // 设置文本样式
                Map<String, Object> textStyle = new HashMap<>();
                textStyle.put("color", getContrastColor(backgroundColor));
                textStyle.put("fontSize", 12);
                textStyle.put("fontFamily", "Arial");
                option.setTextStyle(textStyle);
            }
        };
    }
    
    /**
     * 创建单色主题
     * @param name 主题名称
     * @param baseColor 基础颜色
     * @param backgroundColor 背景颜色
     * @return 自定义主题对象
     */
    public static CustomTheme createMonochromeTheme(String name, String baseColor, String backgroundColor) {
        return new CustomTheme(name) {
            @Override
            public void applyToOption(Option option) {
                option.setBackgroundColor(backgroundColor);
                
                // 创建单色系颜色数组
                String[] monochromeColors = createMonochromeColors(baseColor, 10);
                option.setColor(monochromeColors);
                
                // 设置文本样式
                Map<String, Object> textStyle = new HashMap<>();
                textStyle.put("color", getContrastColor(backgroundColor));
                textStyle.put("fontSize", 12);
                textStyle.put("fontFamily", "Arial");
                option.setTextStyle(textStyle);
            }
        };
    }
    
    /**
     * 创建渐变颜色数组
     * @param startColor 起始颜色
     * @param endColor 结束颜色
     * @param count 颜色数量
     * @return 颜色数组
     */
    private static String[] createGradientColors(String startColor, String endColor, int count) {
        String[] colors = new String[count];
        // 简化实现，实际应该计算渐变
        for (int i = 0; i < count; i++) {
            colors[i] = (i % 2 == 0) ? startColor : endColor;
        }
        return colors;
    }
    
    /**
     * 创建单色系颜色数组
     * @param baseColor 基础颜色
     * @param count 颜色数量
     * @return 颜色数组
     */
    private static String[] createMonochromeColors(String baseColor, int count) {
        String[] colors = new String[count];
        // 简化实现，实际应该计算不同明度的颜色
        for (int i = 0; i < count; i++) {
            colors[i] = baseColor;
        }
        return colors;
    }
    
    /**
     * 获取对比色
     * @param backgroundColor 背景颜色
     * @return 对比色
     */
    private static String getContrastColor(String backgroundColor) {
        return ColorPalette.getContrastColor(backgroundColor);
    }
    
    // ========== 抽象自定义主题类 ==========
    
    /**
     * 自定义主题抽象类
     */
    public static abstract class CustomTheme implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private final String name;
        
        public CustomTheme(String name) {
            this.name = name;
        }
        
        public String getName() {
            return name;
        }
        
        /**
         * 应用主题到ECharts选项
         * @param option ECharts选项对象
         */
        public abstract void applyToOption(Option option);
        
        @Override
        public String toString() {
            return "CustomTheme{name='" + name + "'}";
        }
    }
    
    // ========== 增强的主题集成方法 ==========
    
    /**
     * 为样式对象应用主题配置（增强版）
     * @param style 基础样式对象
     * @param themeName 主题名称
     * @return 应用主题后的样式对象
     */
    public static PlotStyle applyThemeToStyle(PlotStyle style, String themeName) {
        if (style == null) {
            style = PlotStyle.defaultStyle();
        }
        
        Map<String, Object> themeConfig = getBuiltinTheme(themeName);
        if (themeConfig == null) {
            return style;
        }
        
        // 创建新的样式对象，保留原始样式
        PlotStyle themedStyle = new PlotStyle(style);
        
        // 应用主题颜色
        String[] colors = (String[]) themeConfig.get("color");
        if (colors != null && colors.length > 0) {
            // 如果当前使用默认颜色，则应用主题颜色
            if (style.getColor().equals(PlotStyle.defaultStyle().getColor())) {
                themedStyle.setColor(colors[0]);
            }
        }
        
        // 应用主题字体
        @SuppressWarnings("unchecked")
        Map<String, Object> textStyle = (Map<String, Object>) themeConfig.get("textStyle");
        if (textStyle != null) {
            String fontFamily = (String) textStyle.get("fontFamily");
            if (fontFamily != null) {
                themedStyle.setFontFamily(fontFamily);
            }
            
            Number fontSize = (Number) textStyle.get("fontSize");
            if (fontSize != null) {
                themedStyle.setFontSize(fontSize.doubleValue());
            }
        }
        
        // 应用主题特定样式
        applyThemeSpecificStyles(themedStyle, themeName);
        
        return themedStyle;
    }
    
    /**
     * 应用主题特定样式
     * @param style 样式对象
     * @param themeName 主题名称
     */
    private static void applyThemeSpecificStyles(PlotStyle style, String themeName) {
        switch (themeName.toLowerCase()) {
            case THEME_ACADEMIC:
                // 学术主题：简洁、专业
                style.setLineWidth(1.5);
                style.setMarkerSize(4);
                style.setAnimationDuration(800);
                break;
                
            case THEME_BUSINESS:
                // 商务主题：现代、专业
                style.setLineWidth(2.5);
                style.setMarkerSize(6);
                style.setAnimationDuration(1200);
                style.setProperty("shadowBlur", 5);
                style.setProperty("shadowColor", "rgba(0, 0, 0, 0.1)");
                break;
                
            case THEME_MINIMAL:
                // 极简主题：精简、清晰
                style.setLineWidth(1);
                style.setMarkerSize(3);
                style.setAnimationDuration(600);
                style.setAlpha(0.9);
                break;
                
            case THEME_RAINBOW:
                // 彩虹主题：生动、多彩
                style.setLineWidth(3);
                style.setMarkerSize(8);
                style.setAnimationDuration(1500);
                style.setProperty("shadowBlur", 8);
                break;
                
            case THEME_VINTAGE:
                // 复古主题：温暖、经典
                style.setLineWidth(2);
                style.setMarkerSize(5);
                style.setAnimationDuration(1000);
                style.setAlpha(0.85);
                break;
                
            case THEME_FUTURISTIC:
                // 未来主题：科技、现代
                style.setLineWidth(2);
                style.setMarkerSize(6);
                style.setAnimationDuration(1800);
                style.setProperty("shadowBlur", 15);
                style.setProperty("shadowColor", "rgba(0, 255, 255, 0.3)");
                break;
        }
    }
    
    /**
     * 获取主题适合的调色板
     * @param themeName 主题名称
     * @return 调色板名称
     */
    public static String getThemePreferredPalette(String themeName) {
        switch (themeName.toLowerCase()) {
            case THEME_ACADEMIC:
                return "muted";
            case THEME_BUSINESS:
                return "echarts";
            case THEME_MINIMAL:
                return "pastel";
            case THEME_RAINBOW:
                return "seaborn";
            case THEME_VINTAGE:
                return "reds";
            case THEME_FUTURISTIC:
                return "blues";
            default:
                return "echarts";
        }
    }
    
    /**
     * 智能主题推荐系统
     * @param dataType 数据类型："business", "scientific", "creative", "academic"
     * @param chartType 图表类型
     * @param userPreference 用户偶好："professional", "colorful", "minimal"
     * @return 推荐的主题名称
     */
    public static String recommendTheme(String dataType, String chartType, String userPreference) {
        // 根据数据类型推荐
        if ("business".equalsIgnoreCase(dataType)) {
            return userPreference.equals("minimal") ? THEME_MINIMAL : THEME_BUSINESS;
        }
        
        if ("scientific".equalsIgnoreCase(dataType) || "academic".equalsIgnoreCase(dataType)) {
            return THEME_ACADEMIC;
        }
        
        // 根据用户偶好推荐
        switch (userPreference.toLowerCase()) {
            case "colorful":
                return THEME_RAINBOW;
            case "minimal":
                return THEME_MINIMAL;
            case "professional":
                return THEME_BUSINESS;
            default:
                return THEME_DEFAULT;
        }
    }
}



