package com.yishape.lab.math.viz;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 颜色调色板类，提供预定义的颜色方案
 * 支持类似matplotlib的颜色映射和调色板
 * 
 * @author lteb2
 */
public class ColorPalette implements Serializable {
    
    // ========== 预定义颜色常量 ==========
    
    // 基础颜色（CSS标准颜色）
    public static final String RED = "#FF0000";
    public static final String GREEN = "#00FF00";
    public static final String BLUE = "#0000FF";
    public static final String BLACK = "#000000";
    public static final String WHITE = "#FFFFFF";
    public static final String YELLOW = "#FFFF00";
    public static final String CYAN = "#00FFFF";
    public static final String MAGENTA = "#FF00FF";
    public static final String ORANGE = "#FFA500";
    public static final String PURPLE = "#800080";
    public static final String PINK = "#FFC0CB";
    public static final String BROWN = "#A52A2A";
    public static final String GRAY = "#808080";
    public static final String GREY = "#808080";
    
    // matplotlib默认颜色序列 (C0-C9)
    public static final String[] MATPLOTLIB_COLORS = {
        "#1f77b4", // C0 - 蓝色
        "#ff7f0e", // C1 - 橙色
        "#2ca02c", // C2 - 绿色
        "#d62728", // C3 - 红色
        "#9467bd", // C4 - 紫色
        "#8c564b", // C5 - 棕色
        "#e377c2", // C6 - 粉色
        "#7f7f7f", // C7 - 灰色
        "#bcbd22", // C8 - 橄榄绿
        "#17becf"  // C9 - 青色
    };
    
    // ECharts默认颜色序列
    public static final String[] ECHARTS_COLORS = {
        "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
        "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#8c8c8c"
    };
    
    // Seaborn风格调色板
    public static final String[] SEABORN_DEEP = {
        "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
        "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"
    };
    
    public static final String[] SEABORN_MUTED = {
        "#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4",
        "#8c613c", "#dc7ec0", "#797979", "#d5bb67", "#82c6e2"
    };
    
    public static final String[] SEABORN_PASTEL = {
        "#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff",
        "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"
    };
    
    // 颜色盲友好调色板
    public static final String[] COLORBLIND_FRIENDLY = {
        "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999"
    };
    
    // 单色调色板
    public static final String[] BLUES = {
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
        "#4292c6", "#2171b5", "#08519c", "#08306b"
    };
    
    public static final String[] REDS = {
        "#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a",
        "#ef3b2c", "#cb181d", "#a50f15", "#67000d"
    };
    
    public static final String[] GREENS = {
        "#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476",
        "#41ab5d", "#238b45", "#006d2c", "#00441b"
    };
    
    // ========== 调色板映射 ==========
    
    private static final Map<String, String[]> PALETTE_MAP = new HashMap<>();
    
    static {
        PALETTE_MAP.put("default", ECHARTS_COLORS);
        PALETTE_MAP.put("matplotlib", MATPLOTLIB_COLORS);
        PALETTE_MAP.put("echarts", ECHARTS_COLORS);
        PALETTE_MAP.put("seaborn", SEABORN_DEEP);
        PALETTE_MAP.put("deep", SEABORN_DEEP);
        PALETTE_MAP.put("muted", SEABORN_MUTED);
        PALETTE_MAP.put("pastel", SEABORN_PASTEL);
        PALETTE_MAP.put("colorblind", COLORBLIND_FRIENDLY);
        PALETTE_MAP.put("blues", BLUES);
        PALETTE_MAP.put("reds", REDS);
        PALETTE_MAP.put("greens", GREENS);
    }
    
    // ========== 颜色名称映射 ==========
    
    public static final Map<String, String> COLOR_NAME_MAP = new HashMap<>();
    
    static {
        // 基础颜色
        COLOR_NAME_MAP.put("r", RED);
        COLOR_NAME_MAP.put("g", GREEN);
        COLOR_NAME_MAP.put("b", BLUE);
        COLOR_NAME_MAP.put("k", BLACK);
        COLOR_NAME_MAP.put("w", WHITE);
        COLOR_NAME_MAP.put("y", YELLOW);
        COLOR_NAME_MAP.put("c", CYAN);
        COLOR_NAME_MAP.put("m", MAGENTA);
        
        // 完整颜色名称
        COLOR_NAME_MAP.put("red", RED);
        COLOR_NAME_MAP.put("green", GREEN);
        COLOR_NAME_MAP.put("blue", BLUE);
        COLOR_NAME_MAP.put("black", BLACK);
        COLOR_NAME_MAP.put("white", WHITE);
        COLOR_NAME_MAP.put("yellow", YELLOW);
        COLOR_NAME_MAP.put("cyan", CYAN);
        COLOR_NAME_MAP.put("magenta", MAGENTA);
        COLOR_NAME_MAP.put("orange", ORANGE);
        COLOR_NAME_MAP.put("purple", PURPLE);
        COLOR_NAME_MAP.put("pink", PINK);
        COLOR_NAME_MAP.put("brown", BROWN);
        COLOR_NAME_MAP.put("gray", GRAY);
        COLOR_NAME_MAP.put("grey", GREY);
        
        // matplotlib风格的C0-C9颜色
        for (int i = 0; i < MATPLOTLIB_COLORS.length; i++) {
            COLOR_NAME_MAP.put("C" + i, MATPLOTLIB_COLORS[i]);
        }
    }
    
    /**
     * 获取指定调色板的颜色数组
     * @param paletteName 调色板名称
     * @return 颜色数组，如果调色板不存在则返回默认调色板
     */
    public static String[] getPalette(String paletteName) {
        return PALETTE_MAP.getOrDefault(paletteName.toLowerCase(), ECHARTS_COLORS);
    }
    
    /**
     * 获取调色板中指定索引的颜色
     * @param paletteName 调色板名称
     * @param index 颜色索引
     * @return 颜色值，索引超出范围时循环使用
     */
    public static String getColor(String paletteName, int index) {
        String[] palette = getPalette(paletteName);
        return palette[index % palette.length];
    }
    
    /**
     * 根据颜色名称或代码获取颜色值
     * @param colorSpec 颜色规范（可以是颜色名称、单字符代码、十六进制值等）
     * @return 颜色值，无法识别时返回默认颜色
     */
    public static String parseColor(String colorSpec) {
        if (colorSpec == null || colorSpec.trim().isEmpty()) {
            return ECHARTS_COLORS[0]; // 默认颜色
        }
        
        String spec = colorSpec.trim().toLowerCase();
        
        // 检查是否是十六进制颜色
        if (spec.startsWith("#") && (spec.length() == 7 || spec.length() == 4)) {
            return colorSpec.toUpperCase();
        }
        
        // 检查是否是RGB颜色
        if (spec.startsWith("rgb(") && spec.endsWith(")")) {
            return colorSpec;
        }
        
        // 检查是否是颜色名称映射
        if (COLOR_NAME_MAP.containsKey(spec)) {
            return COLOR_NAME_MAP.get(spec);
        }
        
        // 检查是否是matplotlib风格的颜色引用 (如 "C0", "C1")
        if (spec.matches("^c\\d+$")) {
            String upperSpec = spec.toUpperCase();
            if (COLOR_NAME_MAP.containsKey(upperSpec)) {
                return COLOR_NAME_MAP.get(upperSpec);
            }
        }
        
        // 无法识别的颜色，返回默认颜色
        return ECHARTS_COLORS[0];
    }
    
    /**
     * 生成渐变色序列（增强版，支持多种算法）
     * @param startColor 起始颜色
     * @param endColor 结束颜色
     * @param steps 步数
     * @param algorithm 算法类型："linear", "quadratic", "cubic"
     * @return 渐变色数组
     */
    public static String[] generateAdvancedGradient(String startColor, String endColor, int steps, String algorithm) {
        if (steps <= 0) {
            return new String[]{startColor};
        }
        if (steps == 1) {
            return new String[]{startColor};
        }
        if (steps == 2) {
            return new String[]{startColor, endColor};
        }
        
        String[] gradient = new String[steps];
        gradient[0] = startColor;
        gradient[steps - 1] = endColor;
        
        // 解析起始和结束颜色
        int[] startRGB = parseColorToRGB(startColor);
        int[] endRGB = parseColorToRGB(endColor);
        
        if (startRGB == null || endRGB == null) {
            // 如果解析失败，使用简单的交替模式
            for (int i = 1; i < steps - 1; i++) {
                gradient[i] = (i % 2 == 0) ? startColor : endColor;
            }
            return gradient;
        }
        
        // 根据算法计算中间颜色
        for (int i = 1; i < steps - 1; i++) {
            double t = (double) i / (steps - 1);
            
            // 应用不同的算法
            switch (algorithm.toLowerCase()) {
                case "quadratic":
                    t = t * t; // 二次曲线
                    break;
                case "cubic":
                    t = t * t * t; // 三次曲线
                    break;
                case "ease-in":
                    t = 1 - Math.cos(t * Math.PI / 2); // 缓入
                    break;
                case "ease-out":
                    t = Math.sin(t * Math.PI / 2); // 缓出
                    break;
                case "ease-in-out":
                    t = 0.5 * (1 - Math.cos(Math.PI * t)); // 缓入缓出
                    break;
                default: // linear
                    // t 保持不变
                    break;
            }
            
            // 计算中间颜色
            int r = (int) (startRGB[0] + (endRGB[0] - startRGB[0]) * t);
            int g = (int) (startRGB[1] + (endRGB[1] - startRGB[1]) * t);
            int b = (int) (startRGB[2] + (endRGB[2] - startRGB[2]) * t);
            
            gradient[i] = String.format("#%02x%02x%02x", 
                Math.max(0, Math.min(255, r)),
                Math.max(0, Math.min(255, g)),
                Math.max(0, Math.min(255, b)));
        }
        
        return gradient;
    }
    
    /**
     * 检查颜色组合是否色盲友好
     * @param colors 颜色数组
     * @return 是否色盲友好
     */
    public static boolean isColorBlindFriendly(String[] colors) {
        if (colors == null || colors.length < 2) {
            return true; // 单色或空数组认为是友好的
        }
        
        // 计算所有颜色对之间的对比度
        for (int i = 0; i < colors.length; i++) {
            for (int j = i + 1; j < colors.length; j++) {
                double contrast = calculateColorContrast(colors[i], colors[j]);
                if (contrast < 3.0) { // WCAG AA 标准的最小对比度
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /**
     * 调整颜色以适应色盲
     * @param colors 原始颜色数组
     * @param colorBlindType 色盲类型："protanopia", "deuteranopia", "tritanopia"
     * @return 调整后的颜色数组
     */
    public static String[] adjustForColorBlindness(String[] colors, String colorBlindType) {
        if (colors == null || colors.length == 0) {
            return colors;
        }
        
        String[] adjustedColors = new String[colors.length];
        
        for (int i = 0; i < colors.length; i++) {
            adjustedColors[i] = adjustColorForColorBlindness(colors[i], colorBlindType);
        }
        
        return adjustedColors;
    }
    
    /**
     * 计算两个颜色之间的对比度
     * @param color1 颜色1
     * @param color2 颜色2
     * @return 对比度比率
     */
    public static double calculateColorContrast(String color1, String color2) {
        double[] lab1 = rgbToLab(parseColorToRGB(color1));
        double[] lab2 = rgbToLab(parseColorToRGB(color2));
        
        if (lab1 == null || lab2 == null) {
            return 1.0; // 默认对比度
        }
        
        // 使用 Delta E CIE76 算法计算颜色差异
        double deltaL = lab1[0] - lab2[0];
        double deltaA = lab1[1] - lab2[1];
        double deltaB = lab1[2] - lab2[2];
        
        double deltaE = Math.sqrt(deltaL * deltaL + deltaA * deltaA + deltaB * deltaB);
        
        // 将 Delta E 转换为对比度比率（简化算法）
        return Math.max(1.0, deltaE / 20.0);
    }
    
    /**
     * 创建按温度分类的色彩方案
     * @param baseTemp 基础温度："warm", "cool", "neutral"
     * @param count 颜色数量
     * @return 颜色数组
     */
    public static String[] createTemperatureBasedPalette(String baseTemp, int count) {
        String[] palette = new String[Math.max(1, count)];
        
        switch (baseTemp.toLowerCase()) {
            case "warm":
                // 暖色调：红、橙、黄色系
                String[] warmBase = {"#ff6b6b", "#ffa726", "#ffeb3b", "#ff8a65", "#ffab40"};
                return expandPalette(warmBase, count);
                
            case "cool":
                // 冷色调：蓝、绿、紫色系
                String[] coolBase = {"#42a5f5", "#66bb6a", "#ab47bc", "#26c6da", "#7e57c2"};
                return expandPalette(coolBase, count);
                
            case "neutral":
                // 中性色调：灰、棕色系
                String[] neutralBase = {"#78909c", "#8d6e63", "#90a4ae", "#a1887f", "#9e9e9e"};
                return expandPalette(neutralBase, count);
                
            default:
                // 默认使用混合色调
                return Arrays.copyOf(ECHARTS_COLORS, Math.min(count, ECHARTS_COLORS.length));
        }
    }
    
    /**
     * 创建语义化颜色方案
     * @param semantic 语义类型："success", "warning", "error", "info"
     * @param intensity 强度级别：1-5
     * @return 颜色值
     */
    public static String getSemanticColor(String semantic, int intensity) {
        intensity = Math.max(1, Math.min(5, intensity)); // 限制在 1-5 范围内
        
        Map<String, String[]> semanticColors = new HashMap<>();
        semanticColors.put("success", new String[]{"#e8f5e8", "#c8e6c9", "#4caf50", "#388e3c", "#2e7d32"});
        semanticColors.put("warning", new String[]{"#fff3e0", "#ffcc02", "#ff9800", "#f57c00", "#ef6c00"});
        semanticColors.put("error", new String[]{"#ffebee", "#ffcdd2", "#f44336", "#d32f2f", "#c62828"});
        semanticColors.put("info", new String[]{"#e3f2fd", "#bbdefb", "#2196f3", "#1976d2", "#1565c0"});
        
        String[] colors = semanticColors.get(semantic.toLowerCase());
        if (colors != null) {
            return colors[intensity - 1];
        }
        
        return ECHARTS_COLORS[0]; // 默认颜色
    }
    // ========== 私有辅助方法 ==========
    
    /**
     * 解析颜色为RGB数组
     * @param color 颜色字符串
     * @return RGB数组 [r, g, b] 或 null
     */
    public static int[] parseColorToRGB(String color) {
        if (color == null) return null;
        
        // 处理十六进制颜色
        if (color.startsWith("#")) {
            String hex = color.substring(1);
            if (hex.length() == 3) {
                // 扩偕3位十六进制到6位
                hex = String.format("%c%c%c%c%c%c", 
                    hex.charAt(0), hex.charAt(0),
                    hex.charAt(1), hex.charAt(1),
                    hex.charAt(2), hex.charAt(2));
            }
            
            if (hex.length() == 6) {
                try {
                    int r = Integer.parseInt(hex.substring(0, 2), 16);
                    int g = Integer.parseInt(hex.substring(2, 4), 16);
                    int b = Integer.parseInt(hex.substring(4, 6), 16);
                    return new int[]{r, g, b};
                } catch (NumberFormatException e) {
                    return null;
                }
            }
        }
        
        // 处理RGB格式
        if (color.toLowerCase().startsWith("rgb(")) {
            String rgbPart = color.substring(4, color.length() - 1);
            String[] parts = rgbPart.split(",");
            if (parts.length == 3) {
                try {
                    int r = Integer.parseInt(parts[0].trim());
                    int g = Integer.parseInt(parts[1].trim());
                    int b = Integer.parseInt(parts[2].trim());
                    return new int[]{r, g, b};
                } catch (NumberFormatException e) {
                    return null;
                }
            }
        }
        
        // 处理颜色名称
        String colorValue = parseColor(color);
        if (colorValue.startsWith("#")) {
            return parseColorToRGB(colorValue);
        }
        
        return null;
    }
    
    /**
     * 将RGB转换为LAB颜色空间
     * @param rgb RGB数组
     * @return LAB数组 [L, a, b] 或 null
     */
    private static double[] rgbToLab(int[] rgb) {
        if (rgb == null || rgb.length != 3) return null;
        
        // 简化的RGB到LAB转换（实际上需要经过XYZ色彩空间）
        // 这里使用简化算法
        double r = rgb[0] / 255.0;
        double g = rgb[1] / 255.0;
        double b = rgb[2] / 255.0;
        
        // 简化的亮度计算
        double l = 0.299 * r + 0.587 * g + 0.114 * b;
        
        // 简化的a和b通道
        double a = (r - g) * 127;
        double reb = (g - b) * 127;
        
        return new double[]{l * 100, a, reb};
    }
    
    /**
     * 调整单个颜色以适应色盲
     * @param color 原始颜色
     * @param colorBlindType 色盲类型
     * @return 调整后的颜色
     */
    private static String adjustColorForColorBlindness(String color, String colorBlindType) {
        int[] rgb = parseColorToRGB(color);
        if (rgb == null) return color;
        
        // 简化的色盲调整算法
        switch (colorBlindType.toLowerCase()) {
            case "protanopia": // 红色盲
                // 减少红色成分，增强绿蓝对比
                rgb[0] = (int) (rgb[0] * 0.3 + rgb[1] * 0.7);
                break;
                
            case "deuteranopia": // 绿色盲
                // 减少绿色成分，增强红蓝对比
                rgb[1] = (int) (rgb[1] * 0.3 + rgb[0] * 0.4 + rgb[2] * 0.3);
                break;
                
            case "tritanopia": // 蓝色盲
                // 减少蓝色成分，增强红绿对比
                rgb[2] = (int) (rgb[2] * 0.3 + rgb[0] * 0.4 + rgb[1] * 0.3);
                break;
                
            default:
                return color; // 不支持的类型，返回原色
        }
        
        // 确保值在有效范围内
        for (int i = 0; i < 3; i++) {
            rgb[i] = Math.max(0, Math.min(255, rgb[i]));
        }
        
        return String.format("#%02x%02x%02x", rgb[0], rgb[1], rgb[2]);
    }
    
    /**
     * 扩展调色板到指定数量
     * @param basePalette 基础调色板
     * @param targetCount 目标数量
     * @return 扩展后的调色板
     */
    private static String[] expandPalette(String[] basePalette, int targetCount) {
        if (basePalette == null || basePalette.length == 0) {
            return new String[0];
        }
        
        if (targetCount <= basePalette.length) {
            return Arrays.copyOf(basePalette, targetCount);
        }
        
        String[] expanded = new String[targetCount];
        
        // 复制基础颜色
        for (int i = 0; i < basePalette.length; i++) {
            expanded[i] = basePalette[i];
        }
        
        // 使用渐变填充剩余位置
        int remaining = targetCount - basePalette.length;
        int gradientSteps = remaining + 1;
        
        for (int i = 0; i < remaining; i++) {
            int baseIndex = i % (basePalette.length - 1);
            String startColor = basePalette[baseIndex];
            String endColor = basePalette[baseIndex + 1];
            
            String[] gradient = generateAdvancedGradient(startColor, endColor, gradientSteps, "linear");
            expanded[basePalette.length + i] = gradient[1 + (i % (gradientSteps - 2))];
        }
        
        return expanded;
    }
    
    /**
     * 混合两个颜色（简单实现）
     * @param color1 颜色1
     * @param color2 颜色2
     * @return 混合后的颜色
     */
    private static String mixColors(String color1, String color2) {
        // 简单的颜色混合实现
        // 在实际应用中可以实现更精确的RGB混合算法
        return color1; // 简化实现
    }
    
    /**
     * 获取所有可用的调色板名称
     * @return 调色板名称列表
     */
    public static List<String> getAvailablePalettes() {
        return Arrays.asList(PALETTE_MAP.keySet().toArray(new String[0]));
    }
    
    /**
     * 检查调色板是否存在
     * @param paletteName 调色板名称
     * @return 是否存在
     */
    public static boolean hasPalette(String paletteName) {
        return PALETTE_MAP.containsKey(paletteName.toLowerCase());
    }
    
    /**
     * 创建自定义调色板
     * @param name 调色板名称
     * @param colors 颜色数组
     */
    public static void registerPalette(String name, String[] colors) {
        PALETTE_MAP.put(name.toLowerCase(), colors.clone());
    }
    
    /**
     * 获取对比色（简单实现）
     * @param color 原始颜色
     * @return 对比色
     */
    public static String getContrastColor(String color) {
        // 简单实现：如果是深色返回白色，浅色返回黑色
        if (color == null) return BLACK;
        
        String normalizedColor = parseColor(color);
        
        // 简单的明度判断（在实际应用中可以实现更精确的算法）
        if (normalizedColor.equals(BLACK) || 
            normalizedColor.equals("#000000") ||
            normalizedColor.toLowerCase().contains("dark")) {
            return WHITE;
        } else {
            return BLACK;
        }
    }
}