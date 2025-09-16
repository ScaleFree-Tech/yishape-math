package com.reremouse.lab.math.viz;

import java.io.Serializable;
import java.util.*;
import com.reremouse.lab.math.linalg.IVector;

/**
 * Seaborn风格的分组样式管理器
 * 提供类似seaborn的hue, style, size等分组映射功能
 * 
 * @author lteb2
 */
public class SeabornStyleMapper implements Serializable {
    
    // 默认调色板序列
    private static final String[] DEFAULT_HUE_PALETTE = {
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    };
    
    // 默认线条样式序列
    private static final String[] DEFAULT_STYLE_SEQUENCE = {
        "solid", "dashed", "dotted", "dashdot"
    };
    
    // 默认标记样式序列
    private static final String[] DEFAULT_MARKER_SEQUENCE = {
        "o", "s", "^", "v", "<", ">", "d", "*", "+", "x"
    };
    
    // 默认尺寸序列
    private static final double[] DEFAULT_SIZE_SEQUENCE = {
        6.0f, 8.0f, 10.0f, 12.0f, 14.0f
    };
    
    private String[] huePalette;
    private String[] styleSequence;
    private String[] markerSequence;
    private double[] sizeSequence;
    
    // 映射缓存
    private final Map<String, String> hueColorMap = new HashMap<>();
    private final Map<String, String> styleMap = new HashMap<>();
    private final Map<String, String> markerMap = new HashMap<>();
    private final Map<String, Double> sizeMap = new HashMap<>();
    
    /**
     * 默认构造函数
     */
    public SeabornStyleMapper() {
        this.huePalette = DEFAULT_HUE_PALETTE.clone();
        this.styleSequence = DEFAULT_STYLE_SEQUENCE.clone();
        this.markerSequence = DEFAULT_MARKER_SEQUENCE.clone();
        this.sizeSequence = DEFAULT_SIZE_SEQUENCE.clone();
    }
    
    /**
     * 设置色调调色板
     * @param palette 调色板名称或颜色数组
     * @return 当前实例
     */
    public SeabornStyleMapper setHuePalette(String palette) {
        this.huePalette = ColorPalette.getPalette(palette);
        clearHueCache();
        return this;
    }
    
    /**
     * 设置色调调色板
     * @param colors 颜色数组
     * @return 当前实例
     */
    public SeabornStyleMapper setHuePalette(String[] colors) {
        this.huePalette = colors.clone();
        clearHueCache();
        return this;
    }
    
    /**
     * 设置样式序列
     * @param styles 样式数组
     * @return 当前实例
     */
    public SeabornStyleMapper setStyleSequence(String[] styles) {
        this.styleSequence = styles.clone();
        clearStyleCache();
        return this;
    }
    
    /**
     * 设置标记序列
     * @param markers 标记数组
     * @return 当前实例
     */
    public SeabornStyleMapper setMarkerSequence(String[] markers) {
        this.markerSequence = markers.clone();
        clearMarkerCache();
        return this;
    }
    
    /**
     * 设置尺寸序列
     * @param sizes 尺寸数组
     * @return 当前实例
     */
    public SeabornStyleMapper setSizeSequence(double[] sizes) {
        this.sizeSequence = sizes.clone();
        clearSizeCache();
        return this;
    }
    
    /**
     * 为分组数据生成样式映射
     * @param hue 色调分组（可选）
     * @param style 样式分组（可选）
     * @param size 尺寸分组（可选）
     * @param marker 标记分组（可选）
     * @return 样式映射结果
     */
    public GroupStyleMapping createMapping(List<String> hue, List<String> style, 
                                         List<String> size, List<String> marker) {
        GroupStyleMapping mapping = new GroupStyleMapping();
        
        // 处理色调映射
        if (hue != null) {
            Set<String> uniqueHues = new LinkedHashSet<>(hue);
            int i = 0;
            for (String h : uniqueHues) {
                if (!hueColorMap.containsKey(h)) {
                    hueColorMap.put(h, huePalette[i % huePalette.length]);
                    i++;
                }
            }
            mapping.setHueMapping(new HashMap<>(hueColorMap));
        }
        
        // 处理样式映射
        if (style != null) {
            Set<String> uniqueStyles = new LinkedHashSet<>(style);
            int i = 0;
            for (String s : uniqueStyles) {
                if (!styleMap.containsKey(s)) {
                    styleMap.put(s, styleSequence[i % styleSequence.length]);
                    i++;
                }
            }
            mapping.setStyleMapping(new HashMap<>(styleMap));
        }
        
        // 处理标记映射
        if (marker != null) {
            Set<String> uniqueMarkers = new LinkedHashSet<>(marker);
            int i = 0;
            for (String m : uniqueMarkers) {
                if (!markerMap.containsKey(m)) {
                    markerMap.put(m, markerSequence[i % markerSequence.length]);
                    i++;
                }
            }
            mapping.setMarkerMapping(new HashMap<>(markerMap));
        }
        
        // 处理尺寸映射
        if (size != null) {
            Set<String> uniqueSizes = new LinkedHashSet<>(size);
            int i = 0;
            for (String sz : uniqueSizes) {
                if (!sizeMap.containsKey(sz)) {
                    sizeMap.put(sz, sizeSequence[i % sizeSequence.length]);
                    i++;
                }
            }
            mapping.setSizeMapping(new HashMap<>(sizeMap));
        }
        
        return mapping;
    }
    
    /**
     * 为特定数据点生成PlotStyle
     * @param mapping 样式映射
     * @param hueValue 色调值
     * @param styleValue 样式值
     * @param sizeValue 尺寸值
     * @param markerValue 标记值
     * @return 生成的样式
     */
    public PlotStyle createStyle(GroupStyleMapping mapping, String hueValue, 
                               String styleValue, String sizeValue, String markerValue) {
        PlotStyle style = PlotStyle.defaultStyle();
        
        // 应用色调
        if (hueValue != null && mapping.getHueMapping() != null) {
            String color = mapping.getHueMapping().get(hueValue);
            if (color != null) {
                style.color(color);
            }
        }
        
        // 应用样式
        if (styleValue != null && mapping.getStyleMapping() != null) {
            String lineStyle = mapping.getStyleMapping().get(styleValue);
            if (lineStyle != null) {
                style.lineStyle(lineStyle);
            }
        }
        
        // 应用标记
        if (markerValue != null && mapping.getMarkerMapping() != null) {
            String marker = mapping.getMarkerMapping().get(markerValue);
            if (marker != null) {
                style.marker(marker);
            }
        }
        
        // 应用尺寸
        if (sizeValue != null && mapping.getSizeMapping() != null) {
            Double size = mapping.getSizeMapping().get(sizeValue);
            if (size != null) {
                style.markerSize(size).lineWidth(size / 3.0);
            }
        }
        
        return style;
    }
    
    /**
     * 数据分组辅助方法
     * @param x X轴数据
     * @param y Y轴数据
     * @param hue 分组标签
     * @param mapping 样式映射
     * @return 分组数据
     */
    public Map<String, GroupedData> groupData(IVector x, IVector y, List<String> hue, 
                                            GroupStyleMapping mapping) {
        if (x.length() != y.length() || (hue != null && x.length() != hue.size())) {
            throw new IllegalArgumentException("数据长度不匹配");
        }
        
        Map<String, GroupedData> groups = new HashMap<>();
        
        for (int i = 0; i < x.length(); i++) {
            String groupKey = hue != null ? hue.get(i) : "default";
            
            if (!groups.containsKey(groupKey)) {
                PlotStyle style = createStyle(mapping, groupKey, null, null, null);
                style.label(groupKey);
                groups.put(groupKey, new GroupedData(groupKey, style));
            }
            
            groups.get(groupKey).addPoint((double)x.get(i), (double)y.get(i));
        }
        
        return groups;
    }
    
    // 清理缓存方法
    private void clearHueCache() { hueColorMap.clear(); }
    private void clearStyleCache() { styleMap.clear(); }
    private void clearMarkerCache() { markerMap.clear(); }
    private void clearSizeCache() { sizeMap.clear(); }
    
    /**
     * 样式映射结果类
     */
    public static class GroupStyleMapping implements Serializable {
        private Map<String, String> hueMapping;
        private Map<String, String> styleMapping;
        private Map<String, String> markerMapping;
        private Map<String, Double> sizeMapping;
        
        // Getters and Setters
        public Map<String, String> getHueMapping() { return hueMapping; }
        public void setHueMapping(Map<String, String> hueMapping) { this.hueMapping = hueMapping; }
        
        public Map<String, String> getStyleMapping() { return styleMapping; }
        public void setStyleMapping(Map<String, String> styleMapping) { this.styleMapping = styleMapping; }
        
        public Map<String, String> getMarkerMapping() { return markerMapping; }
        public void setMarkerMapping(Map<String, String> markerMapping) { this.markerMapping = markerMapping; }
        
        public Map<String, Double> getSizeMapping() { return sizeMapping; }
        public void setSizeMapping(Map<String, Double> sizeMapping) { this.sizeMapping = sizeMapping; }
    }
    
    /**
     * 分组数据类
     */
    public static class GroupedData implements Serializable {
        private final String groupName;
        private final PlotStyle style;
        private final List<Object[]> data;
        
        public GroupedData(String groupName, PlotStyle style) {
            this.groupName = groupName;
            this.style = style;
            this.data = new ArrayList<>();
        }
        
        public void addPoint(double x, double y) {
            data.add(new Number[]{x, y});
        }
        
        public String getGroupName() { return groupName; }
        public PlotStyle getStyle() { return style; }
        public Object[] getData() { return data.toArray(new Object[0]); }
        public int size() { return data.size(); }
    }
}