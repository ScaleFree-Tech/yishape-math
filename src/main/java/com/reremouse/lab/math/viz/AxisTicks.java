package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.IVector;
import java.util.List;
import java.util.ArrayList;

/**
 * 坐标轴刻度配置类
 * 用于配置图表的X轴和Y轴刻度
 * @author lteb2
 */
public class AxisTicks {
    
    private IVector tickValues;
    private List<String> tickLabels;
    
    /**
     * 默认构造函数
     */
    public AxisTicks() {
        this.tickValues = null;
        this.tickLabels = new ArrayList<>();
    }
    
    /**
     * 带刻度值的构造函数
     * @param tickValues 刻度值向量
     */
    public AxisTicks(IVector tickValues) {
        this.tickValues = tickValues;
        this.tickLabels = new ArrayList<>();
    }
    
    /**
     * 带刻度值和标签的构造函数
     * @param tickValues 刻度值向量
     * @param tickLabels 刻度标签列表
     */
    public AxisTicks(IVector tickValues, List<String> tickLabels) {
        this.tickValues = tickValues;
        this.tickLabels = tickLabels != null ? new ArrayList<>(tickLabels) : new ArrayList<>();
    }
    
    /**
     * 获取刻度值
     * @return 刻度值向量
     */
    public IVector getTickValues() {
        return tickValues;
    }
    
    /**
     * 设置刻度值
     * @param tickValues 刻度值向量
     */
    public void setTickValues(IVector tickValues) {
        this.tickValues = tickValues;
    }
    
    /**
     * 获取刻度标签
     * @return 刻度标签列表
     */
    public List<String> getTickLabels() {
        return tickLabels;
    }
    
    /**
     * 设置刻度标签
     * @param tickLabels 刻度标签列表
     */
    public void setTickLabels(List<String> tickLabels) {
        this.tickLabels = tickLabels != null ? new ArrayList<>(tickLabels) : new ArrayList<>();
    }
    
    /**
     * 添加刻度标签
     * @param label 标签
     */
    public void addTickLabel(String label) {
        if (this.tickLabels == null) {
            this.tickLabels = new ArrayList<>();
        }
        this.tickLabels.add(label);
    }
    
    /**
     * 检查是否有刻度值
     * @return 是否有刻度值
     */
    public boolean hasTickValues() {
        return tickValues != null && tickValues.length() > 0;
    }
    
    /**
     * 检查是否有刻度标签
     * @return 是否有刻度标签
     */
    public boolean hasTickLabels() {
        return tickLabels != null && !tickLabels.isEmpty();
    }
}
