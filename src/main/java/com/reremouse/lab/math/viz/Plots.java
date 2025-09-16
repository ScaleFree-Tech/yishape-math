package com.reremouse.lab.math.viz;

import java.util.List;
import com.reremouse.lab.math.linalg.IDoubleMatrix;
import com.reremouse.lab.math.linalg.IDoubleVector;

/**
 * 绘图静态工厂类，提供创建各种图表类型的静态方法
 * @author lteb2
 */
public final class Plots {
    
//    private Plots() {
//        // 工具类，防止实例化
//    }
    
    // ========== 基础工厂方法 ==========
    
    /**
     * 创建默认绘图对象
     * @return RerePlot实例
     */
    public static RerePlot of() {
        return new RerePlot();
    }

    /**
     * 创建指定尺寸的绘图对象
     * @param width 图表宽度
     * @param height 图表高度
     * @return RerePlot实例
     */
    public static RerePlot of(int width, int height) {
        return new RerePlot(width, height);
    }

    /**
     * 创建指定尺寸和主题的绘图对象
     * @param width 图表宽度
     * @param height 图表高度
     * @param theme 主题名称
     * @return RerePlot实例
     */
    public static RerePlot of(int width, int height, String theme) {
        return new RerePlot(width, height, theme);
    }
    
    // ========== 图表类型专用工厂方法 ==========
    
}
