package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IMatrix;
import java.time.LocalDateTime;
import java.util.Arrays;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 时间序列分析入口 / Time Series Analysis entry point.
 *
 * <p>创建数据（类似 {@code Linalg.vector} / {@code Linalg.matrix} 模式）:
 * <pre>{@code
 *   IVector<Double> y = Linalg.vector(new double[]{...});
 *   TimeSeriesData ts = TSA.data(y, "price");
 * }</pre>
 *
 * <p>单变量分析:
 * <pre>{@code
 *   ForecastResult f = TSA.forecast.arima(y, 1, 0, 1, 10);
 *   FilterResult flt = TSA.filter.lowPass(y, 0.1, 4);
 *   DecompositionResult d = TSA.decompose.stl(y, 12, 7, 21);
 * }</pre>
 *
 * @author lteb2
 * @version 2.1
 * @since 1.0
 */
public final class TSA {

    private TSA() {
    }

    // ========== 数据创建 / Data creation（Linalg.vector / Linalg.matrix 模式）==========

    public static TimeSeriesData data(IVector<Double> values, String name) {
        return TimeSeriesData.of(values, name);
    }

    public static TimeSeriesData data(double[] values, String name) {
        return TimeSeriesData.of(Linalg.vector(values), name);
    }

    public static TimeSeriesData data(IVector<Double> values, String name, LocalDateTime[] timestamps) {
        return new TimeSeriesData( values, name,Arrays.asList(timestamps));
    }
    
        public static TimeSeriesData data(IMatrix<Double> values, String[] names, LocalDateTime[] timestamps) {
        return new TimeSeriesData( values, names,Arrays.asList(timestamps));
    }

    // ========== 领域分组 / Domain groupings ==========

    /** 预测 / Forecasting */
    public static final ForecastWrapper forecast = new ForecastWrapper();

    /** 滤波 / Filtering */
    public static final FilterWrapper filter = new FilterWrapper();

    /** 分解 / Decomposition */
    public static final DecomposeWrapper decompose = new DecomposeWrapper();

    /** 协整 / Cointegration */
    public static final CointegrateWrapper cointegrate = new CointegrateWrapper();

    /** 可视化 / Plots */
    public static final PlotWrapper plot = new PlotWrapper();
}
