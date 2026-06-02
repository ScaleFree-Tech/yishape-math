/*
 * The MIT License
 *
 * Copyright 2026 lteb2.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.timeseries.model.ITimeSeriesForecastResult;

import java.util.Map;

/**
 * 时间序列分析最小示例 / Minimal time series analysis example.
 *
 * @author lteb2
 */
public class SimpleExample {

    public static void main(String[] args) {
        IVector<Double> timeIndex = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        IVector<Double> values = Linalg.vector(new double[]{10, 12, 13, 15, 18, 20, 22, 25, 28, 30});

        // 1. 创建时间序列数据 / Create time series data
        TimeSeriesData ts = TSA.data(values, "test");

        // 2. 创建分析器 / Create analyzer
        TimeSeriesAnalyzer analyzer = new TimeSeriesAnalyzer(values, "test");

        // 3. 先拟合/选模，再预测 / Fit then forecast
        analyzer.quickAnalyze();
        Map<String, Object> trend = analyzer.getTrendAnalysis();
        double slope = (Double) trend.get("slope");

        ITimeSeriesForecastResult forecastResult = analyzer.forecast(3, 0.95);
        IVector<Double> forecast = forecastResult.getForecastVector();

        // 也可用一步预测门面 / Or one-step forecast facade:
        ForecastResult quick = TSA.forecast.expSmooth(values, 0.3, 3);

        System.out.println("趋势斜率 / trend slope: " + slope);
        System.out.println("预测 / forecast: " + forecast);

        IVector<Double> futureTime = Linalg.vector(new double[]{11, 12, 13});
        Plots.of(800, 400)
                .line(timeIndex, values, "bo-")
                .scatter(futureTime, forecast, "ro-")
                .xlabel("时间 / Time")
                .ylabel("数值 / Value")
                .title("时间序列分析示例 / Time Series Analysis Example")
                .show();
    }
}
