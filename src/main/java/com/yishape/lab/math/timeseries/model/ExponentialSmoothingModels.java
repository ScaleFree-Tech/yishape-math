package com.yishape.lab.math.timeseries.model;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.List;

/**
 * 指数平滑模型类 / Exponential Smoothing Models Class
 * <p>
 * 提供各种指数平滑模型的实现，包括简单指数平滑、双指数平滑、Holt-Winters等。
 * 使用项目现有的linalg包功能进行数值计算。
 * </p>
 * <p>
 * Provides implementation of various exponential smoothing models including simple exponential smoothing,
 * double exponential smoothing, Holt-Winters, etc. Uses existing linalg package functionality for numerical computation.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ExponentialSmoothingModels {
    
    /**
     * 简单指数平滑模型 / Simple Exponential Smoothing Model
     * <p>
     * 适用于无趋势和季节性的平稳时间序列，使用单一平滑参数alpha。
     * Suitable for stationary time series without trend and seasonality, using single smoothing parameter alpha.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class SimpleExponentialSmoothing {
        private final double alpha; // 平滑参数 / Smoothing parameter
        private final IVector<Double> level; // 水平分量 / Level component
        private final double mse; // 均方误差 / Mean squared error
        private final double mae; // 平均绝对误差 / Mean absolute error

        /**
         * 构造函数 / Constructor
         *
         * @param alpha 平滑参数 / Smoothing parameter
         * @param level 水平分量 / Level component
         * @param mse 均方误差 / Mean squared error
         * @param mae 平均绝对误差 / Mean absolute error
         */
        public SimpleExponentialSmoothing(double alpha, IVector<Double> level, double mse, double mae) {
            this.alpha = alpha;
            this.level = level;
            this.mse = mse;
            this.mae = mae;
        }
        
        /**
         * 拟合简单指数平滑模型 / Fit simple exponential smoothing model
         *
         * @param data 输入时间序列 / Input time series
         * @param alpha 平滑参数 / Smoothing parameter
         * @return 拟合的模型 / Fitted model
         */
        public static SimpleExponentialSmoothing fit(IVector<Double> data, double alpha) {
            int n = data.length();
            IVector<Double> level = Linalg.zeros(n);
            
            // 初始化 / Initialize
            level.set(0, data.get(0));
            
            // 递归计算水平分量 / Recursively calculate level component
            for (int i = 1; i < n; i++) {
                level.set(i, alpha * data.get(i) + (1 - alpha) * level.get(i - 1));
            }
            
            // 计算误差 / Calculate errors
            IVector<Double> fitted = level.slice(0, n - 1);
            IVector<Double> actual = data.slice(1, n);
            IVector<Double> errors = actual.sub(fitted);
            
            double mse = errors.multiply(errors).meanValue();
            double mae = errors.apply(Math::abs).meanValue();
            
            return new SimpleExponentialSmoothing(alpha, level, mse, mae);
        }
        
        /**
         * 预测 / Forecast
         *
         * @param steps 预测步数 / Forecast steps
         * @return 预测值 / Forecast values
         */
        public IVector<Double> forecast(int steps) {
            IVector<Double> forecast = Linalg.zeros(steps);
            double lastLevel = level.get(level.length() - 1);
            
            for (int i = 0; i < steps; i++) {
                forecast.set(i, lastLevel);
            }
            
            return forecast;
        }
        
        public double getAlpha() { return alpha; }
        public IVector<Double> getLevel() { return level; }
        public double getMse() { return mse; }
        public double getMae() { return mae; }
    }
    
    /**
     * 双指数平滑模型（Holt方法） / Double Exponential Smoothing Model (Holt's Method)
     * <p>
     * 适用于有趋势但无季节性的时间序列，使用水平和趋势两个平滑参数。
     * Suitable for time series with trend but no seasonality, using two smoothing parameters for level and trend.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class DoubleExponentialSmoothing {
        private final double alpha; // 水平平滑参数 / Level smoothing parameter
        private final double beta;  // 趋势平滑参数 / Trend smoothing parameter
        private final IVector<Double> level; // 水平分量 / Level component
        private final IVector<Double> trend; // 趋势分量 / Trend component
        private final double mse; // 均方误差 / Mean squared error
        private final double mae; // 平均绝对误差 / Mean absolute error

        /**
         * 构造函数 / Constructor
         *
         * @param alpha 水平平滑参数 / Level smoothing parameter
         * @param beta 趋势平滑参数 / Trend smoothing parameter
         * @param level 水平分量 / Level component
         * @param trend 趋势分量 / Trend component
         * @param mse 均方误差 / Mean squared error
         * @param mae 平均绝对误差 / Mean absolute error
         */
        public DoubleExponentialSmoothing(double alpha, double beta, IVector<Double> level,
                                        IVector<Double> trend, double mse, double mae) {
            this.alpha = alpha;
            this.beta = beta;
            this.level = level;
            this.trend = trend;
            this.mse = mse;
            this.mae = mae;
        }
        
        /**
         * 拟合双指数平滑模型 / Fit double exponential smoothing model
         *
         * @param data 输入时间序列 / Input time series
         * @param alpha 水平平滑参数 / Level smoothing parameter
         * @param beta 趋势平滑参数 / Trend smoothing parameter
         * @return 拟合的模型 / Fitted model
         */
        public static DoubleExponentialSmoothing fit(IVector<Double> data, double alpha, double beta) {
            int n = data.length();
            IVector<Double> level = Linalg.zeros(n);
            IVector<Double> trend = Linalg.zeros(n);
            
            // 初始化 / Initialize
            level.set(0, data.get(0));
            trend.set(0, data.get(1) - data.get(0));
            
            // 递归计算水平和趋势分量 / Recursively calculate level and trend components
            for (int i = 1; i < n; i++) {
                double prevLevel = level.get(i - 1);
                double prevTrend = trend.get(i - 1);
                
                level.set(i, alpha * data.get(i) + (1 - alpha) * (prevLevel + prevTrend));
                trend.set(i, beta * (level.get(i) - prevLevel) + (1 - beta) * prevTrend);
            }
            
            // 计算误差 / Calculate errors
            IVector<Double> fitted = Linalg.zeros(n - 1);
            for (int i = 1; i < n; i++) {
                fitted.set(i - 1, level.get(i - 1) + trend.get(i - 1));
            }
            
            IVector<Double> actual = data.slice(1, n);
            IVector<Double> errors = actual.sub(fitted);
            
            double mse = errors.multiply(errors).meanValue();
            double mae = errors.apply(Math::abs).meanValue();
            
            return new DoubleExponentialSmoothing(alpha, beta, level, trend, mse, mae);
        }
        
        /**
         * 预测 / Forecast
         *
         * @param steps 预测步数 / Forecast steps
         * @return 预测值 / Forecast values
         */
        public IVector<Double> forecast(int steps) {
            IVector<Double> forecast = Linalg.zeros(steps);
            double lastLevel = level.get(level.length() - 1);
            double lastTrend = trend.get(trend.length() - 1);
            
            for (int i = 0; i < steps; i++) {
                forecast.set(i, lastLevel + (i + 1) * lastTrend);
            }
            
            return forecast;
        }
        
        public double getAlpha() { return alpha; }
        public double getBeta() { return beta; }
        public IVector<Double> getLevel() { return level; }
        public IVector<Double> getTrend() { return trend; }
        public double getMse() { return mse; }
        public double getMae() { return mae; }
    }
    
    /**
     * Holt-Winters三参数指数平滑模型 / Holt-Winters Triple Exponential Smoothing Model
     * <p>
     * 适用于有趋势和季节性的时间序列，使用水平、趋势和季节性三个平滑参数。
     * Suitable for time series with trend and seasonality, using three smoothing parameters for level, trend, and seasonal components.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class HoltWintersSmoothing {
        private final double alpha; // 水平平滑参数 / Level smoothing parameter
        private final double beta;  // 趋势平滑参数 / Trend smoothing parameter
        private final double gamma; // 季节性平滑参数 / Seasonal smoothing parameter
        private final int period;   // 季节周期 / Seasonal period
        private final IVector<Double> level;     // 水平分量 / Level component
        private final IVector<Double> trend;     // 趋势分量 / Trend component
        private final IVector<Double> seasonal;  // 季节性分量 / Seasonal component
        private final double mse; // 均方误差 / Mean squared error
        private final double mae; // 平均绝对误差 / Mean absolute error

        /**
         * 构造函数 / Constructor
         *
         * @param alpha 水平平滑参数 / Level smoothing parameter
         * @param beta 趋势平滑参数 / Trend smoothing parameter
         * @param gamma 季节性平滑参数 / Seasonal smoothing parameter
         * @param period 季节周期 / Seasonal period
         * @param level 水平分量 / Level component
         * @param trend 趋势分量 / Trend component
         * @param seasonal 季节性分量 / Seasonal component
         * @param mse 均方误差 / Mean squared error
         * @param mae 平均绝对误差 / Mean absolute error
         */
        public HoltWintersSmoothing(double alpha, double beta, double gamma, int period,
                                  IVector<Double> level, IVector<Double> trend, IVector<Double> seasonal,
                                  double mse, double mae) {
            this.alpha = alpha;
            this.beta = beta;
            this.gamma = gamma;
            this.period = period;
            this.level = level;
            this.trend = trend;
            this.seasonal = seasonal;
            this.mse = mse;
            this.mae = mae;
        }
        
        /**
         * 拟合Holt-Winters模型 / Fit Holt-Winters model
         *
         * @param data 输入时间序列 / Input time series
         * @param alpha 水平平滑参数 / Level smoothing parameter
         * @param beta 趋势平滑参数 / Trend smoothing parameter
         * @param gamma 季节性平滑参数 / Seasonal smoothing parameter
         * @param period 季节周期 / Seasonal period
         * @return 拟合的模型 / Fitted model
         */
        public static HoltWintersSmoothing fit(IVector<Double> data, double alpha, double beta, 
                                             double gamma, int period) {
            int n = data.length();
            if (n < 2 * period) {
                throw new IllegalArgumentException("数据长度不足以拟合Holt-Winters模型");
            }
            
            IVector<Double> level = Linalg.zeros(n);
            IVector<Double> trend = Linalg.zeros(n);
            IVector<Double> seasonal = Linalg.zeros(n);
            
            // 初始化季节性分量 / Initialize seasonal component
            for (int i = 0; i < period; i++) {
                double sum = 0.0;
                int count = 0;
                for (int j = i; j < n; j += period) {
                    sum += data.get(j);
                    count++;
                }
                double avg = sum / count;
                for (int j = i; j < n; j += period) {
                    seasonal.set(j, data.get(j) - avg);
                }
            }
            
            // 初始化水平和趋势分量 / Initialize level and trend components
            level.set(0, data.get(0) - seasonal.get(0));
            trend.set(0, (data.get(period) - data.get(0)) / period);
            
            // 递归计算各分量 / Recursively calculate components
            for (int i = 1; i < n; i++) {
                double prevLevel = level.get(i - 1);
                double prevTrend = trend.get(i - 1);
                double prevSeasonal = seasonal.get(i - period >= 0 ? i - period : i);
                
                level.set(i, alpha * (data.get(i) - prevSeasonal) + (1 - alpha) * (prevLevel + prevTrend));
                trend.set(i, beta * (level.get(i) - prevLevel) + (1 - beta) * prevTrend);
                seasonal.set(i, gamma * (data.get(i) - level.get(i)) + (1 - gamma) * prevSeasonal);
            }
            
            // 计算误差 / Calculate errors
            IVector<Double> fitted = Linalg.zeros(n - period);
            for (int i = period; i < n; i++) {
                fitted.set(i - period, level.get(i - 1) + trend.get(i - 1) + seasonal.get(i - period));
            }
            
            IVector<Double> actual = data.slice(period, n);
            IVector<Double> errors = actual.sub(fitted);
            
            double mse = errors.multiply(errors).meanValue();
            double mae = errors.apply(Math::abs).meanValue();
            
            return new HoltWintersSmoothing(alpha, beta, gamma, period, level, trend, seasonal, mse, mae);
        }
        
        /**
         * 预测 / Forecast
         *
         * @param steps 预测步数 / Forecast steps
         * @return 预测值 / Forecast values
         */
        public IVector<Double> forecast(int steps) {
            IVector<Double> forecast = Linalg.zeros(steps);
            double lastLevel = level.get(level.length() - 1);
            double lastTrend = trend.get(trend.length() - 1);
            
            for (int i = 0; i < steps; i++) {
                int seasonalIndex = (level.length() - period + i) % period;
                double seasonalValue = seasonal.get(level.length() - period + seasonalIndex);
                forecast.set(i, lastLevel + (i + 1) * lastTrend + seasonalValue);
            }
            
            return forecast;
        }
        
        public double getAlpha() { return alpha; }
        public double getBeta() { return beta; }
        public double getGamma() { return gamma; }
        public int getPeriod() { return period; }
        public IVector<Double> getLevel() { return level; }
        public IVector<Double> getTrend() { return trend; }
        public IVector<Double> getSeasonal() { return seasonal; }
        public double getMse() { return mse; }
        public double getMae() { return mae; }
    }
    
    /**
     * 自适应指数平滑模型 / Adaptive Exponential Smoothing Model
     * <p>
     * 平滑参数随时间自适应调整，适用于非平稳时间序列。
     * Smoothing parameter adjusts adaptively over time, suitable for non-stationary time series.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class AdaptiveExponentialSmoothing {
        private final double initialAlpha; // 初始平滑参数 / Initial smoothing parameter
        private final IVector<Double> alpha; // 自适应平滑参数序列 / Adaptive smoothing parameter sequence
        private final IVector<Double> level; // 水平分量 / Level component
        private final double mse; // 均方误差 / Mean squared error
        private final double mae; // 平均绝对误差 / Mean absolute error

        /**
         * 构造函数 / Constructor
         *
         * @param initialAlpha 初始平滑参数 / Initial smoothing parameter
         * @param alpha 自适应平滑参数序列 / Adaptive smoothing parameter sequence
         * @param level 水平分量 / Level component
         * @param mse 均方误差 / Mean squared error
         * @param mae 平均绝对误差 / Mean absolute error
         */
        public AdaptiveExponentialSmoothing(double initialAlpha, IVector<Double> alpha, 
                                          IVector<Double> level, double mse, double mae) {
            this.initialAlpha = initialAlpha;
            this.alpha = alpha;
            this.level = level;
            this.mse = mse;
            this.mae = mae;
        }
        
        /**
         * 拟合自适应指数平滑模型 / Fit adaptive exponential smoothing model
         *
         * @param data 输入时间序列 / Input time series
         * @param initialAlpha 初始平滑参数 / Initial smoothing parameter
         * @param adaptationRate 自适应率 / Adaptation rate
         * @return 拟合的模型 / Fitted model
         */
        public static AdaptiveExponentialSmoothing fit(IVector<Double> data, double initialAlpha, double adaptationRate) {
            int n = data.length();
            IVector<Double> alpha = Linalg.zeros(n);
            IVector<Double> level = Linalg.zeros(n);
            
            // 初始化 / Initialize
            alpha.set(0, initialAlpha);
            level.set(0, data.get(0));
            
            // 递归计算 / Recursively calculate
            for (int i = 1; i < n; i++) {
                // 计算预测误差 / Calculate forecast error
                double error = data.get(i) - level.get(i - 1);
                
                // 自适应调整平滑参数 / Adaptively adjust smoothing parameter
                double newAlpha = alpha.get(i - 1) + adaptationRate * error * (data.get(i) - level.get(i - 1));
                newAlpha = Math.max(0.01, Math.min(0.99, newAlpha)); // 限制在合理范围内 / Limit to reasonable range
                alpha.set(i, newAlpha);
                
                // 更新水平分量 / Update level component
                level.set(i, newAlpha * data.get(i) + (1 - newAlpha) * level.get(i - 1));
            }
            
            // 计算误差 / Calculate errors
            IVector<Double> fitted = level.slice(0, n - 1);
            IVector<Double> actual = data.slice(1, n);
            IVector<Double> errors = actual.sub(fitted);
            
            double mse = errors.multiply(errors).meanValue();
            double mae = errors.apply(Math::abs).meanValue();
            
            return new AdaptiveExponentialSmoothing(initialAlpha, alpha, level, mse, mae);
        }
        
        /**
         * 预测 / Forecast
         *
         * @param steps 预测步数 / Forecast steps
         * @return 预测值 / Forecast values
         */
        public IVector<Double> forecast(int steps) {
            IVector<Double> forecast = Linalg.zeros(steps);
            double lastLevel = level.get(level.length() - 1);
            double lastAlpha = alpha.get(alpha.length() - 1);
            
            for (int i = 0; i < steps; i++) {
                forecast.set(i, lastLevel);
            }
            
            return forecast;
        }
        
        public double getInitialAlpha() { return initialAlpha; }
        public IVector<Double> getAlpha() { return alpha; }
        public IVector<Double> getLevel() { return level; }
        public double getMse() { return mse; }
        public double getMae() { return mae; }
    }
    
    /**
     * 指数平滑模型选择器 / Exponential Smoothing Model Selector
     * <p>
     * 自动搜索并选择最优的指数平滑模型，包括简单、双和Holt-Winters模型。
     * Automatically searches and selects optimal exponential smoothing model including simple, double, and Holt-Winters models.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class ModelSelector {
        
        /**
         * 自动选择最优指数平滑模型 / Automatically select optimal exponential smoothing model
         *
         * @param data 输入时间序列 / Input time series
         * @param maxPeriod 最大季节周期 / Maximum seasonal period
         * @return 最优模型信息 / Optimal model information
         */
        public static ModelSelectionResult selectBestModel(IVector<Double> data, int maxPeriod) {
            List<ModelCandidate> candidates = new ArrayList<>();
            
            // 简单指数平滑 / Simple exponential smoothing
            for (double alpha = 0.1; alpha <= 0.9; alpha += 0.1) {
                try {
                    SimpleExponentialSmoothing model = SimpleExponentialSmoothing.fit(data, alpha);
                    candidates.add(new ModelCandidate("Simple", alpha, 0, 0, 0, model.getMse(), model.getMae()));
                } catch (Exception e) {
                    // 跳过无效参数 / Skip invalid parameters
                }
            }
            
            // 双指数平滑 / Double exponential smoothing
            for (double alpha = 0.1; alpha <= 0.9; alpha += 0.1) {
                for (double beta = 0.1; beta <= 0.9; beta += 0.1) {
                    try {
                        DoubleExponentialSmoothing model = DoubleExponentialSmoothing.fit(data, alpha, beta);
                        candidates.add(new ModelCandidate("Double", alpha, beta, 0, 0, model.getMse(), model.getMae()));
                    } catch (Exception e) {
                        // 跳过无效参数 / Skip invalid parameters
                    }
                }
            }
            
            // Holt-Winters模型 / Holt-Winters model
            for (int period = 2; period <= Math.min(maxPeriod, data.length() / 2); period++) {
                for (double alpha = 0.1; alpha <= 0.9; alpha += 0.2) {
                    for (double beta = 0.1; beta <= 0.9; beta += 0.2) {
                        for (double gamma = 0.1; gamma <= 0.9; gamma += 0.2) {
                            try {
                                HoltWintersSmoothing model = HoltWintersSmoothing.fit(data, alpha, beta, gamma, period);
                                candidates.add(new ModelCandidate("HoltWinters", alpha, beta, gamma, period, model.getMse(), model.getMae()));
                            } catch (Exception e) {
                                // 跳过无效参数 / Skip invalid parameters
                            }
                        }
                    }
                }
            }
            
            // 选择MSE最小的模型 / Select model with minimum MSE
            ModelCandidate bestModel = candidates.stream()
                .min((a, b) -> Double.compare(a.mse, b.mse))
                .orElse(null);
            
            if (bestModel == null) {
                throw new RuntimeException("无法找到合适的指数平滑模型");
            }
            
            return new ModelSelectionResult(bestModel, candidates);
        }
    }
    
    // ========== 结果类 / Result Classes ==========
    
    /**
     * 模型候选类 / Model Candidate Class
     * <p>
     * 存储指数平滑模型的候选信息，包括模型类型和误差指标。
     * Stores candidate information for exponential smoothing models including model type and error metrics.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class ModelCandidate {
        public final String type;
        public final double alpha;
        public final double beta;
        public final double gamma;
        public final int period;
        public final double mse;
        public final double mae;

        /**
         * 构造函数 / Constructor
         *
         * @param type 模型类型 / Model type
         * @param alpha 水平平滑参数 / Level smoothing parameter
         * @param beta 趋势平滑参数 / Trend smoothing parameter
         * @param gamma 季节性平滑参数 / Seasonal smoothing parameter
         * @param period 季节周期 / Seasonal period
         * @param mse 均方误差 / Mean squared error
         * @param mae 平均绝对误差 / Mean absolute error
         */
        public ModelCandidate(String type, double alpha, double beta, double gamma, int period, double mse, double mae) {
            this.type = type;
            this.alpha = alpha;
            this.beta = beta;
            this.gamma = gamma;
            this.period = period;
            this.mse = mse;
            this.mae = mae;
        }
    }
    
    /**
     * 模型选择结果类 / Model Selection Result Class
     * <p>
     * 存储模型选择的结果，包括最优模型和所有候选模型。
     * Stores model selection result including optimal model and all candidates.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class ModelSelectionResult {
        public final ModelCandidate bestModel;
        public final List<ModelCandidate> allCandidates;

        /**
         * 构造函数 / Constructor
         *
         * @param bestModel 最优模型 / Best model
         * @param allCandidates 所有候选模型列表 / List of all candidate models
         */
        public ModelSelectionResult(ModelCandidate bestModel, List<ModelCandidate> allCandidates) {
            this.bestModel = bestModel;
            this.allCandidates = allCandidates;
        }
    }
}
