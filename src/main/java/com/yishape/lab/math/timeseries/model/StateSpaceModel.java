package com.yishape.lab.math.timeseries.model;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * 状态空间模型实现类 / State Space Model Implementation Class
 * <p>
 * 提供状态空间模型的实现，包括Kalman滤波、状态估计、预测等。
 * 使用项目现有的linalg包功能进行数值计算。
 * </p>
 * <p>
 * Provides state space model implementation including Kalman filtering, state estimation, forecasting, etc.
 * Uses existing linalg package functionality for numerical computation.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class StateSpaceModel {
    
    private final IMatrix<Double> F; // 状态转移矩阵 / State transition matrix
    private final IMatrix<Double> H; // 观测矩阵 / Observation matrix
    private final IMatrix<Double> Q; // 过程噪声协方差矩阵 / Process noise covariance matrix
    private final IMatrix<Double> R; // 观测噪声协方差矩阵 / Observation noise covariance matrix
    private final IVector<Double> x0; // 初始状态 / Initial state
    private final IMatrix<Double> P0; // 初始状态协方差 / Initial state covariance
    private final List<IVector<Double>> filteredStates; // 滤波状态 / Filtered states
    private final List<IMatrix<Double>> filteredCovariances; // 滤波协方差 / Filtered covariances
    private final List<IVector<Double>> predictedStates; // 预测状态 / Predicted states
    private final List<IMatrix<Double>> predictedCovariances; // 预测协方差 / Predicted covariances
    private final List<IVector<Double>> innovations; // 新息 / Innovations
    private final List<Double> logLikelihoods; // 对数似然 / Log likelihoods
    
    /**
     * 构造函数 / Constructor
     *
     * @param F 状态转移矩阵 / State transition matrix
     * @param H 观测矩阵 / Observation matrix
     * @param Q 过程噪声协方差矩阵 / Process noise covariance matrix
     * @param R 观测噪声协方差矩阵 / Observation noise covariance matrix
     * @param x0 初始状态 / Initial state
     * @param P0 初始状态协方差 / Initial state covariance
     */
    public StateSpaceModel(IMatrix<Double> F, IMatrix<Double> H, IMatrix<Double> Q, IMatrix<Double> R,
                          IVector<Double> x0, IMatrix<Double> P0) {
        this.F = F;
        this.H = H;
        this.Q = Q;
        this.R = R;
        this.x0 = x0;
        this.P0 = P0;
        this.filteredStates = new ArrayList<>();
        this.filteredCovariances = new ArrayList<>();
        this.predictedStates = new ArrayList<>();
        this.predictedCovariances = new ArrayList<>();
        this.innovations = new ArrayList<>();
        this.logLikelihoods = new ArrayList<>();
    }
    
    /**
     * 运行Kalman滤波 / Run Kalman Filter
     * <p>
     * 对观测序列运行Kalman滤波算法。
     * Run Kalman filter algorithm on observation sequence.
     * </p>
     *
     * @param observations 观测序列 / Observation sequence
     * @return 滤波结果 / Filtering results
     */
    public KalmanFilterResult runKalmanFilter(IVector<Double> observations) {
        int n = observations.length();
        int stateDim = x0.length();
        int obsDim = H.getRowNum();
        
        // 初始化 / Initialize
        IVector<Double> x = x0.copy();
        IMatrix<Double> P = P0.copy();
        
        // 存储结果 / Store results
        List<IVector<Double>> filteredStates = new ArrayList<>();
        List<IMatrix<Double>> filteredCovariances = new ArrayList<>();
        List<IVector<Double>> predictedStates = new ArrayList<>();
        List<IMatrix<Double>> predictedCovariances = new ArrayList<>();
        List<IVector<Double>> innovations = new ArrayList<>();
        List<Double> logLikelihoods = new ArrayList<>();
        
        for (int t = 0; t < n; t++) {
            // 预测步骤 / Prediction step
            IVector<Double> xPred = F.mmul(x);
            IMatrix<Double> PPred = F.mmul(P).mmul(F.transpose()).add(Q);
            
            // 更新步骤 / Update step
            IVector<Double> y = Linalg.vector(new double[]{observations.get(t)});
            IVector<Double> yPred = H.mmul(xPred);
            IVector<Double> innovation = y.sub(yPred);
            
            IMatrix<Double> S = H.mmul(PPred).mmul(H.transpose()).add(R);
            IMatrix<Double> K = PPred.mmul(H.transpose()).mmul(S.pinv());
            
            x = xPred.add(K.mmul(innovation));
            IMatrix<Double> I = Linalg.eye(stateDim);
            P = I.sub(K.mmul(H)).mmul(PPred);
            
            // 计算对数似然 / Calculate log likelihood
            double logLikelihood = calculateLogLikelihood(innovation, S);
            
            // 存储结果 / Store results
            filteredStates.add(x.copy());
            filteredCovariances.add(P.copy());
            predictedStates.add(xPred.copy());
            predictedCovariances.add(PPred.copy());
            innovations.add(innovation.copy());
            logLikelihoods.add(logLikelihood);
        }
        
        return new KalmanFilterResult(filteredStates, filteredCovariances, predictedStates, 
                                    predictedCovariances, innovations, logLikelihoods);
    }
    
    /**
     * 预测未来状态 / Predict Future States
     * <p>
     * 使用滤波后的状态预测未来的状态。
     * Use filtered states to predict future states.
     * </p>
     *
     * @param steps 预测步数 / Forecast steps
     * @return 预测结果 / Forecast results
     */
    public StateSpaceForecastResult forecast(int steps) {
        if (filteredStates.isEmpty()) {
            throw new IllegalStateException("必须先运行Kalman滤波");
        }
        
        int stateDim = x0.length();
        IVector<Double> currentState = filteredStates.get(filteredStates.size() - 1);
        IMatrix<Double> currentCovariance = filteredCovariances.get(filteredCovariances.size() - 1);
        
        List<IVector<Double>> forecastStates = new ArrayList<>();
        List<IMatrix<Double>> forecastCovariances = new ArrayList<>();
        List<IVector<Double>> forecastObservations = new ArrayList<>();
        List<IVector<Double>> forecastStd = new ArrayList<>();
        
        IVector<Double> state = currentState.copy();
        IMatrix<Double> covariance = currentCovariance.copy();
        
        for (int t = 0; t < steps; t++) {
            // 状态预测 / State prediction
            state = F.mmul(state);
            covariance = F.mmul(covariance).mmul(F.transpose()).add(Q);
            
            // 观测预测 / Observation prediction
            IVector<Double> obs = H.mmul(state);
            IMatrix<Double> obsCovariance = H.mmul(covariance).mmul(H.transpose()).add(R);
            IVector<Double> obsStd = Linalg.vector(new double[]{Math.sqrt(obsCovariance.get(0, 0))});
            
            // 存储结果 / Store results
            forecastStates.add(state.copy());
            forecastCovariances.add(covariance.copy());
            forecastObservations.add(obs.copy());
            forecastStd.add(obsStd.copy());
        }
        
        return new StateSpaceForecastResult(forecastStates, forecastCovariances, 
                                          forecastObservations, forecastStd);
    }
    
    /**
     * 平滑 / Smoothing
     * <p>
     * 使用Rauch-Tung-Striebel平滑算法对状态进行平滑。
     * Use Rauch-Tung-Striebel smoothing algorithm to smooth states.
     * </p>
     *
     * @return 平滑结果 / Smoothing results
     */
    public StateSpaceSmoothResult smooth() {
        if (filteredStates.isEmpty()) {
            throw new IllegalStateException("必须先运行Kalman滤波");
        }
        
        int n = filteredStates.size();
        int stateDim = x0.length();
        
        List<IVector<Double>> smoothedStates = new ArrayList<>();
        List<IMatrix<Double>> smoothedCovariances = new ArrayList<>();
        
        // 初始化 / Initialize
        for (int i = 0; i < n; i++) {
            smoothedStates.add(filteredStates.get(i).copy());
            smoothedCovariances.add(filteredCovariances.get(i).copy());
        }
        
        // 反向平滑 / Backward smoothing
        for (int t = n - 2; t >= 0; t--) {
            IVector<Double> xFiltered = filteredStates.get(t);
            IMatrix<Double> PFiltered = filteredCovariances.get(t);
            IVector<Double> xPredicted = predictedStates.get(t + 1);
            IMatrix<Double> PPredicted = predictedCovariances.get(t + 1);
            IVector<Double> xSmoothedNext = smoothedStates.get(t + 1);
            IMatrix<Double> PSmoothedNext = smoothedCovariances.get(t + 1);
            
            // 计算平滑增益 / Calculate smoothing gain
            IMatrix<Double> C = PFiltered.mmul(F.transpose()).mmul(PPredicted.pinv());
            
            // 平滑状态和协方差 / Smooth state and covariance
            IVector<Double> xSmoothed = xFiltered.add(C.mmul(xSmoothedNext.sub(xPredicted)));
            IMatrix<Double> PSmoothed = PFiltered.add(C.mmul(PSmoothedNext.sub(PPredicted)).mmul(C.transpose()));
            
            smoothedStates.set(t, xSmoothed);
            smoothedCovariances.set(t, PSmoothed);
        }
        
        return new StateSpaceSmoothResult(smoothedStates, smoothedCovariances);
    }
    
    /**
     * 创建局部线性趋势模型 / Create Local Linear Trend Model
     * <p>
     * 创建局部线性趋势状态空间模型。
     * Create local linear trend state space model.
     * </p>
     *
     * @param sigmaEta 水平噪声标准差 / Level noise standard deviation
     * @param sigmaZeta 趋势噪声标准差 / Trend noise standard deviation
     * @param sigmaEpsilon 观测噪声标准差 / Observation noise standard deviation
     * @return 局部线性趋势模型 / Local linear trend model
     */
    public static StateSpaceModel createLocalLinearTrend(double sigmaEta, double sigmaZeta, double sigmaEpsilon) {
        // 状态转移矩阵 / State transition matrix
        IMatrix<Double> F = Linalg.matrix(new double[][]{
            {1, 1},
            {0, 1}
        });
        
        // 观测矩阵 / Observation matrix
        IMatrix<Double> H = Linalg.matrix(new double[][]{
            {1, 0}
        });
        
        // 过程噪声协方差矩阵 / Process noise covariance matrix
        IMatrix<Double> Q = Linalg.matrix(new double[][]{
            {sigmaEta * sigmaEta, 0.0},
            {0.0, sigmaZeta * sigmaZeta}
        });
        
        // 观测噪声协方差矩阵 / Observation noise covariance matrix
        IMatrix<Double> R = Linalg.matrix(new double[][]{
            {sigmaEpsilon * sigmaEpsilon}
        });
        
        // 初始状态 / Initial state
        IVector<Double> x0 = Linalg.vector(new double[]{0.0, 0.0});
        
        // 初始状态协方差 / Initial state covariance
        IMatrix<Double> P0 = Linalg.matrix(new double[][]{
            {1000.0, 0.0},
            {0.0, 1000.0}
        });
        
        return new StateSpaceModel(F, H, Q, R, x0, P0);
    }
    
    /**
     * 创建季节性模型 / Create Seasonal Model
     * <p>
     * 创建季节性状态空间模型。
     * Create seasonal state space model.
     * </p>
     *
     * @param period 季节周期 / Seasonal period
     * @param sigmaOmega 季节噪声标准差 / Seasonal noise standard deviation
     * @param sigmaEpsilon 观测噪声标准差 / Observation noise standard deviation
     * @return 季节性模型 / Seasonal model
     */
    public static StateSpaceModel createSeasonalModel(int period, double sigmaOmega, double sigmaEpsilon) {
        int stateDim = period;
        
        // 状态转移矩阵 / State transition matrix
        IMatrix<Double> F = Linalg.zeros(stateDim, stateDim);
        F.set(0, 0, -1.0);
        for (int i = 0; i < stateDim - 1; i++) {
            F.set(i + 1, i, 1.0);
        }
        
        // 观测矩阵 / Observation matrix
        IMatrix<Double> H = Linalg.zeros(1, stateDim);
        H.set(0, 0, 1.0);
        
        // 过程噪声协方差矩阵 / Process noise covariance matrix
        IMatrix<Double> Q = Linalg.zeros(stateDim, stateDim);
        Q.set(0, 0, sigmaOmega * sigmaOmega);
        
        // 观测噪声协方差矩阵 / Observation noise covariance matrix
        IMatrix<Double> R = Linalg.matrix(new double[][]{
            {sigmaEpsilon * sigmaEpsilon}
        });
        
        // 初始状态 / Initial state
        IVector<Double> x0 = Linalg.zeros(stateDim);
        
        // 初始状态协方差 / Initial state covariance
        IMatrix<Double> P0 = Linalg.eye(stateDim).multiplyByScalar(1000.0);
        
        return new StateSpaceModel(F, H, Q, R, x0, P0);
    }
    
    // ========== Getter方法 / Getter Methods ==========
    
    public IMatrix<Double> getF() { return F; }
    public IMatrix<Double> getH() { return H; }
    public IMatrix<Double> getQ() { return Q; }
    public IMatrix<Double> getR() { return R; }
    public IVector<Double> getX0() { return x0; }
    public IMatrix<Double> getP0() { return P0; }
    
    // ========== 结果类 / Result Classes ==========
    
    /**
     * Kalman滤波结果类 / Kalman Filter Result Class
     */
    public static class KalmanFilterResult {
        public final List<IVector<Double>> filteredStates;
        public final List<IMatrix<Double>> filteredCovariances;
        public final List<IVector<Double>> predictedStates;
        public final List<IMatrix<Double>> predictedCovariances;
        public final List<IVector<Double>> innovations;
        public final List<Double> logLikelihoods;
        
        public KalmanFilterResult(List<IVector<Double>> filteredStates, List<IMatrix<Double>> filteredCovariances,
                                List<IVector<Double>> predictedStates, List<IMatrix<Double>> predictedCovariances,
                                List<IVector<Double>> innovations, List<Double> logLikelihoods) {
            this.filteredStates = filteredStates;
            this.filteredCovariances = filteredCovariances;
            this.predictedStates = predictedStates;
            this.predictedCovariances = predictedCovariances;
            this.innovations = innovations;
            this.logLikelihoods = logLikelihoods;
        }
    }
    
    /**
     * 状态空间预测结果类 / State Space Forecast Result Class
     */
    public static class StateSpaceForecastResult {
        public final List<IVector<Double>> forecastStates;
        public final List<IMatrix<Double>> forecastCovariances;
        public final List<IVector<Double>> forecastObservations;
        public final List<IVector<Double>> forecastStd;
        
        public StateSpaceForecastResult(List<IVector<Double>> forecastStates, List<IMatrix<Double>> forecastCovariances,
                                      List<IVector<Double>> forecastObservations, List<IVector<Double>> forecastStd) {
            this.forecastStates = forecastStates;
            this.forecastCovariances = forecastCovariances;
            this.forecastObservations = forecastObservations;
            this.forecastStd = forecastStd;
        }
    }
    
    /**
     * 状态空间平滑结果类 / State Space Smooth Result Class
     */
    public static class StateSpaceSmoothResult {
        public final List<IVector<Double>> smoothedStates;
        public final List<IMatrix<Double>> smoothedCovariances;
        
        public StateSpaceSmoothResult(List<IVector<Double>> smoothedStates, List<IMatrix<Double>> smoothedCovariances) {
            this.smoothedStates = smoothedStates;
            this.smoothedCovariances = smoothedCovariances;
        }
    }
    
    // ========== 私有辅助方法 / Private Helper Methods ==========
    
    /**
     * 计算对数似然 / Calculate log likelihood
     */
    private double calculateLogLikelihood(IVector<Double> innovation, IMatrix<Double> covariance) {
        try {
            IMatrix<Double> invCov = covariance.pinv();
            double det = Math.abs(covariance.get(0, 0)); // 简化处理 / Simplified handling
            
            if (det <= 0) return Double.NEGATIVE_INFINITY;
            
            double quadratic = innovation.get(0) * innovation.get(0) / det; // 简化处理 / Simplified handling
            return -0.5 * (Math.log(2 * Math.PI * det) + quadratic);
        } catch (Exception e) {
            return Double.NEGATIVE_INFINITY;
        }
    }
    
    @Override
    public String toString() {
        return String.format("StateSpaceModel{stateDim=%d, obsDim=%d}", 
                           x0.length(), H.getRowNum());
    }
}
