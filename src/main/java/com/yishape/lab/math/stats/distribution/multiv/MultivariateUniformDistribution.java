package com.yishape.lab.math.stats.distribution.multiv;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * 多元均匀分布实现 / Multivariate Uniform Distribution Implementation
 * 
 * <p>多元均匀分布是在指定的多维区域内均匀分布的概率分布。
 * 在超矩形区域 [a₁,b₁] × [a₂,b₂] × ... × [aₖ,bₖ] 内，概率密度函数为：</p>
 * <p>f(x) = 1 / ∏(bᵢ - aᵢ) 如果 x ∈ [a₁,b₁] × ... × [aₖ,bₖ]</p>
 * <p>f(x) = 0 其他情况</p>
 * 
 * <p>Multivariate uniform distribution is a probability distribution that is uniform
 * over a specified multidimensional region.
 * Over the hyperrectangular region [a₁,b₁] × [a₂,b₂] × ... × [aₖ,bₖ], the probability density function is:</p>
 * <p>f(x) = 1 / ∏(bᵢ - aᵢ) if x ∈ [a₁,b₁] × ... × [aₖ,bₖ]</p>
 * <p>f(x) = 0 otherwise</p>
 * 
 * <h3>主要特性 / Key Properties:</h3>
 * <ul>
 *   <li>在支撑区域内概率密度恒定 / Constant probability density over support region</li>
 *   <li>独立的边际分布 / Independent marginal distributions</li>
 *   <li>有界支撑 / Bounded support</li>
 *   <li>最大熵分布（在给定支撑下） / Maximum entropy distribution (given support)</li>
 * </ul>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MultivariateUniformDistribution implements IMultivariateDistribution<Double> {
    
    private static final long serialVersionUID = 1L;
    
    /** 下界向量 / Lower bounds vector */
    private final IVector<Double> lowerBounds;
    
    /** 上界向量 / Upper bounds vector */
    private final IVector<Double> upperBounds;
    
    /** 区间长度向量 / Interval lengths vector */
    private final IVector<Double> intervals;
    
    /** 维度 / Dimensionality */
    private final int dimension;
    
    /** 超体积（所有区间长度的乘积） / Hypervolume (product of all interval lengths) */
    private final double hypervolume;
    
    /** 概率密度值 / Probability density value */
    private final double densityValue;
    
    /** 随机数生成器 / Random number generator */
    private final Random random;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param lowerBounds 下界向量 / Lower bounds vector
     * @param upperBounds 上界向量 / Upper bounds vector
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public MultivariateUniformDistribution(IVector<Double> lowerBounds, IVector<Double> upperBounds) {
        this(lowerBounds, upperBounds, new Random());
    }
    
    /**
     * 构造函数（带随机种子）
     * Constructor with random seed
     * 
     * @param lowerBounds 下界向量 / Lower bounds vector
     * @param upperBounds 上界向量 / Upper bounds vector
     * @param random 随机数生成器 / Random number generator
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public MultivariateUniformDistribution(IVector<Double> lowerBounds, IVector<Double> upperBounds, Random random) {
        validateParameters(lowerBounds, upperBounds);
        
        this.dimension = lowerBounds.length();
        this.lowerBounds = lowerBounds.copy();
        this.upperBounds = upperBounds.copy();
        this.random = random;
        
        // 计算区间长度
        this.intervals = upperBounds.sub(lowerBounds);
        
        // 计算超体积
        this.hypervolume = computeHypervolume();
        
        // 计算概率密度值
        this.densityValue = 1.0 / hypervolume;
    }
    
    /**
     * 验证参数有效性
     * Validate parameter validity
     */
    private void validateParameters(IVector<Double> lowerBounds, IVector<Double> upperBounds) {
        if (lowerBounds == null || upperBounds == null) {
            throw new IllegalArgumentException("下界和上界向量不能为null");
        }
        
        if (lowerBounds.length() != upperBounds.length()) {
            throw new IllegalArgumentException("下界和上界向量维度必须相同");
        }
        
        // 验证所有维度上下界都有效
        for (int i = 0; i < lowerBounds.length(); i++) {
            if (lowerBounds.get(i) >= upperBounds.get(i)) {
                throw new IllegalArgumentException(
                    String.format("维度 %d 的下界 %.6f 必须小于上界 %.6f", 
                                i, lowerBounds.get(i), upperBounds.get(i)));
            }
        }
    }
    
    /**
     * 计算超体积
     * Compute hypervolume
     */
    private double computeHypervolume() {
        double volume = 1.0;
        for (int i = 0; i < dimension; i++) {
            volume *= intervals.get(i);
        }
        return volume;
    }
    
    /**
     * 检查点是否在支撑区域内
     * Check if point is within support region
     */
    private boolean isInSupport(IVector<Double> x) {
        for (int i = 0; i < dimension; i++) {
            double value = x.get(i);
            if (value < lowerBounds.get(i) || value > upperBounds.get(i)) {
                return false;
            }
        }
        return true;
    }
    
    // ==================== IMultivariateDistribution 接口实现 ====================
    
    @Override
    public int getDimension() {
        return dimension;
    }
    
    @Override
    public String getDistributionName() {
        return "Multivariate Uniform Distribution";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("Dimension: %d, Lower bounds: %s, Upper bounds: %s, Hypervolume: %.6f", 
                           dimension, lowerBounds.toString(), upperBounds.toString(), hypervolume);
    }
    
    @Override
    public double pdf(IVector<Double> x) {
        validateDimension(x);
        return isInSupport(x) ? densityValue : 0.0;
    }
    
    @Override
    public double logPdf(IVector<Double> x) {
        validateDimension(x);
        return isInSupport(x) ? Math.log(densityValue) : Double.NEGATIVE_INFINITY;
    }
    
    @Override
    public double[] pdf(List<IVector<Double>> samples) {
        double[] results = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            results[i] = pdf(samples.get(i));
        }
        return results;
    }
    
    @Override
    public double[] logPdf(List<IVector<Double>> samples) {
        double[] results = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            results[i] = logPdf(samples.get(i));
        }
        return results;
    }
    
    @Override
    public IVector<Double> getMean() {
        // 均值是每个维度上下界的中点
        double[] meanArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            meanArray[i] = 0.5 * (lowerBounds.get(i) + upperBounds.get(i));
        }
        return Linalg.vector(meanArray);
    }
    
    @Override
    public IMatrix<Double> getCovariance() {
        // 协方差矩阵是对角矩阵，对角元素为 (b-a)²/12
        double[][] covArray = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            double interval = intervals.get(i);
            covArray[i][i] = interval * interval / 12.0;
        }
        return Linalg.matrix(covArray);
    }
    
    @Override
    public IMatrix<Double> getCorrelation() {
        // 由于各维度独立，相关矩阵是单位矩阵
        return Linalg.eye(dimension);
    }
    
    @Override
    public IMatrix<Double> getPrecision() {
        return getCovariance().inv();
    }
    
    @Override
    public IVector<Double> getStandardDeviation() {
        double[] stdDevArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            double interval = intervals.get(i);
            stdDevArray[i] = interval / Math.sqrt(12.0);
        }
        return Linalg.vector(stdDevArray);
    }
    
    @Override
    public double mahalanobisDistance(IVector<Double> x) {
        return Math.sqrt(squaredMahalanobisDistance(x));
    }
    
    @Override
    public double squaredMahalanobisDistance(IVector<Double> x) {
        validateDimension(x);
        
        IVector<Double> mean = getMean();
        IVector<Double> diff = x.sub(mean);
        IMatrix<Double> precision = getPrecision();
        IVector<Double> temp = precision.mmul(diff);
        return diff.dot(temp);
    }
    
    @Override
    public IVector<Double> sample() {
        double[] sampleArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            double lower = lowerBounds.get(i);
            double upper = upperBounds.get(i);
            sampleArray[i] = lower + random.nextDouble() * (upper - lower);
        }
        return Linalg.vector(sampleArray);
    }
    
    @Override
    public List<IVector<Double>> sample(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("样本数量必须大于0");
        }
        
        List<IVector<Double>> samples = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            samples.add(sample());
        }
        return samples;
    }
    
    @Override
    public IMatrix<Double> sampleMatrix(int n) {
        List<IVector<Double>> samples = sample(n);
        double[][] sampleArray = new double[n][dimension];
        
        for (int i = 0; i < n; i++) {
            IVector<Double> sample = samples.get(i);
            for (int j = 0; j < dimension; j++) {
                sampleArray[i][j] = sample.get(j);
            }
        }
        
        return Linalg.matrix(sampleArray);
    }
    
    @Override
    public IMultivariateDistribution<Double> getMarginal(int... indices) {
        if (indices == null || indices.length == 0) {
            throw new IllegalArgumentException("索引不能为空");
        }
        
        // 验证索引有效性
        for (int index : indices) {
            if (index < 0 || index >= dimension) {
                throw new IllegalArgumentException("索引超出范围: " + index);
            }
        }
        
        // 提取边际下界和上界
        double[] marginalLowerArray = new double[indices.length];
        double[] marginalUpperArray = new double[indices.length];
        
        for (int i = 0; i < indices.length; i++) {
            marginalLowerArray[i] = lowerBounds.get(indices[i]);
            marginalUpperArray[i] = upperBounds.get(indices[i]);
        }
        
        IVector<Double> marginalLower = Linalg.vector(marginalLowerArray);
        IVector<Double> marginalUpper = Linalg.vector(marginalUpperArray);
        
        return new MultivariateUniformDistribution(marginalLower, marginalUpper, random);
    }
    
    @Override
    public IMultivariateDistribution<Double> getConditional(int[] conditionIndices, IVector<Double> conditionValues) {
        // 对于均匀分布，条件分布仍然是均匀分布
        if (conditionIndices == null || conditionValues == null) {
            throw new IllegalArgumentException("条件索引和条件值不能为null");
        }
        if (conditionIndices.length != conditionValues.length()) {
            throw new IllegalArgumentException("条件索引和条件值的长度必须相同");
        }
        
        // 验证条件值在支撑区域内
        for (int i = 0; i < conditionIndices.length; i++) {
            int index = conditionIndices[i];
            double value = conditionValues.get(i);
            if (value < lowerBounds.get(index) || value > upperBounds.get(index)) {
                throw new IllegalArgumentException(
                    String.format("条件值 %.6f 在维度 %d 上超出支撑区域 [%.6f, %.6f]", 
                                value, index, lowerBounds.get(index), upperBounds.get(index)));
            }
        }
        
        // 构建剩余维度的边际分布
        List<Integer> remainingIndices = new ArrayList<>();
        for (int i = 0; i < dimension; i++) {
            boolean isConditioned = false;
            for (int condIndex : conditionIndices) {
                if (i == condIndex) {
                    isConditioned = true;
                    break;
                }
            }
            if (!isConditioned) {
                remainingIndices.add(i);
            }
        }
        
        if (remainingIndices.isEmpty()) {
            throw new IllegalArgumentException("条件化后没有剩余维度");
        }
        
        int[] remainingArray = remainingIndices.stream().mapToInt(Integer::intValue).toArray();
        return getMarginal(remainingArray);
    }
    
    @Override
    public IMultivariateDistribution<Double> linearTransform(IMatrix<Double> A, IVector<Double> b) {
        if (A.rows() != dimension || A.cols() != dimension) {
            throw new IllegalArgumentException("变换矩阵必须为 " + dimension + "×" + dimension);
        }
        if (b.length() != dimension) {
            throw new IllegalArgumentException("平移向量维度必须为 " + dimension);
        }
        if (!isPositiveDiagonal(A)) {
            throw new UnsupportedOperationException(
                    "仅支持正对角缩放加平移；一般线性变换不保持超矩形均匀分布 / "
                            + "Only positive diagonal scaling preserves rectangular uniform family");
        }
        double[] nl = new double[dimension];
        double[] nu = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            double di = A.get(i, i);
            nl[i] = di * lowerBounds.get(i) + b.get(i);
            nu[i] = di * upperBounds.get(i) + b.get(i);
            if (nl[i] > nu[i]) {
                double t = nl[i];
                nl[i] = nu[i];
                nu[i] = t;
            }
        }
        return new MultivariateUniformDistribution(Linalg.vector(nl), Linalg.vector(nu), random);
    }

    private boolean isPositiveDiagonal(IMatrix<Double> A) {
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (i != j && Math.abs(A.get(i, j)) > 1e-12) {
                    return false;
                }
            }
            if (A.get(i, i) <= 0) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public IMultivariateDistribution<Double> affineTransform(IMatrix<Double> A) {
        return linearTransform(A, Linalg.zeros(A.rows()));
    }
    
    @Override
    public double klDivergence(IMultivariateDistribution<Double> other) {
        if (!(other instanceof MultivariateUniformDistribution)) {
            throw new IllegalArgumentException("只支持与其他多元均匀分布计算KL散度");
        }
        
        MultivariateUniformDistribution otherUniform = (MultivariateUniformDistribution) other;
        if (otherUniform.dimension != this.dimension) {
            throw new IllegalArgumentException("分布维度必须相同");
        }
        
        // 检查支撑区域的包含关系
        boolean thisContainsOther = true;
        boolean otherContainsThis = true;
        
        for (int i = 0; i < dimension; i++) {
            double thisLower = this.lowerBounds.get(i);
            double thisUpper = this.upperBounds.get(i);
            double otherLower = otherUniform.lowerBounds.get(i);
            double otherUpper = otherUniform.upperBounds.get(i);
            
            if (thisLower > otherLower || thisUpper < otherUpper) {
                thisContainsOther = false;
            }
            if (otherLower > thisLower || otherUpper < thisUpper) {
                otherContainsThis = false;
            }
        }
        
        if (!otherContainsThis) {
            return Double.POSITIVE_INFINITY; // KL散度无穷大
        }
        
        // 如果other包含this，KL散度为log(V_other / V_this)
        return Math.log(otherUniform.hypervolume / this.hypervolume);
    }
    
    @Override
    public double wassersteinDistance(IMultivariateDistribution<Double> other) {
        if (!(other instanceof MultivariateUniformDistribution)) {
            throw new IllegalArgumentException("只支持与其他多元均匀分布计算Wasserstein距离");
        }
        
        MultivariateUniformDistribution otherUniform = (MultivariateUniformDistribution) other;
        if (otherUniform.dimension != this.dimension) {
            throw new IllegalArgumentException("分布维度必须相同");
        }
        
        // 对于均匀分布，Wasserstein距离可以通过区间端点的距离计算
        double distance = 0.0;
        for (int i = 0; i < dimension; i++) {
            double lowerDiff = Math.abs(this.lowerBounds.get(i) - otherUniform.lowerBounds.get(i));
            double upperDiff = Math.abs(this.upperBounds.get(i) - otherUniform.upperBounds.get(i));
            distance += Math.max(lowerDiff, upperDiff);
        }
        
        return distance / dimension;
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples) {
        return fitFromSamples(samples);
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples, List<Double> weights) {
        // 对于均匀分布，权重不影响参数估计
        return fitFromSamples(samples);
    }
    
    @Override
    public boolean isElliptical() {
        return false; // 均匀分布不是椭圆分布
    }
    
    @Override
    public boolean isSymmetric() {
        return true; // 均匀分布是对称分布
    }
    
    @Override
    public boolean isPositiveDefinite() {
        return true; // 协方差矩阵是对角正定的
    }
    
    @Override
    public void validateDimension(IVector<Double> x) {
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        if (x.length() != dimension) {
            throw new IllegalArgumentException(
                String.format("输入向量维度不匹配：期望 %d，实际 %d", dimension, x.length()));
        }
    }
    
    @Override
    public double entropy() {
        return Math.log(hypervolume);
    }
    
    @Override
    public IMatrix<Double> informationMatrix() {
        // 信息矩阵是协方差矩阵的逆
        return getPrecision();
    }
    
    @Override
    public IMultivariateDistribution<Double> conjugateUpdate(IVector<Double> observations) {
        // For multivariate uniform distribution, conjugate update with observations
        // This is a simplified placeholder implementation
        double[] newLowerArray = new double[dimension];
        double[] newUpperArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            newLowerArray[i] = lowerBounds.get(i) - observations.get(i) * 0.01;
            newUpperArray[i] = upperBounds.get(i) + observations.get(i) * 0.01;
        }
        IVector<Double> newLower = Linalg.vector(newLowerArray);
        IVector<Double> newUpper = Linalg.vector(newUpperArray);
        return new MultivariateUniformDistribution(newLower, newUpper);
    }
    
    @Override
    public double marginalLikelihood(IVector<Double> observations) {
        // For multivariate uniform distribution, compute marginal likelihood of observations
        // This is a simplified placeholder implementation
        double logLikelihood = 0.0;
        for (int i = 0; i < dimension; i++) {
            double obs = observations.get(i);
            double lower = lowerBounds.get(i);
            double upper = upperBounds.get(i);
            if (obs < lower || obs > upper) {
                return 0.0; // Outside support
            }
            logLikelihood -= Math.log(upper - lower);
        }
        return Math.exp(logLikelihood);
    }
    
    @Override
    public List<IVector<Double>> posteriorSample(IVector<Double> observations, int n) {
        // Sample from posterior distribution after conjugate update
        IMultivariateDistribution<Double> posterior = conjugateUpdate(observations);
        return posterior.sample(n);
    }
    
    @Override
    public ConfidenceEllipse getConfidenceEllipse(double confidence) {
        if (dimension < 2) {
            throw new UnsupportedOperationException("置信椭圆需要维度至少为 2");
        }
        if (confidence <= 0 || confidence >= 1) {
            throw new IllegalArgumentException("置信水平必须在(0,1)范围内");
        }
        return MultivariateDistributionMath.confidenceEllipseMarginalPlane(
                getMean(), getCovariance(), 0, 1, confidence);
    }
    
    // ==================== 静态工厂方法 ====================
    
    /**
     * 从样本数据估计多元均匀分布参数
     * Estimate multivariate uniform distribution parameters from sample data
     */
    public static MultivariateUniformDistribution fitFromSamples(List<IVector<Double>> samples) {
        if (samples == null || samples.isEmpty()) {
            throw new IllegalArgumentException("样本数据不能为空");
        }
        
        int dimension = samples.get(0).length();
        
        // 验证所有样本维度一致
        for (IVector<Double> sample : samples) {
            if (sample.length() != dimension) {
                throw new IllegalArgumentException("所有样本维度必须一致");
            }
        }
        
        // 计算每个维度的最小值和最大值
        double[] minValues = new double[dimension];
        double[] maxValues = new double[dimension];
        
        // 初始化
        for (int i = 0; i < dimension; i++) {
            minValues[i] = samples.get(0).get(i);
            maxValues[i] = samples.get(0).get(i);
        }
        
        // 找到每个维度的最小值和最大值
        for (IVector<Double> sample : samples) {
            for (int i = 0; i < dimension; i++) {
                double value = sample.get(i);
                if (value < minValues[i]) {
                    minValues[i] = value;
                }
                if (value > maxValues[i]) {
                    maxValues[i] = value;
                }
            }
        }
        
        // 添加小的边距以确保所有样本都在支撑区域内
        double margin = 1e-6;
        for (int i = 0; i < dimension; i++) {
            double range = maxValues[i] - minValues[i];
            if (range == 0) {
                range = 1.0; // 如果所有值相同，设置默认范围
            }
            minValues[i] -= margin * range;
            maxValues[i] += margin * range;
        }
        
        IVector<Double> lowerBounds = Linalg.vector(minValues);
        IVector<Double> upperBounds = Linalg.vector(maxValues);
        
        return new MultivariateUniformDistribution(lowerBounds, upperBounds);
    }
    
    /**
     * 创建标准多元均匀分布（每个维度在[0,1]上均匀分布）
     * Create standard multivariate uniform distribution (uniform on [0,1] in each dimension)
     */
    public static MultivariateUniformDistribution standard(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("维度必须大于0");
        }
        
        double[] zeros = new double[dimension];
        double[] ones = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            zeros[i] = 0.0;
            ones[i] = 1.0;
        }
        
        IVector<Double> lowerBounds = Linalg.vector(zeros);
        IVector<Double> upperBounds = Linalg.vector(ones);
        
        return new MultivariateUniformDistribution(lowerBounds, upperBounds);
    }
    
    /**
     * 创建对称多元均匀分布（每个维度在[-a,a]上均匀分布）
     * Create symmetric multivariate uniform distribution (uniform on [-a,a] in each dimension)
     */
    public static MultivariateUniformDistribution symmetric(int dimension, double bound) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("维度必须大于0");
        }
        if (bound <= 0) {
            throw new IllegalArgumentException("边界必须大于0");
        }
        
        double[] lowerArray = new double[dimension];
        double[] upperArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            lowerArray[i] = -bound;
            upperArray[i] = bound;
        }
        
        IVector<Double> lowerBounds = Linalg.vector(lowerArray);
        IVector<Double> upperBounds = Linalg.vector(upperArray);
        
        return new MultivariateUniformDistribution(lowerBounds, upperBounds);
    }
    
    /**
     * 获取下界向量
     * Get lower bounds vector
     */
    public IVector<Double> getLowerBounds() {
        return lowerBounds.copy();
    }
    
    /**
     * 获取上界向量
     * Get upper bounds vector
     */
    public IVector<Double> getUpperBounds() {
        return upperBounds.copy();
    }
    
    /**
     * 获取区间长度向量
     * Get interval lengths vector
     */
    public IVector<Double> getIntervals() {
        return intervals.copy();
    }
    
    /**
     * 获取超体积
     * Get hypervolume
     */
    public double getHypervolume() {
        return hypervolume;
    }
    
    @Override
    public String toString() {
        return String.format("MultivariateUniformDistribution(dimension=%d, bounds=[%s, %s], volume=%.6f)", 
                           dimension, lowerBounds.toString(), upperBounds.toString(), hypervolume);
    }
}