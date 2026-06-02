package com.yishape.lab.math.stats.anova;

/**
 * 重复测量方差分析结果类 / Repeated Measures ANOVA Result Class
 * <p>
 * 用于封装重复测量ANOVA分析的所有重要统计量。
 * Encapsulates all important statistics from repeated measures ANOVA analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RepeatedMeasuresANOVAResult {

    /** 时间效应的F统计量 / F-statistic for time effect */
    double timeF;
    /** 时间效应的p值 / p-value for time effect */
    double timeP;
    /** 被试效应的F统计量 / F-statistic for subject effect */
    double subjectF;
    /** 被试效应的p值 / p-value for subject effect */
    double subjectP;

    /**
     * 创建重复测量方差分析结果对象
     * Create Repeated Measures ANOVA Result Object
     *
     * @param timeF 时间效应的F统计量 / F-statistic for time effect
     * @param timeP 时间效应的p值 / p-value for time effect
     * @param subjectF 被试效应的F统计量 / F-statistic for subject effect
     * @param subjectP 被试效应的p值 / p-value for subject effect
     */
    RepeatedMeasuresANOVAResult(double timeF, double timeP, double subjectF, double subjectP) {
        this.timeF = timeF;
        this.timeP = timeP;
        this.subjectF = subjectF;
        this.subjectP = subjectP;
    }
}
