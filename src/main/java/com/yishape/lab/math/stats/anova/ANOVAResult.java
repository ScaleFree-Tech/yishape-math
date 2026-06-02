package com.yishape.lab.math.stats.anova;

/**
 * 单因素方差分析结果类 / One-Way ANOVA Result Class
 * <p>
 * 用于封装单因素ANOVA分析的所有重要统计量。
 * Encapsulates all important statistics from one-way ANOVA analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ANOVAResult {

    /** 组间平方和 / Sum of Squares Between groups */
    double ssBetween;
    /** 组内平方和 / Sum of Squares Within groups */
    double ssWithin;
    /** 总平方和 / Total Sum of Squares */
    double ssTotal;
    /** F统计量 / F-statistic */
    double fStatistic;
    /** p值 / p-value */
    double p;

    /**
     * 创建单因素方差分析结果对象 / Create One-Way ANOVA Result Object
     *
     * @param ssBetween 组间平方和 / Sum of squares between groups
     * @param ssWithin 组内平方和 / Sum of squares within groups
     * @param ssTotal 总平方和 / Total sum of squares
     * @param fStatistic F统计量 / F-statistic
     * @param pValue p值 / p-value
     */
    ANOVAResult(double ssBetween, double ssWithin, double ssTotal, double fStatistic, double pValue) {
        this.ssBetween = ssBetween;
        this.ssWithin = ssWithin;
        this.ssTotal = ssTotal;
        this.fStatistic = fStatistic;
        this.p = pValue;
    }
}
