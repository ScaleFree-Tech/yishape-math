package com.yishape.lab.math.stats.anova;

/**
 * 两因素方差分析结果类 / Two-Way ANOVA Result Class
 * <p>
 * 用于封装两因素ANOVA分析的所有重要统计量。
 * Encapsulates all important statistics from two-way ANOVA analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TwoWayANOVAResult {

    /** 因素A的F统计量 / F-statistic for factor A */
    double factorAF;
    /** 因素A的p值 / p-value for factor A */
    double factorAP;
    /** 因素B的F统计量 / F-statistic for factor B */
    double factorBF;
    /** 因素B的p值 / p-value for factor B */
    double factorBP;
    /** 交互效应的F统计量 / F-statistic for interaction effect */
    double interactionF;
    /** 交互效应的p值 / p-value for interaction effect */
    double interactionP;

    /**
     * 创建两因素方差分析结果对象 / Create Two-Way ANOVA Result Object
     *
     * @param factorAF 因素A的F统计量 / F-statistic for factor A
     * @param factorAP 因素A的p值 / p-value for factor A
     * @param factorBF 因素B的F统计量 / F-statistic for factor B
     * @param factorBP 因素B的p值 / p-value for factor B
     * @param interactionF 交互效应的F统计量 / F-statistic for interaction effect
     * @param interactionP 交互效应的p值 / p-value for interaction effect
     */
    TwoWayANOVAResult(double factorAF, double factorAP, double factorBF, double factorBP,
            double interactionF, double interactionP) {
        this.factorAF = factorAF;
        this.factorAP = factorAP;
        this.factorBF = factorBF;
        this.factorBP = factorBP;
        this.interactionF = interactionF;
        this.interactionP = interactionP;
    }
}
