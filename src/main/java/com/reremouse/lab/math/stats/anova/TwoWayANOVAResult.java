package com.reremouse.lab.math.stats.anova;

/**
 * TwoWayANOVAResult - 两因素方差分析结果类 用于封装两因素ANOVA分析的所有重要统计量
 *
 * @author lteb2
 */
public class TwoWayANOVAResult {

    double factorAF;        // 因素A的F统计量
    double factorAP;        // 因素A的p值
    double factorBF;        // 因素B的F统计量
    double factorBP;        // 因素B的p值
    double interactionF;    // 交互效应的F统计量
    double interactionP;    // 交互效应的p值

    // 构造函数
    // 参数说明：
    //   - factorAF: 因素A的F统计量，用于检验因素A的主效应
    //   - factorAP: 因素A的p值，表示因素A主效应的显著性
    //   - factorBF: 因素B的F统计量，用于检验因素B的主效应
    //   - factorBP: 因素B的p值，表示因素B主效应的显著性
    //   - interactionF: 交互效应的F统计量，用于检验A×B交互效应
    //   - interactionP: 交互效应的p值，表示交互效应的显著性
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
