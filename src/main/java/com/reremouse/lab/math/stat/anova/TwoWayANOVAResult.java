package com.reremouse.lab.math.stat.anova;

/**
 * TwoWayANOVAResult - 两因素方差分析结果类 用于封装两因素ANOVA分析的所有重要统计量
 *
 * @author lteb2
 */
public class TwoWayANOVAResult {

    float factorAF;        // 因素A的F统计量
    float factorAP;        // 因素A的p值
    float factorBF;        // 因素B的F统计量
    float factorBP;        // 因素B的p值
    float interactionF;    // 交互效应的F统计量
    float interactionP;    // 交互效应的p值

    // 构造函数
    // 参数说明：
    //   - factorAF: 因素A的F统计量，用于检验因素A的主效应
    //   - factorAP: 因素A的p值，表示因素A主效应的显著性
    //   - factorBF: 因素B的F统计量，用于检验因素B的主效应
    //   - factorBP: 因素B的p值，表示因素B主效应的显著性
    //   - interactionF: 交互效应的F统计量，用于检验A×B交互效应
    //   - interactionP: 交互效应的p值，表示交互效应的显著性
    TwoWayANOVAResult(float factorAF, float factorAP, float factorBF, float factorBP,
            float interactionF, float interactionP) {
        this.factorAF = factorAF;
        this.factorAP = factorAP;
        this.factorBF = factorBF;
        this.factorBP = factorBP;
        this.interactionF = interactionF;
        this.interactionP = interactionP;
    }
}
