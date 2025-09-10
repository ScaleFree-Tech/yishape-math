package com.reremouse.lab.math.stat.anova;

/**
 * RepeatedMeasuresANOVAResult - 重复测量方差分析结果类 用于封装重复测量ANOVA分析的所有重要统计量
 *
 * @author lteb2
 */
public class RepeatedMeasuresANOVAResult {

    float timeF;        // 时间效应的F统计量
    float timeP;        // 时间效应的p值
    float subjectF;     // 被试效应的F统计量
    float subjectP;     // 被试效应的p值

    // 构造函数
    // 参数说明：
    //   - timeF: 时间效应的F统计量，用于检验时间因素的主效应
    //   - timeP: 时间效应的p值，表示时间效应的显著性
    //   - subjectF: 被试效应的F统计量，用于检验被试间差异
    //   - subjectP: 被试效应的p值，表示被试间差异的显著性
    RepeatedMeasuresANOVAResult(float timeF, float timeP, float subjectF, float subjectP) {
        this.timeF = timeF;
        this.timeP = timeP;
        this.subjectF = subjectF;
        this.subjectP = subjectP;
    }
}
