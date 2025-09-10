package com.reremouse.lab.math.stat.anova;

/**
 * ANOVAResult - 单因素方差分析结果类 用于封装ANOVA分析的所有重要统计量
 *
 * @author lteb2
 */
public class ANOVAResult {

    float ssBetween;    // 组间平方和（Sum of Squares Between）
    float ssWithin;     // 组内平方和（Sum of Squares Within）
    float ssTotal;      // 总平方和（Sum of Squares Total）
    float fStatistic;   // F统计量
    float pValue;       // p值

    // 构造函数
    // 参数说明：
    //   - ssBetween: 组间平方和，衡量组间差异
    //   - ssWithin: 组内平方和，衡量组内变异
    //   - ssTotal: 总平方和，等于ssBetween + ssWithin
    //   - fStatistic: F统计量，用于假设检验
    //   - pValue: p值，表示在原假设为真时观察到当前F值或更极端值的概率
    ANOVAResult(float ssBetween, float ssWithin, float ssTotal, float fStatistic, float pValue) {
        this.ssBetween = ssBetween;
        this.ssWithin = ssWithin;
        this.ssTotal = ssTotal;
        this.fStatistic = fStatistic;
        this.pValue = pValue;
    }
}
