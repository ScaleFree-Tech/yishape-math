package com.yishape.lab.math.ml.cls.tree;

/**
 * XGBoost 风格建树策略：精确贪心（排序分裂）与直方图近似（工业默认）。
 *
 * @author lteb2
 */
public enum XGBoostTreeMethod {

    /**
     * 精确贪心：对每个特征排序后枚举候选分裂（与本库早期实现一致，数值基准）。
     */
    EXACT,

    /**
     * 直方图近似：按全局 min/max 均匀分箱后在箱边界上扫描分裂（复杂度近似 O(#bins)，大规模更快）。
     */
    HIST
}
