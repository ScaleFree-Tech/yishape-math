package com.yishape.lab.math.ml.clf.tree;

/**
 * 直方图建树时的分箱策略（仅 {@link XGBoostTreeMethod#HIST}）。
 * <p>
 * {@link #UNIFORM}：全局 min/max 均匀切分（兼容旧行为）。<br>
 * {@link #QUANTILE_WEIGHTED_SKETCH}：按样本二阶梯度（Hessian）加权的分位数切分，
 * 在当前实现中对参与建树的行做一次精确加权分位（与 XGBoost/LightGBM 的 weighted quantile sketch 目标一致，
 * 可作为稠密数据上的参考 sketch；超大规模时可再叠近似 sketch）。
 * </p>
 */
public enum XGBoostHistogramBinning {

    UNIFORM,

    /** Hessian 加权分位数边界（每棵树、基于传入样本行与当前 Hessian） */
    QUANTILE_WEIGHTED_SKETCH
}
