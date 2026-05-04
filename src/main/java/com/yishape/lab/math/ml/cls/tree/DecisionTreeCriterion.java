package com.yishape.lab.math.ml.cls.tree;

/**
 * 单棵决策树分裂准则。
 * <ul>
 *   <li>{@link #CART_GINI} — CART：基尼不纯度下降</li>
 *   <li>{@link #CART_ENTROPY} — CART：信息熵（信息增益）</li>
 *   <li>{@link #C45_GAIN_RATIO} — C4.5 <em>风格</em>：仅在<strong>连续数值特征</strong>上按相邻取值中点二分，
 *       用熵定义的信息增益除以二元分裂信息得到增益率（与 Quinlan C4.5 在此设定下一致）。
 *       <strong>不包含</strong>完整 C4.5/J48：名目属性多路分裂、缺失值、置信度剪枝等。</li>
 * </ul>
 */
public enum DecisionTreeCriterion {

    CART_GINI,

    CART_ENTROPY,

    /** Quinlan C4.5：数值属性上等价于按 Entropy 算信息增益，再用二元分裂信息归一化（增益率） */
    C45_GAIN_RATIO
}
