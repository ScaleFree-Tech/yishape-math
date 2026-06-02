package com.yishape.lab.math.ml.clf.tree;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.clf.ClfResult;

/**
 * 单棵决策树（CART/C4.5 风格）训练结果摘要。
 */
public class DecisionTreeResult extends ClfResult {

    private DecisionTreeCriterion criterion;
    private int maxDepthParam;
    private int treeDepth;
    private int leafCount;
    private boolean trained;

    public DecisionTreeResult() {
        super();
    }

    public DecisionTreeCriterion getCriterion() {
        return criterion;
    }

    public void setCriterion(DecisionTreeCriterion criterion) {
        this.criterion = criterion;
    }

    public int getMaxDepthParam() {
        return maxDepthParam;
    }

    public void setMaxDepthParam(int maxDepthParam) {
        this.maxDepthParam = maxDepthParam;
    }

    public int getTreeDepth() {
        return treeDepth;
    }

    public void setTreeDepth(int treeDepth) {
        this.treeDepth = treeDepth;
    }

    public int getLeafCount() {
        return leafCount;
    }

    public void setLeafCount(int leafCount) {
        this.leafCount = leafCount;
    }

    public void setTrained(boolean trained) {
        this.trained = trained;
    }

    @Override
    public String getModelTypeDescription() {
        return "单棵决策树 (CART/C4.5 风格) / Decision Tree";
    }

    @Override
    public String getModelSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 决策树摘要 ===\n");
        sb.append(getBasicStats()).append("\n");
        sb.append("准则: ").append(criterion != null ? criterion : "?").append("\n");
        sb.append("参数 maxDepth: ").append(maxDepthParam).append("\n");
        sb.append("实际深度: ").append(treeDepth).append("\n");
        sb.append("叶子数: ").append(leafCount).append("\n");
        IVector fi = getFeatureImportance();
        if (fi != null) {
            sb.append("特征重要性 ∞-范数: ").append(String.format("%.6g", fi.normInf())).append("\n");
        }
        return sb.toString();
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    /**
     * 用一组重要性权值填充分类结果（与随机森林接口一致）。
     */
    public void setFeatureImportanceFromArray(double[] importance) {
        if (importance == null) {
            return;
        }
        setFeatureImportance(Linalg.vector(importance.clone()));
    }
}
