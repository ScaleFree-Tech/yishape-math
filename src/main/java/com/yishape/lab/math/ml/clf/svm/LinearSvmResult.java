package com.yishape.lab.math.ml.clf.svm;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.clf.ClfResult;

/**
 * 线性 SVM（OV R、平方铰链 + L2）训练结果摘要。
 */
public class LinearSvmResult extends ClfResult {

    private boolean trained;
    private double cParam;
    private boolean standardized;

    public LinearSvmResult() {
        super();
    }

    public boolean isStandardized() {
        return standardized;
    }

    public void setStandardized(boolean standardized) {
        this.standardized = standardized;
    }

    public double getCParam() {
        return cParam;
    }

    public void setCParam(double cParam) {
        this.cParam = cParam;
    }

    public void setTrained(boolean trained) {
        this.trained = trained;
    }

    @Override
    public String getModelTypeDescription() {
        return "线性 SVM (Linear SVM, squared hinge, OvR)";
    }

    @Override
    public String getModelSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 线性 SVM 摘要 ===\n");
        sb.append(getBasicStats()).append("\n");
        sb.append("C (铰链损失权重): ").append(cParam).append("\n");
        sb.append("标准化特征: ").append(standardized ? "是" : "否").append("\n");
        return sb.toString();
    }

    @Override
    public boolean isTrained() {
        return trained;
    }
}
