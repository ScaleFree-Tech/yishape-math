package com.yishape.lab.math.autodiff;

/**
 * Result of finite-difference gradient checking against analytical AD.
 * 解析梯度与有限差分数值梯度的校验结果。
 *
 * @param passed whether max relative error is within tolerance / 最大相对误差是否在容差内
 * @param maxAbsoluteError max |analytical − numerical| / 最大绝对误差
 * @param maxRelativeError max relative error / 最大相对误差
 * @param meanAbsoluteError mean absolute error / 平均绝对误差
 * @param suspiciousIndices indices exceeding tolerance / 超出容差的下标
 * @param analyticalGrad analytical gradient vector / 解析梯度
 * @param numericalGrad central-difference gradient / 中心差分数值梯度
 */
public record GradientCheckResult(
        boolean passed,
        double maxAbsoluteError,
        double maxRelativeError,
        double meanAbsoluteError,
        int[] suspiciousIndices,
        double[] analyticalGrad,
        double[] numericalGrad) {

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("GradientCheckResult{passed=").append(passed);
        sb.append(", maxAbsError=").append(String.format("%.2e", maxAbsoluteError));
        sb.append(", maxRelError=").append(String.format("%.2e", maxRelativeError));
        sb.append(", meanAbsError=").append(String.format("%.2e", meanAbsoluteError));
        sb.append(", suspiciousCount=").append(suspiciousIndices.length);
        sb.append("}");
        return sb.toString();
    }

    /** Human-readable report with per-index analytical vs numerical values. / 含逐分量对比的详细报告。 */
    public String detailedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append(toString()).append("\n");
        if (suspiciousIndices.length > 0) {
            sb.append("Suspicious indices (analytical vs numerical):\n");
            int show = Math.min(10, suspiciousIndices.length);
            for (int k = 0; k < show; k++) {
                int idx = suspiciousIndices[k];
                sb.append(String.format("  [%d] analytical=%12.8f  numerical=%12.8f  diff=%+.2e%n",
                        idx, analyticalGrad[idx], numericalGrad[idx],
                        analyticalGrad[idx] - numericalGrad[idx]));
            }
            if (suspiciousIndices.length > show) {
                sb.append("  ... and ").append(suspiciousIndices.length - show).append(" more\n");
            }
        }
        return sb.toString();
    }
}
