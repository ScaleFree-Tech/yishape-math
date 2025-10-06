package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

/**
 * 线性规划问题的工具类
 * @author lteb2
 */
public class LinProgUtil {
    
    /**
     * 将不等式约束转换为等式约束（不包含目标函数向量扩展）
     * 
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（不等式右方）
     * @return 返回转换后的约束矩阵和约束向量
     */
    public static Tuple2<IMatrix, IVector> convertUbEqToEqConstraits(IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq) {
        // 获取约束维度信息
        int ubRows = A_ub == null ? 0 : A_ub.rows();
        int eqRows = A_eq == null ? 0 : A_eq.rows();
        
        // 构建新的等式约束矩阵和向量
        if (ubRows == 0) {
            // 没有不等式约束，直接返回等式约束
            return new Tuple2<>(A_eq, b_eq);
        } else if (eqRows == 0) {
            // 只有不等式约束
            // 创建新的约束矩阵 [A_ub | I]
            IMatrix newA = A_ub.hstack(IMatrix.eye(ubRows));
            return new Tuple2<>(newA, b_ub);
        } else {
            // 同时存在不等式约束和等式约束
            // 等式约束需要扩展列数以匹配新变量数
            IMatrix extendedAeq = A_eq.hstack(IMatrix.zeros(eqRows, ubRows));
            
            // 不等式约束转换为等式约束（添加单位矩阵）
            IMatrix newAub = A_ub.hstack(IMatrix.eye(ubRows));
            
            // 合并约束矩阵
            IMatrix combinedA = extendedAeq.vstack(newAub);
            
            // 合并约束向量
            IVector combinedB = b_eq.concat(b_ub);
            
            return new Tuple2<>(combinedA, combinedB);
        }
    }
    
    /**
     * 将solve中的参数值（不等式约束、不等式约束值、等式约束、等式约束值）转换为solveWithEqualConstraits需要的等式形式
     *
     * @param c 目标函数系数向量
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（不等式右方）
     * @return 返回扩展后的目标函数向量、约束矩阵和约束向量
     */
    public static Tuple3<IVector, IMatrix, IVector> convertUbEqToEqConstraits(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq) {
        // 获取约束维度信息
        int ubRows = A_ub == null ? 0 : A_ub.rows();
        int ubCols = A_ub == null ? 0 : A_ub.cols();
        int eqRows = A_eq == null ? 0 : A_eq.rows();
        int eqCols = A_eq == null ? 0 : A_eq.cols();
        
        // 验证输入一致性
        if (A_ub != null && b_ub != null && A_ub.rows() != b_ub.length()) {
            throw new IllegalArgumentException("A_ub rows (" + A_ub.rows() + ") must match b_ub length (" + b_ub.length() + ")");
        }
        if (A_eq != null && b_eq != null && A_eq.rows() != b_eq.length()) {
            throw new IllegalArgumentException("A_eq rows (" + A_eq.rows() + ") must match b_eq length (" + b_eq.length() + ")");
        }
        if (A_ub != null && A_ub.cols() != c.length()) {
            throw new IllegalArgumentException("A_ub cols (" + A_ub.cols() + ") must match c length (" + c.length() + ")");
        }
        if (A_eq != null && A_eq.cols() != c.length()) {
            throw new IllegalArgumentException("A_eq cols (" + A_eq.cols() + ") must match c length (" + c.length() + ")");
        }
        
        // 构建新的等式约束矩阵和向量
        if (ubRows == 0) {
            // 没有不等式约束，直接返回等式约束
            return new Tuple3<>(c, A_eq, b_eq);
        } else if (eqRows == 0) {
            // 只有不等式约束
            // 先添加松弛变量转换为等式约束：A_ub * x + s = b_ub
            IMatrix newA = A_ub.hstack(IMatrix.eye(ubRows));
            
            // 扩展目标函数向量，为松弛变量添加0系数
            IVector extendedC = extendObjectiveVector(c, ubRows);
            
            // 然后处理负的b_ub值（现在是等式约束了）
            Tuple2<IMatrix, IVector> processed = processNegativeBUb(newA, b_ub);
            IMatrix processedA = processed.getFirst();
            IVector processedB = processed.getSecond();
            
            return new Tuple3<>(extendedC, processedA, processedB);
        } else {
            // 同时存在不等式约束和等式约束
            // 先添加松弛变量转换不等式约束为等式约束
            IMatrix newAub = A_ub.hstack(IMatrix.eye(ubRows));
            
            // 等式约束需要扩展列数以匹配新变量数
            IMatrix extendedAeq = A_eq.hstack(IMatrix.zeros(eqRows, ubRows));
            
            // 扩展目标函数向量，为松弛变量添加0系数
            IVector extendedC = extendObjectiveVector(c, ubRows);
            
            // 分别处理等式约束和不等式约束的负b值
            Tuple2<IMatrix, IVector> processedEq = processNegativeBEq(extendedAeq, b_eq);
            Tuple2<IMatrix, IVector> processedUb = processNegativeBUb(newAub, b_ub);
            
            // 合并约束矩阵和向量：等式约束在前，不等式约束在后
            IMatrix combinedA = processedEq.getFirst().vstack(processedUb.getFirst());
            IVector combinedB = processedEq.getSecond().concat(processedUb.getSecond());
            
            return new Tuple3<>(extendedC, combinedA, combinedB);
        }
    }
    
    /**
     * 扩展目标函数向量，为松弛变量添加0系数
     * 
     * @param c 原始目标函数向量
     * @param slackVarCount 松弛变量数量
     * @return 扩展后的目标函数向量
     */
    private static IVector extendObjectiveVector(IVector c, int slackVarCount) {
        if (slackVarCount == 0) {
            return c;
        }
        
        // 创建松弛变量的系数向量（全为0）
        double[] slackCoeffs = new double[slackVarCount];
        // 松弛变量在目标函数中的系数为0
        for (int i = 0; i < slackVarCount; i++) {
            slackCoeffs[i] = 0.0;
        }
        
        // 将原始目标函数向量与松弛变量系数向量连接
        return c.concat(IVector.of(slackCoeffs));
    }
    
    /**
     * 处理不等式约束中b_ub向量的负值
     * 如果b_ub的某个元素为负，则将对应的约束行（A_ub）和b_ub值同时乘以-1
     * 
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束向量
     * @return 处理后的约束矩阵和向量
     */
    public static Tuple2<IMatrix, IVector> processNegativeBUb(IMatrix A_ub, IVector b_ub) {
        if (A_ub == null || b_ub == null) {
            return new Tuple2<>(A_ub, b_ub);
        }
        
        int ubRows = A_ub.rows();
        int ubCols = A_ub.cols();
        
        // 复制原始矩阵和向量
        IMatrix processedAub = A_ub.copy();
        IVector processedBub = b_ub.copy();
        
        for (int i = 0; i < ubRows; i++) {
            if (processedBub.get(i).doubleValue() < 0) {
                // 将约束行和b值同时乘以-1
                for (int j = 0; j < ubCols; j++) {
                    processedAub.set(i, j, -processedAub.get(i, j).doubleValue());
                }
                processedBub.set(i, -processedBub.get(i).doubleValue());
            }
        }
        
        return new Tuple2<>(processedAub, processedBub);
    }

    /**
     * 处理负的b_eq值：当b_eq中有负值时，将对应的约束行乘以-1
     * 这确保了单纯形法的初始基本可行解是可行的
     * 
     * @param A_eq 等式约束矩阵
     * @param b_eq 等式约束右端向量
     * @return 返回处理后的约束矩阵和向量
     */
    public static Tuple2<IMatrix, IVector> processNegativeBEq(IMatrix A_eq, IVector b_eq) {
        if (A_eq == null || b_eq == null) {
            return new Tuple2<>(A_eq, b_eq);
        }
        
        IMatrix processedA_eq = A_eq.copy();
        IVector processedB_eq = b_eq.copy();
        
        for (int i = 0; i < processedB_eq.length(); i++) {
            double bValue = processedB_eq.get(i).doubleValue();
            if (bValue < 0) {
                // 将第i行约束乘以-1
                // A_eq[i] * x = b_eq[i] 变为 -A_eq[i] * x = -b_eq[i]
                for (int j = 0; j < processedA_eq.cols(); j++) {
                    double aValue = processedA_eq.get(i, j).doubleValue();
                    processedA_eq.set(i, j, -aValue);
                }
                processedB_eq.set(i, -bValue);
            }
        }
        
        return new Tuple2<>(processedA_eq, processedB_eq);
    }
}
