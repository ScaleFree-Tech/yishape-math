package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;

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
        
        // 构建新的等式约束矩阵和向量
        if (ubRows == 0) {
            // 没有不等式约束，直接返回等式约束
            return new Tuple3<>(c, A_eq, b_eq);
        } else if (eqRows == 0) {
            // 只有不等式约束
            // 创建新的约束矩阵 [A_ub | I]
            IMatrix newA = A_ub.hstack(IMatrix.eye(ubRows));
            
            // 扩展目标函数向量，为松弛变量添加0系数
            IVector extendedC = extendObjectiveVector(c, ubRows);
            
            return new Tuple3<>(extendedC, newA, b_ub);
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
            
            // 扩展目标函数向量，为松弛变量添加0系数
            IVector extendedC = extendObjectiveVector(c, ubRows);
            
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
}
