package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;

/**
 * 线性规划问题的工具类
 * @author lteb2
 */
public class LinProgUtil {
    /**
     * 将solve中的参数值（不等式约束、不等式约束值、等式约束、等式约束值）转换为solveWithEqualConstraits需要的等式形式
     *
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（不等式右方）
     * @return 返回solveWithEqualConstraits需要的等式形式
     */
    public static Tuple2<IMatrix, IVector> convertUbEqToEqConstraits(IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq) {
        // 获取约束维度信息
        int ubRows = A_ub == null ? 0 : A_ub.rows();
        int ubCols = A_ub == null ? 0 : A_ub.cols();
        int eqRows = A_eq == null ? 0 : A_eq.rows();
        int eqCols = A_eq == null ? 0 : A_eq.cols();
        
        // 计算变量总数（原始变量 + 松弛变量）
        int totalCols = ubCols + ubRows;  // 原始变量数 + 松弛变量数
        
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
}
