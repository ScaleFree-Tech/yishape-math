package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LinProgUtilTest {

    /**
     * 测试不等式约束转换为等式约束的功能
     */
    @Test
    public void testConvertUbEqToEqConstraits() {
        // 创建一个简单的线性规划问题:
        // minimize: x1
        // subject to: 
        //   x1 <= 1
        //   x1 >= 0
        
        IVector c = IVector.of(1.0);  // 目标函数系数
        IMatrix A_ub = IMatrix.of(new double[][]{{1.0}});  // 不等式约束矩阵
        IVector b_ub = IVector.of(1.0);  // 不等式约束值
        
        // 转换约束
        Tuple3<IVector, IMatrix, IVector> result = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, null, null);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result._1, "扩展的目标函数向量不应为null");
        assertNotNull(result._2, "扩展的约束矩阵不应为null");
        assertNotNull(result._3, "扩展的约束向量不应为null");
        
        // 验证扩展的目标函数向量维度（应该是2：1个原始变量 + 1个松弛变量）
        assertEquals(2, result._1.length(), "扩展的目标函数向量应该包含原始变量和松弛变量");
        
        // 验证扩展的约束矩阵维度
        assertEquals(1, result._2.rows(), "约束矩阵应该有1行");
        assertEquals(2, result._2.cols(), "约束矩阵应该有2列（原始变量+松弛变量）");
        
        // 验证扩展的约束向量维度
        assertEquals(1, result._3.length(), "约束向量应该有1个元素");
        
        // 验证目标函数系数：原始变量系数为1，松弛变量系数为0
        assertEquals(1.0, (Double)result._1.get(0), 1e-9, "原始变量系数应为1");
        assertEquals(0.0, (Double)result._1.get(1), 1e-9, "松弛变量系数应为0");
        
        // 验证约束矩阵：应该是 [1, 1]（原始变量系数 + 松弛变量系数）
        assertEquals(1.0, (Double)result._2.get(0, 0), 1e-9, "原始变量约束系数应为1");
        assertEquals(1.0, (Double)result._2.get(0, 1), 1e-9, "松弛变量约束系数应为1");
        
        // 验证约束向量：应该是 [1]
        assertEquals(1.0, (Double)result._3.get(0), 1e-9, "约束值应为1");
    }
    
    /**
     * 测试不等式约束转换为等式约束的功能（原测试）
     */
    @Test
    public void testConvertUbEqToEqConstraitsOriginal() {
        // 创建一个简单的线性规划问题:
        // minimize: -x1 - x2 (equivalent to maximizing x1 + x2)
        // subject to: 
        //   x1 + x2 <= 2
        //   x1, x2 >= 0
        
        IVector c = IVector.of(-1.0, -1.0);  // 目标函数系数
        IMatrix A_ub = IMatrix.of(new double[][]{{1.0, 1.0}});  // 不等式约束矩阵
        IVector b_ub = IVector.of(2.0);  // 不等式约束值
        
        // 转换约束
        Tuple3<IVector, IMatrix, IVector> result = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, null, null);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result._1, "扩展的目标函数向量不应为null");
        assertNotNull(result._2, "扩展的约束矩阵不应为null");
        assertNotNull(result._3, "扩展的约束向量不应为null");
        
        // 验证扩展的目标函数向量维度（应该是3：2个原始变量 + 1个松弛变量）
        assertEquals(3, result._1.length(), "扩展的目标函数向量应该包含原始变量和松弛变量");
        
        // 验证扩展的约束矩阵维度
        assertEquals(1, result._2.rows(), "约束矩阵应该有1行");
        assertEquals(3, result._2.cols(), "约束矩阵应该有3列（原始变量+松弛变量）");
        
        // 验证扩展的约束向量维度
        assertEquals(1, result._3.length(), "约束向量应该有1个元素");
        
        // 验证目标函数系数：原始变量系数为-1, -1，松弛变量系数为0
        assertEquals(-1.0, (Double)result._1.get(0), 1e-9, "第一个原始变量系数应为-1");
        assertEquals(-1.0, (Double)result._1.get(1), 1e-9, "第二个原始变量系数应为-1");
        assertEquals(0.0, (Double)result._1.get(2), 1e-9, "松弛变量系数应为0");
        
        // 验证约束矩阵：应该是 [1, 1, 1]（原始变量系数 + 松弛变量系数）
        assertEquals(1.0, (Double)result._2.get(0, 0), 1e-9, "第一个原始变量约束系数应为1");
        assertEquals(1.0, (Double)result._2.get(0, 1), 1e-9, "第二个原始变量约束系数应为1");
        assertEquals(1.0, (Double)result._2.get(0, 2), 1e-9, "松弛变量约束系数应为1");
        
        // 验证约束向量：应该是 [2]
        assertEquals(2.0, (Double)result._3.get(0), 1e-9, "约束值应为2");
    }
}