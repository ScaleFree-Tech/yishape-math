package com.yishape.lab.math.test;

import com.yishape.lab.math.ml.dr.RereFA;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 因子分析(FA)算法测试类
 */
public class RereFATest {

    @Test
    public void testFA() {
        System.out.println("=== 因子分析(FA)算法测试 ===");

        // 创建测试数据 (6x4矩阵: 6个样本, 4个特征)
        // 模拟一个潜在因子结构：前3个特征主要由因子1影响，第4个特征主要由因子2影响
        double[][] testData = {
            {2.5, 2.4, 1.0, 3.0},  // 样本1
            {0.5, 0.7, 0.2, 1.0},   // 样本2
            {2.2, 2.9, 1.1, 2.5},   // 样本3
            {1.9, 2.2, 0.9, 2.0},   // 样本4
            {3.0, 3.5, 1.5, 3.5},   // 样本5
            {1.0, 1.2, 0.5, 1.5}    // 样本6
        };

        IMatrix<Double> originalData = Linalg.matrix(testData);
        System.out.println("原始数据:");
        System.out.println("矩阵维度: " + originalData.getRowNum() + "x" + originalData.getColNum());
        printMatrix(originalData);

        // 创建FA对象并进行降维
        RereFA fa = new RereFA();

        // 降维到2因子
        System.out.println("\n开始FA降维...");
        IMatrix<?> reducedData = fa.dimensionReduction(originalData, 2);
        System.out.println("降维结果维度: " + reducedData.getRowNum() + "x" + reducedData.getColNum());
        System.out.println("降维后的数据（因子得分）:");
        printMatrix(reducedData);

        // 验证降维结果维度
        assertEquals(6, reducedData.getRowNum(), "样本数应保持不变");
        assertEquals(2, reducedData.getColNum(), "降维后应为2列（2个因子）");

        // 验证模型参数
        System.out.println("\n模型参数:");
        System.out.println("因子载荷矩阵维度: " + fa.getLoadings().getRowNum() + "x" + fa.getLoadings().getColNum());
        assertEquals(4, fa.getLoadings().getRowNum(), "载荷矩阵行数应为特征数");
        assertEquals(2, fa.getLoadings().getColNum(), "载荷矩阵列数应为因子数");
        System.out.println("因子载荷矩阵:");
        printMatrix(fa.getLoadings());

        System.out.println("\n唯一性方差:");
        IVector<Double> uniquenesses = fa.getUniquenesses();
        assertEquals(4, uniquenesses.size(), "唯一性方差数应为特征数");
        for (int i = 0; i < uniquenesses.size(); i++) {
            System.out.printf("  特征%d: %.4f%n", i + 1, uniquenesses.get(i));
            // 唯一性方差应为正数
            assertTrue(uniquenesses.get(i) > 0, "唯一性方差应为正数");
        }

        assertTrue(fa.isFitted(), "模型应已拟合");
        assertTrue(fa.getIterations() > 0, "应有迭代次数");

        // 测试保存功能
        String testPath = "test_fa_model.tmp";
        fa.save(testPath);
        System.out.println("\n模型已保存到: " + testPath);

        java.io.File f = new java.io.File(testPath);
        assertTrue(f.exists(), "模型文件应存在");
        System.out.println("模型文件大小: " + f.length() + " bytes");
        f.delete(); // 清理测试文件

        System.out.println("\n=== FA测试完成 ===");
    }

    @Test
    public void testFAWithDifferentFactors() {
        System.out.println("=== 测试不同因子数量 ===");

        double[][] testData = {
            {2.5, 2.4, 1.0, 3.0, 1.5},
            {0.5, 0.7, 0.2, 1.0, 0.8},
            {2.2, 2.9, 1.1, 2.5, 1.3},
            {1.9, 2.2, 0.9, 2.0, 1.1},
            {3.0, 3.5, 1.5, 3.5, 2.0},
            {1.0, 1.2, 0.5, 1.5, 0.9}
        };

        IMatrix<Double> data = Linalg.matrix(testData);
        RereFA fa = new RereFA();

        // 降维到1因子
        IMatrix<?> reduced1 = fa.dimensionReduction(data, 1);
        assertEquals(6, reduced1.getRowNum());
        assertEquals(1, reduced1.getColNum());

        // 降维到2因子
        IMatrix<?> reduced2 = fa.dimensionReduction(data, 2);
        assertEquals(6, reduced2.getRowNum());
        assertEquals(2, reduced2.getColNum());

        System.out.println("1因子降维结果维度: " + reduced1.getRowNum() + "x" + reduced1.getColNum());
        System.out.println("2因子降维结果维度: " + reduced2.getRowNum() + "x" + reduced2.getColNum());
    }

    @Test
    public void testFAValidation() {
        System.out.println("=== 测试参数验证 ===");

        double[][] testData = {{1.0, 2.0}, {3.0, 4.0}};
        IMatrix<Double> data = Linalg.matrix(testData);
        RereFA fa = new RereFA();

        // 测试空数据
        assertThrows(IllegalArgumentException.class, () -> {
            fa.dimensionReduction(null, 1);
        });

        // 测试目标维度为0
        assertThrows(IllegalArgumentException.class, () -> {
            fa.dimensionReduction(data, 0);
        });

        // 测试目标维度 >= 原始维度
        assertThrows(IllegalArgumentException.class, () -> {
            fa.dimensionReduction(data, 2);  // 原始是2维
        });

        // 测试样本数过少
        double[][] singleSample = {{1.0, 2.0}};
        IMatrix<Double> singleSampleData = Linalg.matrix(singleSample);
        assertThrows(IllegalArgumentException.class, () -> {
            fa.dimensionReduction(singleSampleData, 1);
        });

        System.out.println("参数验证测试通过!");
    }

    private static void printMatrix(IMatrix<?> matrix) {
        try {
            double[][] data = matrix.toDoubleArray();
            System.out.println("矩阵维度: " + data.length + "x" + data[0].length);
            for (double[] row : data) {
                for (int j = 0; j < row.length; j++) {
                    System.out.printf("%10.4f ", row[j]);
                }
                System.out.println();
            }
        } catch (Exception e) {
            System.err.println("打印矩阵时出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
