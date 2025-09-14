package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.RereMatrix;

/**
 * 快速伪逆测试类
 * 用于验证伪逆计算的性能优化
 */
public class QuickPinvTest {
    
    public static void main(String[] args) {
        System.out.println("开始测试Matrix Pseudo-Inverse性能优化...");
        
        // 测试不同大小的矩阵
        int[] sizes = {10, 20, 30, 40, 50, 100};
        
        for (int size : sizes) {
            System.out.println("\n正在测试: Matrix Pseudo-Inverse (大小: " + size + ")");
            
            try {
                // 创建随机矩阵
                IMatrix matrix = IMatrix.rand(size, size);
                
                // 记录开始时间
                long startTime = System.currentTimeMillis();
                
                // 计算伪逆
                IMatrix pseudoInverse = matrix.pinv();
                
                // 记录结束时间
                long endTime = System.currentTimeMillis();
                long duration = endTime - startTime;
                
                System.out.println("✓ 成功完成，耗时: " + duration + "ms");
                System.out.println("  伪逆矩阵维度: " + pseudoInverse.getRows() + "x" + pseudoInverse.getColumns());
                
                // 验证伪逆的基本性质：A * A⁺ * A ≈ A
                IMatrix verification = matrix.mmul(pseudoInverse).mmul(matrix);
                IMatrix diff = matrix.sub(verification);
                float maxError = 0.0f;
                for (int i = 0; i < diff.getRows(); i++) {
                    for (int j = 0; j < diff.getColumns(); j++) {
                        maxError = Math.max(maxError, Math.abs(diff.get(i, j)));
                    }
                }
                System.out.println("  最大误差: " + maxError);
                
            } catch (Exception e) {
                System.err.println("✗ 测试失败: " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        System.out.println("\n测试完成！");
    }
}
