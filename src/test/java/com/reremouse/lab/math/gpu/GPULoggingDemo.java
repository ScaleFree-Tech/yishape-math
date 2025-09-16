package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUComputeFloatUtils;
import com.reremouse.lab.math.linalg.RereFloatMatrix;
import com.reremouse.lab.math.linalg.RereFloatVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IFloatVector;

/**
 * GPU日志控制演示程序
 * 展示如何使用GPUComputeUtils的日志控制功能
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GPULoggingDemo {
    
    public static void main(String[] args) {
        System.out.println("=== GPU日志控制演示程序 ===\n");
        
        // 创建测试数据
        int size = 200; // 200x200 = 40000 > 10000 (GPU阈值)
        float[][] dataA = new float[size][size];
        float[][] dataB = new float[size][size];
        float[] vectorDataA = new float[size * size];
        float[] vectorDataB = new float[size * size];
        
        // 初始化测试数据
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                dataA[i][j] = (float) (Math.random() * 10);
                dataB[i][j] = (float) (Math.random() * 10);
                vectorDataA[i * size + j] = (float) (Math.random() * 10);
                vectorDataB[i * size + j] = (float) (Math.random() * 10);
            }
        }
        
        IMatrix matrixA = new RereFloatMatrix(dataA);
        IMatrix matrixB = new RereFloatMatrix(dataB);
        IFloatVector vectorA = new RereFloatVector(vectorDataA);
        IFloatVector vectorB = new RereFloatVector(vectorDataB);
        
        // 演示1：关闭日志
        System.out.println("1. 关闭日志模式:");
        System.out.println("----------------");
        GPUComputeFloatUtils.setLoggingEnabled(false);
        matrixA.add(matrixB);
        vectorA.add(vectorB);
        System.out.println("(应该看不到任何GPU/CPU日志)\n");
        
        // 演示2：基本日志模式
        System.out.println("2. 基本日志模式:");
        System.out.println("----------------");
        GPUComputeFloatUtils.setLoggingEnabled(true);
        GPUComputeFloatUtils.setDetailedLoggingEnabled(false);
        matrixA.add(matrixB);
        vectorA.add(vectorB);
        System.out.println("(只显示GPU/CPU标识)\n");
        
        // 演示3：详细日志模式
        System.out.println("3. 详细日志模式:");
        System.out.println("----------------");
        GPUComputeFloatUtils.setDetailedLoggingEnabled(true);
        matrixA.add(matrixB);
        vectorA.add(vectorB);
        matrixA.mmul(2.0f);
        vectorA.sum();
        System.out.println("(显示GPU/CPU标识、操作详情和性能统计)\n");
        
        // 演示4：小数据量CPU回退
        System.out.println("4. 小数据量CPU回退:");
        System.out.println("------------------");
        float[][] smallData = {{1, 2}, {3, 4}};
        IMatrix smallMatrix = new RereFloatMatrix(smallData);
        IMatrix smallMatrix2 = new RereFloatMatrix(smallData);
        smallMatrix.add(smallMatrix2);
        System.out.println("(小数据量应该显示CPU回退日志)\n");
        
        // 演示5：各种GPU操作
        System.out.println("5. 各种GPU操作:");
        System.out.println("---------------");
        System.out.println("矩阵操作:");
        matrixA.sub(matrixB);
        matrixA.mmul(1.5f);
        matrixA.sub(0.5f);
        matrixA.transposeNew();
        
        System.out.println("\n向量操作:");
        vectorA.sub(vectorB);
        vectorA.multiply(vectorB);
        vectorA.addScalar(1.0f);
        vectorA.subScalar(0.5f);
        vectorA.multiplyScalar(2.0f);
        vectorA.square();
        vectorA.sqrt();
        vectorA.innerProduct(vectorB);
        
        System.out.println("\n=== 演示完成 ===");
        System.out.println("您可以通过以下方法控制日志:");
        System.out.println("- GPUComputeUtils.setLoggingEnabled(true/false)  // 启用/关闭日志");
        System.out.println("- GPUComputeUtils.setDetailedLoggingEnabled(true/false)  // 启用/关闭详细日志");
        System.out.println("- GPUComputeUtils.isLoggingEnabled()  // 检查日志状态");
        System.out.println("- GPUComputeUtils.isDetailedLoggingEnabled()  // 检查详细日志状态");
    }
}
