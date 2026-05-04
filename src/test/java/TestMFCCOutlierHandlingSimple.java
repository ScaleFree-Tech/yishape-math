import com.yishape.lab.audio.Audios;
import com.yishape.lab.audio.AudioPlots;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * 简单测试MFCC异常值处理功能
 */
public class TestMFCCOutlierHandlingSimple {
    public static void main(String[] args) {
        try {
            // 创建一个简单的测试矩阵来验证异常值处理功能
            System.out.println("测试MFCC异常值处理功能...");
            
            // 创建一个包含异常值的测试矩阵
            double[][] testData = {
                {1.0, 2.0, 3.0, 4.0},
                {2.0, 3.0, 4.0, 5.0},
                {3.0, 4.0, 5.0, 6.0},
                {100.0, 4.0, 5.0, 6.0}, // 异常值
                {3.0, 4.0, 5.0, 6.0},
                {3.0, 4.0, 50.0, 6.0},  // 异常值
                {3.0, 4.0, 5.0, 6.0}
            };
            
            IMatrix<Double> testMatrix = Linalg.matrix(testData);
            System.out.println("原始矩阵:");
            printMatrix(testMatrix);
            
            // 测试异常值处理功能
            IMatrix<Double> processedMatrix = removeOutliers(testMatrix, 2.0);
            System.out.println("\n处理后的矩阵 (阈值=2.0):");
            printMatrix(processedMatrix);
            
            System.out.println("\n测试完成！");
            
        } catch (Exception e) {
            System.err.println("测试过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 移除矩阵中的异常值 / Remove outliers from matrix
     * 
     * @param matrix 输入矩阵 / Input matrix
     * @param threshold 标准差倍数阈值 / Standard deviation multiplier threshold
     * @return 处理后的矩阵 / Processed matrix
     */
    private static IMatrix<Double> removeOutliers(IMatrix<Double> matrix, double threshold) {
        // 计算矩阵的均值和标准差
        double mean = matrix.mean().doubleValue();
        double std = matrix.std().doubleValue();
        
        System.out.println("矩阵均值: " + mean);
        System.out.println("矩阵标准差: " + std);
        
        // 定义异常值范围
        double lowerBound = mean - threshold * std;
        double upperBound = mean + threshold * std;
        
        System.out.println("异常值范围: [" + lowerBound + ", " + upperBound + "]");
        
        // 创建新的矩阵存储处理后的数据
        IMatrix<Double> processedMatrix = Linalg.zeros(matrix.rows(), matrix.cols());
        
        // 处理每个元素
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                double value = matrix.get(i, j).doubleValue();
                // 将异常值替换为边界值
                if (value < lowerBound) {
                    System.out.println("发现下界异常值: " + value + " at (" + i + ", " + j + ")");
                    processedMatrix.set(i, j, lowerBound);
                } else if (value > upperBound) {
                    System.out.println("发现上界异常值: " + value + " at (" + i + ", " + j + ")");
                    processedMatrix.set(i, j, upperBound);
                } else {
                    processedMatrix.set(i, j, value);
                }
            }
        }
        
        return processedMatrix;
    }
    
    /**
     * 打印矩阵
     */
    private static void printMatrix(IMatrix<Double> matrix) {
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                System.out.printf("%8.2f ", matrix.get(i, j).doubleValue());
            }
            System.out.println();
        }
    }
}