package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecomposition;
import com.yishape.lab.util.Tuple3;
import org.apache.commons.math4.legacy.linear.SingularValueDecomposition;
import org.apache.commons.math4.legacy.linear.MatrixUtils;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SVDAccuracyTest {

    @Test
    public void testSmallMatrixSVD() {
        // 测试小矩阵，应该使用traditionalSVD方法
        double[][] smallData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(smallData);
        RereSVDDecomposition svd_simple = new RereSVDDecomposition();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd_simple.decompose(matrix);
        
        // 验证分解结果的基本属性
        assertNotNull(result.getFirst());
        assertNotNull(result.getSecond());
        assertNotNull(result.getThird());
        
        System.out.println("Small matrix SVD completed");
        System.out.println("U dimensions: " + result.getFirst().getRowNum() + "x" + result.getFirst().getColNum());
        System.out.println("S length: " + result.getSecond().length());
        System.out.println("VT dimensions: " + result.getThird().getRowNum() + "x" + result.getThird().getColNum());
    }
    
    @Test
    public void testMediumMatrixSVD() {
        // 测试中等矩阵，应该使用bidiagonalSVD方法
        double[][] mediumData = new double[50][30];
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 30; j++) {
                mediumData[i][j] = Math.random();
            }
        }
        
        IMatrix<Double> matrix = Linalg.matrix(mediumData);
        RereSVDDecomposition svd_orig = new RereSVDDecomposition();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd_orig.decompose(matrix);
        
        // 验证分解结果的基本属性
        assertNotNull(result.getFirst());
        assertNotNull(result.getSecond());
        assertNotNull(result.getThird());
        
        System.out.println("Medium matrix SVD completed");
        System.out.println("U dimensions: " + result.getFirst().getRowNum() + "x" + result.getFirst().getColNum());
        System.out.println("S length: " + result.getSecond().length());
        System.out.println("VT dimensions: " + result.getThird().getRowNum() + "x" + result.getThird().getColNum());
    }
    
    @Test
    public void testLargeMatrixSVD() {
        // 测试大矩阵，应该使用optimizedSVD方法
        double[][] largeData = new double[150][100];
        for (int i = 0; i < 150; i++) {
            for (int j = 0; j < 100; j++) {
                largeData[i][j] = Math.random();
            }
        }
        
        IMatrix<Double> matrix = Linalg.matrix(largeData);
        RereSVDDecomposition svd_opt = new RereSVDDecomposition();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd_opt.decompose(matrix);
        
        // 验证分解结果的基本属性
        assertNotNull(result.getFirst());
        assertNotNull(result.getSecond());
        assertNotNull(result.getThird());
        
        System.out.println("Large matrix SVD completed");
        System.out.println("U dimensions: " + result.getFirst().getRowNum() + "x" + result.getFirst().getColNum());
        System.out.println("S length: " + result.getSecond().length());
        System.out.println("VT dimensions: " + result.getThird().getRowNum() + "x" + result.getThird().getColNum());
    }
    
    @Test
    public void testSVDReconstruction() {
        // 测试SVD分解的准确性，通过重构原始矩阵验证
        // 使用一个非奇异矩阵进行测试
        double[][] testData = {
            {2.0, 1.0, 1.0},
            {1.0, 3.0, 2.0},
            {1.0, 0.0, 1.0}
        };
        
        IMatrix<Double> originalMatrix = Linalg.matrix(testData);
        RereSVDDecomposition rereSVD = new RereSVDDecomposition();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = rereSVD.decompose(originalMatrix);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        // 重构矩阵: A = U * S * V^T
        int m = U.getRowNum();
        int n = VT.getColNum();
        int rank = S.length();
        
        // 创建对角矩阵S
        IMatrix<Double> S_matrix = Linalg.zeros(rank, rank);
        for (int i = 0; i < rank; i++) {
            S_matrix.put(i, i, S.get(i));
        }
        
        // 计算 U * S
        IMatrix<Double> US = U.mmul(S_matrix);
        
        // 计算 (U * S) * V^T
        IMatrix<Double> reconstructed = US.mmul(VT);
        
        // 验证重构矩阵与原始矩阵是否相等（在一定误差范围内）
        double tolerance = 1e-10;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double diff = Math.abs(originalMatrix.get(i, j) - reconstructed.get(i, j));
                assertTrue(diff < tolerance, "Matrix reconstruction failed at position (" + i + "," + j + 
                          "), difference: " + diff);
            }
        }
        
        System.out.println("SVD reconstruction test passed");
    }
    
    @Test
    public void testComparisonWithCommonsMath() {
        // 与Apache Commons Math的结果进行比较
        // 使用一个更合适的测试矩阵
        double[][] testData = {
            {2.0, 1.0, 0.0},
            {1.0, 2.0, 1.0},
            {0.0, 1.0, 2.0}
        };
        
        // RereMouse SVD
        IMatrix<Double> matrix = Linalg.matrix(testData);
        RereSVDDecomposition rereSVD = new RereSVDDecomposition();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> rereResult = rereSVD.decompose(matrix);
        
        // Commons Math SVD
        RealMatrix commonsMatrix = MatrixUtils.createRealMatrix(testData);
        SingularValueDecomposition commonsSVD = new SingularValueDecomposition(commonsMatrix);
        double[] commonsSingularValues = commonsSVD.getSingularValues();
        
        // 比较奇异值
        IVector<Double> rereSingularValues = rereResult.getSecond();
        assertEquals(commonsSingularValues.length, rereSingularValues.length(), 
                    "Number of singular values should match");
        
        // 使用更宽松的容差，因为不同的SVD实现可能会有微小差异
        double tolerance = 1e-10;
        for (int i = 0; i < commonsSingularValues.length; i++) {
            double diff = Math.abs(commonsSingularValues[i] - rereSingularValues.get(i));
            assertTrue(diff < tolerance, "Singular value " + i + " differs: RereMouse=" + rereSingularValues.get(i) + 
                      ", CommonsMath=" + commonsSingularValues[i] + ", diff=" + diff);
        }
        
        System.out.println("Comparison with Commons Math test passed");
    }
    
    // Helper method to compare vectors
    private void assertVectorClose(IVector<Double> rereVector, RealVector commonsVector, double tolerance) {
        assertEquals(commonsVector.getDimension(), rereVector.length(), "Vector dimensions should match");
        for (int i = 0; i < rereVector.length(); i++) {
            double diff = Math.abs(rereVector.get(i) - commonsVector.getEntry(i));
            assertTrue(diff < tolerance, "Vector element " + i + " differs: RereMouse=" + rereVector.get(i) + 
                      ", CommonsMath=" + commonsVector.getEntry(i) + ", diff=" + diff);
        }
    }
    
    // Helper method to compare matrices
    private void assertMatrixClose(IMatrix<Double> rereMatrix, RealMatrix commonsMatrix, double tolerance) {
        assertEquals(commonsMatrix.getRowDimension(), rereMatrix.getRowNum(), "Matrix row dimensions should match");
        assertEquals(commonsMatrix.getColumnDimension(), rereMatrix.getColNum(), "Matrix column dimensions should match");
        for (int i = 0; i < rereMatrix.getRowNum(); i++) {
            for (int j = 0; j < rereMatrix.getColNum(); j++) {
                double diff = Math.abs(rereMatrix.get(i, j) - commonsMatrix.getEntry(i, j));
                assertTrue(diff < tolerance, "Matrix element (" + i + "," + j + ") differs: RereMouse=" + rereMatrix.get(i, j) + 
                          ", CommonsMath=" + commonsMatrix.getEntry(i, j) + ", diff=" + diff);
            }
        }
    }
}