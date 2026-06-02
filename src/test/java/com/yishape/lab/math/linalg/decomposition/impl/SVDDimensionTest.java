package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.util.Tuple3;

public class SVDDimensionTest {
    public static void main(String[] args) {
        ISVDDecomposition svd = new RereSVDDecompBlas2();
        
        // 创建一个非方阵 (2x4)
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        System.out.println("Original matrix dimensions: " + matrix.rows() + "x" + matrix.cols());
        
        Tuple3<IMatrix<Double>, com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> result = 
            svd.decompose(matrix);
        
        System.out.println("U matrix dimensions: " + result.getFirst().rows() + "x" + result.getFirst().cols());
        System.out.println("Singular values length: " + result.getSecond().length());
        System.out.println("V transpose matrix dimensions: " + result.getThird().rows() + "x" + result.getThird().cols());
    }
}