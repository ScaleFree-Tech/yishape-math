package com.yishape.lab.math.linalg;

import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecomposition;
import com.yishape.lab.util.Tuple3;
import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.SingularValueDecomposition;
import org.junit.jupiter.api.Test;

/**
 * Debug test for SVD computation issues
 */
public class SVDDebugTest {
    
    @Test
    void debugSVDComparison() {
        System.out.println("=== SVD Debug Comparison ===");
        
        // Test matrix from SolverComparisonTest
        double[][] testData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}
        };
        
        System.out.println("Test matrix (4x3):");
        for (int i = 0; i < testData.length; i++) {
            for (int j = 0; j < testData[i].length; j++) {
                System.out.printf("%.1f ", testData[i][j]);
            }
            System.out.println();
        }
        System.out.println();
        
        // RereMouse SVD
        System.out.println("=== RereMouse SVD Results ===");
        IMatrix<Double> rereMatrix = Linalg.matrix(testData);
        RereSVDDecomposition rereSVD = new RereSVDDecomposition();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> rereResult = rereSVD.decompose(rereMatrix);
        
        IMatrix<Double> rereU = rereResult._1;
        IVector<Double> rereS = rereResult._2;
        IMatrix<Double> rereVT = rereResult._3;
        
        System.out.println("Singular values:");
        for (int i = 0; i < rereS.length(); i++) {
            System.out.printf("%.10f ", rereS.get(i));
        }
        System.out.println();
        
        System.out.println("U matrix:");
        for (int i = 0; i < rereU.rows(); i++) {
            for (int j = 0; j < rereU.cols(); j++) {
                System.out.printf("%.6f ", rereU.get(i, j));
            }
            System.out.println();
        }
        
        System.out.println("V^T matrix:");
        for (int i = 0; i < rereVT.rows(); i++) {
            for (int j = 0; j < rereVT.cols(); j++) {
                System.out.printf("%.6f ", rereVT.get(i, j));
            }
            System.out.println();
        }
        System.out.println();
        
        // Commons Math SVD
        System.out.println("=== Commons Math SVD Results ===");
        RealMatrix commonsMatrix = new Array2DRowRealMatrix(testData);
        SingularValueDecomposition commonsSVD = new SingularValueDecomposition(commonsMatrix);
        
        double[] commonsSingularValues = commonsSVD.getSingularValues();
        RealMatrix commonsU = commonsSVD.getU();
        RealMatrix commonsVT = commonsSVD.getVT();
        
        System.out.println("Singular values:");
        for (double sv : commonsSingularValues) {
            System.out.printf("%.10f ", sv);
        }
        System.out.println();
        
        System.out.println("U matrix:");
        for (int i = 0; i < commonsU.getRowDimension(); i++) {
            for (int j = 0; j < commonsU.getColumnDimension(); j++) {
                System.out.printf("%.6f ", commonsU.getEntry(i, j));
            }
            System.out.println();
        }
        
        System.out.println("V^T matrix:");
        for (int i = 0; i < commonsVT.getRowDimension(); i++) {
            for (int j = 0; j < commonsVT.getColumnDimension(); j++) {
                System.out.printf("%.6f ", commonsVT.getEntry(i, j));
            }
            System.out.println();
        }
        System.out.println();
        
        // Verification: Reconstruct original matrix
        System.out.println("=== Matrix Reconstruction Verification ===");
        
        // RereMouse reconstruction
        System.out.println("RereMouse reconstruction:");
        IMatrix<Double> rereV = rereVT.transpose();
        IMatrix<Double> rereSMatrix = Linalg.zeros(rereU.cols(), rereV.rows());
        for (int i = 0; i < Math.min(rereS.length(), Math.min(rereSMatrix.rows(), rereSMatrix.cols())); i++) {
            rereSMatrix.put(i, i, rereS.get(i));
        }
        IMatrix<Double> rereReconstructed = rereU.mmul(rereSMatrix).mmul(rereVT);
        
        for (int i = 0; i < rereReconstructed.rows(); i++) {
            for (int j = 0; j < rereReconstructed.cols(); j++) {
                System.out.printf("%.6f ", rereReconstructed.get(i, j));
            }
            System.out.println();
        }
        
        // Commons Math reconstruction
        System.out.println("Commons Math reconstruction:");
        RealMatrix commonsSMatrix = commonsSVD.getS();
        RealMatrix commonsReconstructed = commonsU.multiply(commonsSMatrix).multiply(commonsVT);
        
        for (int i = 0; i < commonsReconstructed.getRowDimension(); i++) {
            for (int j = 0; j < commonsReconstructed.getColumnDimension(); j++) {
                System.out.printf("%.6f ", commonsReconstructed.getEntry(i, j));
            }
            System.out.println();
        }
    }
}