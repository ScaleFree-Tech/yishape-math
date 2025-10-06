package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.decomposition.ICholeskyDecomposition;
import com.yishape.lab.math.linalg.decomposition.Decomps;

public class ImportTest {
    public static void main(String[] args) {
        // This is just a test to verify imports work
        ICholeskyDecomposition chol = Decomps.createCholesky();
        System.out.println("Import test successful");
    }
}