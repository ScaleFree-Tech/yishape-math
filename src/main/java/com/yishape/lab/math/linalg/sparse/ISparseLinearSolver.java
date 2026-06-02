package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IVector;

public interface ISparseLinearSolver {

    IVector<Double> solve(ISparseMatrix A, IVector<Double> b);

    IVector<Double> solve(ISparseMatrix A, IVector<Double> b, IVector<Double> x0);

    int getIterationCount();

    double getResidual();
}
