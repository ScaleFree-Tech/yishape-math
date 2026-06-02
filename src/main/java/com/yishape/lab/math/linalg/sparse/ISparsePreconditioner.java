package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IVector;

public interface ISparsePreconditioner {

    IVector<Double> apply(IVector<Double> r);

    void factor(ISparseMatrix A);
}
