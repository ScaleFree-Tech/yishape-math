package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

public class SparseILUPreconditioner implements ISparsePreconditioner {

    private final double dropTolerance;
    private int[] rowPtr;
    private int[] colInd;
    private double[] values;
    private int n;

    public SparseILUPreconditioner(double dropTolerance) {
        this.dropTolerance = dropTolerance;
    }

    public SparseILUPreconditioner() {
        this(0.0);
    }

    @Override
    public void factor(ISparseMatrix A) {
        if (!(A instanceof ISparseMatrix.RereSparseDoubleMatrix r)) {
            ISparseMatrix csr = A.toFormat(SparseFormat.CSR);
            if (!(csr instanceof ISparseMatrix.RereSparseDoubleMatrix r2)) {
                throw new IllegalArgumentException("Unsupported sparse matrix type");
            }
            factorInternal(r2);
            return;
        }
        factorInternal(r);
    }

    private void factorInternal(ISparseMatrix.RereSparseDoubleMatrix A) {
        n = A.rows();
        int[] aRowPtr = A.rowPtrRef();
        int[] aColInd = A.colIndRef();
        double[] aValues = A.valuesRef();

        rowPtr = aRowPtr.clone();
        int nnz = aValues.length;
        colInd = aColInd.clone();
        values = aValues.clone();

        for (int i = 0; i < n; i++) {
            for (int jp = rowPtr[i]; jp < rowPtr[i + 1]; jp++) {
                int j = colInd[jp];
                if (j >= i) break;

                double multiplier = values[jp];
                if (Math.abs(multiplier) < dropTolerance) continue;

                int diagIdx = findDiagonal(j);
                if (diagIdx < 0) continue;
                double diagVal = values[diagIdx];
                if (Math.abs(diagVal) < 1e-15) continue;
                multiplier /= diagVal;
                values[jp] = multiplier;

                for (int kp = rowPtr[i]; kp < rowPtr[i + 1]; kp++) {
                    int k = colInd[kp];
                    if (k <= j) continue;
                    int ukIdx = findInRow(j, k);
                    if (ukIdx >= 0) {
                        values[kp] -= multiplier * values[ukIdx];
                        if (Math.abs(values[kp]) < dropTolerance) {
                            values[kp] = 0.0;
                        }
                    }
                }
            }
        }
    }

    private int findInRow(int row, int col) {
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; j++) {
            if (colInd[j] == col) return j;
        }
        return -1;
    }

    private int findDiagonal(int row) {
        int start = rowPtr[row];
        int end = rowPtr[row + 1];
        int lo = start, hi = end - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            int c = colInd[mid];
            if (c < row) lo = mid + 1;
            else if (c > row) hi = mid - 1;
            else return mid;
        }
        return -1;
    }

    @Override
    public IVector<Double> apply(IVector<Double> r) {
        double[] y = new double[n];
        double[] z = new double[n];
        double[] rArr = new double[n];
        for (int i = 0; i < n; i++) rArr[i] = r.get(i);

        // Forward: L y = r  (L unit diagonal)
        for (int i = 0; i < n; i++) {
            double sum = 0;
            int start = rowPtr[i];
            int end = rowPtr[i + 1];
            for (int j = start; j < end; j++) {
                int col = colInd[j];
                if (col >= i) break;
                sum += values[j] * y[col];
            }
            y[i] = rArr[i] - sum;
        }

        // Backward: U z = y
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            int start = rowPtr[i];
            int end = rowPtr[i + 1];
            for (int j = start; j < end; j++) {
                int col = colInd[j];
                if (col <= i) continue;
                sum += values[j] * z[col];
            }
            int diagIdx = findDiagonal(i);
            double diag = (diagIdx >= 0) ? values[diagIdx] : 0.0;
            if (Math.abs(diag) < 1e-15) {
                z[i] = y[i];
            } else {
                z[i] = (y[i] - sum) / diag;
            }
        }

        return Linalg.vector(z);
    }
}
