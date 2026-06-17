package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

public interface ISparseMatrix {

    static ISparseMatrix fromDense(double[][] data) {
        return fromDense(data, 1e-10);
    }

    static ISparseMatrix fromDense(double[][] data, double tolerance) {
        int rows = data.length;
        int cols = data[0].length;
        int[] rowIdx = new int[rows * cols];
        int[] colIdx = new int[rows * cols];
        double[] values = new double[rows * cols];
        int nnz = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(data[i][j]) > tolerance) {
                    rowIdx[nnz] = i;
                    colIdx[nnz] = j;
                    values[nnz] = data[i][j];
                    nnz++;
                }
            }
        }

        int[] rowIdxTrim = new int[nnz];
        int[] colIdxTrim = new int[nnz];
        double[] valuesTrim = new double[nnz];
        System.arraycopy(rowIdx, 0, rowIdxTrim, 0, nnz);
        System.arraycopy(colIdx, 0, colIdxTrim, 0, nnz);
        System.arraycopy(values, 0, valuesTrim, 0, nnz);

        return new RereSparseDoubleMatrix(rows, cols, rowIdxTrim, colIdxTrim, valuesTrim, SparseFormat.COO);
    }

    static ISparseMatrix fromCSC(int[] rowPtr, int[] colInd, double[] values, int rows, int cols) {
        return new RereSparseDoubleMatrix(rows, cols, rowPtr.clone(), colInd.clone(), values.clone(), SparseFormat.CSC);
    }

    static ISparseMatrix fromCSR(int[] rowPtr, int[] colInd, double[] values, int rows, int cols) {
        return new RereSparseDoubleMatrix(rows, cols, rowPtr.clone(), colInd.clone(), values.clone(), SparseFormat.CSR);
    }

    static ISparseMatrix fromCOO(int[] rowIdx, int[] colIdx, double[] values, int rows, int cols) {
        return new RereSparseDoubleMatrix(rows, cols, rowIdx.clone(), colIdx.clone(), values.clone(), SparseFormat.COO);
    }

    static ISparseMatrix eye(int size) {
        return eye(size, size);
    }

    static ISparseMatrix eye(int rows, int cols) {
        int n = Math.min(rows, cols);
        int[] rowIdx = new int[n];
        int[] colIdx = new int[n];
        double[] values = new double[n];
        for (int i = 0; i < n; i++) {
            rowIdx[i] = i;
            colIdx[i] = i;
            values[i] = 1.0;
        }
        ISparseMatrix matrix = new RereSparseDoubleMatrix(rows, cols, rowIdx, colIdx, values, SparseFormat.COO);
        return matrix;
    }

    /** Create sparse matrix with all entries set to 1.0 (COO dense-sparse). */
    static ISparseMatrix ones(int rows, int cols) {
        int total = rows * cols;
        int[] rowIdx = new int[total];
        int[] colIdx = new int[total];
        double[] vals = new double[total];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                rowIdx[idx] = i;
                colIdx[idx] = j;
                vals[idx] = 1.0;
            }
        }
        return new RereSparseDoubleMatrix(rows, cols, rowIdx, colIdx, vals, SparseFormat.COO);
    }

    static ISparseMatrix diag(double[] values) {
        return diag(values, values.length);
    }

    static ISparseMatrix diag(double[] values, int size) {
        int n = values.length;
        int[] rowIdx = new int[n];
        int[] colIdx = new int[n];
        double[] vals = new double[n];
        for (int i = 0; i < n; i++) {
            rowIdx[i] = i;
            colIdx[i] = i;
            vals[i] = values[i];
        }
        return new RereSparseDoubleMatrix(size, size, rowIdx, colIdx, vals, SparseFormat.COO);
    }

    int rows();

    int cols();

    int nnz();

    double sparsity();

    SparseFormat format();

    ISparseMatrix toFormat(SparseFormat targetFormat);

    IMatrix<Double> toDense();

    double[][] toDenseArray();

    double get(int row, int col);

    ISparseMatrix add(ISparseMatrix other);

    ISparseMatrix sub(ISparseMatrix other);

    ISparseMatrix scale(double scalar);

    ISparseMatrix multiply(ISparseMatrix other);

    IVector<Double> multiply(IVector<Double> vector);

    default IMatrix<Double> multiplyDense(IMatrix<Double> dense) {
        if (cols() != dense.nrow()) {
            throw new IllegalArgumentException("Sparse columns must match dense rows for multiplication");
        }
        int m = rows();
        int n = dense.ncol();
        double[][] result = new double[m][n];
        for (int j = 0; j < n; j++) {
            IVector<Double> col = dense.getColumn(j);
            IVector<Double> prod = this.multiply(col);
            for (int i = 0; i < m; i++) {
                result[i][j] = prod.get(i);
            }
        }
        return Linalg.matrix(result);
    }

    default IMatrix<Double> multiplyDenseFromLeft(IMatrix<Double> dense) {
        if (dense.ncol() != rows()) {
            throw new IllegalArgumentException("Dense columns must match sparse rows for multiplication");
        }
        ISparseMatrix t = this.transpose();
        IMatrix<Double> dt = dense.transpose();
        IMatrix<Double> prodT = t.multiplyDense(dt);
        return prodT.transpose();
    }

    ISparseMatrix transpose();

    ISparseMatrix conjugateTranspose();

    double frobeniusNorm();

    ISparseMatrix hadamard(ISparseMatrix other);

    ISparseMatrix copy();

    class RereSparseDoubleMatrix implements ISparseMatrix {
        private final int rows;
        private final int cols;
        private int[] rowPtr;
        private int[] colInd;
        private double[] values;
        private SparseFormat format;
        private int cachedNnz = -1;

        public RereSparseDoubleMatrix(int rows, int cols, int[] rowPtr, int[] colInd, double[] values, SparseFormat format) {
            this.rows = rows;
            this.cols = cols;
            this.rowPtr = rowPtr;
            this.colInd = colInd;
            this.values = values;
            this.format = format;
        }

        @Override
        public int rows() { return rows; }

        @Override
        public int cols() { return cols; }

        @Override
        public int nnz() {
            if (cachedNnz == -1) {
                cachedNnz = values.length;
            }
            return cachedNnz;
        }

        @Override
        public double sparsity() {
            return 1.0 - (double) nnz() / (rows * cols);
        }

        @Override
        public SparseFormat format() { return format; }

        private void ensureCSR() {
            if (format == SparseFormat.CSR) return;
            if (format == SparseFormat.CSC) {
                toCSR();
            } else if (format == SparseFormat.COO) {
                cooToCsr();
            }
        }

        private void ensureCSC() {
            if (format == SparseFormat.CSC) return;
            if (format == SparseFormat.CSR) {
                csrToCsc();
            } else if (format == SparseFormat.COO) {
                cooToCsr();
                csrToCsc();
            }
        }

        private void ensureCOO() {
            if (format == SparseFormat.COO) return;
            if (format == SparseFormat.CSR) {
                csrToCoo();
            } else if (format == SparseFormat.CSC) {
                cscToCoo();
            }
        }

        private void cooToCsr() {
            int[] rowPtrNew = new int[rows + 1];
            int[] colIndNew = new int[nnz()];
            double[] valuesNew = new double[nnz()];

            for (int i = 0; i < nnz(); i++) {
                rowPtrNew[rowPtr[i] + 1]++;
            }
            for (int i = 0; i < rows; i++) {
                rowPtrNew[i + 1] += rowPtrNew[i];
            }

            int[] counter = new int[rows + 1];
            System.arraycopy(rowPtrNew, 0, counter, 0, rows + 1);

            for (int i = 0; i < nnz(); i++) {
                int row = this.rowPtr[i];
                int idx = counter[row]++;
                colIndNew[idx] = colInd[i];
                valuesNew[idx] = values[i];
            }

            this.rowPtr = rowPtrNew;
            this.colInd = colIndNew;
            this.values = valuesNew;
            this.format = SparseFormat.CSR;
        }

        private void csrToCsc() {
            int[] colPtrNew = new int[cols + 1];
            int[] rowIndNew = new int[nnz()];
            double[] valuesNew = new double[nnz()];

            for (int i = 0; i < nnz(); i++) {
                colPtrNew[colInd[i] + 1]++;
            }
            for (int i = 0; i < cols; i++) {
                colPtrNew[i + 1] += colPtrNew[i];
            }

            int[] counter = new int[cols + 1];
            System.arraycopy(colPtrNew, 0, counter, 0, cols + 1);

            for (int i = 0; i < rows; i++) {
                for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
                    int col = colInd[j];
                    int idx = counter[col]++;
                    rowIndNew[idx] = i;
                    valuesNew[idx] = values[j];
                }
            }

            this.rowPtr = colPtrNew;
            this.colInd = rowIndNew;
            this.values = valuesNew;
            this.format = SparseFormat.CSC;
        }

        private void csrToCoo() {
            int[] rowIdxNew = new int[nnz()];
            int[] colIdxNew = new int[nnz()];
            double[] valuesNew = new double[nnz()];

            int idx = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
                    rowIdxNew[idx] = i;
                    colIdxNew[idx] = colInd[j];
                    valuesNew[idx] = values[j];
                    idx++;
                }
            }

            this.rowPtr = rowIdxNew;
            this.colInd = colIdxNew;
            this.values = valuesNew;
            this.format = SparseFormat.COO;
        }

        private void cscToCoo() {
            int[] colIdxNew = new int[nnz()];
            int[] rowIdxNew = new int[nnz()];
            double[] valuesNew = new double[nnz()];

            int idx = 0;
            for (int j = 0; j < cols; j++) {
                for (int i = rowPtr[j]; i < rowPtr[j + 1]; i++) {
                    colIdxNew[idx] = j;
                    rowIdxNew[idx] = colInd[i];
                    valuesNew[idx] = values[i];
                    idx++;
                }
            }

            this.rowPtr = rowIdxNew;
            this.colInd = colIdxNew;
            this.values = valuesNew;
            this.format = SparseFormat.COO;
        }

        private void toCSR() {
            if (format == SparseFormat.CSR) return;
            if (format == SparseFormat.COO) {
                cooToCsr();
            } else if (format == SparseFormat.CSC) {
                cscToCoo();
                cooToCsr();
            }
        }

        @Override
        public ISparseMatrix toFormat(SparseFormat targetFormat) {
            if (format == targetFormat) {
                return copy();
            }

            switch (targetFormat) {
                case CSR:
                    ensureCSR();
                    return copy();
                case CSC:
                    ensureCSC();
                    return copy();
                case COO:
                    ensureCOO();
                    return copy();
                default:
                    throw new IllegalArgumentException("Unknown format: " + targetFormat);
            }
        }

        @Override
        public IMatrix<Double> toDense() {
            return Linalg.matrix(toDenseArray());
        }

        @Override
        public double[][] toDenseArray() {
            double[][] result = new double[rows][cols];
            ensureCOO();
            for (int i = 0; i < nnz(); i++) {
                result[rowPtr[i]][colInd[i]] = values[i];
            }
            return result;
        }

        @Override
        public double get(int row, int col) {
            if (row < 0 || row >= rows || col < 0 || col >= cols) {
                throw new IndexOutOfBoundsException("Index (" + row + "," + col + ") out of bounds for " + rows + "x" + cols);
            }
            if (format == SparseFormat.CSR) {
                for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
                    if (colInd[i] == col) {
                        return values[i];
                    }
                }
            } else if (format == SparseFormat.CSC) {
                for (int i = rowPtr[col]; i < rowPtr[col + 1]; i++) {
                    if (colInd[i] == row) {
                        return values[i];
                    }
                }
            } else {
                for (int i = 0; i < nnz(); i++) {
                    if (rowPtr[i] == row && colInd[i] == col) {
                        return values[i];
                    }
                }
            }
            return 0.0;
        }

        @Override
        public ISparseMatrix add(ISparseMatrix other) {
            if (rows != other.rows() || cols != other.cols()) {
                throw new IllegalArgumentException("Matrix dimensions must match for addition");
            }
            // Sparse addition via hash accumulation — avoids O(rows*cols) dense intermediate.
            // Iterates only over non-zero entries (structural), not element-wise numerical.
            java.util.HashMap<Long, Double> acc = new java.util.HashMap<>(nnz() + other.nnz());
            // Accumulate this matrix's non-zero entries
            RereSparseDoubleMatrix selfCopy = (RereSparseDoubleMatrix) this.copy();
            selfCopy.ensureCOO();
            for (int i = 0; i < selfCopy.values.length; i++) { // structural: only non-zeros
                long key = (long) selfCopy.rowPtr[i] * cols + selfCopy.colInd[i];
                acc.merge(key, selfCopy.values[i], Double::sum);
            }
            // Accumulate other matrix's non-zero entries (avoid mutating other)
            RereSparseDoubleMatrix o = (RereSparseDoubleMatrix) other.copy();
            o.ensureCOO();
            for (int i = 0; i < o.values.length; i++) { // structural: only non-zeros
                long key = (long) o.rowPtr[i] * cols + o.colInd[i];
                acc.merge(key, o.values[i], Double::sum);
            }
            // Build result from accumulated map
            int nnz = acc.size();
            int[] resRow = new int[nnz];
            int[] resCol = new int[nnz];
            double[] resVal = new double[nnz];
            int idx = 0;
            for (var e : acc.entrySet()) {
                double v = e.getValue();
                if (Math.abs(v) < 1e-30) { nnz--; continue; } // drop numerical zeros
                long key = e.getKey();
                resRow[idx] = (int) (key / cols);
                resCol[idx] = (int) (key % cols);
                resVal[idx] = v;
                idx++;
            }
            if (idx < nnz) {
                resRow = java.util.Arrays.copyOf(resRow, idx);
                resCol = java.util.Arrays.copyOf(resCol, idx);
                resVal = java.util.Arrays.copyOf(resVal, idx);
            }
            return new RereSparseDoubleMatrix(rows, cols, resRow, resCol, resVal, SparseFormat.COO);
        }

        @Override
        public ISparseMatrix sub(ISparseMatrix other) {
            return add(other.scale(-1.0));
        }

        @Override
        public ISparseMatrix scale(double scalar) {
            if (Math.abs(scalar) < 1e-12) {
                return new RereSparseDoubleMatrix(rows, cols, new int[0], new int[0], new double[0], SparseFormat.COO);
            }

            ensureCOO();
            double[] valuesNew = new double[nnz()];
            for (int i = 0; i < nnz(); i++) {
                valuesNew[i] = values[i] * scalar;
            }
            return new RereSparseDoubleMatrix(rows, cols, rowPtr.clone(), colInd.clone(), valuesNew, SparseFormat.COO);
        }

        @Override
        public ISparseMatrix multiply(ISparseMatrix other) {
            if (cols != other.rows()) {
                throw new IllegalArgumentException("Matrix dimensions not compatible for multiplication");
            }

            ensureCSR();
            ISparseMatrix otherCsr = other.toFormat(SparseFormat.CSR);

            int[] rowPtrB = ((RereSparseDoubleMatrix) otherCsr).rowPtr;
            int[] colIndB = ((RereSparseDoubleMatrix) otherCsr).colInd;
            double[] valuesB = ((RereSparseDoubleMatrix) otherCsr).values;

            int resultCols = other.cols();
            int[] rowPtrC = new int[rows + 1];
            int[] colIndC = new int[nnz() * other.nnz()];
            double[] valuesC = new double[nnz() * other.nnz()];

            int nnzC = 0;
            for (int i = 0; i < rows; i++) {
                rowPtrC[i] = nnzC;
                for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
                    int k = colInd[j];
                    double valA = values[j];

                    for (int p = rowPtrB[k]; p < rowPtrB[k + 1]; p++) {
                        int colC = colIndB[p];
                        double valC = valA * valuesB[p];

                        int existingIdx = findInRow(rowPtrC, colIndC, valuesC, rowPtrC[i], nnzC, colC);
                        if (existingIdx >= 0) {
                            valuesC[existingIdx] += valC;
                        } else {
                            colIndC[nnzC] = colC;
                            valuesC[nnzC] = valC;
                            nnzC++;
                        }
                    }
                }
            }
            rowPtrC[rows] = nnzC;

            int[] colIndTrim = new int[nnzC];
            double[] valuesTrim = new double[nnzC];
            System.arraycopy(colIndC, 0, colIndTrim, 0, nnzC);
            System.arraycopy(valuesC, 0, valuesTrim, 0, nnzC);

            return new RereSparseDoubleMatrix(rows, resultCols, rowPtrC, colIndTrim, valuesTrim, SparseFormat.CSR);
        }

        private int findInRow(int[] rowPtr, int[] colInd, double[] values, int start, int end, int col) {
            for (int i = start; i < end; i++) {
                if (colInd[i] == col) {
                    return i;
                }
            }
            return -1;
        }

        @Override
        public IVector<Double> multiply(IVector<Double> vector) {
            if (cols != vector.length()) {
                throw new IllegalArgumentException("Matrix columns must match vector length");
            }

            ensureCSR();
            double[] result = new double[rows];

            for (int i = 0; i < rows; i++) {
                double sum = 0.0;
                for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
                    sum += values[j] * vector.get(colInd[j]);
                }
                result[i] = sum;
            }

            return Linalg.vector(result);
        }

        @Override
        public ISparseMatrix transpose() {
            ensureCOO();
            return new RereSparseDoubleMatrix(cols, rows, colInd.clone(), rowPtr.clone(), values.clone(), SparseFormat.COO);
        }

        @Override
        public ISparseMatrix conjugateTranspose() {
            return transpose();
        }

        @Override
        public double frobeniusNorm() {
            double sum = 0;
            for (int i = 0; i < nnz(); i++) {
                sum += values[i] * values[i];
            }
            return Math.sqrt(sum);
        }

        @Override
        public ISparseMatrix hadamard(ISparseMatrix other) {
            if (rows != other.rows() || cols != other.cols()) {
                throw new IllegalArgumentException("Matrix dimensions must match for Hadamard product");
            }

            ensureCOO();
            ISparseMatrix otherCoo = other.toFormat(SparseFormat.COO);
            int[] otherRowIdx = ((RereSparseDoubleMatrix) otherCoo).rowPtr;
            int[] otherColIdx = ((RereSparseDoubleMatrix) otherCoo).colInd;
            double[] otherVals = ((RereSparseDoubleMatrix) otherCoo).values;

            int[] rowIdxNew = new int[nnz()];
            int[] colIdxNew = new int[nnz()];
            double[] valuesNew = new double[nnz()];
            int nnzNew = 0;

            for (int i = 0; i < nnz(); i++) {
                double otherVal = 0;
                for (int j = 0; j < otherCoo.nnz(); j++) {
                    if (otherRowIdx[j] == rowPtr[i] && otherColIdx[j] == colInd[i]) {
                        otherVal = otherVals[j];
                        break;
                    }
                }
                if (otherVal != 0) {
                    rowIdxNew[nnzNew] = rowPtr[i];
                    colIdxNew[nnzNew] = colInd[i];
                    valuesNew[nnzNew] = values[i] * otherVal;
                    nnzNew++;
                }
            }

            int[] rowIdxTrim = new int[nnzNew];
            int[] colIdxTrim = new int[nnzNew];
            double[] valuesTrim = new double[nnzNew];
            System.arraycopy(rowIdxNew, 0, rowIdxTrim, 0, nnzNew);
            System.arraycopy(colIdxNew, 0, colIdxTrim, 0, nnzNew);
            System.arraycopy(valuesNew, 0, valuesTrim, 0, nnzNew);

            return new RereSparseDoubleMatrix(rows, cols, rowIdxTrim, colIdxTrim, valuesTrim, SparseFormat.COO);
        }

        @Override
        public ISparseMatrix copy() {
            return new RereSparseDoubleMatrix(rows, cols, rowPtr.clone(), colInd.clone(), values.clone(), format);
        }

        int[] rowPtrRef() { ensureCSR(); return rowPtr; }
        int[] colIndRef() { ensureCSR(); return colInd; }
        double[] valuesRef() { ensureCSR(); return values; }

        @Override
        public String toString() {
            return String.format("SparseMatrix[%d x %d, nnz=%d, format=%s]", rows, cols, nnz(), format);
        }
    }
}