package com.yishape.lab.math.linalg.sparse;

public interface ISpecialSparseMatrix {

    class DiagonalSparseMatrix implements ISparseMatrix {
        private final int size;
        private final double[] diagonal;
        private final int rows;
        private final int cols;

        public DiagonalSparseMatrix(double[] diagonal) {
            this.diagonal = diagonal.clone();
            this.size = diagonal.length;
            this.rows = size;
            this.cols = size;
        }

        public DiagonalSparseMatrix(double[] diagonal, int rows, int cols) {
            this.diagonal = diagonal.clone();
            this.size = diagonal.length;
            this.rows = rows;
            this.cols = cols;
        }

        @Override
        public int rows() { return rows; }

        @Override
        public int cols() { return cols; }

        @Override
        public int nnz() { return size; }

        @Override
        public double sparsity() {
            return 1.0 - (double) size / (rows * cols);
        }

        @Override
        public SparseFormat format() { return SparseFormat.COO; }

        @Override
        public ISparseMatrix toFormat(SparseFormat targetFormat) {
            if (targetFormat == SparseFormat.COO) {
                return this;
            } else if (targetFormat == SparseFormat.CSR || targetFormat == SparseFormat.CSC) {
                return toCsrCsc(targetFormat);
            }
            throw new IllegalArgumentException("Unsupported format: " + targetFormat);
        }

        private ISparseMatrix toCsrCsc(SparseFormat targetFormat) {
            int[] rowPtr = targetFormat == SparseFormat.CSR ? new int[rows + 1] : new int[cols + 1];
            int[] colInd = new int[size];
            double[] values = diagonal.clone();

            // Fill rowPtr[0..size] with 0,1,...,size
            for (int i = 0; i <= size; i++) {
                rowPtr[i] = i;
            }
            // For non-square matrices, pad remaining rowPtr entries with size
            // (rows beyond diagonal have zero non-zero elements)
            int ptrLen = targetFormat == SparseFormat.CSR ? rows + 1 : cols + 1;
            for (int i = size + 1; i < ptrLen; i++) {
                rowPtr[i] = size;
            }
            for (int i = 0; i < size; i++) {
                colInd[i] = i;
            }

            return new ISparseMatrix.RereSparseDoubleMatrix(rows, cols, rowPtr, colInd, values, targetFormat);
        }

        @Override
        public com.yishape.lab.math.linalg.IMatrix<Double> toDense() {
            double[][] result = new double[rows][cols];
            for (int i = 0; i < size; i++) {
                result[i][i] = diagonal[i];
            }
            return com.yishape.lab.math.linalg.Linalg.matrix(result);
        }

        @Override
        public double[][] toDenseArray() {
            double[][] result = new double[rows][cols];
            for (int i = 0; i < size; i++) {
                result[i][i] = diagonal[i];
            }
            return result;
        }

        @Override
        public double get(int row, int col) {
            if (row == col && row < size) {
                return diagonal[row];
            }
            return 0.0;
        }

        @Override
        public ISparseMatrix add(ISparseMatrix other) {
            if (rows != other.rows() || cols != other.cols()) {
                throw new IllegalArgumentException("Dimensions must match");
            }
            double[][] result = toDenseArray();
            double[][] otherDense = other.toDenseArray();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] += otherDense[i][j];
                }
            }
            return ISparseMatrix.fromDense(result);
        }

        @Override
        public ISparseMatrix sub(ISparseMatrix other) {
            return add(other.scale(-1.0));
        }

        @Override
        public ISparseMatrix scale(double scalar) {
            double[] newDiag = new double[size];
            for (int i = 0; i < size; i++) {
                newDiag[i] = diagonal[i] * scalar;
            }
            return new DiagonalSparseMatrix(newDiag, rows, cols);
        }

        @Override
        public ISparseMatrix multiply(ISparseMatrix other) {
            if (cols != other.rows()) {
                throw new IllegalArgumentException("Dimensions not compatible");
            }
            double[][] result = toDenseArray();
            double[][] otherDense = other.toDenseArray();
            int m = rows;
            int n = other.cols();
            int k = cols;
            double[][] product = new double[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int p = 0; p < k; p++) {
                        sum += result[i][p] * otherDense[p][j];
                    }
                    product[i][j] = sum;
                }
            }
            return ISparseMatrix.fromDense(product);
        }

        @Override
        public com.yishape.lab.math.linalg.IVector<Double> multiply(com.yishape.lab.math.linalg.IVector<Double> vector) {
            if (cols != vector.length()) {
                throw new IllegalArgumentException("Dimension mismatch");
            }
            double[] result = new double[rows];
            for (int i = 0; i < size && i < vector.length(); i++) {
                result[i] = diagonal[i] * vector.get(i);
            }
            return com.yishape.lab.math.linalg.Linalg.vector(result);
        }

        @Override
        public ISparseMatrix transpose() {
            return this;
        }

        @Override
        public ISparseMatrix conjugateTranspose() {
            return this;
        }

        @Override
        public double frobeniusNorm() {
            double sum = 0;
            for (double v : diagonal) {
                sum += v * v;
            }
            return Math.sqrt(sum);
        }

        @Override
        public ISparseMatrix hadamard(ISparseMatrix other) {
            double[][] otherDense = other.toDenseArray();
            double[][] result = toDenseArray();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] *= otherDense[i][j];
                }
            }
            return ISparseMatrix.fromDense(result);
        }

        @Override
        public ISparseMatrix copy() {
            return new DiagonalSparseMatrix(diagonal, rows, cols);
        }

        public double[] getDiagonal() {
            return diagonal.clone();
        }

        @Override
        public String toString() {
            return String.format("DiagonalSparseMatrix[%d x %d, nnz=%d]", rows, cols, size);
        }
    }

    class TridiagonalSparseMatrix implements ISparseMatrix {
        private final int rows;
        private final int cols;
        private final double[] lower;
        private final double[] main;
        private final double[] upper;

        public TridiagonalSparseMatrix(double[] lower, double[] main, double[] upper) {
            this.lower = lower.clone();
            this.main = main.clone();
            this.upper = upper.clone();
            this.rows = main.length;
            this.cols = main.length;
        }

        public TridiagonalSparseMatrix(double[] main, double[] upper) {
            this.lower = new double[main.length - 1];
            this.main = main.clone();
            this.upper = upper.clone();
            this.rows = main.length;
            this.cols = main.length;
        }

        @Override
        public int rows() { return rows; }

        @Override
        public int cols() { return cols; }

        @Override
        public int nnz() {
            int count = main.length;
            if (lower.length > 0) count += lower.length;
            if (upper.length > 0) count += upper.length;
            return count;
        }

        @Override
        public double sparsity() {
            return 1.0 - (double) nnz() / (rows * cols);
        }

        @Override
        public SparseFormat format() { return SparseFormat.COO; }

        @Override
        public ISparseMatrix toFormat(SparseFormat targetFormat) {
            if (targetFormat == SparseFormat.COO) {
                return toCOO();
            }
            int n = main.length;
            int count = n + (lower.length > 0 ? n - 1 : 0) + (upper.length > 0 ? n - 1 : 0);
            int[] rowIdx = new int[count];
            int[] colIdx = new int[count];
            double[] values = new double[count];
            int idx = 0;

            for (int i = 0; i < n; i++) {
                rowIdx[idx] = i;
                colIdx[idx] = i;
                values[idx++] = main[i];
            }
            for (int i = 0; i < lower.length; i++) {
                rowIdx[idx] = i + 1;
                colIdx[idx] = i;
                values[idx++] = lower[i];
            }
            for (int i = 0; i < upper.length; i++) {
                rowIdx[idx] = i;
                colIdx[idx] = i + 1;
                values[idx++] = upper[i];
            }

            if (targetFormat == SparseFormat.CSR) {
                int[] rowPtr = new int[rows + 1];
                for (int i = 0; i <= n; i++) rowPtr[i] = i;
                return new ISparseMatrix.RereSparseDoubleMatrix(rows, cols, rowPtr, colIdx, values, SparseFormat.CSR);
            } else {
                int[] colPtr = new int[cols + 1];
                for (int i = 0; i <= n; i++) colPtr[i] = i;
                return new ISparseMatrix.RereSparseDoubleMatrix(rows, cols, colPtr, rowIdx, values, SparseFormat.CSC);
            }
        }

        private ISparseMatrix toCOO() {
            int n = main.length;
            int count = n + (lower.length > 0 ? n - 1 : 0) + (upper.length > 0 ? n - 1 : 0);
            int[] rowIdx = new int[count];
            int[] colIdx = new int[count];
            double[] values = new double[count];
            int idx = 0;

            for (int i = 0; i < n; i++) {
                rowIdx[idx] = i;
                colIdx[idx] = i;
                values[idx++] = main[i];
            }
            for (int i = 0; i < lower.length; i++) {
                rowIdx[idx] = i + 1;
                colIdx[idx] = i;
                values[idx++] = lower[i];
            }
            for (int i = 0; i < upper.length; i++) {
                rowIdx[idx] = i;
                colIdx[idx] = i + 1;
                values[idx++] = upper[i];
            }

            return new ISparseMatrix.RereSparseDoubleMatrix(rows, cols, rowIdx, colIdx, values, SparseFormat.COO);
        }

        @Override
        public com.yishape.lab.math.linalg.IMatrix<Double> toDense() {
            return com.yishape.lab.math.linalg.Linalg.matrix(toDenseArray());
        }

        @Override
        public double[][] toDenseArray() {
            double[][] result = new double[rows][cols];
            for (int i = 0; i < main.length; i++) {
                result[i][i] = main[i];
            }
            for (int i = 0; i < lower.length; i++) {
                result[i + 1][i] = lower[i];
            }
            for (int i = 0; i < upper.length; i++) {
                result[i][i + 1] = upper[i];
            }
            return result;
        }

        @Override
        public double get(int row, int col) {
            if (row == col) return main[row];
            if (col == row - 1 && row > 0) return lower[row - 1];
            if (col == row + 1 && row < main.length - 1) return upper[row];
            return 0.0;
        }

        @Override
        public ISparseMatrix add(ISparseMatrix other) {
            double[][] result = toDenseArray();
            double[][] otherDense = other.toDenseArray();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] += otherDense[i][j];
                }
            }
            return ISparseMatrix.fromDense(result);
        }

        @Override
        public ISparseMatrix sub(ISparseMatrix other) {
            return add(other.scale(-1.0));
        }

        @Override
        public ISparseMatrix scale(double scalar) {
            double[] newLower = new double[lower.length];
            double[] newMain = new double[main.length];
            double[] newUpper = new double[upper.length];
            for (int i = 0; i < lower.length; i++) newLower[i] = lower[i] * scalar;
            for (int i = 0; i < main.length; i++) newMain[i] = main[i] * scalar;
            for (int i = 0; i < upper.length; i++) newUpper[i] = upper[i] * scalar;
            return new TridiagonalSparseMatrix(newLower, newMain, newUpper);
        }

        @Override
        public ISparseMatrix multiply(ISparseMatrix other) {
            double[][] a = toDenseArray();
            double[][] b = other.toDenseArray();
            double[][] result = new double[rows][other.cols()];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < other.cols(); j++) {
                    double sum = 0;
                    for (int k = 0; k < cols; k++) {
                        sum += a[i][k] * b[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            return ISparseMatrix.fromDense(result);
        }

        @Override
        public com.yishape.lab.math.linalg.IVector<Double> multiply(com.yishape.lab.math.linalg.IVector<Double> vector) {
            double[] result = new double[rows];
            for (int i = 0; i < main.length; i++) {
                result[i] = main[i] * vector.get(i);
                if (i > 0) result[i] += lower[i - 1] * vector.get(i - 1);
                if (i < main.length - 1) result[i] += upper[i] * vector.get(i + 1);
            }
            return com.yishape.lab.math.linalg.Linalg.vector(result);
        }

        @Override
        public ISparseMatrix transpose() {
            return new TridiagonalSparseMatrix(upper, main.clone(), lower);
        }

        @Override
        public ISparseMatrix conjugateTranspose() {
            return transpose();
        }

        @Override
        public double frobeniusNorm() {
            double sum = 0;
            for (double v : main) sum += v * v;
            for (double v : lower) sum += v * v;
            for (double v : upper) sum += v * v;
            return Math.sqrt(sum);
        }

        @Override
        public ISparseMatrix hadamard(ISparseMatrix other) {
            double[][] result = toDenseArray();
            double[][] otherDense = other.toDenseArray();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] *= otherDense[i][j];
                }
            }
            return ISparseMatrix.fromDense(result);
        }

        @Override
        public ISparseMatrix copy() {
            return new TridiagonalSparseMatrix(lower.clone(), main.clone(), upper.clone());
        }

        public double[] getLower() { return lower.clone(); }
        public double[] getMain() { return main.clone(); }
        public double[] getUpper() { return upper.clone(); }

        @Override
        public String toString() {
            return String.format("TridiagonalSparseMatrix[%d x %d, nnz=%d]", rows, cols, nnz());
        }
    }

    class IdentitySparseMatrix implements ISparseMatrix {
        private final int size;
        private final int rows;
        private final int cols;

        public IdentitySparseMatrix(int size) {
            this.size = size;
            this.rows = size;
            this.cols = size;
        }

        public IdentitySparseMatrix(int rows, int cols) {
            this.size = Math.min(rows, cols);
            this.rows = rows;
            this.cols = cols;
        }

        @Override
        public int rows() { return rows; }

        @Override
        public int cols() { return cols; }

        @Override
        public int nnz() { return size; }

        @Override
        public double sparsity() {
            return 1.0 - (double) size / (rows * cols);
        }

        @Override
        public SparseFormat format() { return SparseFormat.COO; }

        @Override
        public ISparseMatrix toFormat(SparseFormat targetFormat) {
            return new DiagonalSparseMatrix(new double[size], rows, cols).toFormat(targetFormat);
        }

        @Override
        public com.yishape.lab.math.linalg.IMatrix<Double> toDense() {
            return com.yishape.lab.math.linalg.Linalg.eye(rows);
        }

        @Override
        public double[][] toDenseArray() {
            double[][] result = new double[rows][cols];
            for (int i = 0; i < size; i++) {
                result[i][i] = 1.0;
            }
            return result;
        }

        @Override
        public double get(int row, int col) {
            return (row == col && row < size) ? 1.0 : 0.0;
        }

        @Override
        public ISparseMatrix add(ISparseMatrix other) {
            double[][] result = toDenseArray();
            double[][] otherDense = other.toDenseArray();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] += otherDense[i][j];
                }
            }
            return ISparseMatrix.fromDense(result);
        }

        @Override
        public ISparseMatrix sub(ISparseMatrix other) {
            return add(other.scale(-1.0));
        }

        @Override
        public ISparseMatrix scale(double scalar) {
            double[] diag = new double[size];
            for (int i = 0; i < size; i++) {
                diag[i] = scalar;
            }
            return new DiagonalSparseMatrix(diag, rows, cols);
        }

        @Override
        public ISparseMatrix multiply(ISparseMatrix other) {
            return other.copy();
        }

        @Override
        public com.yishape.lab.math.linalg.IVector<Double> multiply(com.yishape.lab.math.linalg.IVector<Double> vector) {
            return vector.copy();
        }

        @Override
        public ISparseMatrix transpose() {
            return this;
        }

        @Override
        public ISparseMatrix conjugateTranspose() {
            return this;
        }

        @Override
        public double frobeniusNorm() {
            return Math.sqrt(size);
        }

        @Override
        public ISparseMatrix hadamard(ISparseMatrix other) {
            return other.copy();
        }

        @Override
        public ISparseMatrix copy() {
            return new IdentitySparseMatrix(rows, cols);
        }

        @Override
        public String toString() {
            return String.format("IdentitySparseMatrix[%d x %d]", rows, cols);
        }
    }

    class ZeroSparseMatrix implements ISparseMatrix {
        private final int rows;
        private final int cols;

        public ZeroSparseMatrix(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
        }

        @Override
        public int rows() { return rows; }

        @Override
        public int cols() { return cols; }

        @Override
        public int nnz() { return 0; }

        @Override
        public double sparsity() { return 1.0; }

        @Override
        public SparseFormat format() { return SparseFormat.COO; }

        @Override
        public ISparseMatrix toFormat(SparseFormat targetFormat) {
            return this;
        }

        @Override
        public com.yishape.lab.math.linalg.IMatrix<Double> toDense() {
            return com.yishape.lab.math.linalg.Linalg.zeros(rows, cols);
        }

        @Override
        public double[][] toDenseArray() {
            return new double[rows][cols];
        }

        @Override
        public double get(int row, int col) { return 0.0; }

        @Override
        public ISparseMatrix add(ISparseMatrix other) {
            return other.copy();
        }

        @Override
        public ISparseMatrix sub(ISparseMatrix other) {
            return other.scale(-1.0);
        }

        @Override
        public ISparseMatrix scale(double scalar) {
            return this;
        }

        @Override
        public ISparseMatrix multiply(ISparseMatrix other) {
            return this;
        }

        @Override
        public com.yishape.lab.math.linalg.IVector<Double> multiply(com.yishape.lab.math.linalg.IVector<Double> vector) {
            return com.yishape.lab.math.linalg.Linalg.zeros(rows);
        }

        @Override
        public ISparseMatrix transpose() {
            return new ZeroSparseMatrix(cols, rows);
        }

        @Override
        public ISparseMatrix conjugateTranspose() {
            return transpose();
        }

        @Override
        public double frobeniusNorm() { return 0.0; }

        @Override
        public ISparseMatrix hadamard(ISparseMatrix other) {
            return this;
        }

        @Override
        public ISparseMatrix copy() {
            return new ZeroSparseMatrix(rows, cols);
        }

        @Override
        public String toString() {
            return String.format("ZeroSparseMatrix[%d x %d]", rows, cols);
        }
    }
}