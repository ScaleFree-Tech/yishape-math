package com.yishape.lab.math.linalg;

import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

/**
 * Zero-copy view over a {@link RereDoubleMatrix} (or another {@link DoubleMatrixView}).
 * Delegates get/set to the parent's internal {@code double[][]} with offset translation.
 *
 * <p>Mutating the view mutates the parent. Intended for internal use within
 * decomposition algorithms to avoid allocating intermediate matrix copies.</p>
 */
public final class DoubleMatrixView implements IMatrixView<Double>, IDoubleMatrix {

    private final double[][] parentData;
    private final IDoubleMatrix parent;
    private final int rowOff;
    private final int colOff;
    private final int rows;
    private final int cols;

    /**
     * Create a view of {@code parentMatrix} starting at {@code (rowOff, colOff)}.
     * The view has dimensions {@code rows x cols}.
     *
     * @param parentMatrix the owning matrix (must be RereDoubleMatrix or DoubleMatrixView)
     * @param rowOff       row offset within parent
     * @param colOff       column offset within parent
     * @param rows         number of rows in view
     * @param cols         number of columns in view
     * @throws IllegalArgumentException if parent is not a supported type or offsets invalid
     */
    public DoubleMatrixView(IDoubleMatrix parentMatrix, int rowOff, int colOff, int rows, int cols) {
        if (parentMatrix instanceof DoubleMatrixView v) {
            this.parentData = v.parentData;
            this.parent = v.parent;
            this.rowOff = v.rowOff + rowOff;
            this.colOff = v.colOff + colOff;
        } else if (parentMatrix instanceof IDoubleMatrix dm) {
            this.parentData = dm.getData();
            this.parent = parentMatrix;
            this.rowOff = rowOff;
            this.colOff = colOff;
        } else {
            throw new IllegalArgumentException(
                "DoubleMatrixView requires RereDoubleMatrix or DoubleMatrixView parent");
        }
        if (rowOff < 0 || colOff < 0 || rows < 0 || cols < 0
                || rowOff + rows > parentData.length
                || (parentData.length > 0 && colOff + cols > parentData[0].length)) {
            throw new IllegalArgumentException("View dimensions out of bounds");
        }
        this.rows = rows;
        this.cols = cols;
    }

    @Override
    public double get(int i, int j) {
        return parentData[rowOff + i][colOff + j];
    }

    @Override
    public void set(int i, int j, double value) {
        parentData[rowOff + i][colOff + j] = value;
    }

    @Override
    public int getRowNum() {
        return rows;
    }

    @Override
    public int getColNum() {
        return cols;
    }

    @Override
    public  IDoubleMatrix copy() {
        double[][] copyData = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(parentData[rowOff + i], colOff, copyData[i], 0, cols);
        }
        return new RereDoubleMatrix(copyData);
    }

    @Override
    public double[][] getData() {
        double[][] copyData = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(parentData[rowOff + i], colOff, copyData[i], 0, cols);
        }
        return copyData;
    }

    @Override
    public  IDoubleMatrix parent() {
        return parent;
    }

    @Override
    public int rowOffset() {
        return rowOff;
    }

    @Override
    public int colOffset() {
        return colOff;
    }

    @Override
    public IMatrixView<Double> subView(int startRow, int endRow, int startCol, int endCol) {
        return new DoubleMatrixView(parent, rowOff + startRow, colOff + startCol,
                endRow - startRow, endCol - startCol);
    }

    @Override
    public boolean isView() {
        return true;
    }

    @Override
    public void put(int row, int col, double value) {
        parentData[rowOff + row][colOff + col] = value;
    }

    @Override
    public  IDoubleMatrix multiply(IMatrix<Double> other) {
        return copy().multiply(other);
    }

    @Override
    public  IDoubleMatrix multiplyByScalar(double scalar) {
        return copy().multiplyByScalar(scalar);
    }

    @Override
    public  IDoubleMatrix add(IMatrix<Double> other) {
        return copy().add(other);
    }

    @Override
    public  IDoubleMatrix sub(double scalar) {
        return copy().sub(scalar);
    }

    @Override
    public  IDoubleMatrix sub(IMatrix<Double> other) {
        return copy().sub(other);
    }

    @Override
    public  IDoubleMatrix mmul(IMatrix<Double> other) {
        return copy().mmul(other);
    }

    @Override
    public IVector<Double> mmul(IVector<Double> other) {
        return copy().mmul(other);
    }

    @Override
    public  IDoubleMatrix divide(IMatrix<Double> other) {
        return copy().divide(other);
    }

    @Override
    public  IDoubleMatrix divideByScalar(double scalar) {
        return copy().divideByScalar(scalar);
    }

    @Override
    public double frobeniusInnerProduct(IMatrix<Double> other) {
        return copy().frobeniusInnerProduct(other);
    }

    @Override
    public  IDoubleMatrix kron(IMatrix<Double> other) {
        return (IDoubleMatrix)copy().kron(other);
    }

    @Override
    public IDoubleMatrix sum() {
        return copy().sum();
    }

    @Override
    public IDoubleMatrix mean() {
        return copy().mean();
    }

    @Override
    public double max() {
        return copy().max();
    }

    @Override
    public double min() {
        return copy().min();
    }

    @Override
    public int[] shape() {
        return new int[]{rows, cols};
    }

    @Override
    public int rows() {
        return rows;
    }

    @Override
    public int cols() {
        return cols;
    }

    @Override
    public  IDoubleMatrix transposeInPlace() {
        throw new UnsupportedOperationException("View does not support in-place transpose");
    }

    @Override
    public  IDoubleMatrix transposeNew() {
        return copy().transposeNew();
    }

    @Override
    public  IDoubleMatrix inv() {
        return copy().inv();
    }

    @Override
    public  IDoubleMatrix pinv() {
        return copy().pinv();
    }

    @Override
    public IVector<Double> rowSums() {
        return copy().rowSums();
    }

    @Override
    public IVector<Double> rowMeans() {
        return copy().rowMeans();
    }

    @Override
    public IVector<Double> colSums() {
        return copy().colSums();
    }

    @Override
    public IVector<Double> colMeans() {
        return copy().colMeans();
    }

    @Override
    public IVector<Double> min(int axis) {
        return copy().min(axis);
    }

    @Override
    public IVector<Double> max(int axis) {
        return copy().max(axis);
    }

    @Override
    public IVector<Double> sum(int axis) {
        return copy().sum(axis);
    }

    @Override
    public IVector<Double> mean(int axis) {
        return copy().mean(axis);
    }

    @Override
    public IVector<Double> rowMins() {
        return copy().rowMins();
    }

    @Override
    public IVector<Double> rowMaxs() {
        return copy().rowMaxs();
    }

    @Override
    public IVector<Double> colMins() {
        return copy().colMins();
    }

    @Override
    public IVector<Double> colMaxs() {
        return copy().colMaxs();
    }

    @Override
    public IVector<Double> getColumn(int colIndex) {
        double[] col = new double[rows];
        for (int i = 0; i < rows; i++) {
            col[i] = parentData[rowOff + i][colOff + colIndex];
        }
        return IDoubleVector.of(col);
    }

    @Override
    public  IDoubleMatrix getColumnAsCloumnVector(int colIndex) {
        double[][] col = new double[rows][1];
        for (int i = 0; i < rows; i++) {
            col[i][0] = parentData[rowOff + i][colOff + colIndex];
        }
        return new RereDoubleMatrix(col);
    }

    @Override
    public IVector<Double> getRow(int rowIndex) {
        double[] row = new double[cols];
        System.arraycopy(parentData[rowOff + rowIndex], colOff, row, 0, cols);
        return IDoubleVector.of(row);
    }

    @Override
    public  IDoubleMatrix getColumnMatrix(int colIndex) {
        return getColumnAsCloumnVector(colIndex);
    }

    @Override
    public void putColumn(int colIndex, IMatrix<Double> column) {
        for (int i = 0; i < rows; i++) {
            parentData[rowOff + i][colOff + colIndex] = column.get(i, 0);
        }
    }

    @Override
    public void setColumn(int colIndex, IVector<Double> column) {
        for (int i = 0; i < rows; i++) {
            parentData[rowOff + i][colOff + colIndex] = column.get(i);
        }
    }

    @Override
    public void setRow(int rowIndex, IVector<Double> row) {
        for (int j = 0; j < cols; j++) {
            parentData[rowOff + rowIndex][colOff + j] = row.get(j);
        }
    }

    @Override
    public IVector<Double>[] getColumns(int[] colIndices) {
        return copy().getColumns(colIndices);
    }

    @Override
    public  IDoubleMatrix sqrt() {
        return copy().sqrt();
    }

    @Override
    public  IDoubleMatrix pow(Double exponent) {
        return copy().pow(exponent);
    }

    @Override
    public  IDoubleMatrix exp() {
        return copy().exp();
    }

    @Override
    public  IDoubleMatrix log() {
        return copy().log();
    }

    @Override
    public double frobeniusNorm() {
        return copy().frobeniusNorm();
    }

    @Override
    public boolean isSymmetric() {
        return copy().isSymmetric();
    }

    @Override
    public boolean isPositiveDefinite() {
        return copy().isPositiveDefinite();
    }

    @Override
    public IVector<Double> flatten() {
        return copy().flatten();
    }

    @Override
    public double frobeniusDistance(IMatrix<Double> other) {
        return copy().frobeniusDistance(other);
    }

    @Override
    public  IDoubleMatrix normalizeRows() {
        return copy().normalizeRows();
    }

    @Override
    public  IDoubleMatrix normalizeColumns() {
        return copy().normalizeColumns();
    }

    @Override
    public  IDoubleMatrix normalize() {
        return copy().normalize();
    }

    @Override
    public  IDoubleMatrix center() {
        return copy().center();
    }

    @Override
    public  IDoubleMatrix covariance() {
        return copy().covariance();
    }

    @Override
    public  IDoubleMatrix cov() {
        return copy().cov();
    }

    @Override
    public  IDoubleMatrix covarianceFromCentered() {
        return copy().covarianceFromCentered();
    }

    @Override
    public IVector<Double> diag() {
        return copy().diag();
    }

    @Override
    public  IDoubleMatrix hstack(IMatrix<Double> other) {
        return copy().hstack(other);
    }

    @Override
    public  IDoubleMatrix vstack(IMatrix<Double> other) {
        return copy().vstack(other);
    }

    @Override
    public  IDoubleMatrix[] hsplit(int... widths) {
        return (IDoubleMatrix[])copy().hsplit(widths);
    }

    @Override
    public  IDoubleMatrix[] vsplit(int... heights) {
        return (IDoubleMatrix[])copy().vsplit(heights);
    }

    @Override
    public  IDoubleMatrix reshape(int newRows, int newCols) {
        return copy().reshape(newRows, newCols);
    }

    @Override
    public  IDoubleMatrix abs() {
        return copy().abs();
    }

    @Override
    public  IDoubleMatrix sign() {
        return copy().sign();
    }

    @Override
    public  IDoubleMatrix sin() {
        return copy().sin();
    }

    @Override
    public  IDoubleMatrix cos() {
        return copy().cos();
    }

    @Override
    public  IDoubleMatrix tan() {
        return copy().tan();
    }

    @Override
    public  IDoubleMatrix sinh() {
        return copy().sinh();
    }

    @Override
    public  IDoubleMatrix cosh() {
        return copy().cosh();
    }

    @Override
    public  IDoubleMatrix tanh() {
        return copy().tanh();
    }

    @Override
    public double std() {
        return copy().std();
    }

    @Override
    public double var() {
        return copy().var();
    }

    @Override
    public double det() {
        return copy().det();
    }

    @Override
    public double trace() {
        return copy().trace();
    }

    @Override
    public double cond() {
        return copy().cond();
    }

    @Override
    public int rank() {
        return copy().rank();
    }

    @Override
    public com.yishape.lab.util.Tuple2<IMatrix<Double>, IMatrix<Double>> lu() {
        return copy().lu();
    }

    @Override
    public  IDoubleMatrix cholesky() {
        return copy().cholesky();
    }

    @Override
    public com.yishape.lab.util.Tuple2<IMatrix<Double>, IMatrix<Double>> schur() {
        return copy().schur();
    }

    @Override
    public com.yishape.lab.util.Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> biDiag() {
        return copy().biDiag();
    }

    @Override
    public com.yishape.lab.util.Tuple2<IMatrix<Double>, IMatrix<Double>> triDiag() {
        return copy().triDiag();
    }

    @Override
    public com.yishape.lab.util.Tuple2<IMatrix<Double>, IMatrix<Double>> hessenberg() {
        return copy().hessenberg();
    }

    @Override
    public com.yishape.lab.util.Tuple2<IVector<Double>, IMatrix<Double>> eigen() {
        return copy().eigen();
    }

    @Override
    public com.yishape.lab.util.Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svd() {
        return copy().svd();
    }

    @Override
    public com.yishape.lab.util.Tuple2<IMatrix<Double>, IMatrix<Double>> qr() {
        return copy().qr();
    }

    @Override
    public IVector<Double> solve(IVector<Double> b) {
        return copy().solve(b);
    }

    @Override
    public  IDoubleMatrix solve(IMatrix<Double> B) {
        return copy().solve(B);
    }

    @Override
    public  IDoubleMatrix slice(String rowSlice, String colSlice) {
        return copy().slice(rowSlice, colSlice);
    }

    @Override
    public  IDoubleMatrix sliceRows(String sliceExpression) {
        return copy().sliceRows(sliceExpression);
    }

    @Override
    public  IDoubleMatrix sliceColumns(String sliceExpression) {
        return copy().sliceColumns(sliceExpression);
    }

    @Override
    public  IDoubleMatrix fancyGet(int[] rowIndices, int[] colIndices) {
        return copy().fancyGet(rowIndices, colIndices);
    }

    @Override
    public  IDoubleMatrix booleanGet(boolean[] rowMask) {
        return copy().booleanGet(rowMask);
    }

    @Override
    public  IDoubleMatrix booleanGet(boolean[] rowMask, boolean[] colMask) {
        return copy().booleanGet(rowMask, colMask);
    }

    @Override
    public void fancySet(int[] rowIndices, int[] colIndices, Double[] values) {
        // Update through the view offset
        IndexExpressionParser.FancyIndexResult resolvedRows =
            IndexExpressionParser.resolveFancyIndex(rowIndices, rows);
        IndexExpressionParser.FancyIndexResult resolvedCols =
            IndexExpressionParser.resolveFancyIndex(colIndices, cols);
        int idx = 0;
        for (int ri : resolvedRows.indices) {
            for (int ci : resolvedCols.indices) {
                parentData[rowOff + ri][colOff + ci] = values[idx++];
            }
        }
    }

    @Override
    public void fancySetScalar(int[] rowIndices, int[] colIndices, Double value) {
        IndexExpressionParser.FancyIndexResult resolvedRows =
            IndexExpressionParser.resolveFancyIndex(rowIndices, rows);
        IndexExpressionParser.FancyIndexResult resolvedCols =
            IndexExpressionParser.resolveFancyIndex(colIndices, cols);
        for (int ri : resolvedRows.indices) {
            for (int ci : resolvedCols.indices) {
                parentData[rowOff + ri][colOff + ci] = value;
            }
        }
    }

    @Override
    public  IDoubleMatrix subMatrix(int startRow, int endRow, int startCol, int endCol) {
        return copy().subMatrix(startRow, endRow, startCol, endCol);
    }

    @Override
    public void setSubMatrix(int startRow, int endRow, int startCol, int endCol, IMatrix<Double> subMatrix) {
        for (int i = startRow; i < endRow; i++) {
            for (int j = startCol; j < endCol; j++) {
                parentData[rowOff + i][colOff + j] = subMatrix.get(i - startRow, j - startCol);
            }
        }
    }

    @Override
    public double[][] toDoubleArray() {
        return copy().toDoubleArray();
    }

    @Override
    public float[][] toFloatArray() {
        return copy().toFloatArray();
    }

    @Override
    public int[][] toIntArray() {
        return copy().toIntArray();
    }

    @Override
    public IVector<Double> apply(java.util.function.Function<IVector<Double>, Double> fun, int axis) {
        return copy().apply(fun, axis);
    }

    @Override
    public  IDoubleMatrix applyMap(java.util.function.Function<Double, Double> function) {
        return copy().applyMap(function);
    }

    @Override
    public void save(String filePath) {
        copy().save(filePath);
    }

    @Override
    public void setDiag(IVector<Double> diagonal) {
        for (int i = 0; i < Math.min(rows, cols); i++) {
            parentData[rowOff + i][colOff + i] = diagonal.get(i);
        }
    }

    @Override
    public  IDoubleMatrix broadcastColumn(IVector<Double> colVector,
            java.util.function.BiFunction<IVector<Double>, IVector<Double>, IVector<Double>> fun) {
        return copy().broadcastColumn(colVector, fun);
    }

    @Override
    public  IDoubleMatrix broadcastRow(IVector<Double> rowVector,
            java.util.function.BiFunction<IVector<Double>, IVector<Double>, IVector<Double>> fun) {
        return copy().broadcastRow(rowVector, fun);
    }

    private double[][] otherData(IMatrix<Double> other) {
        return other instanceof RereDoubleMatrix ? ((RereDoubleMatrix) other).data : other.toDoubleArray();
    }

    @Override
    public boolean[][] eq(IMatrix<Double> other) {
        double[][] od = otherData(other);
        boolean[][] result = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = parentData[rowOff + i][colOff + j] == od[i][j];
            }
        }
        return result;
    }

    @Override
    public boolean[][] lt(IMatrix<Double> other) {
        double[][] od = otherData(other);
        boolean[][] result = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = parentData[rowOff + i][colOff + j] < od[i][j];
            }
        }
        return result;
    }

    @Override
    public boolean[][] gt(IMatrix<Double> other) {
        double[][] od = otherData(other);
        boolean[][] result = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = parentData[rowOff + i][colOff + j] > od[i][j];
            }
        }
        return result;
    }

    @Override
    public boolean[][] le(IMatrix<Double> other) {
        double[][] od = otherData(other);
        boolean[][] result = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = parentData[rowOff + i][colOff + j] <= od[i][j];
            }
        }
        return result;
    }

    @Override
    public boolean[][] ge(IMatrix<Double> other) {
        double[][] od = otherData(other);
        boolean[][] result = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = parentData[rowOff + i][colOff + j] >= od[i][j];
            }
        }
        return result;
    }
}
