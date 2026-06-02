package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 矩阵分块操作工具类 / Block Operations Utility Class for Matrix Decompositions
 * <p>
 * 提供矩阵分解中常用分块操作的优化实现。使用分块技术提高大型矩阵运算的缓存性能。
 * </p>
 * <p>
 * Utility class for block operations in matrix decompositions.
 * <p>
 * This class provides optimized implementations of common matrix operations
 * using blocking techniques to improve cache performance for large matrices.
 * </p>
 *
 * <h3>性能优化 / Performance Optimizations:</h3>
 * <ul>
 *   <li>缓存友好的分块算法 / Cache-friendly blocking algorithms</li>
 *   <li>减少内存访问模式 / Reduced memory access patterns</li>
 *   <li>提高数值稳定性 / Improved numerical stability</li>
 *   <li>可配置的分块大小适应不同架构 / Configurable block sizes for different architectures</li>
 * </ul>
 *
 * <h3>参考文献 / References:</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Dongarra, J. J., Duff, I. S., Sorensen, D. C., &amp; van der Vorst, H. A. (1998). Numerical linear algebra for high-performance computers. SIAM.</li>
 * </ul>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class BlockOperationUtils {
    
    /** Default block size for cache-friendly operations. */
    public static final int DEFAULT_BLOCK_SIZE = 64;
    
    /** Block size for cache-friendly operations. */
    private final int blockSize;
    
    /**
     * Create a block operation utility with default block size.
     */
    public BlockOperationUtils() {
        this.blockSize = DEFAULT_BLOCK_SIZE;
    }
    
    /**
     * Create a block operation utility with specified block size.
     * 
     * @param blockSize the block size to use for operations
     */
    public BlockOperationUtils(int blockSize) {
        if (blockSize <= 0) {
            throw new IllegalArgumentException("Block size must be positive");
        }
        this.blockSize = blockSize;
    }
    
    /**
     * Perform block matrix multiplication: C = A * B
     * <p>
     * This method uses a cache-friendly blocking algorithm to improve performance
     * for large matrices by reducing cache misses.
     * </p>
     * 
     * @param a the left matrix
     * @param b the right matrix
     * @return the result matrix C = A * B
     */
    public IMatrix<Double> blockMultiply(IMatrix<Double> a, IMatrix<Double> b) {
        final int m = a.rows();
        final int n = b.cols();
        final int k = a.cols();
        
        if (k != b.rows()) {
            throw new IllegalArgumentException("Matrix dimensions do not match for multiplication");
        }
        
        // Create result matrix
        IMatrix<Double> c = Linalg.zeros(m, n);
        
        // Perform block multiplication
        for (int ii = 0; ii < m; ii += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {
                for (int kk = 0; kk < k; kk += blockSize) {
                    // Define block boundaries
                    final int iEnd = Math.min(ii + blockSize, m);
                    final int jEnd = Math.min(jj + blockSize, n);
                    final int kEnd = Math.min(kk + blockSize, k);
                    
                    // Perform multiplication for this block
                    multiplyBlock(a, b, c, ii, iEnd, jj, jEnd, kk, kEnd);
                }
            }
        }
        
        return c;
    }
    
    /**
     * Multiply a block of matrices: C[ii:iEnd, jj:jEnd] += A[ii:iEnd, kk:kEnd] * B[kk:kEnd, jj:jEnd]
     * 
     * @param a the left matrix
     * @param b the right matrix
     * @param c the result matrix
     * @param ii start row index for C and A
     * @param iEnd end row index for C and A
     * @param jj start column index for C and B
     * @param jEnd end column index for C and B
     * @param kk start column index for A, start row index for B
     * @param kEnd end column index for A, end row index for B
     */
    private void multiplyBlock(IMatrix<Double> a, IMatrix<Double> b, IMatrix<Double> c,
                              int ii, int iEnd, int jj, int jEnd, int kk, int kEnd) {
        for (int i = ii; i < iEnd; i++) {
            for (int j = jj; j < jEnd; j++) {
                double sum = c.get(i, j);
                for (int k = kk; k < kEnd; k++) {
                    sum += a.get(i, k) * b.get(k, j);
                }
                c.put(i, j, sum);
            }
        }
    }
    
    /**
     * Perform block matrix addition: C = A + B
     * 
     * @param a the first matrix
     * @param b the second matrix
     * @return the result matrix C = A + B
     */
    public IMatrix<Double> blockAdd(IMatrix<Double> a, IMatrix<Double> b) {
        final int m = a.rows();
        final int n = a.cols();
        
        if (m != b.rows() || n != b.cols()) {
            throw new IllegalArgumentException("Matrix dimensions do not match for addition");
        }
        
        // Create result matrix
        IMatrix<Double> c = Linalg.zeros(m, n);
        
        // Perform block addition
        for (int ii = 0; ii < m; ii += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {
                // Define block boundaries
                final int iEnd = Math.min(ii + blockSize, m);
                final int jEnd = Math.min(jj + blockSize, n);
                
                // Perform addition for this block
                addBlock(a, b, c, ii, iEnd, jj, jEnd);
            }
        }
        
        return c;
    }
    
    /**
     * Add a block of matrices: C[ii:iEnd, jj:jEnd] = A[ii:iEnd, jj:jEnd] + B[ii:iEnd, jj:jEnd]
     * 
     * @param a the first matrix
     * @param b the second matrix
     * @param c the result matrix
     * @param ii start row index
     * @param iEnd end row index
     * @param jj start column index
     * @param jEnd end column index
     */
    private void addBlock(IMatrix<Double> a, IMatrix<Double> b, IMatrix<Double> c,
                         int ii, int iEnd, int jj, int jEnd) {
        for (int i = ii; i < iEnd; i++) {
            for (int j = jj; j < jEnd; j++) {
                c.put(i, j, a.get(i, j) + b.get(i, j));
            }
        }
    }
    
    /**
     * Perform block matrix subtraction: C = A - B
     * 
     * @param a the first matrix
     * @param b the second matrix
     * @return the result matrix C = A - B
     */
    public IMatrix<Double> blockSubtract(IMatrix<Double> a, IMatrix<Double> b) {
        final int m = a.rows();
        final int n = a.cols();
        
        if (m != b.rows() || n != b.cols()) {
            throw new IllegalArgumentException("Matrix dimensions do not match for subtraction");
        }
        
        // Create result matrix
        IMatrix<Double> c = Linalg.zeros(m, n);
        
        // Perform block subtraction
        for (int ii = 0; ii < m; ii += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {
                // Define block boundaries
                final int iEnd = Math.min(ii + blockSize, m);
                final int jEnd = Math.min(jj + blockSize, n);
                
                // Perform subtraction for this block
                subtractBlock(a, b, c, ii, iEnd, jj, jEnd);
            }
        }
        
        return c;
    }
    
    /**
     * Subtract a block of matrices: C[ii:iEnd, jj:jEnd] = A[ii:iEnd, jj:jEnd] - B[ii:iEnd, jj:jEnd]
     * 
     * @param a the first matrix
     * @param b the second matrix
     * @param c the result matrix
     * @param ii start row index
     * @param iEnd end row index
     * @param jj start column index
     * @param jEnd end column index
     */
    private void subtractBlock(IMatrix<Double> a, IMatrix<Double> b, IMatrix<Double> c,
                              int ii, int iEnd, int jj, int jEnd) {
        for (int i = ii; i < iEnd; i++) {
            for (int j = jj; j < jEnd; j++) {
                c.put(i, j, a.get(i, j) - b.get(i, j));
            }
        }
    }
    
    /**
     * Perform block Householder transformation.
     * 
     * @param matrix the matrix to transform
     * @param v the Householder vector
     * @param rowStart the starting row index
     * @param colStart the starting column index
     * @return the transformed matrix
     */
    public IMatrix<Double> blockHouseholderTransform(IMatrix<Double> matrix, IMatrix<Double> v,
                                                   int rowStart, int colStart) {
        final int m = matrix.rows();
        final int n = matrix.cols();
        final int vLen = v.rows();
        
        // Create a copy of the matrix
        IMatrix<Double> result = matrix.copy();
        
        // Compute 2 * v^T * A
        IMatrix<Double> vTranspose = v.transpose();
        IMatrix<Double> temp = vTranspose.mmul(result);
        
        // Compute 2 * v * (v^T * A)
        IMatrix<Double> update = v.mmul(temp);
        
        // Apply the transformation: A = A - 2 * v * (v^T * A)
        for (int i = rowStart; i < Math.min(rowStart + vLen, m); i++) {
            for (int j = colStart; j < n; j++) {
                result.put(i, j, result.get(i, j) - 2.0 * update.get(i - rowStart, j));
            }
        }
        
        return result;
    }
    
    /**
     * Get the block size used for operations.
     * 
     * @return the block size
     */
    public int getBlockSize() {
        return blockSize;
    }
}