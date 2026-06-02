package com.yishape.lab.math.linalg.decomposition;

/**
 * Exception thrown when attempting to decompose a singular matrix.
 * <p>
 * A singular matrix is a square matrix that does not have an inverse,
 * meaning its determinant is zero. Many matrix decomposition algorithms
 * require non-singular matrices to produce meaningful results.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class SingularMatrixException extends MatrixDecompositionException {
    
    /**
     * Constructs a new singular matrix exception with the specified detail message.
     * 
     * @param message the detail message
     */
    public SingularMatrixException(String message) {
        super(message);
    }
    
    /**
     * Constructs a new singular matrix exception with the specified detail message
     * and cause.
     * 
     * @param message the detail message
     * @param cause the cause
     */
    public SingularMatrixException(String message, Throwable cause) {
        super(message, cause);
    }
    
    /**
     * Constructs a new singular matrix exception with the specified detail message,
     * decomposition name, and matrix context.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     */
    public SingularMatrixException(String message, String decompositionName, String matrixContext) {
        super(message, decompositionName, matrixContext);
    }
    
    /**
     * Constructs a new singular matrix exception with the specified detail message,
     * decomposition name, matrix context, and cause.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     * @param cause the cause
     */
    public SingularMatrixException(String message, String decompositionName, String matrixContext, Throwable cause) {
        super(message, decompositionName, matrixContext, cause);
    }
}