package com.yishape.lab.math.linalg.decomposition;

/**
 * Base exception class for matrix decomposition errors.
 * <p>
 * This exception hierarchy provides more detailed error reporting for matrix
 * decomposition operations, including context information about the matrix
 * that caused the error and the specific decomposition algorithm being used.
 * </p>
 *
 * <h3>Exception Hierarchy</h3>
 * <pre>
 * Exception
 *  └── RuntimeException
 *       └── MatrixDecompositionException
 *            ├── SingularMatrixException
 *            ├── NonSymmetricMatrixException
 *            ├── NonPositiveDefiniteMatrixException
 *            ├── NonSquareMatrixException
 *            └── DecompositionFailedException
 * </pre>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class MatrixDecompositionException extends RuntimeException {
    
    /** The name of the decomposition algorithm that failed. */
    private final String decompositionName;
    
    /** Context information about the matrix. */
    private final String matrixContext;
    
    /**
     * Constructs a new matrix decomposition exception with the specified detail message.
     * 
     * @param message the detail message
     */
    public MatrixDecompositionException(String message) {
        super(message);
        this.decompositionName = null;
        this.matrixContext = null;
    }
    
    /**
     * Constructs a new matrix decomposition exception with the specified detail message
     * and cause.
     * 
     * @param message the detail message
     * @param cause the cause
     */
    public MatrixDecompositionException(String message, Throwable cause) {
        super(message, cause);
        this.decompositionName = null;
        this.matrixContext = null;
    }
    
    /**
     * Constructs a new matrix decomposition exception with the specified detail message,
     * decomposition name, and matrix context.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     */
    public MatrixDecompositionException(String message, String decompositionName, String matrixContext) {
        super(message);
        this.decompositionName = decompositionName;
        this.matrixContext = matrixContext;
    }
    
    /**
     * Constructs a new matrix decomposition exception with the specified detail message,
     * decomposition name, matrix context, and cause.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     * @param cause the cause
     */
    public MatrixDecompositionException(String message, String decompositionName, String matrixContext, Throwable cause) {
        super(message, cause);
        this.decompositionName = decompositionName;
        this.matrixContext = matrixContext;
    }
    
    /**
     * Gets the name of the decomposition algorithm that failed.
     * 
     * @return the decomposition name, or null if not specified
     */
    public String getDecompositionName() {
        return decompositionName;
    }
    
    /**
     * Gets context information about the matrix.
     * 
     * @return the matrix context, or null if not specified
     */
    public String getMatrixContext() {
        return matrixContext;
    }
    
    @Override
    public String getMessage() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getMessage());
        
        if (decompositionName != null) {
            sb.append(" (Decomposition: ").append(decompositionName).append(")");
        }
        
        if (matrixContext != null) {
            sb.append(" (Matrix: ").append(matrixContext).append(")");
        }
        
        return sb.toString();
    }
}