package com.yishape.lab.math.linalg.decomposition;

/**
 * Exception thrown when attempting to decompose a non-square matrix
 * with an algorithm that requires a square matrix.
 * <p>
 * Some matrix decomposition algorithms (such as Cholesky, LU, and QR decompositions)
 * require the input matrix to be square. This exception is thrown when that 
 * requirement is not met.
 * </p>
 * 
 * @since 2.0
 */
public class NonSquareMatrixException extends MatrixDecompositionException {
    
    /** The number of rows in the matrix. */
    private final int rows;
    
    /** The number of columns in the matrix. */
    private final int columns;
    
    /**
     * Constructs a new non-square matrix exception with the specified detail message.
     * 
     * @param message the detail message
     */
    public NonSquareMatrixException(String message) {
        super(message);
        this.rows = -1;
        this.columns = -1;
    }
    
    /**
     * Constructs a new non-square matrix exception with the specified detail message
     * and cause.
     * 
     * @param message the detail message
     * @param cause the cause
     */
    public NonSquareMatrixException(String message, Throwable cause) {
        super(message, cause);
        this.rows = -1;
        this.columns = -1;
    }
    
    /**
     * Constructs a new non-square matrix exception with detailed information.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     */
    public NonSquareMatrixException(String message, String decompositionName, String matrixContext,
                                 int rows, int columns) {
        super(message, decompositionName, matrixContext);
        this.rows = rows;
        this.columns = columns;
    }
    
    /**
     * Constructs a new non-square matrix exception with detailed information and cause.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param cause the cause
     */
    public NonSquareMatrixException(String message, String decompositionName, String matrixContext,
                                 int rows, int columns, Throwable cause) {
        super(message, decompositionName, matrixContext, cause);
        this.rows = rows;
        this.columns = columns;
    }
    
    /**
     * Gets the number of rows in the matrix.
     * 
     * @return the number of rows, or -1 if not specified
     */
    public int getRows() {
        return rows;
    }
    
    /**
     * Gets the number of columns in the matrix.
     * 
     * @return the number of columns, or -1 if not specified
     */
    public int getColumns() {
        return columns;
    }
    
    @Override
    public String getMessage() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getMessage());
        
        if (rows >= 0 && columns >= 0) {
            sb.append(" (Matrix dimensions: ").append(rows).append("x").append(columns).append(")");
        }
        
        return sb.toString();
    }
}