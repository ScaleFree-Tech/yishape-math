package com.yishape.lab.math.linalg.decomposition;

/**
 * Exception thrown when attempting to decompose a non-positive definite matrix
 * with an algorithm that requires positive definiteness.
 * <p>
 * Some matrix decomposition algorithms (such as Cholesky decomposition)
 * require the input matrix to be positive definite. This exception is thrown
 * when that requirement is not met.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class NonPositiveDefiniteMatrixException extends MatrixDecompositionException {
    
    /** The threshold used to determine positive definiteness. */
    private final double positivityThreshold;
    
    /** The minimum diagonal element found in the matrix. */
    private final double minDiagonal;
    
    /**
     * Constructs a new non-positive definite matrix exception with the specified detail message.
     * 
     * @param message the detail message
     */
    public NonPositiveDefiniteMatrixException(String message) {
        super(message);
        this.positivityThreshold = Double.NaN;
        this.minDiagonal = Double.NaN;
    }
    
    /**
     * Constructs a new non-positive definite matrix exception with the specified detail message
     * and cause.
     * 
     * @param message the detail message
     * @param cause the cause
     */
    public NonPositiveDefiniteMatrixException(String message, Throwable cause) {
        super(message, cause);
        this.positivityThreshold = Double.NaN;
        this.minDiagonal = Double.NaN;
    }
    
    /**
     * Constructs a new non-positive definite matrix exception with detailed information.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     * @param positivityThreshold the threshold used to determine positive definiteness
     * @param minDiagonal the minimum diagonal element found in the matrix
     */
    public NonPositiveDefiniteMatrixException(String message, String decompositionName, String matrixContext,
                                           double positivityThreshold, double minDiagonal) {
        super(message, decompositionName, matrixContext);
        this.positivityThreshold = positivityThreshold;
        this.minDiagonal = minDiagonal;
    }
    
    /**
     * Constructs a new non-positive definite matrix exception with detailed information and cause.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     * @param positivityThreshold the threshold used to determine positive definiteness
     * @param minDiagonal the minimum diagonal element found in the matrix
     * @param cause the cause
     */
    public NonPositiveDefiniteMatrixException(String message, String decompositionName, String matrixContext,
                                           double positivityThreshold, double minDiagonal, Throwable cause) {
        super(message, decompositionName, matrixContext, cause);
        this.positivityThreshold = positivityThreshold;
        this.minDiagonal = minDiagonal;
    }
    
    /**
     * Gets the threshold used to determine positive definiteness.
     * 
     * @return the positivity threshold
     */
    public double getPositivityThreshold() {
        return positivityThreshold;
    }
    
    /**
     * Gets the minimum diagonal element found in the matrix.
     * 
     * @return the minimum diagonal element
     */
    public double getMinDiagonal() {
        return minDiagonal;
    }
    
    @Override
    public String getMessage() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getMessage());
        
        if (!Double.isNaN(positivityThreshold) && !Double.isNaN(minDiagonal)) {
            sb.append(" (Positivity threshold: ").append(positivityThreshold)
              .append(", Min diagonal: ").append(minDiagonal).append(")");
        }
        
        return sb.toString();
    }
}