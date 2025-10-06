package com.yishape.lab.math.linalg.decomposition;

/**
 * Exception thrown when a matrix decomposition algorithm fails to converge
 * or encounters a numerical error during computation.
 * <p>
 * This exception is used when the decomposition algorithm itself fails,
 * rather than when preconditions are not met. Examples include:
 * <ul>
 *   <li>Failure to converge in iterative algorithms</li>
 *   <li>Numerical instability causing loss of precision</li>
 *   <li>Division by zero or overflow during computation</li>
 * </ul>
 * </p>
 * 
 * @since 2.0
 */
public class DecompositionFailedException extends MatrixDecompositionException {
    
    /** The iteration count when the failure occurred, if applicable. */
    private final int iterationCount;
    
    /** The error tolerance at the time of failure, if applicable. */
    private final double errorTolerance;
    
    /**
     * Constructs a new decomposition failed exception with the specified detail message.
     * 
     * @param message the detail message
     */
    public DecompositionFailedException(String message) {
        super(message);
        this.iterationCount = -1;
        this.errorTolerance = Double.NaN;
    }
    
    /**
     * Constructs a new decomposition failed exception with the specified detail message
     * and cause.
     * 
     * @param message the detail message
     * @param cause the cause
     */
    public DecompositionFailedException(String message, Throwable cause) {
        super(message, cause);
        this.iterationCount = -1;
        this.errorTolerance = Double.NaN;
    }
    
    /**
     * Constructs a new decomposition failed exception with detailed information.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     * @param iterationCount the iteration count when the failure occurred
     * @param errorTolerance the error tolerance at the time of failure
     */
    public DecompositionFailedException(String message, String decompositionName, String matrixContext,
                                     int iterationCount, double errorTolerance) {
        super(message, decompositionName, matrixContext);
        this.iterationCount = iterationCount;
        this.errorTolerance = errorTolerance;
    }
    
    /**
     * Constructs a new decomposition failed exception with detailed information and cause.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     * @param iterationCount the iteration count when the failure occurred
     * @param errorTolerance the error tolerance at the time of failure
     * @param cause the cause
     */
    public DecompositionFailedException(String message, String decompositionName, String matrixContext,
                                     int iterationCount, double errorTolerance, Throwable cause) {
        super(message, decompositionName, matrixContext, cause);
        this.iterationCount = iterationCount;
        this.errorTolerance = errorTolerance;
    }
    
    /**
     * Gets the iteration count when the failure occurred.
     * 
     * @return the iteration count, or -1 if not applicable
     */
    public int getIterationCount() {
        return iterationCount;
    }
    
    /**
     * Gets the error tolerance at the time of failure.
     * 
     * @return the error tolerance, or NaN if not applicable
     */
    public double getErrorTolerance() {
        return errorTolerance;
    }
    
    @Override
    public String getMessage() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getMessage());
        
        if (iterationCount >= 0) {
            sb.append(" (Iteration: ").append(iterationCount).append(")");
        }
        
        if (!Double.isNaN(errorTolerance)) {
            sb.append(" (Error tolerance: ").append(errorTolerance).append(")");
        }
        
        return sb.toString();
    }
}