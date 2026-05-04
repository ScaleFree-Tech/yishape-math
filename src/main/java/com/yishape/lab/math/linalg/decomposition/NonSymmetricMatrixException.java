package com.yishape.lab.math.linalg.decomposition;

/**
 * Exception thrown when attempting to decompose a non-symmetric matrix
 * with an algorithm that requires symmetry.
 * <p>
 * Some matrix decomposition algorithms (such as Cholesky decomposition)
 * require the input matrix to be symmetric. This exception is thrown
 * when that requirement is not met.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class NonSymmetricMatrixException extends MatrixDecompositionException {
    
    /** The threshold used to determine symmetry. */
    private final double symmetryThreshold;
    
    /** The maximum asymmetry found in the matrix. */
    private final double maxAsymmetry;
    
    /**
     * Constructs a new non-symmetric matrix exception with the specified detail message.
     * 
     * @param message the detail message
     */
    public NonSymmetricMatrixException(String message) {
        super(message);
        this.symmetryThreshold = Double.NaN;
        this.maxAsymmetry = Double.NaN;
    }
    
    /**
     * Constructs a new non-symmetric matrix exception with the specified detail message
     * and cause.
     * 
     * @param message the detail message
     * @param cause the cause
     */
    public NonSymmetricMatrixException(String message, Throwable cause) {
        super(message, cause);
        this.symmetryThreshold = Double.NaN;
        this.maxAsymmetry = Double.NaN;
    }
    
    /**
     * Constructs a new non-symmetric matrix exception with detailed information.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     * @param symmetryThreshold the threshold used to determine symmetry
     * @param maxAsymmetry the maximum asymmetry found in the matrix
     */
    public NonSymmetricMatrixException(String message, String decompositionName, String matrixContext,
                                     double symmetryThreshold, double maxAsymmetry) {
        super(message, decompositionName, matrixContext);
        this.symmetryThreshold = symmetryThreshold;
        this.maxAsymmetry = maxAsymmetry;
    }
    
    /**
     * Constructs a new non-symmetric matrix exception with detailed information and cause.
     * 
     * @param message the detail message
     * @param decompositionName the name of the decomposition algorithm
     * @param matrixContext context information about the matrix
     * @param symmetryThreshold the threshold used to determine symmetry
     * @param maxAsymmetry the maximum asymmetry found in the matrix
     * @param cause the cause
     */
    public NonSymmetricMatrixException(String message, String decompositionName, String matrixContext,
                                     double symmetryThreshold, double maxAsymmetry, Throwable cause) {
        super(message, decompositionName, matrixContext, cause);
        this.symmetryThreshold = symmetryThreshold;
        this.maxAsymmetry = maxAsymmetry;
    }
    
    /**
     * Gets the threshold used to determine symmetry.
     * 
     * @return the symmetry threshold
     */
    public double getSymmetryThreshold() {
        return symmetryThreshold;
    }
    
    /**
     * Gets the maximum asymmetry found in the matrix.
     * 
     * @return the maximum asymmetry
     */
    public double getMaxAsymmetry() {
        return maxAsymmetry;
    }
    
    @Override
    public String getMessage() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getMessage());
        
        if (!Double.isNaN(symmetryThreshold) && !Double.isNaN(maxAsymmetry)) {
            sb.append(" (Symmetry threshold: ").append(symmetryThreshold)
              .append(", Max asymmetry: ").append(maxAsymmetry).append(")");
        }
        
        return sb.toString();
    }
}