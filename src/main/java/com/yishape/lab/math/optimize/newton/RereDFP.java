package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.RereLineSearch;
import com.yishape.lab.math.util.Precision;

import java.util.ArrayList;
import java.util.List;

/**
 * DFP拟牛顿优化算法
 * <p>
 * DFP (Davidon-Fletcher-Powell) algorithm is a quasi-Newton optimization method
 * for solving unconstrained nonlinear optimization problems. It approximates
 * the inverse of the Hessian matrix and updates this approximation at each iteration.
 * </p>
 * 
 * <h3>Algorithm Features:</h3>
 * <ul>
 *   <li>Superlinear convergence: Fast convergence near optimal solution</li>
 *   <li>No Hessian computation: Only requires objective function and gradient</li>
 *   <li>Positive definite updates: Maintains positive definiteness of Hessian approximation</li>
 *   <li>Suitable for medium-scale problems</li>
 * </ul>
 * 
 * @author lteb2
 */
public class RereDFP implements IOptimizer {

    // DFP algorithm parameters
    private double tolerance = 1e-6;       // Convergence tolerance
    private int maxIterations = 1000;      // Maximum iterations

    /**
     * Constructor with default parameters
     */
    public RereDFP() {
    }
    
    /**
     * Constructor with custom parameters
     * 
     * @param tolerance Convergence tolerance
     * @param maxIterations Maximum iterations
     */
    public RereDFP(double tolerance, int maxIterations) {
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
    }

    @Override
    public OptResult optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun) {
        // Parameter validation
        if (initX == null) {
            throw new IllegalArgumentException("Initial point cannot be null");
        }
        if (objFun == null) {
            throw new IllegalArgumentException("Objective function cannot be null");
        }
        if (grdFun == null) {
            throw new IllegalArgumentException("Gradient function cannot be null");
        }
        
        // Record start time
        long startTime = System.currentTimeMillis();
        
        // Initialize variables
        IVector x = initX.copy();  // Current point
        IVector initialPoint = initX.copy(); // Save initial point
        int n = x.length();       // Problem dimension
        
        // Compute initial function value
        double initialValue = objFun.computeObjective(x);
        
        // Initialize inverse Hessian approximation as identity matrix
        IMatrix<Double> H = Linalg.eye(n);
        
        // Convergence history tracking
        List<Double> functionValueHistory = new ArrayList<>();
        List<Double> gradientNormHistory = new ArrayList<>();
        List<IVector> parameterHistory = new ArrayList<>();
        
        // Evaluation counters
        int functionEvaluations = 1; // Initial function value computation
        int gradientEvaluations = 0; // Gradient evaluations will start counting in loop
        
        // Compute initial gradient
        IVector grad = grdFun.computeGradient(x);
        gradientEvaluations++;
        double initialGradNorm = (Double) grad.norm2();
        double finalGradientNorm = initialGradNorm;
        
        // Add initial history records
        functionValueHistory.add(initialValue);
        gradientNormHistory.add(initialGradNorm);
        parameterHistory.add(x.copy());
        
        boolean converged = false;
        String convergenceReason = "Maximum iterations reached";
        int actualIterations = 0;
        
        // Main iteration loop
        for (int iter = 0; iter < maxIterations; iter++) {
            actualIterations = iter + 1;
            
            // Check convergence: gradient norm is small enough
            double gradNorm = (Double) grad.norm2();
            finalGradientNorm = gradNorm;
            double convergenceThreshold = tolerance * Math.max(1.0, initialGradNorm);
            if (Precision.compareTo(gradNorm, convergenceThreshold, tolerance) < 0) {
                converged = true;
                convergenceReason = "Gradient norm below tolerance";
                double optimalValue = objFun.computeObjective(x);
                functionEvaluations++;
                
                // Build rich OptResult
                OptResult.Builder builder = new OptResult.Builder(optimalValue, x)
                    .initialPoint(initialPoint)
                    .initialValue(initialValue)
                    .converged(converged)
                    .convergenceReason(convergenceReason)
                    .iterations(actualIterations)
                    .maxIterations(maxIterations)
                    .finalGradientNorm(finalGradientNorm)
                    .tolerance(tolerance)
                    .executionTimeMs(System.currentTimeMillis() - startTime)
                    .functionEvaluations(functionEvaluations)
                    .gradientEvaluations(gradientEvaluations)
                    .functionValueHistory(functionValueHistory)
                    .gradientNormHistory(gradientNormHistory)
                    .parameterHistory(parameterHistory);
                
                return builder.build();
            }
            
            // Compute search direction: d = -H * g
            IVector searchDirection = H.mmul(grad).multiplyScalar(-1.0);
            
            // Line search to determine step size
            double stepSize = new RereLineSearch().search(x, searchDirection, objFun, grdFun, grad);
            
            // Update position
            IVector newX = x.add(searchDirection.multiplyScalar(stepSize));
            IVector newGrad = grdFun.computeGradient(newX);
            gradientEvaluations++;
            
            // Compute new function value and record
            double newValue = objFun.computeObjective(newX);
            functionEvaluations++;
            functionValueHistory.add(newValue);
            gradientNormHistory.add((Double) newGrad.norm2());
            parameterHistory.add(newX.copy());
            
            // Compute differences
            IVector s = newX.sub(x);           // s = x_{k+1} - x_k
            IVector y = newGrad.sub(grad);     // y = g_{k+1} - g_k
            
            // Update inverse Hessian approximation using DFP formula
            H = updateInverseHessian(H, s, y);
            
            // Update current point and gradient
            x = newX;
            grad = newGrad;
        }
        
        // Maximum iterations reached, return current best solution
        double finalValue = objFun.computeObjective(x);
        functionEvaluations++;
        
        // Build rich OptResult
        OptResult.Builder builder = new OptResult.Builder(finalValue, x)
            .initialPoint(initialPoint)
            .initialValue(initialValue)
            .converged(converged)
            .convergenceReason(convergenceReason)
            .iterations(actualIterations)
            .maxIterations(maxIterations)
            .finalGradientNorm(finalGradientNorm)
            .tolerance(tolerance)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .functionEvaluations(functionEvaluations)
            .gradientEvaluations(gradientEvaluations)
            .functionValueHistory(functionValueHistory)
            .gradientNormHistory(gradientNormHistory)
            .parameterHistory(parameterHistory);
        
        return builder.build();
    }
    
    /**
     * Update inverse Hessian approximation using DFP formula
     * <p>
     * DFP update formula:
     * H_{k+1} = H_k + (s * s^T) / (s^T * y) - (H_k * y * y^T * H_k) / (y^T * H_k * y)
     * 
     * where:
     * - s = x_{k+1} - x_k (position difference)
     * - y = g_{k+1} - g_k (gradient difference)
     * - H_k is the inverse Hessian approximation at iteration k
     * </p>
     * 
     * @param H Current inverse Hessian approximation
     * @param s Position difference vector
     * @param y Gradient difference vector
     * @return Updated inverse Hessian approximation
     */
    private IMatrix<Double> updateInverseHessian(IMatrix<Double> H, IVector s, IVector y) {
        // Compute s^T * y (needed for DFP update)
        double sTy = (Double) s.innerProduct(y);
        
        // Check curvature condition: s^T * y > 0, required for positive definiteness
        if (Precision.compareTo(sTy, 1e-10, tolerance) <= 0) {
            // If curvature condition is not satisfied, return current Hessian approximation
            return H;
        }
        
        // Convert vectors to column matrices for matrix operations
        IMatrix<Double> sMatrix = s.asColumnVector();
        IMatrix<Double> yMatrix = y.asColumnVector();
        
        // Compute s * s^T
        IMatrix<Double> sOuter = sMatrix.mmul(sMatrix.transposeNew());
        
        // Compute (s * s^T) / (s^T * y)
        IMatrix<Double> term1 = sOuter.multiplyScalar(1.0 / sTy);
        
        // Compute H * y
        IVector Hy = H.mmul(y);
        IMatrix<Double> HyMatrix = Hy.asColumnVector();
        
        // Compute (H * y) * (H * y)^T
        IMatrix<Double> HyOuter = HyMatrix.mmul(HyMatrix.transposeNew());
        
        // Compute y^T * H * y
        double yTHy = (Double) y.innerProduct(Hy);
        
        // Compute (H * y * y^T * H) / (y^T * H * y)
        IMatrix<Double> term2 = HyOuter.multiplyScalar(1.0 / yTHy);
        
        // Apply DFP update formula: H_{k+1} = H_k + term1 - term2
        IMatrix<Double> newH = H.add(term1).sub(term2);
        
        return newH;
    }
    
    // Getter and Setter methods
    
    /**
     * Get convergence tolerance
     * @return Convergence tolerance
     */
    public double getTolerance() {
        return tolerance;
    }
    
    /**
     * Set convergence tolerance
     * @param tolerance Convergence tolerance
     */
    public void setTolerance(double tolerance) {
        this.tolerance = Math.max(1e-12, tolerance);
    }
    
    /**
     * Get maximum iterations
     * @return Maximum iterations
     */
    public int getMaxIterations() {
        return maxIterations;
    }
    
    /**
     * Set maximum iterations
     * @param maxIterations Maximum iterations
     */
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = Math.max(1, maxIterations);
    }
}