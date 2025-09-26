package com.reremouse.lab.math.optimize.newton;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IOptimizer;
import com.reremouse.lab.math.optimize.RereLineSearch;
import com.reremouse.lab.util.Tuple2;

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
    public Tuple2<Double, IVector> optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun) {
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
        
        // Initialize variables
        IVector x = initX.copy();  // Current point
        int n = x.length();       // Problem dimension
        
        // Initialize inverse Hessian approximation as identity matrix
        IMatrix<Double> H = Linalg.eye(n);
        
        // Compute initial gradient
        IVector grad = grdFun.computeGradient(x);
        double initialGradNorm = (Double) grad.norm2();
        
        // Main iteration loop
        for (int iter = 0; iter < maxIterations; iter++) {
            // Check convergence: gradient norm is small enough
            double gradNorm = (Double) grad.norm2();
            if (gradNorm < tolerance * Math.max(1.0, initialGradNorm)) {
                double optimalValue = objFun.computeObjective(x);
                return new Tuple2<>(optimalValue, x);
            }
            
            // Compute search direction: d = -H * g
            IVector searchDirection = H.mmul(grad).multiplyScalar(-1.0);
            
            // Line search to determine step size
            double stepSize = new RereLineSearch().search(x, searchDirection, objFun, grdFun, grad);
            
            // Update position
            IVector newX = x.add(searchDirection.multiplyScalar(stepSize));
            IVector newGrad = grdFun.computeGradient(newX);
            
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
        return new Tuple2<>(finalValue, x);
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
        if (sTy <= 1e-10) {
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