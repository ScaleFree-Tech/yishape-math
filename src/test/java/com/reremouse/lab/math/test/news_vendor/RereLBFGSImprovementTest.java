package com.reremouse.lab.math.test.news_vendor;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify if RereLBFGS has similar issues and needs improvements
 */
public class RereLBFGSImprovementTest {
    
    @Test
    public void testLBFGSWithPoorInitialGuess() {
        // Simple quadratic function: f(x) = (x-5)^2
        // Minimum at x=5, gradient = 2(x-5)
        
        IObjectiveFunction objFunc = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double val = x.get(0).doubleValue();
                return (val - 5.0) * (val - 5.0);
            }
        };
        
        IGradientFunction gradFunc = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double val = x.get(0).doubleValue();
                return Linalg.vector(new double[]{2.0 * (val - 5.0)});
            }
        };
        
        // Very poor initial guess
        IVector initialGuess = Linalg.vector(new double[]{100.0});
        
        // Test with standard LBFGS
        RereLBFGS optimizer = new RereLBFGS(10, 1e-8, 100);
        
        OptResult result = optimizer.optimize(initialGuess, objFunc, gradFunc);
        double optimalPoint = result.getOptimalPoint().get(0).doubleValue();
        
        System.out.println("LBFGS test with poor initial guess:");
        System.out.println("  Converged: " + result.isConverged());
        System.out.println("  Iterations: " + result.getIterations());
        System.out.println("  Optimal point: " + optimalPoint);
        System.out.println("  Expected: 5.0");
        System.out.println("  Difference: " + Math.abs(5.0 - optimalPoint));
        
        // LBFGS should converge even with poor initial guess
        assertTrue(result.isConverged(), "LBFGS should converge with poor initial guess");
        assertEquals(5.0, optimalPoint, 1e-6, "LBFGS should find minimum at x=5");
    }
    
    /**
     * Simple quadratic function for testing: f(x) = (x-3)^2
     * Minimum at x=3, gradient = 2(x-3)
     */
    static class QuadraticObjective implements IObjectiveFunction {
        @Override
        public double computeObjective(IVector x) {
            double val = x.get(0).doubleValue();
            return (val - 3.0) * (val - 3.0);
        }
    }
    
    static class QuadraticGradient implements IGradientFunction {
        @Override
        public IVector computeGradient(IVector x) {
            double val = x.get(0).doubleValue();
            return Linalg.vector(new double[]{2.0 * (val - 3.0)});
        }
    }
    
    @Test
    public void testLBFGSOnSimpleQuadratic() {
        // Test on a simple quadratic function
        QuadraticObjective objFunc = new QuadraticObjective();
        QuadraticGradient gradFunc = new QuadraticGradient();
        
        // Poor initial guess
        IVector<Double> initialGuess = Linalg.vector(new double[]{10.0});
        
        // Test with LBFGS
        RereLBFGS optimizer = new RereLBFGS(10, 1e-10, 100);
        
        OptResult result = optimizer.optimize(initialGuess, objFunc, gradFunc);
        double optimalPoint = result.getOptimalPoint().get(0).doubleValue();
        
        System.out.println("LBFGS quadratic test:");
        System.out.println("  Converged: " + result.isConverged());
        System.out.println("  Iterations: " + result.getIterations());
        System.out.println("  Optimal point: " + optimalPoint);
        System.out.println("  Expected: 3.0");
        System.out.println("  Difference: " + Math.abs(3.0 - optimalPoint));
        
        assertTrue(result.isConverged(), "LBFGS should converge on quadratic function");
        assertEquals(3.0, optimalPoint, 1e-6, "LBFGS should find minimum at x=3");
    }
    
    @Test
    public void testLBFGSConvergenceWithSmallGradient() {
        // Test function where gradient is naturally small
        // f(x) = 1e-10 * x^2, minimum at x=0, gradient = 2e-10 * x
        // At x=1e-5, gradient = 2e-15 which is very small
        
        IObjectiveFunction objFunc = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double val = x.get(0).doubleValue();
                return 1e-10 * val * val;
            }
        };
        
        IGradientFunction gradFunc = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double val = x.get(0).doubleValue();
                return Linalg.vector(new double[]{2e-10 * val});
            }
        };
        
        // Initial point where gradient is small
        IVector initialGuess = Linalg.vector(new double[]{1e-5});
        
        // Test with standard LBFGS using tight tolerance
        RereLBFGS optimizer = new RereLBFGS(10, 1e-12, 100);
        
        OptResult result = optimizer.optimize(initialGuess, objFunc, gradFunc);
        double optimalPoint = result.getOptimalPoint().get(0).doubleValue();
        double gradientNorm = result.getFinalGradientNorm();
        
        System.out.println("LBFGS convergence test with small gradient:");
        System.out.println("  Converged: " + result.isConverged());
        System.out.println("  Iterations: " + result.getIterations());
        System.out.println("  Optimal point: " + optimalPoint);
        System.out.println("  Gradient norm: " + gradientNorm);
        System.out.println("  Tolerance: " + result.getTolerance());
        System.out.println("  Initial gradient norm: " + (2e-10 * 1e-5));
        
        // With the relative tolerance only, this might not converge properly
        // because gradient norm (2e-15) < tolerance (1e-12) * initialGradNorm (2e-15) = 2e-27
        // So the condition is 2e-15 < 2e-27 which is false
        // This means it might not converge even though the gradient is small
    }
    
    @Test
    public void testLBFGSConvergenceCondition() {
        // Demonstrate the convergence condition issue in RereLBFGS
        // The condition is: gradNorm < tolerance * Math.max(1.0, initialGradNorm)
        
        // Case 1: Small initial gradient
        double tolerance = 1e-6;
        double initialGradNorm = 1e-8; // Very small initial gradient
        double currentGradNorm = 1e-7; // Current gradient (larger than initial)
        
        boolean wouldConverge = currentGradNorm < tolerance * Math.max(1.0, initialGradNorm);
        System.out.println("Convergence test case 1:");
        System.out.println("  Initial gradient norm: " + initialGradNorm);
        System.out.println("  Current gradient norm: " + currentGradNorm);
        System.out.println("  Tolerance: " + tolerance);
        System.out.println("  Would converge: " + wouldConverge);
        System.out.println("  Threshold: " + (tolerance * Math.max(1.0, initialGradNorm)));
        
        // Case 2: Large initial gradient
        initialGradNorm = 100.0; // Large initial gradient
        currentGradNorm = 1e-4; // Small current gradient
        
        wouldConverge = currentGradNorm < tolerance * Math.max(1.0, initialGradNorm);
        System.out.println("\nConvergence test case 2:");
        System.out.println("  Initial gradient norm: " + initialGradNorm);
        System.out.println("  Current gradient norm: " + currentGradNorm);
        System.out.println("  Tolerance: " + tolerance);
        System.out.println("  Would converge: " + wouldConverge);
        System.out.println("  Threshold: " + (tolerance * Math.max(1.0, initialGradNorm)));
        
        // The issue is that in case 1, even though the gradient is very small (1e-7),
        // it won't converge because the threshold is tolerance * 1.0 = 1e-6
        // But 1e-7 < 1e-6, so it should converge!
        
        // Our improved version uses:
        // threshold = Math.max(tolerance, tolerance * Math.max(1.0, initialGradNorm))
        // In case 1: threshold = Math.max(1e-6, 1e-6 * 1.0) = 1e-6
        // In case 2: threshold = Math.max(1e-6, 1e-6 * 100.0) = 1e-4
        
        // Actually, let's check case 1 more carefully:
        // currentGradNorm = 1e-7
        // threshold = 1e-6
        // 1e-7 < 1e-6 -> TRUE, so it WOULD converge
        // 
        // Let's try a case where it wouldn't converge:
        initialGradNorm = 1e-10; // Extremely small initial gradient
        currentGradNorm = 1e-8; // Current gradient
        
        wouldConverge = currentGradNorm < tolerance * Math.max(1.0, initialGradNorm);
        System.out.println("\nConvergence test case 3 (problematic case):");
        System.out.println("  Initial gradient norm: " + initialGradNorm);
        System.out.println("  Current gradient norm: " + currentGradNorm);
        System.out.println("  Tolerance: " + tolerance);
        System.out.println("  Would converge: " + wouldConverge);
        System.out.println("  Threshold: " + (tolerance * Math.max(1.0, initialGradNorm)));
        
        // In this case: threshold = 1e-6 * 1.0 = 1e-6
        // But currentGradNorm = 1e-8, so 1e-8 < 1e-6 -> TRUE, it WOULD converge
        // 
        // Let me think of a case where it wouldn't converge:
        // We want: currentGradNorm >= tolerance * Math.max(1.0, initialGradNorm)
        // And we want currentGradNorm to be "small enough" that it should converge
        // 
        // Let's say we want convergence when gradient < 1e-10
        // But initialGradNorm = 1e-15
        // Then threshold = 1e-6 * 1.0 = 1e-6
        // If currentGradNorm = 1e-10, then 1e-10 < 1e-6 -> TRUE, so it converges
        // 
        // Actually, the current implementation is fine for these cases.
        // The issue might be more subtle.
    }
    
    @Test
    public void testLBFGSWithVeryPoorInitialGuess() {
        // Very challenging quadratic function
        IObjectiveFunction objFunc = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double val = x.get(0).doubleValue();
                return 0.5 * val * val - 1000.0 * val + 500000.0; // min at x=1000
            }
        };
        
        IGradientFunction gradFunc = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double val = x.get(0).doubleValue();
                return Linalg.vector(new double[]{val - 1000.0});
            }
        };
        
        // Very poor initial guess - far from optimal
        IVector initialGuess = Linalg.vector(new double[]{1000000.0});
        
        // Test with LBFGS
        RereLBFGS optimizer = new RereLBFGS(10, 1e-8, 1000);
        
        OptResult result = optimizer.optimize(initialGuess, objFunc, gradFunc);
        double optimalPoint = result.getOptimalPoint().get(0).doubleValue();
        
        System.out.println("LBFGS with very poor initial guess:");
        System.out.println("  Converged: " + result.isConverged());
        System.out.println("  Iterations: " + result.getIterations());
        System.out.println("  Optimal point: " + optimalPoint);
        System.out.println("  Expected: 1000.0");
        System.out.println("  Difference: " + Math.abs(1000.0 - optimalPoint));
        
        // LBFGS should still converge
        assertTrue(result.isConverged(), "LBFGS should converge with very poor initial guess");
        assertEquals(1000.0, optimalPoint, 1e-3, "LBFGS should find minimum at x=1000");
    }
    
    @Test
    public void testLBFGSImprovements() {
        // Test to demonstrate the improvements made to RereLBFGS:
        // 1. Combined absolute/relative tolerance
        // 2. Best solution tracking
        // 3. Stagnation detection
        
        // Create a function that might cause stagnation
        IObjectiveFunction objFunc = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double val = x.get(0).doubleValue();
                // A function with a very flat region
                if (Math.abs(val - 10.0) < 1e-10) {
                    return 0.0;
                }
                return Math.pow(val - 10.0, 2);
            }
        };
        
        IGradientFunction gradFunc = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double val = x.get(0).doubleValue();
                double grad;
                // Gradient in the flat region
                if (Math.abs(val - 10.0) < 1e-10) {
                    grad = 0.0;
                } else {
                    grad = 2 * (val - 10.0);
                }
                return Linalg.vector(new double[]{grad});
            }
        };
        
        // Initial point
        IVector initialGuess = Linalg.vector(new double[]{0.0});
        
        // Test with improved LBFGS
        RereLBFGS optimizer = new RereLBFGS(10, 1e-12, 100);
        
        OptResult result = optimizer.optimize(initialGuess, objFunc, gradFunc);
        double optimalPoint = result.getOptimalPoint().get(0).doubleValue();
        
        System.out.println("LBFGS improvements test:");
        System.out.println("  Converged: " + result.isConverged());
        System.out.println("  Reason: " + result.getConvergenceReason());
        System.out.println("  Iterations: " + result.getIterations());
        System.out.println("  Optimal point: " + optimalPoint);
        System.out.println("  Expected: 10.0");
        System.out.println("  Difference: " + Math.abs(10.0 - optimalPoint));
        System.out.println("  Final gradient norm: " + result.getFinalGradientNorm());
        
        // The improved LBFGS should converge to the correct solution
        assertTrue(result.isConverged(), "LBFGS should converge with improvements");
        assertEquals(10.0, optimalPoint, 1e-6, "LBFGS should find minimum at x=10");
    }
    
    @Test
    public void testLBFGSConvergenceWithCombinedTolerance() {
        // Test the improved convergence criteria with combined absolute/relative tolerance
        
        // Function with very small gradient
        IObjectiveFunction objFunc = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double val = x.get(0).doubleValue();
                return Math.pow(val, 2); // Simple quadratic function
            }
        };
        
        IGradientFunction gradFunc = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double val = x.get(0).doubleValue();
                return Linalg.vector(new double[]{2 * val});
            }
        };
        
        // Initial point at the minimum
        IVector initialGuess = Linalg.vector(new double[]{0.0});
        
        // Test with tight tolerance
        RereLBFGS optimizer = new RereLBFGS(10, 1e-10, 50);
        
        OptResult result = optimizer.optimize(initialGuess, objFunc, gradFunc);
        double optimalPoint = result.getOptimalPoint().get(0).doubleValue();
        double gradientNorm = result.getFinalGradientNorm();
        
        System.out.println("LBFGS combined tolerance test:");
        System.out.println("  Converged: " + result.isConverged());
        System.out.println("  Reason: " + result.getConvergenceReason());
        System.out.println("  Iterations: " + result.getIterations());
        System.out.println("  Optimal point: " + optimalPoint);
        System.out.println("  Gradient norm: " + gradientNorm);
        System.out.println("  Tolerance: " + result.getTolerance());
        
        // With the improved convergence criteria, this should converge properly
        assertTrue(result.isConverged(), "LBFGS should converge with combined tolerance");
        assertEquals(0.0, optimalPoint, 1e-6, "LBFGS should find minimum at x=0");
        assertTrue(gradientNorm < 1e-10, "Gradient norm should be very small");
    }
}