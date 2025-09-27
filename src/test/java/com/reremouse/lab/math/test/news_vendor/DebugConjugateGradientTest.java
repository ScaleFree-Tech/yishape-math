package com.reremouse.lab.math.test.news_vendor;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.math.optimize.newton.RereConjugateGradient;
import com.reremouse.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Debug test for conjugate gradient optimizer
 */
public class DebugConjugateGradientTest {
    
    @Test
    public void testSimpleOptimization() {
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
        
        // Initial guess
        IVector initialGuess = Linalg.vector(new double[]{0.0});
        
        // Optimize
        RereConjugateGradient optimizer = new RereConjugateGradient(1e-6, 100, 0.1);
        OptResult result = optimizer.optimize(initialGuess, objFunc, gradFunc);
        
        double optimalPoint = result.getOptimalPoint().get(0).doubleValue();
        
        System.out.println("Simple quadratic test:");
        System.out.println("  Converged: " + result.isConverged());
        System.out.println("  Iterations: " + result.getIterations());
        System.out.println("  Optimal point: " + optimalPoint);
        System.out.println("  Expected: 5.0");
        System.out.println("  Difference: " + Math.abs(5.0 - optimalPoint));
        
        assertTrue(result.isConverged(), "Should converge on simple quadratic");
        assertEquals(5.0, optimalPoint, 1e-3, "Should find minimum at x=5");
    }
    
    @Test
    public void testNewsvendorGradientDirection() {
        // Test that the newsvendor gradient points in the right direction
        
        // Define problem parameters
        double purchaseCost = 5.0;    // Purchase cost per unit
        double sellingPrice = 10.0;   // Selling price per unit
        double shortageCost = 3.0;    // Shortage cost per unit
        double demandMean = 100.0;    // Demand mean
        double demandStd = 20.0;      // Demand standard deviation
        
        // Create newsvendor model instance
        NewsvendorModel model = new NewsvendorModel(
            purchaseCost, sellingPrice, shortageCost, demandMean, demandStd);
        
        // Compute theoretical optimal solution
        double theoreticalOptimal = model.computeTheoreticalOptimalQuantity();
        
        // Test gradient at different points
        System.out.println("Newsvendor gradient test:");
        System.out.println("  Theoretical optimal: " + theoreticalOptimal);
        
        // Test at a point below optimal
        double belowOptimal = theoreticalOptimal - 10.0;
        IVector pointBelow = Linalg.vector(new double[]{belowOptimal});
        NewsvendorGradientFunction gradFunc = new NewsvendorGradientFunction(model);
        IVector gradientBelow = gradFunc.computeGradient(pointBelow);
        double gradBelowValue = gradientBelow.get(0).doubleValue();
        
        System.out.println("  Point below optimal: " + belowOptimal);
        System.out.println("  Gradient below: " + gradBelowValue);
        
        // Test at a point above optimal
        double aboveOptimal = theoreticalOptimal + 10.0;
        IVector pointAbove = Linalg.vector(new double[]{aboveOptimal});
        IVector gradientAbove = gradFunc.computeGradient(pointAbove);
        double gradAboveValue = gradientAbove.get(0).doubleValue();
        
        System.out.println("  Point above optimal: " + aboveOptimal);
        System.out.println("  Gradient above: " + gradAboveValue);
        
        // For the negative profit function we're minimizing:
        // Gradient should be negative below optimal (need to increase Q to reduce -profit)
        // Gradient should be positive above optimal (need to decrease Q to reduce -profit)
        assertTrue(gradBelowValue < 0, "Gradient should be negative below optimal (minimizing -profit)");
        assertTrue(gradAboveValue > 0, "Gradient should be positive above optimal (minimizing -profit)");
    }
    
    // Inner classes for accessing private classes in NewsvendorModel
    static class NewsvendorGradientFunction implements IGradientFunction {
        private final NewsvendorModel model;
        
        public NewsvendorGradientFunction(NewsvendorModel model) {
            this.model = model;
        }
        
        @Override
        public IVector computeGradient(IVector x) {
            double quantity = x.get(0).doubleValue();
            double cdf = model.getDemandDistribution().cdf(quantity);
            double gradient = model.getPurchaseCost() - (model.getSellingPrice() + model.getShortageCost()) * (1 - cdf);
            return Linalg.vector(new double[]{gradient});
        }
    }
}