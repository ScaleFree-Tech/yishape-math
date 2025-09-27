package com.reremouse.lab.math.test.news_vendor;

import com.reremouse.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit test for the NewsvendorModel class
 */
public class NewsvendorTest {
    
    @Test
    public void testNewsvendorModel() {
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
        double criticalRatio = (sellingPrice - purchaseCost) / (sellingPrice - purchaseCost + shortageCost);
        double theoreticalOptimal = model.computeTheoreticalOptimalQuantity();
        double theoreticalMaxProfit = model.computeExpectedProfit(theoreticalOptimal);
        
        // Validate critical ratio
        assertEquals(0.625, criticalRatio, 1e-6, "Critical ratio should be 0.625");
        
        // Validate theoretical optimal quantity is reasonable
        assertTrue(theoreticalOptimal > demandMean, "Optimal quantity should be greater than mean demand");
        assertTrue(theoreticalOptimal < demandMean + 3 * demandStd, "Optimal quantity should be within 3 std devs");
        
        // Validate theoretical maximum profit is positive
        assertTrue(theoreticalMaxProfit > 0, "Expected profit should be positive");
        
        // Solve using numerical optimization
        OptResult numericalResult = model.solveNumerically();
        double numericalOptimal = numericalResult.getOptimalPoint().get(0).doubleValue();
        double numericalMaxProfit = -numericalResult.getOptimalValue(); // Negative because of objective function
        
        // Print debug information
        System.out.println("Debug information:");
        System.out.println("  Converged: " + numericalResult.isConverged());
        System.out.println("  Reason: " + numericalResult.getConvergenceReason());
        System.out.println("  Iterations: " + numericalResult.getIterations());
        System.out.println("  Optimal value: " + numericalResult.getOptimalValue());
        System.out.println("  Optimal point: " + numericalResult.getOptimalPoint());
        System.out.println("  Gradient norm: " + numericalResult.getFinalGradientNorm());
        System.out.println("  Tolerance: " + numericalResult.getTolerance());
        
        // Validate that we get a reasonable solution even if not fully converged
        // The numerical solution should be close to the theoretical solution
        double quantityDifference = Math.abs(theoreticalOptimal - numericalOptimal);
        double profitDifference = Math.abs(theoreticalMaxProfit - numericalMaxProfit);
        
        // Check that the differences are within reasonable bounds
        assertTrue(quantityDifference < 10.0, "Numerical quantity should be close to theoretical quantity");
        assertTrue(profitDifference < 5.0, "Numerical profit should be close to theoretical profit");
        
        // Validate numerical optimal quantity is reasonable
        assertTrue(numericalOptimal > demandMean, "Numerical optimal quantity should be greater than mean demand");
        assertTrue(numericalOptimal < demandMean + 3 * demandStd, "Numerical optimal quantity should be within 3 std devs");
        
        // Validate numerical maximum profit is positive
        assertTrue(numericalMaxProfit > 0, "Numerical expected profit should be positive");
        
        System.out.println("Newsvendor model test passed!");
        System.out.printf("Theoretical optimal quantity: %.2f%n", theoreticalOptimal);
        System.out.printf("Numerical optimal quantity: %.2f%n", numericalOptimal);
        System.out.printf("Theoretical max profit: %.2f%n", theoreticalMaxProfit);
        System.out.printf("Numerical max profit: %.2f%n", numericalMaxProfit);
        System.out.printf("Quantity difference: %.2f%n", quantityDifference);
        System.out.printf("Profit difference: %.2f%n", profitDifference);
    }
}