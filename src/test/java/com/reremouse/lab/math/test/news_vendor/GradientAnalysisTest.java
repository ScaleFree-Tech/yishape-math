package com.reremouse.lab.math.test.news_vendor;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;
import org.junit.jupiter.api.Test;

/**
 * Gradient analysis for the newsvendor model to understand convergence issues
 */
public class GradientAnalysisTest {
    
    @Test
    public void testGradientAnalysis() {
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
        double criticalRatio = (sellingPrice + shortageCost - purchaseCost) / (sellingPrice + shortageCost);
        double theoreticalOptimal = model.computeTheoreticalOptimalQuantity();
        double theoreticalMaxProfit = model.computeExpectedProfit(theoreticalOptimal);
        
        System.out.println("Problem parameters:");
        System.out.println("  Purchase cost (c): " + purchaseCost);
        System.out.println("  Selling price (p): " + sellingPrice);
        System.out.println("  Shortage cost (s): " + shortageCost);
        System.out.println("  Demand mean (μ): " + demandMean);
        System.out.println("  Demand std (σ): " + demandStd);
        System.out.println();
        System.out.println("Theoretical calculation:");
        System.out.println("  Critical ratio (p+s-c)/(p+s): " + criticalRatio);
        System.out.println("  Theoretical optimal quantity: " + theoreticalOptimal);
        System.out.println("  Theoretical max profit: " + theoreticalMaxProfit);
        System.out.println();
        
        // Verify the critical ratio calculation
        NormalDistribution demandDist = new NormalDistribution(demandMean, demandStd);
        double verifyQuantity = demandDist.ppf(criticalRatio);
        System.out.println("Verification:");
        System.out.println("  F(Q*) = " + demandDist.cdf(theoreticalOptimal));
        System.out.println("  (p+s-c)/(p+s) = " + criticalRatio);
        System.out.println("  Match: " + (Math.abs(demandDist.cdf(theoreticalOptimal) - criticalRatio) < 1e-10));
        System.out.println();
        
        // Analyze gradient around the optimal point
        System.out.println("Gradient analysis around optimal point:");
        System.out.printf("%-10s %-15s %-15s %-15s %-15s%n", "Quantity", "Profit", "CDF", "Gradient", "Grad Norm");
        System.out.println("-".repeat(75));
        
        // Check points around the optimal quantity
        for (int i = -10; i <= 10; i++) {
            double quantity = theoreticalOptimal + i * 0.5;
            double profit = model.computeExpectedProfit(quantity);
            double cdf = demandDist.cdf(quantity);
            // Corrected gradient formula: d(-E[Profit])/dQ = c - (p + s) * (1 - F(Q))
            double gradient = purchaseCost - (sellingPrice + shortageCost) * (1 - cdf);
            
            System.out.printf("%-10.2f %-15.2f %-15.6f %-15.6f %-15.6f%n", 
                quantity, profit, cdf, gradient, Math.abs(gradient));
        }
        
        // Find where gradient is zero (or close to zero)
        System.out.println("\nSearching for zero gradient:");
        double epsilon = 1e-6;
        double low = theoreticalOptimal - 20;
        double high = theoreticalOptimal + 20;
        
        // Binary search for zero gradient
        while (high - low > epsilon) {
            double mid = (low + high) / 2;
            double cdf = demandDist.cdf(mid);
            // Corrected gradient formula: d(-E[Profit])/dQ = c - (p + s) * (1 - F(Q))
            double gradient = purchaseCost - (sellingPrice + shortageCost) * (1 - cdf);
            
            if (gradient > 0) {
                low = mid;
            } else {
                high = mid;
            }
        }
        
        double zeroGradientPoint = (low + high) / 2;
        double zeroGradientProfit = model.computeExpectedProfit(zeroGradientPoint);
        double zeroGradientCDF = demandDist.cdf(zeroGradientPoint);
        
        System.out.println("Zero gradient point: " + zeroGradientPoint);
        System.out.println("Profit at zero gradient: " + zeroGradientProfit);
        System.out.println("CDF at zero gradient: " + zeroGradientCDF);
        System.out.println("Theoretical CDF: " + criticalRatio);
        System.out.println("Difference: " + Math.abs(zeroGradientCDF - criticalRatio));
        
        // Test the gradient function directly
        System.out.println("\nTesting gradient function directly:");
        NewsvendorModel.NewsvendorGradientFunction gradFunc = 
            new NewsvendorModel.NewsvendorGradientFunction(model);
        
        for (int i = -5; i <= 5; i++) {
            double quantity = theoreticalOptimal + i * 1.0;
            IVector<Double> x = Linalg.vector(new double[]{quantity});
            IVector gradient = gradFunc.computeGradient(x);
            double gradValue = gradient.get(0).doubleValue();
            double profit = model.computeExpectedProfit(quantity);
            
            System.out.printf("Quantity: %8.2f, Profit: %8.2f, Gradient: %8.6f%n", 
                quantity, profit, gradValue);
        }
    }
}