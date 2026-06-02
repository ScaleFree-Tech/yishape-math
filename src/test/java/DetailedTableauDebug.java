import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * Detailed debug test to trace the simplex tableau state
 */
public class DetailedTableauDebug {
    public static void main(String[] args) {
        System.out.println("=== Detailed Tableau Debug ===\n");
        
        // Knapsack problem data
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // Print problem data
        System.out.println("Problem: Maximize sum(values[i] * x[i])");
        System.out.println("Subject to: sum(weights[i] * x[i]) <= " + capacity);
        System.out.println("\nItem data:");
        for (int i = 0; i < values.length; i++) {
            double density = values[i] / weights[i];
            System.out.printf("  x%d: value=%.1f, weight=%.1f, density=%.3f\n", 
                i, values[i], weights[i], density);
        }
        
        System.out.println("\nOptimal LP relaxation should select items by density:");
        System.out.println("  x0: density=6.000 (best)");
        System.out.println("  x1: density=5.000");
        System.out.println("  x2: density=4.000");
        System.out.println("  x5: density=3.333");
        System.out.println("  x4: density=3.000");
        System.out.println("  x3: density=2.000");
        System.out.println("  x6: density=5.000");
        
        System.out.println("\nExpected optimal LP solution:");
        System.out.println("  Select x0=1 (weight=10), x1=1 (weight=20), x2=1 (weight=30)");
        System.out.println("  Remaining capacity: 100 - 60 = 40");
        System.out.println("  Next best is x3 (weight=40) or x6 (weight=10)");
        System.out.println("  Choose x3=1 (weight=40) -> total weight=100");
        System.out.println("  Total value: 60+100+120+80 = 360");
        System.out.println("\nOR:");
        System.out.println("  Select x0=1, x1=1, x6=1 (weight=40), x2=1 (weight=30)");
        System.out.println("  Total weight=100, value=60+100+50+120=330");
        System.out.println("\nOR better:");
        System.out.println("  Select x1=1 (20), x2=1 (30), x4=1 (50) -> weight=100");
        System.out.println("  Total value: 100+120+150 = 370");
        System.out.println("\nOR even better:");
        System.out.println("  Select x1=1 (20), x5=1 (60), x0=1 (10), x6=1 (10) -> weight=100");
        System.out.println("  Total value: 100+200+60+50 = 410");
        
        // Convert to minimization problem (simplex maximizes by negating)
        var c = Linalg.vector(values).multiplyByScalar(-1.0);
        var A_ub = Linalg.matrix(new double[][]{weights});
        var b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("\n=== Solving with BetterSimplexLinProgSolver ===");
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        if (result != null) {
            System.out.println("\n=== RESULT ===");
            System.out.println("Optimal value (minimization): " + result.getOptimalValue());
            System.out.println("Optimal value (maximization): " + (-result.getOptimalValue()));
            System.out.println("Solution: " + result.getOptimalPoint());
            
            // Verify
            double totalValue = 0.0;
            double totalWeight = 0.0;
            System.out.println("\nSolution breakdown:");
            for (int i = 0; i < values.length; i++) {
                double x = result.getOptimalPoint().get(i);
                if (x > 1e-6) {
                    totalValue += values[i] * x;
                    totalWeight += weights[i] * x;
                    System.out.printf("  x%d = %.6f -> value=%.1f, weight=%.1f\n", 
                        i, x, values[i] * x, weights[i] * x);
                }
            }
            System.out.printf("\nTotal value: %.1f\n", totalValue);
            System.out.printf("Total weight: %.1f (capacity: %.1f)\n", totalWeight, capacity);
            
            // Check if this is optimal
            if (Math.abs(totalValue - 410.0) < 1e-6) {
                System.out.println("\n✓ CORRECT! Found optimal value 410");
            } else if (Math.abs(totalValue - 600.0) < 1e-6) {
                System.out.println("\n✗ WRONG! Got 600 (only x0=10), should be 410");
                System.out.println("This suggests the solver stopped too early after 1 pivot.");
            } else {
                System.out.println("\n? Got unexpected value: " + totalValue);
            }
        } else {
            System.out.println("Solver returned null!");
        }
    }
}
