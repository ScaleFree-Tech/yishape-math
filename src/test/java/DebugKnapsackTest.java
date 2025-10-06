import com.yishape.lab.math.optimize.linpg.ComMath4LinProgSolver;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * Debug test for knapsack problem with RereSimplexLinProgSolver
 */
public class DebugKnapsackTest {
    public static void main(String[] args) {
        System.out.println("=== Debug Knapsack Problem ===");
        
        // Item values and weights
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // Convert to minimization problem
        var c = Linalg.vector(values).multiplyScalar(-1.0);
        var A_ub = Linalg.matrix(new double[][]{weights});
        var b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("Objective coefficients (for minimization): " + c);
        System.out.println("Constraint matrix: " + A_ub);
        System.out.println("Constraint RHS: " + b_ub);
        System.out.println();
        
        // Test with RereSimplexLinProgSolver (verbose)
        System.out.println("=== Testing BetterSimplexLinProgSolver ===");
        RereSimplexLinProgSolver betterSolver = new RereSimplexLinProgSolver();
        betterSolver.setVerbose(true);
        
        // Test LP relaxation first (without integer constraints)
        System.out.println("\n--- LP Relaxation (without integer constraints) ---");
        OptResult lpResult = betterSolver.solve(c, A_ub, b_ub);
        
        if (lpResult != null) {
            System.out.println("LP Relaxation result:");
            System.out.println("  Optimal value (minimization): " + lpResult.getOptimalValue());
            System.out.println("  Optimal value (maximization): " + (-lpResult.getOptimalValue()));
            System.out.println("  Solution: " + lpResult.getOptimalPoint());
            
            // Verify the solution
            double totalValue = 0.0;
            double totalWeight = 0.0;
            for (int i = 0; i < values.length; i++) {
                double x = lpResult.getOptimalPoint().get(i).doubleValue();
                totalValue += values[i] * x;
                totalWeight += weights[i] * x;
            }
            System.out.println("  Verification - Total value: " + totalValue);
            System.out.println("  Verification - Total weight: " + totalWeight);
        } else {
            System.out.println("LP Relaxation failed!");
        }
        
        // Test with integer constraints
        System.out.println("\n--- Integer Programming (with binary constraints) ---");
        RereSimplexLinProgSolver base = new RereSimplexLinProgSolver();
        base.setVerbose(false); // Turn off verbose for IP solver
        RereIntegerProg ipSolver = new RereIntegerProg(base);
        ipSolver.setAllVariablesBinary();
        ipSolver.setMaxDepth(100);
        ipSolver.setGapTolerance(1e-10);
        ipSolver.setTolerance(1e-10);
        ipSolver.setMaxIterations(10000);
        ipSolver.setVerbose(true);
        
        OptResult ipResult = ipSolver.solve(c, A_ub, b_ub);
        
        if (ipResult != null) {
            System.out.println("Integer Programming result:");
            System.out.println("  Optimal value (minimization): " + ipResult.getOptimalValue());
            System.out.println("  Optimal value (maximization): " + (-ipResult.getOptimalValue()));
            System.out.println("  Solution: " + ipResult.getOptimalPoint());
            
            // Verify the solution
            double totalValue = 0.0;
            double totalWeight = 0.0;
            for (int i = 0; i < values.length; i++) {
                double x = ipResult.getOptimalPoint().get(i).doubleValue();
                totalValue += values[i] * x;
                totalWeight += weights[i] * x;
                System.out.printf("  x%d = %.6f (value=%.1f, weight=%.1f)%n", i, x, values[i], weights[i]);
            }
            System.out.println("  Verification - Total value: " + totalValue);
            System.out.println("  Verification - Total weight: " + totalWeight);
        } else {
            System.out.println("Integer Programming failed!");
        }
        
        // Test with ComMath4LinProgSolver for comparison
        System.out.println("\n=== Testing ComMath4LinProgSolver for comparison ===");
        ILinProgSolver comMathBase = new ComMath4LinProgSolver();
        RereIntegerProg comMathIPSolver = new RereIntegerProg(comMathBase);
        comMathIPSolver.setAllVariablesBinary();
        comMathIPSolver.setMaxDepth(100);
        comMathIPSolver.setGapTolerance(1e-10);
        comMathIPSolver.setTolerance(1e-10);
        comMathIPSolver.setMaxIterations(10000);
        comMathIPSolver.setVerbose(false);
        
        OptResult comMathResult = comMathIPSolver.solve(c, A_ub, b_ub);
        
        if (comMathResult != null) {
            System.out.println("ComMath4 result:");
            System.out.println("  Optimal value (minimization): " + comMathResult.getOptimalValue());
            System.out.println("  Optimal value (maximization): " + (-comMathResult.getOptimalValue()));
            System.out.println("  Solution: " + comMathResult.getOptimalPoint());
            
            // Verify the solution
            double totalValue = 0.0;
            double totalWeight = 0.0;
            for (int i = 0; i < values.length; i++) {
                double x = comMathResult.getOptimalPoint().get(i).doubleValue();
                totalValue += values[i] * x;
                totalWeight += weights[i] * x;
                System.out.printf("  x%d = %.6f (value=%.1f, weight=%.1f)%n", i, x, values[i], weights[i]);
            }
            System.out.println("  Verification - Total value: " + totalValue);
            System.out.println("  Verification - Total weight: " + totalWeight);
        } else {
            System.out.println("ComMath4 failed!");
        }
    }
}
