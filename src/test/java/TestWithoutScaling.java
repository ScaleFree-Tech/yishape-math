import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * Test without numerical scaling
 */
public class TestWithoutScaling {
    public static void main(String[] args) {
        System.out.println("=== Test WITHOUT Scaling ===\n");
        
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        var c = Linalg.vector(values);
        var A_ub = Linalg.matrix(new double[][]{weights});
        var b_ub = Linalg.vector(new double[]{capacity});
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        solver.setUseNumericalScaling(false);  // DISABLE SCALING
        
        OptResult result = solver.maximize(c, A_ub, b_ub, null, null, null);
        
        if (result != null) {
            System.out.println("\n=== RESULT (without scaling) ===");
            System.out.println("Optimal value: " + result.getOptimalValue());
            System.out.println("Solution: " + result.getOptimalPoint());
            
            double totalValue = 0.0;
            double totalWeight = 0.0;
            for (int i = 0; i < values.length; i++) {
                double x = result.getOptimalPoint().get(i).doubleValue();
                if (x > 1e-6) {
                    totalValue += values[i] * x;
                    totalWeight += weights[i] * x;
                    System.out.printf("  x%d = %.6f\n", i, x);
                }
            }
            System.out.printf("Total value: %.1f, Total weight: %.1f\n", totalValue, totalWeight);
            
            if (Math.abs(totalValue - 410.0) < 1) {
                System.out.println("✓ CORRECT!");
            } else {
                System.out.println("✗ WRONG! Expected ~410");
            }
        }
    }
}
