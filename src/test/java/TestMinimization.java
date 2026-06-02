import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

public class TestMinimization {
    public static void main(String[] args) {
        System.out.println("=== Testing Minimization Problem ===\n");
        
        // 问题: min 3*x1 + 2*x2
        // 约束: x1 + x2 - s1 + a1 = 4, 2*x1 + x2 - s2 + a2 = 6
        // 变量: [x1, x2, s1, s2, a1, a2]
        
        double bigM = 10000.0;
        var c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0, bigM, bigM});
        var A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0, 1.0, 0.0},
            {2.0, 1.0, 0.0, -1.0, 0.0, 1.0}
        });
        var b_eq = Linalg.vector(new double[]{4.0, 6.0});
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        
        System.out.println("Calling solveWithNonNegativeEqualConstraints...");
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        System.out.println("\n=== RESULT ===");
        if (result == null) {
            System.out.println("Result is NULL!");
        } else {
            System.out.println("Converged: " + result.isConverged());
            System.out.println("Convergence reason: " + result.getConvergenceReason());
            System.out.println("Optimal value: " + result.getOptimalValue());
            if (result.getOptimalPoint() != null) {
                System.out.println("Solution: " + result.getOptimalPoint());
            }
        }
    }
}
