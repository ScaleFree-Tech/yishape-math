import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

public class TestMinimizationCorrect {
    public static void main(String[] args) {
        System.out.println("=== Testing Minimization with GEQ constraints ===\n");
        
        // 问题: min 3*x1 + 2*x2
        // 约束: x1 + x2 >= 4, 2*x1 + x2 >= 6, x1, x2 >= 0
        
        // 转换为标准形式：使用剩余变量
        // x1 + x2 - s1 = 4
        // 2*x1 + x2 - s2 = 6
        // 其中 s1, s2 >= 0 是剩余变量
        
        var c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0});
        var A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0},
            {2.0, 1.0, 0.0, -1.0}
        });
        var b_eq = Linalg.vector(new double[]{4.0, 6.0});
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(false);
        
        System.out.println("Calling solve (minimization)...");
        OptResult result = solver.solve(c, null, null, A_eq, b_eq, null);
        
        System.out.println("\n=== RESULT ===");
        if (result == null) {
            System.out.println("Result is NULL!");
        } else {
            System.out.println("Converged: " + result.isConverged());
            System.out.println("Convergence reason: " + result.getConvergenceReason());
            System.out.println("Optimal value: " + result.getOptimalValue());
            if (result.getOptimalPoint() != null) {
                System.out.println("Solution: " + result.getOptimalPoint());
                double x1 = result.getOptimalPoint().get(0);
                double x2 = result.getOptimalPoint().get(1);
                System.out.println("x1 = " + x1 + ", x2 = " + x2);
                System.out.println("Objective = 3*" + x1 + " + 2*" + x2 + " = " + (3*x1 + 2*x2));
            }
        }
    }
}
