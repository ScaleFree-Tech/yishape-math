
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
/**
 *
 * @author lteb2
 */
public class TestLP {

    public static void main(String[] args) {

        // 创建单纯形法求解器 / Create simplex solver
        ILinProgSolver solver = Opts.simplexLinProgSolver();

// 定义线性规划问题 / Define linear programming problem
// minimize 2x1 + 3x2
// subject to x1 + x2 = 5, x1 ≥ 0, x2 ≥ 0
        IVector c = Linalg.vector(new double[]{2.0, 3.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1.0, 1.0}});
        IVector b_eq = Linalg.vector(new double[]{5.0});

// 求解(等式约束调用solveWithNonNegativeEqualConstraints，小于等于约束调用solve) / Solve
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);

        double optimalValue = result.getOptimalValue();
        IVector optimalSolution = result.getOptimalPoint();

        System.out.println("最优解: " + optimalSolution);
        System.out.println("最优值: " + optimalValue);

    }
}
