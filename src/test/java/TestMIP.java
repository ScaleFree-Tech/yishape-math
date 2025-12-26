
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.math.optimize.linpg.IIntegerProg;

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
/**
 *
 * @author lteb2
 */
public class TestMIP {

    public static void main(String[] args) {

        // 创建整数规划求解器 / Create integer programming solver
        IIntegerProg solver = Opts.intLinProgSolver();

// 定义整数规划问题 / Define integer programming problem
// minimize x1 + x2
// subject to x1 + x2 = 3, x1 ≥ 0, x2 ≥ 0, x1,x2 ∈ Z
        IVector c = Linalg.vector(new double[]{1.0, 1.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1.0, 1.0}});
        IVector b_eq = Linalg.vector(new double[]{3.0});

// 设置所有变量为整数(如果不是所有变量序号，则为混合整数规划) / Set all variables as integer(if not all variable indexes, it is mixed integer programming)
//        solver.addIntegerVariables(0, 1);
// 另一种方法设置所有变量为整数变量 / Another method to set all variables as integer variables
//solver.setAllVariablesInteger();
// 设置所有变量为二进制变量（0-1变量） / set all variables as binary variables (0-1 variables)
solver.setAllVariablesBinary();

// 求解 / Solve
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);

        double optimalValue = result.getOptimalValue();
        IVector optimalSolution = result.getOptimalPoint();

        System.out.println("最优整数解 / Optimal solution: " + optimalSolution);
        System.out.println("最优值 / Optimal value: " + optimalValue);

    }

}
