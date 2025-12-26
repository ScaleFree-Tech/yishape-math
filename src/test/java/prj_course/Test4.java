package prj_course;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.Opts;

public class Test4 {
    public static void main(String[] args) {
        // 目标：max 90*x1 + 70*x2 + 80*x3 + 60*x4 + 95*x5 +85*x6
        // 约束：40*x1 + 30*x2 + 35*x3 + 25*x4 + 50*x5 + 45*x6 <= 100
        double[] obj = new double[] { 90, 70, 80, 60, 95, 85 };
        double[][] cons = new double[][] { { 40, 30, 35, 25, 50, 45 } };
        double[] b = new double[] { 100 };
        var c = Linalg.vector(obj).multiplyScalar(-1.0);
        var A_ub = Linalg.matrix(cons);
        var b_ub = Linalg.vector(b);
        // 单纯形法求解线性规划问题
        var solver = Opts.interPointLinProgSolver();
        var res = solver.solve(c, A_ub, b_ub);
        System.out.println(res.getOptimalPoint());
        // 0-1整数规划求解
        var solver2 = Opts.intLinProgSolver();
        solver2.setAllVariablesBinary();//设置所有变量为0或者1的二值变量
        var res2 = solver2.solve(c, A_ub, b_ub);
        System.out.println(-res2.getOptimalValue()+":"+res2.getOptimalPoint());
    }

}
