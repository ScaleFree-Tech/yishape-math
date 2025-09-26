import com.reremouse.lab.math.linalg.*;
import com.reremouse.lab.math.optimize.linpg.SimplexLinProgSolver;
import com.reremouse.lab.util.Tuple2;

public class DebugEnteringVariable {
    public static void main(String[] args) {
        // 问题定义
        IVector c = IVector.of(new double[]{1, 2, 3});
        IMatrix A_eq = IMatrix.of(new double[][]{
            {1, 1, 1},
            {2, 1, 0}
        });
        IVector b_eq = IVector.of(new double[]{3, 2});
        
        System.out.println("=== 调试入基变量选择 ===");
        
        // 创建求解器并获取初始表
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        try {
            // 使用反射访问私有方法
            java.lang.reflect.Method buildTableauMethod = SimplexLinProgSolver.class.getDeclaredMethod("buildInitialTableau", IVector.class, IMatrix.class, IVector.class);
            buildTableauMethod.setAccessible(true);
            IMatrix tableau = (IMatrix) buildTableauMethod.invoke(solver, c, A_eq, b_eq);
            
            System.out.println("初始表:");
            printTableau(tableau);
            
            java.lang.reflect.Method isOptimalMethod = SimplexLinProgSolver.class.getDeclaredMethod("isOptimal", IMatrix.class);
            isOptimalMethod.setAccessible(true);
            boolean optimal = (Boolean) isOptimalMethod.invoke(solver, tableau);
            System.out.println("是否最优: " + optimal);
            
            if (!optimal) {
                java.lang.reflect.Method selectEnteringMethod = SimplexLinProgSolver.class.getDeclaredMethod("selectEnteringVariable", IMatrix.class, int.class);
                selectEnteringMethod.setAccessible(true);
                int enteringVar = (Integer) selectEnteringMethod.invoke(solver, tableau, 3);
                System.out.println("入基变量: " + enteringVar);
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    private static void printTableau(IMatrix tableau) {
        int rows = tableau.rows();
        int cols = tableau.cols();
        
        System.out.printf("%8s", "");
        for (int j = 0; j < cols - 1; j++) {
            System.out.printf("%12s", "x" + j);
        }
        System.out.printf("%12s%n", "RHS");
        
        for (int i = 0; i < rows - 1; i++) {
            System.out.printf("R%-6d:", i);
            for (int j = 0; j < cols; j++) {
                System.out.printf("%12.3f", (Double) tableau.get(i, j));
            }
            System.out.println();
        }
        
        System.out.printf("OBJ    :");
        for (int j = 0; j < cols; j++) {
            System.out.printf("%12.3f", (Double) tableau.get(rows - 1, j));
        }
        System.out.println();
    }
}