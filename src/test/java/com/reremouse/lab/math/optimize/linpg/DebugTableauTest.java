package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.*;
import org.junit.jupiter.api.Test;
import java.lang.reflect.Method;

/**
 * 调试Tableau构建过程的测试类
 */
public class DebugTableauTest {
    
    @Test
    public void debugTableauConstruction() throws Exception {
        System.out.println("=== 调试Tableau构建过程 ===");
        
        // 问题：minimize 2*x1 + x2
        // subject to x1 + x2 = 2.5
        
        IVector c = Linalg.vector(new double[]{2, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2.5});
        
        System.out.println("目标函数系数: " + c);
        System.out.println("约束矩阵: " + A_eq);
        System.out.println("约束右端: " + b_eq);
        
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        // 使用反射调用私有方法buildInitialTableau
        Method buildInitialTableauMethod = SimplexLinProgSolver.class.getDeclaredMethod(
            "buildInitialTableau", IVector.class, IMatrix.class, IVector.class);
        buildInitialTableauMethod.setAccessible(true);
        
        IMatrix initialTableau = (IMatrix) buildInitialTableauMethod.invoke(solver, c, A_eq, b_eq);
        
        System.out.println("\n初始Tableau:");
        printTableau(initialTableau);
        
        // 分析初始tableau
        int rows = initialTableau.rows();
        int cols = initialTableau.cols();
        
        System.out.println("\nTableau分析:");
        System.out.println("行数: " + rows + " (约束数 + 1个目标函数行)");
        System.out.println("列数: " + cols + " (变量数 + 人工变量数 + 1个RHS列)");
        
        // 显示目标函数行
        System.out.println("\n目标函数行 (最后一行):");
        for (int j = 0; j < cols; j++) {
            double value = (Double) initialTableau.get(rows - 1, j);
            System.out.printf("%.6f ", value);
        }
        System.out.println();
        
        // 分析基变量
        System.out.println("\n基变量分析:");
        for (int i = 0; i < rows - 1; i++) {
            System.out.print("约束 " + i + ": ");
            for (int j = 0; j < cols - 1; j++) {
                double value = (Double) initialTableau.get(i, j);
                if (Math.abs(value - 1.0) < 1e-9) {
                    // 检查是否为基变量
                    boolean isBasic = true;
                    for (int k = 0; k < rows - 1; k++) {
                        if (k != i) {
                            double otherValue = (Double) initialTableau.get(k, j);
                            if (Math.abs(otherValue) > 1e-9) {
                                isBasic = false;
                                break;
                            }
                        }
                    }
                    if (isBasic) {
                        double rhsValue = (Double) initialTableau.get(i, cols - 1);
                        if (j < 2) {
                            System.out.printf("x%d = %.6f (原始变量) ", j + 1, rhsValue);
                        } else {
                            System.out.printf("s%d = %.6f (人工变量) ", j - 1, rhsValue);
                        }
                    }
                }
            }
            System.out.println();
        }
    }
    
    private void printTableau(IMatrix tableau) {
        int rows = tableau.rows();
        int cols = tableau.cols();
        
        // 打印表头
        System.out.print("     ");
        for (int j = 0; j < cols - 1; j++) {
            if (j < 2) {
                System.out.printf("    x%-6d", j + 1);
            } else {
                System.out.printf("    s%-6d", j - 1);
            }
        }
        System.out.println("       RHS");
        
        // 打印约束行
        for (int i = 0; i < rows - 1; i++) {
            System.out.printf("C%-2d: ", i + 1);
            for (int j = 0; j < cols; j++) {
                double value = (Double) tableau.get(i, j);
                System.out.printf("%10.6f ", value);
            }
            System.out.println();
        }
        
        // 打印目标函数行
        System.out.print("OBJ: ");
        for (int j = 0; j < cols; j++) {
            double value = (Double) tableau.get(rows - 1, j);
            System.out.printf("%10.6f ", value);
        }
        System.out.println();
    }
}