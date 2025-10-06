package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * Test with a fractional solution to see if branching works correctly
 */
public class FractionalKnapsackTest {
    public static void main(String[] args) {
        // Knapsack problem where the LP relaxation has a fractional solution:
        // Maximize: 3*x1 + 5*x2
        // Subject to: 2*x1 + 3*x2 <= 5
        // x1, x2 in {0, 1}
        // LP relaxation solution: x1=1, x2=1/3 with value 3*1 + 5*(1/3) = 3 + 5/3 = 14/3 ≈ 4.67
        // Integer solutions:
        // [0,0]: value 0
        // [0,1]: value 5 (feasible, weight=3)
        // [1,0]: value 3 (feasible, weight=2)
        // [1,1]: value 8 (infeasible, weight=5 > 5)
        // Optimal integer solution: [0,1] with value 5
        
        // Objective function coefficients (negative because we minimize)
        IVector c = Linalg.vector(new double[]{-3.0, -5.0});
        
        // Constraint matrix (weights)
        IMatrix A_ub = IMatrix.of(new double[][]{
            {2.0, 3.0}  // Weight constraint coefficients
        });
        
        // Constraint vector (capacity)
        IVector b_ub = Linalg.vector(new double[]{5.0});
        
        RereIntegerProg solver = new RereIntegerProg();
        solver.setAllVariablesBinary();  // Set all variables as binary (0-1)
        solver.setVerbose(true);
        solver.setMaxIterations(1000);
        solver.setGapTolerance(1e-9);
        solver.setTolerance(1e-9);
        
        System.out.println("Solving fractional knapsack problem with verbose output:");
        System.out.println("Maximize: 3*x1 + 5*x2");
        System.out.println("Subject to: 2*x1 + 3*x2 <= 5");
        System.out.println("x1, x2 in {0, 1}");
        
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        if (result != null) {
            System.out.println("Solution found:");
            System.out.println("Objective value (negative): " + result.getOptimalValue());
            System.out.println("Objective value (positive): " + (-result.getOptimalValue()));
            System.out.println("Solution vector: " + result.getOptimalPoint());
            System.out.println("Converged: " + result.isConverged());
        } else {
            System.out.println("No solution found");
        }
    }
}