package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * Simple test to verify the binary variable methods in RereIntegerProg
 */
public class SimpleBinaryVariableTest {
    public static void main(String[] args) {
        System.out.println("Testing binary variable methods in RereIntegerProg");
        
        // Create a simple problem: maximize x1 + x2 subject to x1 + x2 <= 1.5
        // Solution should be x1=1, x2=1 (but constraint limits to x1+x2 <= 1.5, so one must be 0)
        // With binary constraints, solution should be either x1=1,x2=0 or x1=0,x2=1 with objective=1
        
        IVector<Double> c = Linalg.vector(new double[]{-1.0, -1.0}); // maximize x1 + x2 => minimize -x1 - x2
        IMatrix<Double> A_ub = Linalg.matrix(new double[][]{{1.0, 1.0}});
        IVector<Double> b_ub = Linalg.vector(new double[]{1.5});
        
        RereIntegerProg solver = new RereIntegerProg();
        
        // Test setAllVariablesBinary method
        System.out.println("Testing setAllVariablesBinary method...");
        solver.setAllVariablesBinary();
        
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        if (result != null) {
            System.out.println("Solution found:");
            System.out.println("Objective value: " + (-result.getOptimalValue())); // Convert back to maximization
            System.out.println("Solution vector: " + result.getOptimalPoint());
            
            // Verify it's a binary solution
            IVector solution = result.getOptimalPoint();
            boolean isBinary = true;
            for (int i = 0; i < solution.size(); i++) {
                double value = solution.get(i);
                if (Math.abs(value) > 1e-6 && Math.abs(value - 1.0) > 1e-6) {
                    isBinary = false;
                    break;
                }
            }
            System.out.println("Is binary solution: " + isBinary);
        } else {
            System.out.println("No solution found");
        }
        
        // Test addBinaryVariables method
        System.out.println("\nTesting addBinaryVariables method...");
        RereIntegerProg solver2 = new RereIntegerProg();
        solver2.addBinaryVariables(0, 1);
        
        OptResult result2 = solver2.solve(c, A_ub, b_ub);
        
        if (result2 != null) {
            System.out.println("Solution found:");
            System.out.println("Objective value: " + (-result2.getOptimalValue())); // Convert back to maximization
            System.out.println("Solution vector: " + result2.getOptimalPoint());
        } else {
            System.out.println("No solution found");
        }
        
        // Test setBinaryVariable method
        System.out.println("\nTesting setBinaryVariable method...");
        RereIntegerProg solver3 = new RereIntegerProg();
        solver3.setBinaryVariable(0);
        solver3.setBinaryVariable(1);
        
        OptResult result3 = solver3.solve(c, A_ub, b_ub);
        
        if (result3 != null) {
            System.out.println("Solution found:");
            System.out.println("Objective value: " + (-result3.getOptimalValue())); // Convert back to maximization
            System.out.println("Solution vector: " + result3.getOptimalPoint());
        } else {
            System.out.println("No solution found");
        }
        
        System.out.println("\nAll tests completed.");
    }
}