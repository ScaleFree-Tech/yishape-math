package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.OptResult;

/**
 * Test class for verifying the setAllVariablesInteger() and setAllVariablesBinary() methods
 * can be called before solving a problem without throwing exceptions.
 */
public class SetAllVariablesTest {
    public static void main(String[] args) {
        System.out.println("Testing setAllVariablesInteger() and setAllVariablesBinary() methods...");
        
        // Test setAllVariablesInteger() before solving
        System.out.println("\n1. Testing setAllVariablesInteger() before solving:");
        try {
            RereIntegerProg solver1 = new RereIntegerProg();
            solver1.setAllVariablesInteger(); // Should not throw exception
            System.out.println("   SUCCESS: setAllVariablesInteger() did not throw exception");
            
            // Now solve a problem to verify it works
            IVector c1 = Linalg.vector(new double[]{1, 1});
            IMatrix A_eq1 = Linalg.matrix(new double[][]{{1, 1}});
            IVector b_eq1 = Linalg.vector(new double[]{2});
            
            OptResult result1 = solver1.solveWithNonNegativeEqualConstraints(c1, A_eq1, b_eq1);
            if (result1 != null) {
                System.out.println("   Solution: " + result1.getOptimalPoint());
                // Check if all variables are integers
                boolean allIntegers = true;
                for (int i = 0; i < result1.getOptimalPoint().length(); i++) {
                    double value = (Double) result1.getOptimalPoint().get(i);
                    if (Math.abs(value - Math.round(value)) > 1e-6) {
                        allIntegers = false;
                        break;
                    }
                }
                System.out.println("   All variables are integers: " + allIntegers);
            } else {
                System.out.println("   No solution found");
            }
        } catch (Exception e) {
            System.out.println("   FAILED: " + e.getMessage());
            e.printStackTrace();
        }
        
        // Test setAllVariablesBinary() before solving
        System.out.println("\n2. Testing setAllVariablesBinary() before solving:");
        try {
            RereIntegerProg solver2 = new RereIntegerProg();
            solver2.setAllVariablesBinary(); // Should not throw exception
            System.out.println("   SUCCESS: setAllVariablesBinary() did not throw exception");
            
            // Now solve a problem to verify it works
            IVector c2 = Linalg.vector(new double[]{-1, -1}); // Maximize x1 + x2
            IMatrix A_ub2 = Linalg.matrix(new double[][]{{1, 1}});
            IVector b_ub2 = Linalg.vector(new double[]{1}); // x1 + x2 <= 1
            
            OptResult result2 = solver2.solve(c2, A_ub2, b_ub2);
            if (result2 != null) {
                System.out.println("   Solution: " + result2.getOptimalPoint());
                // Check if all variables are binary (0 or 1)
                boolean allBinary = true;
                for (int i = 0; i < result2.getOptimalPoint().length(); i++) {
                    double value = (Double) result2.getOptimalPoint().get(i);
                    if (Math.abs(value) > 1e-6 && Math.abs(value - 1.0) > 1e-6) {
                        allBinary = false;
                        break;
                    }
                }
                System.out.println("   All variables are binary: " + allBinary);
            } else {
                System.out.println("   No solution found");
            }
        } catch (Exception e) {
            System.out.println("   FAILED: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\nTest completed.");
    }
}