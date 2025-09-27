package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.function.Executable;

/**
 * Test class for verifying the implementation of interface methods in RereIntegerProg
 */
public class RereIntegerProgInterfaceTest {

    /**
     * Test the setAllVariablesInteger() method
     */
    @Test
    public void testSetAllVariablesInteger() {
        // Create a simple problem first to set the variable count
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});
        
        RereIntegerProg solver = new RereIntegerProg();
        
        // Solve a problem first to set originalVariableCount
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // Now test the parameterless method
        solver.setAllVariablesInteger();
        
        // Verify that all variables are marked as integer
        // This is a bit tricky to test directly since the fields are private
        // We'll test by solving another problem and checking that the solution is integer
        IVector c2 = Linalg.vector(new double[]{1, 2});
        IMatrix A_eq2 = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq2 = Linalg.vector(new double[]{3});
        
        Tuple2<Double, IVector> result2 = solver.solveWithNonNegativeEqualConstraints(c2, A_eq2, b_eq2);
        
        assertNotNull(result2, "Result should not be null");
        IVector solution = result2.getSecond();
        assertNotNull(solution, "Solution should not be null");
        
        // Check that all variables are integers
        for (int i = 0; i < solution.length(); i++) {
            double value = (Double) solution.get(i);
            assertEquals(Math.round(value), value, 1e-6, "Variable " + i + " should be integer");
        }
    }
    
    /**
     * Test the setAllVariablesBinary() method
     */
    @Test
    public void testSetAllVariablesBinary() {
        // Create a simple problem first to set the variable count
        IVector c = Linalg.vector(new double[]{-1, -1}); // Maximize x1 + x2
        IMatrix A_ub = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_ub = Linalg.vector(new double[]{1}); // x1 + x2 <= 1
        
        RereIntegerProg solver = new RereIntegerProg();
        
        // Solve a problem first to set originalVariableCount
        Tuple2<Double, IVector> result = solver.solve(c, A_ub, b_ub);
        
        // Now test the parameterless method
        solver.setAllVariablesBinary();
        
        // Verify that all variables are marked as binary by solving another problem
        IVector c2 = Linalg.vector(new double[]{-1, -2}); // Maximize x1 + 2*x2
        IMatrix A_ub2 = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_ub2 = Linalg.vector(new double[]{1}); // x1 + x2 <= 1
        
        Tuple2<Double, IVector> result2 = solver.solve(c2, A_ub2, b_ub2);
        
        assertNotNull(result2, "Result should not be null");
        IVector solution = result2.getSecond();
        assertNotNull(solution, "Solution should not be null");
        
        // Check that all variables are binary (0 or 1)
        for (int i = 0; i < solution.length(); i++) {
            double value = (Double) solution.get(i);
            assertTrue(Math.abs(value) < 1e-6 || Math.abs(value - 1.0) < 1e-6, 
                      "Variable " + i + " should be 0 or 1");
        }
    }
    
    /**
     * Test that setAllVariablesInteger() throws exception when called before solving
     */
    @Test
    public void testSetAllVariablesIntegerBeforeSolving() {
        RereIntegerProg solver = new RereIntegerProg();
        
        // Use assertThrows for JUnit 5
        assertThrows(IllegalStateException.class, new Executable() {
            @Override
            public void execute() throws Throwable {
                solver.setAllVariablesInteger(); // Should throw IllegalStateException
            }
        });
    }
    
    /**
     * Test that setAllVariablesBinary() throws exception when called before solving
     */
    @Test
    public void testSetAllVariablesBinaryBeforeSolving() {
        RereIntegerProg solver = new RereIntegerProg();
        
        // Use assertThrows for JUnit 5
        assertThrows(IllegalStateException.class, new Executable() {
            @Override
            public void execute() throws Throwable {
                solver.setAllVariablesBinary(); // Should throw IllegalStateException
            }
        });
    }
}