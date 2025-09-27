package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Specific test cases for SimplexLinProgSolver issues
 */
public class SimplexLinProgSolverSpecificTest {

    /**
     * Test the specific issue mentioned:
     * Minimize x1 subject to x1 + s1 = 1
     * Expected optimal solution: x1=0, s1=1 (objective value = 0)
     * But solver incorrectly returns: x1=1, s1=0 (objective value = 1)
     */
    @Test
    public void testMinimizeX1WithConstraint() {
        // Objective function: minimize x1 => c = [1, 0]
        IVector c = Linalg.vector(new double[]{1, 0});
        
        // Constraint: x1 + s1 = 1 => A = [[1, 1]], b = [1]
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{1});
        
        // Create solver
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        // Solve
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // Verify result
        assertNotNull(result, "Result should not be null");
        assertNotNull(result.getFirst(), "Optimal value should not be null");
        assertNotNull(result.getSecond(), "Optimal solution should not be null");
        
        // Extract solution
        IVector solution = result.getSecond();
        assertEquals(2, solution.length(), "Solution should have 2 variables");
        
        // Check constraint satisfaction
        IVector constraintValue = A_eq.mmul(solution);
        assertEquals(1.0, (double)constraintValue.get(0), 1e-3, "Constraint should be satisfied");
        
        // Check non-negativity
        assertTrue((double)solution.get(0) >= -1e-6, "x1 should be non-negative");
        assertTrue((double)solution.get(1) >= -1e-6, "s1 should be non-negative");
        
        // Check optimal solution
        // The optimal solution should be x1=0, s1=1 with objective value 0
        assertEquals(0.0, (double)solution.get(0), 1e-3, "x1 should be 0 (optimal)");
        assertEquals(1.0, (double)solution.get(1), 1e-3, "s1 should be 1 (optimal)");
        assertEquals(0.0, (double)result.getFirst(), 1e-3, "Objective value should be 0 (optimal)");
    }
}