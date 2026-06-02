package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Detailed debugging of Phase II optimization issue
 *
 * Note: RereSimplexTableau expects c to contain only original decision variables.
 * It automatically adds slack, surplus, and artificial variables internally.
 */
public class DetailedPhaseIIDebugTest {

    private RereSimplexLinProgSolver solver;
    private static final double TOLERANCE = 1e-6;

    @BeforeEach
    void setUp() {
        solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
    }

    @Test
    @DisplayName("Debug Phase II optimization step by step")
    void testDetailedPhaseIIDebugging() {
        System.out.println("=== DETAILED PHASE II DEBUGGING ===");

        // Problem: min 3*x1 + 2*x2 subject to x1+x2>=4, 2*x1+x2>=6, x1,x2>=0
        // Expected optimal: x1=2, x2=2, objective=10

        // c contains ONLY original decision variables (2 elements)
        // Tableau will internally add slack, surplus, and artificial variables
        IVector c = Linalg.vector(new double[]{3.0, 2.0});

        // A_eq has 2 columns for x1, x2 (Tableau adds slack/artificial internally)
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0},   // x1 + x2 >= 4
            {2.0, 1.0}    // 2*x1 + x2 >= 6
        });

        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});

        System.out.println("Problem setup:");
        System.out.println("  Objective (original vars only): " + c);
        System.out.println("  Constraints (original vars only): " + A_eq);
        System.out.println("  RHS: " + b_eq);
        System.out.println("  Note: Tableau will add slack/surplus/artificial variables internally");
        System.out.println();

        System.out.println("Expected optimal solution analysis:");
        System.out.println("  Point (2,2): constraints 2+2=4>=4, 2*2+2=6>=6, objective=3*2+2*2=10");
        System.out.println("  Therefore (2,2) with objective=10 should be optimal (minimum)");
        System.out.println();

        // Solve with detailed output
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);

        System.out.println("=== SOLVER RESULT ANALYSIS ===");
        System.out.println("Converged: " + result.isConverged());

        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            double x1 = solution.get(0);
            double x2 = solution.get(1);

            System.out.println("Solution found:");
            System.out.println("  x1 = " + x1 + ", x2 = " + x2);
            System.out.println("  Reported objective = " + result.getOptimalValue());

            // Check constraint satisfaction
            double constraint1 = x1 + x2;
            double constraint2 = 2*x1 + x2;
            System.out.println("Constraint verification:");
            System.out.println("  x1 + x2 = " + constraint1 + " >= 4? " + (constraint1 >= 4.0 - TOLERANCE));
            System.out.println("  2*x1 + x2 = " + constraint2 + " >= 6? " + (constraint2 >= 6.0 - TOLERANCE));

            // Assertions
            assertEquals(2.0, x1, TOLERANCE, "x1 should be 2.0");
            assertEquals(2.0, x2, TOLERANCE, "x2 should be 2.0");
            assertEquals(10.0, result.getOptimalValue(), TOLERANCE * 10, "Optimal objective should be 10");

            System.out.println("SUCCESS: Found optimal solution!");
        } else {
            System.out.println("SOLVER FAILED TO CONVERGE");
            fail("Solver should converge for this problem");
        }
    }
}
