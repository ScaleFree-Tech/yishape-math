import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;

/**
 * Test cases for RereSimplexLinProgSolver with inequality constraint support
 测试 RereSimplexLinProgSolver 的不等式约束支持
 */
public class BetterSimplexLinProgSolverTest {
    
    private static final double TOLERANCE = 1e-6;
    
    public static void main(String[] args) {
        BetterSimplexLinProgSolverTest test = new BetterSimplexLinProgSolverTest();
        
        System.out.println("=== Testing BetterSimplexLinProgSolver with Inequality Constraints ===\n");
        
        // Test 1: Only inequality constraints
        test.testOnlyInequalityConstraints();
        
        // Test 2: Only equality constraints  
        test.testOnlyEqualityConstraints();
        
        // Test 3: Mixed inequality and equality constraints
        test.testMixedConstraints();
        
        // Test 4: Standard example from linear programming textbook
        test.testStandardExample();
        
        // Test 5: Large scale problem
        test.testLargeScaleProblem();
        
        System.out.println("\n=== All Tests Completed ===");
    }
    
    /**
     * Test 1: Only inequality constraints
     * 
     * Maximize: 3*x1 + 2*x2
     * Subject to:
     *   2*x1 + x2 <= 4
     *   x1 + 2*x2 <= 3  
     *   x1, x2 >= 0
     *   
     * Expected optimal solution: x1 = 5/3, x2 = 2/3, value = 19/3 ≈ 6.33
     */
    private void testOnlyInequalityConstraints() {
        System.out.println("Test 1: Only Inequality Constraints");
        System.out.println("Maximize: 3*x1 + 2*x2");
        System.out.println("Subject to: 2*x1 + x2 <= 4, x1 + 2*x2 <= 3, x1,x2 >= 0");
        
        try {
            // Set up the problem
            IVector c = IVector.of(3.0, 2.0);  // objective coefficients
            
            // Inequality constraints: A_ub * x <= b_ub
            IMatrix A_ub = IMatrix.of(new double[][]{
                {2.0, 1.0},  // 2*x1 + x2 <= 4
                {1.0, 2.0}   // x1 + 2*x2 <= 3
            });
            IVector b_ub = IVector.of(4.0, 3.0);
            
            // No equality constraints
            IMatrix A_eq = null;
            IVector b_eq = null;
            
            // Solve
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            solver.setVerbose(true);
            
            OptResult result = solver.maximize(c, A_ub, b_ub, A_eq, b_eq, null);
            
            // Check result
            if (result.isConverged()) {
                IVector solution = result.getOptimalPoint();
                double objectiveValue = result.getOptimalValue();
                
                System.out.println("Solution found:");
                System.out.printf("x1 = %.6f, x2 = %.6f\n", 
                    solution.get(0).doubleValue(), solution.get(1).doubleValue());
                System.out.printf("Objective value = %.6f\n", objectiveValue);
                
                // Verify constraints
                verifyInequalityConstraints(solution, A_ub, b_ub);
                
                // Expected optimal solution: x1 = 5/3, x2 = 2/3, value = 19/3
                double expectedX1 = 5.0/3.0;
                double expectedX2 = 2.0/3.0;
                double expectedValue = 19.0/3.0;
                
                if (Math.abs(solution.get(0).doubleValue() - expectedX1) < TOLERANCE &&
                    Math.abs(solution.get(1).doubleValue() - expectedX2) < TOLERANCE &&
                    Math.abs(objectiveValue - expectedValue) < TOLERANCE) {
                    System.out.println("✓ Test 1 PASSED");
                } else {
                    System.out.printf("✗ Test 1 FAILED: Expected x1=%.6f, x2=%.6f, value=%.6f\n", 
                        expectedX1, expectedX2, expectedValue);
                }
            } else {
                System.out.println("✗ Test 1 FAILED: Solver did not converge - " + result.getConvergenceReason());
            }
            
        } catch (Exception e) {
            System.out.println("✗ Test 1 FAILED with exception: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println();
    }
    
    /**
     * Test 2: Only equality constraints
     * 
     * Maximize: x1 + 2*x2
     * Subject to:
     *   x1 + x2 = 3
     *   x1, x2 >= 0
     *   
     * Expected optimal solution: x1 = 0, x2 = 3, value = 6
     */
    private void testOnlyEqualityConstraints() {
        System.out.println("Test 2: Only Equality Constraints");
        System.out.println("Maximize: x1 + 2*x2");
        System.out.println("Subject to: x1 + x2 = 3, x1,x2 >= 0");
        
        try {
            // Set up the problem
            IVector c = IVector.of(1.0, 2.0);  // objective coefficients
            
            // No inequality constraints
            IMatrix A_ub = null;
            IVector b_ub = null;
            
            // Equality constraints: A_eq * x = b_eq
            IMatrix A_eq = IMatrix.of(new double[][]{
                {1.0, 1.0}   // x1 + x2 = 3
            });
            IVector b_eq = IVector.of(3.0);
            
            // Solve
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            solver.setVerbose(true);
            
            OptResult result = solver.maximize(c, A_ub, b_ub, A_eq, b_eq, null);
            
            // Check result
            if (result.isConverged()) {
                IVector solution = result.getOptimalPoint();
                double objectiveValue = result.getOptimalValue();
                
                System.out.println("Solution found:");
                System.out.printf("x1 = %.6f, x2 = %.6f\n", 
                    solution.get(0).doubleValue(), solution.get(1).doubleValue());
                System.out.printf("Objective value = %.6f\n", objectiveValue);
                
                // Verify constraints
                verifyEqualityConstraints(solution, A_eq, b_eq);
                
                // Expected optimal solution: x1 = 0, x2 = 3, value = 6
                double expectedX1 = 0.0;
                double expectedX2 = 3.0;
                double expectedValue = 6.0;
                
                if (Math.abs(solution.get(0).doubleValue() - expectedX1) < TOLERANCE &&
                    Math.abs(solution.get(1).doubleValue() - expectedX2) < TOLERANCE &&
                    Math.abs(objectiveValue - expectedValue) < TOLERANCE) {
                    System.out.println("✓ Test 2 PASSED");
                } else {
                    System.out.printf("✗ Test 2 FAILED: Expected x1=%.6f, x2=%.6f, value=%.6f\n", 
                        expectedX1, expectedX2, expectedValue);
                }
            } else {
                System.out.println("✗ Test 2 FAILED: Solver did not converge - " + result.getConvergenceReason());
            }
            
        } catch (Exception e) {
            System.out.println("✗ Test 2 FAILED with exception: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println();
    }
    
    /**
     * Test 3: Mixed inequality and equality constraints
     * 
     * Maximize: 2*x1 + 3*x2
     * Subject to:
     *   x1 + x2 <= 4      (inequality)
     *   2*x1 + x2 = 5     (equality)
     *   x1, x2 >= 0
     *   
     * Expected optimal solution: x1 = 1, x2 = 3, value = 11
     */
    private void testMixedConstraints() {
        System.out.println("Test 3: Mixed Inequality and Equality Constraints");
        System.out.println("Maximize: 2*x1 + 3*x2");
        System.out.println("Subject to: x1 + x2 <= 4, 2*x1 + x2 = 5, x1,x2 >= 0");
        
        try {
            // Set up the problem
            IVector c = IVector.of(2.0, 3.0);  // objective coefficients
            
            // Inequality constraints: A_ub * x <= b_ub
            IMatrix A_ub = IMatrix.of(new double[][]{
                {1.0, 1.0}   // x1 + x2 <= 4
            });
            IVector b_ub = IVector.of(4.0);
            
            // Equality constraints: A_eq * x = b_eq
            IMatrix A_eq = IMatrix.of(new double[][]{
                {2.0, 1.0}   // 2*x1 + x2 = 5
            });
            IVector b_eq = IVector.of(5.0);
            
            // Solve
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            solver.setVerbose(true);
            
            OptResult result = solver.maximize(c, A_ub, b_ub, A_eq, b_eq, null);
            
            // Check result
            if (result.isConverged()) {
                IVector solution = result.getOptimalPoint();
                double objectiveValue = result.getOptimalValue();
                
                System.out.println("Solution found:");
                System.out.printf("x1 = %.6f, x2 = %.6f\n", 
                    solution.get(0).doubleValue(), solution.get(1).doubleValue());
                System.out.printf("Objective value = %.6f\n", objectiveValue);
                
                // Verify constraints
                verifyInequalityConstraints(solution, A_ub, b_ub);
                verifyEqualityConstraints(solution, A_eq, b_eq);
                
                // Expected optimal solution: x1 = 1, x2 = 3, value = 11
                double expectedX1 = 1.0;
                double expectedX2 = 3.0;
                double expectedValue = 11.0;
                
                if (Math.abs(solution.get(0).doubleValue() - expectedX1) < TOLERANCE &&
                    Math.abs(solution.get(1).doubleValue() - expectedX2) < TOLERANCE &&
                    Math.abs(objectiveValue - expectedValue) < TOLERANCE) {
                    System.out.println("✓ Test 3 PASSED");
                } else {
                    System.out.printf("✗ Test 3 FAILED: Expected x1=%.6f, x2=%.6f, value=%.6f\n", 
                        expectedX1, expectedX2, expectedValue);
                }
            } else {
                System.out.println("✗ Test 3 FAILED: Solver did not converge - " + result.getConvergenceReason());
            }
            
        } catch (Exception e) {
            System.out.println("✗ Test 3 FAILED with exception: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println();
    }
    
    /**
     * Test 4: Standard textbook example
     * 
     * Maximize: 40*x1 + 30*x2
     * Subject to:
     *   x1 <= 40
     *   x2 <= 30  
     *   x1 + x2 <= 50
     *   x1, x2 >= 0
     *   
     * Expected optimal solution: x1 = 20, x2 = 30, value = 1700
     */
    private void testStandardExample() {
        System.out.println("Test 4: Standard Textbook Example");
        System.out.println("Maximize: 40*x1 + 30*x2");
        System.out.println("Subject to: x1 <= 40, x2 <= 30, x1 + x2 <= 50, x1,x2 >= 0");
        
        try {
            // Set up the problem
            IVector c = IVector.of(40.0, 30.0);  // objective coefficients
            
            // Inequality constraints: A_ub * x <= b_ub
            IMatrix A_ub = IMatrix.of(new double[][]{
                {1.0, 0.0},  // x1 <= 40
                {0.0, 1.0},  // x2 <= 30
                {1.0, 1.0}   // x1 + x2 <= 50
            });
            IVector b_ub = IVector.of(new double[]{40.0, 30.0, 50.0});
            
            // No equality constraints
            IMatrix A_eq = null;
            IVector b_eq = null;
            
            // Solve
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            solver.setVerbose(true);
            
            OptResult result = solver.maximize(c, A_ub, b_ub, A_eq, b_eq, null);
            
            // Check result
            if (result.isConverged()) {
                IVector solution = result.getOptimalPoint();
                double objectiveValue = result.getOptimalValue();
                
                System.out.println("Solution found:");
                System.out.printf("x1 = %.6f, x2 = %.6f\n", 
                    solution.get(0).doubleValue(), solution.get(1).doubleValue());
                System.out.printf("Objective value = %.6f\n", objectiveValue);
                
                // Verify constraints
                verifyInequalityConstraints(solution, A_ub, b_ub);
                
                // Expected optimal solution: x1 = 20, x2 = 30, value = 1700
                double expectedX1 = 20.0;
                double expectedX2 = 30.0;
                double expectedValue = 1700.0;
                
                if (Math.abs(solution.get(0).doubleValue() - expectedX1) < TOLERANCE &&
                    Math.abs(solution.get(1).doubleValue() - expectedX2) < TOLERANCE &&
                    Math.abs(objectiveValue - expectedValue) < TOLERANCE) {
                    System.out.println("✓ Test 4 PASSED");
                } else {
                    System.out.printf("✗ Test 4 FAILED: Expected x1=%.6f, x2=%.6f, value=%.6f\n", 
                        expectedX1, expectedX2, expectedValue);
                }
            } else {
                System.out.println("✗ Test 4 FAILED: Solver did not converge - " + result.getConvergenceReason());
            }
            
        } catch (Exception e) {
            System.out.println("✗ Test 4 FAILED with exception: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println();
    }
    
    /**
     * Test 5: Large scale problem to test industrial performance
     */
    private void testLargeScaleProblem() {
        System.out.println("Test 5: Large Scale Problem (100 variables, 50 constraints)");
        
        try {
            int numVars = 100;
            int numIneqConstraints = 30;
            int numEqConstraints = 20;
            
            // Create a random but solvable problem
            IVector c = Linalg.ones(numVars);  // Simple objective: maximize sum of variables
            
            // Random inequality constraints
            double[][] A_ub_data = new double[numIneqConstraints][numVars];
            double[] b_ub_data = new double[numIneqConstraints];
            
            for (int i = 0; i < numIneqConstraints; i++) {
                for (int j = 0; j < numVars; j++) {
                    A_ub_data[i][j] = Math.random() * 2.0; // Random coefficients [0, 2]
                }
                b_ub_data[i] = 10.0 + Math.random() * 20.0; // Random RHS [10, 30]
            }
            
            IMatrix A_ub = IMatrix.of(A_ub_data);
            IVector b_ub = IVector.of(b_ub_data);
            
            // Random equality constraints  
            double[][] A_eq_data = new double[numEqConstraints][numVars];
            double[] b_eq_data = new double[numEqConstraints];
            
            for (int i = 0; i < numEqConstraints; i++) {
                for (int j = 0; j < numVars; j++) {
                    A_eq_data[i][j] = Math.random() * 2.0; // Random coefficients [0, 2]
                }
                b_eq_data[i] = 5.0 + Math.random() * 10.0; // Random RHS [5, 15]
            }
            
            IMatrix A_eq = IMatrix.of(A_eq_data);
            IVector b_eq = IVector.of(b_eq_data);
            
            // Solve
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            solver.setVerbose(false); // Reduce output for large problem
            
            long startTime = System.currentTimeMillis();
            OptResult result = solver.maximize(c, A_ub, b_ub, A_eq, b_eq, null);
            long endTime = System.currentTimeMillis();
            
            // Check result
            if (result.isConverged()) {
                IVector solution = result.getOptimalPoint();
                double objectiveValue = result.getOptimalValue();
                
                System.out.println("Large scale problem solved successfully!");
                System.out.printf("Objective value = %.6f\n", objectiveValue);
                System.out.printf("Solve time = %d ms\n", endTime - startTime);
                System.out.printf("Iterations = %d\n", result.getIterations());
                
                // Verify constraints (basic check)
                boolean constraintsValid = true;
                try {
                    verifyInequalityConstraints(solution, A_ub, b_ub);
                    verifyEqualityConstraints(solution, A_eq, b_eq);
                } catch (AssertionError e) {
                    constraintsValid = false;
                    System.out.println("Constraint violation: " + e.getMessage());
                }
                
                if (constraintsValid) {
                    System.out.println("✓ Test 5 PASSED");
                } else {
                    System.out.println("✗ Test 5 FAILED: Constraint violations");
                }
            } else {
                System.out.println("✗ Test 5 FAILED: Solver did not converge - " + result.getConvergenceReason());
            }
            
        } catch (Exception e) {
            System.out.println("✗ Test 5 FAILED with exception: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println();
    }
    
    /**
     * Verify that a solution satisfies inequality constraints A_ub * x <= b_ub
     */
    private void verifyInequalityConstraints(IVector x, IMatrix A_ub, IVector b_ub) {
        if (A_ub == null || b_ub == null) return;
        
        for (int i = 0; i < A_ub.rows(); i++) {
            double lhs = 0.0;
            for (int j = 0; j < A_ub.cols(); j++) {
                lhs += A_ub.get(i, j).doubleValue() * x.get(j).doubleValue();
            }
            double rhs = b_ub.get(i).doubleValue();
            
            if (lhs > rhs + TOLERANCE) {
                throw new AssertionError(String.format(
                    "Inequality constraint %d violated: %.6f > %.6f", i, lhs, rhs));
            }
        }
    }
    
    /**
     * Verify that a solution satisfies equality constraints A_eq * x = b_eq
     */
    private void verifyEqualityConstraints(IVector x, IMatrix A_eq, IVector b_eq) {
        if (A_eq == null || b_eq == null) return;
        
        for (int i = 0; i < A_eq.rows(); i++) {
            double lhs = 0.0;
            for (int j = 0; j < A_eq.cols(); j++) {
                lhs += A_eq.get(i, j).doubleValue() * x.get(j).doubleValue();
            }
            double rhs = b_eq.get(i).doubleValue();
            
            if (Math.abs(lhs - rhs) > TOLERANCE) {
                throw new AssertionError(String.format(
                    "Equality constraint %d violated: %.6f != %.6f", i, lhs, rhs));
            }
        }
    }
}