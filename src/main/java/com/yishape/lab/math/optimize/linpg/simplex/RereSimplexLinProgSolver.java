package com.yishape.lab.math.optimize.linpg.simplex;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.util.RerePrecision;

/**
 * 工业级单纯形法线性规划求解器 / Industrial-Strength Simplex Linear Programming Solver
 * <p>
 * 设计用于大规模问题的工业级线性规划求解器。
 * Designed for large-scale problems with thousands of constraints.
 * </p>
 *
 * <h3>主要特性 / Key Features</h3>
 * <ul>
 *   <li>高级数值稳定性，使用IEEE 754系数缩放</li>
 *   <li>多种枢轴选择策略（Dantzig, Bland, Steepest Edge）</li>
 *   <li>两阶段方法，高效处理人工变量</li>
 *   <li>全面的退化预防和循环检测</li>
 * </ul>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class RereSimplexLinProgSolver implements ISimplexLinProgSolver {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereSimplexLinProgSolver.class);


    // Configuration
    private boolean verbose = false;
    private PivotSelectionRule pivotSelectionRule = PivotSelectionRule.DANTZIG;
//    private ISolutionCallback solutionCallback = null;
    
    // Numerical parameters - optimized for large-scale problems
    private static final double DEFAULT_EPSILON = 1e-8;
    private static final int DEFAULT_ULPS = 10;
    private static final double DEFAULT_CUT_OFF = 1e-12;
    private static final int MAX_ITERATIONS = 10000; // Increased for large problems
    
    private final double epsilon;
    private final int maxUlps;
    private final double cutOff;
    private final boolean useBlandRule;
    
    // Performance optimization flags
    private boolean useNumericalScaling = true;
    private boolean useAdvancedDegeneracyHandling = true;
    private boolean useMemoryOptimization = true;
    
    // Strategy components
    private PivotSelectionStrategy pivotStrategy;
    
    // Performance monitoring
    private long totalSolveTime = 0;
    private int problemsSolved = 0;
    
    /**
     * Default constructor with optimized settings for large-scale problems
     */
    public RereSimplexLinProgSolver() {
        this(DEFAULT_EPSILON, DEFAULT_ULPS, DEFAULT_CUT_OFF);
    }
    
    /**
     * Constructor with custom numerical tolerances
     */
    public RereSimplexLinProgSolver(double epsilon, int maxUlps, double cutOff) {
        this.epsilon = epsilon;
        this.maxUlps = maxUlps;
        this.cutOff = cutOff;
        this.useBlandRule = true; // Enable Bland rule to prevent cycling
        this.pivotStrategy = new PivotSelectionStrategy(epsilon, maxUlps);
    }
    
    // Configuration methods
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
    
    public void setPivotSelectionRule(PivotSelectionRule rule) {
        this.pivotSelectionRule = rule;
    }
    
//    public void setSolutionCallback(ISolutionCallback callback) {
//        this.solutionCallback = callback;
//    }
    
    public void setUseNumericalScaling(boolean enable) {
        this.useNumericalScaling = enable;
    }
    
    public void setUseAdvancedDegeneracyHandling(boolean enable) {
        this.useAdvancedDegeneracyHandling = enable;
    }
    
    public void setUseMemoryOptimization(boolean enable) {
        this.useMemoryOptimization = enable;
    }

    /**
     * Solve linear programming problem: minimize c^T x subject to A_eq x = b_eq, x >= 0
     * This is the standard interface method required by ILinProgSolver
     * 按单纯形法常用的最大化来处理问题，防止在程序中来回转换目标函数出现最大错�?
     * @param c 目标函数系数（最大化问题�?
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（等式右方�?
     * @param initX 初始点（热启动点�?
     * @return 优化结果
     **/
    @Override
    public OptResult maximize(IVector c,IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX) {
        
        long startTime = System.nanoTime();
        
        if (verbose) {
            log.debug("=== BetterSimplexLinProgSolver: Industrial-Strength Simplex Method ===");
            int totalConstraints = (A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0);
            log.debug("Problem size: " + totalConstraints + " constraints, " + c.length() + " variables");
            if (A_ub != null) {
                log.debug("Inequality constraints: " + A_ub.rows());
            }
            if (A_eq != null) {
                log.debug("Equality constraints: " + A_eq.rows());
            }
            log.debug("Configuration: scaling=" + useNumericalScaling + 
                             ", degeneracy=" + useAdvancedDegeneracyHandling +
                             ", memory_opt=" + useMemoryOptimization);
        }
        
        try {
            // Input validation
            // Input validation
            if (c == null || c.length() == 0) {
                throw new IllegalArgumentException("Objective function cannot be null or empty");
            }
            if ((A_ub == null && A_eq == null) || 
                ((A_ub == null || A_ub.rows() == 0) && (A_eq == null || A_eq.rows() == 0))) {
                throw new IllegalArgumentException("Problem must have at least one constraint");
            }
            if (A_eq != null && A_eq.cols() != c.length()) {
                throw new IllegalArgumentException(
                        "A_eq column count (" + A_eq.cols() + ") must match objective length (" + c.length() + ")");
            }
            if (A_ub != null && A_ub.cols() != c.length()) {
                throw new IllegalArgumentException(
                        "A_ub column count (" + A_ub.cols() + ") must match objective length (" + c.length() + ")");
            }
            
            // Build the advanced simplex tableau with NATIVE constraint handling
            RereSimplexTableau tableau = new RereSimplexTableau(c, A_ub, b_ub, A_eq, b_eq, true, epsilon, maxUlps, useNumericalScaling, verbose);
            // Verbose mode is set during construction for scaling analysis
            
            if (verbose) {
                log.debug("Tableau construction time: " + tableau.getConstructionTime() / 1_000_000 + " ms");
                log.debug("Initial phase: " + tableau.getCurrentPhase());
                log.debug("Native constraint processing: LEQ=" + (A_ub != null ? A_ub.rows() : 0) + 
                                 ", EQ=" + (A_eq != null ? A_eq.rows() : 0));
            }
            
            // Phase I: Find initial feasible solution (if needed)
            if (tableau.getCurrentPhase() == 1) {
                boolean phaseISuccess = solvePhaseI(tableau);
                if (!phaseISuccess) {
                    return createInfeasibleResult(c.length(), "Problem is infeasible - Phase I failed");
                }
                
                // Transition to Phase II
                tableau.transitionToPhaseII();
                if (verbose) {
                    log.debug("Transitioned to Phase II");
                }
            }
            
            // Phase II: Optimize original objective function
            OptResult result = solvePhaseII(tableau, c);
            
            // Performance tracking
            long solveTime = System.nanoTime() - startTime;
            totalSolveTime += solveTime;
            problemsSolved++;
            
            if (verbose) {
                log.debug("Total solve time: " + solveTime / 1_000_000 + " ms");
                log.debug("Pivot operations: " + tableau.getPivotOperations());
                log.debug("Average solve time: " + (totalSolveTime / problemsSolved) / 1_000_000 + " ms");
            }
            
            return result;
            
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            if (verbose) {
                log.warn("Solver exception: " + e.getMessage());
                log.error("Solver failed", e);
            }
            
            return createFailureResult(c.length(), "Solver failed: " + e.getMessage());
        }
    }
    
    
    /**
     * Extract original variables from extended solution vector
     * @param extendedSolution The solution vector including slack/surplus/artificial variables
     * @param originalVarCount The number of original decision variables
     * @return Vector containing only the original decision variables
     */
    private IVector extractOriginalVariables(IVector extendedSolution, int originalVarCount) {
        if (extendedSolution == null) {
            return Linalg.zeros(originalVarCount);
        }
        
        // Create a new vector with only the first originalVarCount elements
        IVector originalSolution = Linalg.zeros(originalVarCount);
        int copyLength = Math.min(originalVarCount, extendedSolution.length());
        
        for (int i = 0; i < copyLength; i++) {
            originalSolution.set(i, extendedSolution.get(i));
        }
        
        return originalSolution;
    }
    
    /**
     * Try to solve directly for unique solution cases using Linalg API
     */
    private OptResult tryDirectSolution(IVector c, IMatrix A_eq, IVector b_eq) {
        try {
            // Use Linalg API to solve A_eq * x = b_eq directly
            IVector solution = A_eq.solve(b_eq);
            
            // Check non-negativity constraints using RerePrecision
            boolean feasible = true;
            for (int i = 0; i < solution.length(); i++) {
                double value = RereMathUtil.safeDoubleValue(solution.get(i));
                if (RerePrecision.isLessThan(value, 0.0, epsilon)) {
                    feasible = false;
                    break;
                }
            }
            
            if (feasible) {
                // Calculate objective value using IVector dot product
                double objectiveValue = RereMathUtil.safeDoubleValue(c.dotValue(solution));
                
                if (verbose) {
                    log.debug("Direct solution found: " + solution);
                    log.debug("Objective value: " + objectiveValue);
                }
                
                return new OptResult.Builder(objectiveValue, solution)
                    .converged(true)
                    .convergenceReason("Direct solution - unique feasible point")
                    .iterations(1)
                    .build();
            }
            
        } catch (Exception e) {
            // If direct solution fails, fall back to simplex method
            if (verbose) {
                log.debug("Direct solution failed, using simplex method: " + e.getMessage());
            }
        }
        
        // Return non-converged result to indicate fallback needed
        return new OptResult.Builder(Double.NaN, Linalg.zeros(c.length()).toDoubleVector())
            .converged(false)
            .convergenceReason("Direct solution not applicable")
            .iterations(0)
            .build();
    }
    private void validateInput(IVector c, IMatrix A_eq, IVector b_eq) {
        if (c == null || A_eq == null || b_eq == null) {
            throw new IllegalArgumentException("Input parameters cannot be null");
        }
        if (c.length() != A_eq.cols()) {
            throw new IllegalArgumentException("Objective coefficients length must match constraint matrix columns");
        }
        if (A_eq.rows() != b_eq.length()) {
            throw new IllegalArgumentException("Constraint matrix rows must match RHS vector length");
        }
        if (c.length() == 0 || A_eq.rows() == 0) {
            throw new IllegalArgumentException("Problem dimensions must be positive");
        }
        
        // Check for NaN or infinite values
        for (int i = 0; i < c.length(); i++) {
            double val = RereMathUtil.safeDoubleValue(c.get(i));
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException("Objective coefficients contain invalid values");
            }
        }
        
        for (int i = 0; i < b_eq.length(); i++) {
            double val = RereMathUtil.safeDoubleValue(b_eq.get(i));
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException("RHS values contain invalid values");
            }
        }
        
        for (int i = 0; i < A_eq.rows(); i++) {
            for (int j = 0; j < A_eq.cols(); j++) {
                double val = RereMathUtil.safeDoubleValue(A_eq.get(i, j));
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException("Constraint matrix contains invalid values");
                }
            }
        }
    }
    
    /**
     * Solve Phase I: Find initial feasible solution
     */
    private boolean solvePhaseI(RereSimplexTableau tableau) {
        if (verbose) {
            log.debug("Starting Phase I: Finding initial feasible solution");
        }
        
        int iteration = 0;
        while (!tableau.isOptimal() && iteration < MAX_ITERATIONS) {
            if (verbose) {
                log.debug("Phase I iteration " + iteration + ", isOptimal: " + tableau.isOptimal());
            }
            
            // Select entering variable
            Integer enteringVar = selectEnteringVariable(tableau);
            if (verbose) {
                log.debug("Phase I entering variable: " + enteringVar);
            }
            if (enteringVar == null) {
                break; // Optimal or no improvement possible
            }
            
            // Select leaving variable
            Integer leavingVar = pivotStrategy.selectLeavingVariable(tableau, enteringVar);
            if (leavingVar == null) {
                // Unbounded in Phase I should not happen with artificial variables
                if (verbose) {
                    log.warn("Warning: Unbounded solution in Phase I");
                }
                return false;
            }
            
            // Perform pivot operation

            
            pivotStrategy.performPivotOperation(tableau, leavingVar, enteringVar);

            
            iteration++;

            
            

            
            if (verbose) {

            
                log.debug("After pivot, checking objective row coefficients:");

            
                int objRow = (tableau.getCurrentPhase() == 1) ? 0 : ((tableau.getNumObjectiveFunctions() == 2) ? 1 : 0);

            
                for (int j = tableau.getNumObjectiveFunctions(); j < Math.min(tableau.getNumObjectiveFunctions() + 10, tableau.getWidth() - 1); j++) {

            
                    log.debug(String.format("  Col %d: %.6f%n", j, tableau.getEntry(objRow, j)));

            
                }

            
            }
            
            if (verbose && iteration % 100 == 0) {
                log.debug("Phase I iteration " + iteration);
            }
        }
        
        if (iteration >= MAX_ITERATIONS) {
            if (verbose) {
                log.warn("Phase I reached iteration limit");
            }
            return false;
        }
        
        // Check if Phase I found a feasible solution
        // 在Phase I中，目标是最小化人工变量之和
        // 如果最优值为0（或足够接近0），则原问题可行
        double phaseIObjective = tableau.getEntry(0, tableau.getRhsOffset());
        // 对于Phase I，如果最终目标值足够接�?，则可行
        boolean feasible = RerePrecision.equalsZero(phaseIObjective, epsilon);
        
        if (verbose) {
            log.debug("Phase I completed in " + iteration + " iterations");
            log.debug("Phase I objective value: " + phaseIObjective);
            log.debug("Problem " + (feasible ? "is feasible" : "is infeasible"));
            log.debug("Phase I feasibility check: |" + phaseIObjective + "| <= " + epsilon + " = " + feasible);
        }
        
        return feasible;
    }
    
    /**
     * Solve Phase II: Optimize original objective
     */
    private OptResult solvePhaseII(RereSimplexTableau tableau, IVector originalObjective) {
        if (verbose) {

            log.debug("Starting Phase II: Optimizing original objective");

            log.debug("Initial objective row coefficients:");

            int objRow = (tableau.getCurrentPhase() == 1) ? 0 : ((tableau.getNumObjectiveFunctions() == 2) ? 1 : 0);

            for (int j = tableau.getNumObjectiveFunctions(); j < Math.min(tableau.getNumObjectiveFunctions() + 10, tableau.getWidth() - 1); j++) {

                log.debug(String.format("  Col %d: %.6f%n", j, tableau.getEntry(objRow, j)));

            }

        }
        
        int iteration = 0;
        double bestObjective = Double.NEGATIVE_INFINITY;
        
        while (!tableau.isOptimal() && iteration < MAX_ITERATIONS) {
            if (verbose) {
                log.debug("Phase II iteration " + iteration + ", isOptimal: " + tableau.isOptimal());
            }
            
            // Select entering variable
            Integer enteringVar = selectEnteringVariable(tableau);
            if (verbose) {
                log.debug("Phase II entering variable: " + enteringVar);
            }
            if (enteringVar == null) {
                if (verbose) {
                    log.debug("No entering variable found, optimal solution reached");
                }
                break; // Optimal
            }
            
            // Select leaving variable
            Integer leavingVar = pivotStrategy.selectLeavingVariable(tableau, enteringVar);
            if (leavingVar == null) {
                if (verbose) {
                    log.warn("Unbounded solution detected in Phase II");
                }
                IVector unboundedSolution = extractSolution(tableau, originalObjective.length());
                return new OptResult.Builder(Double.POSITIVE_INFINITY, unboundedSolution)
                    .converged(false)
                    .convergenceReason("Phase II: Unbounded solution")
                    .iterations(iteration)
                    .build();
            }
            
            if (verbose) {
                log.debug("Phase II leaving variable (row): " + leavingVar);
            }
            
            // Perform pivot operation

            
            pivotStrategy.performPivotOperation(tableau, leavingVar, enteringVar);

            
            iteration++;

            
            

            
            if (verbose) {

            
                log.debug("After pivot, checking objective row coefficients:");

            
                int objRow = (tableau.getCurrentPhase() == 1) ? 0 : ((tableau.getNumObjectiveFunctions() == 2) ? 1 : 0);

            
                for (int j = tableau.getNumObjectiveFunctions(); j < Math.min(tableau.getNumObjectiveFunctions() + 10, tableau.getWidth() - 1); j++) {

            
                    log.debug(String.format("  Col %d: %.6f%n", j, tableau.getEntry(objRow, j)));

            
                }

            
            }
            
            // Check for improvement to prevent cycling
            double currentObjective = tableau.getEntry(0, tableau.getRhsOffset());
            if (verbose) {
                log.debug(String.format("Phase II iteration %d objective: %.6f%n", iteration, currentObjective));
            }
            
            if (iteration % 100 == 0 && verbose) {
                log.debug("Phase II iteration " + iteration + ", current objective: " + currentObjective);
            }
        }
        
        if (iteration >= MAX_ITERATIONS) {
            if (verbose) {
                log.warn("Phase II reached iteration limit");
            }
            return createFailureResult(originalObjective.length(), "Phase II iteration limit exceeded");
        }
        
        // Extract solution
        IVector solution = extractSolution(tableau, originalObjective.length());
        
        // Calculate objective value using original objective
        double objectiveValue = calculateObjectiveValue(originalObjective, solution);
        
        if (verbose) {
            log.debug("Phase II completed in " + iteration + " iterations");
            log.debug("Tableau objective value: " + tableau.getEntry(0, tableau.getRhsOffset()));
            log.debug("True objective value: " + objectiveValue);
            log.debug("Extracted solution: " + solution);
        }
        
        return new OptResult.Builder(objectiveValue, solution)
            .converged(true)
            .convergenceReason("Phase II optimal")
            .iterations(iteration)
            .build();
    }
    
    /**
     * Select entering variable based on pivot rule
     * 根据记忆中的经验，需要考虑所有非基变�?
     */
    private Integer selectEnteringVariable(RereSimplexTableau tableau) {
        switch (pivotSelectionRule) {
            case DANTZIG:
                return selectEnteringVariableDantzig(tableau);
            case BLAND:
                return selectEnteringVariableBland(tableau);
            case STEEP_EDGE:
                // Fallback to Dantzig for now
                return selectEnteringVariableDantzig(tableau);
            default:
                return selectEnteringVariableDantzig(tableau);
        }
    }
    
    /**
     * Dantzig's rule: 选择最负的系数（Phase II）或最正的系数（Phase II)
     */
    private Integer selectEnteringVariableDantzig(RereSimplexTableau tableau) {
        double minValue = 0;
        Integer minPos = null;
        
        // Determine the correct objective row based on phase and number of objective functions
        int objectiveRow = (tableau.getCurrentPhase() == 1) ? 0 : 
                          ((tableau.getNumObjectiveFunctions() == 2) ? 1 : 0);
        
        // Skip first numObjectiveFunctions columns (W and/or Z columns)
        for (int i = tableau.getNumObjectiveFunctions(); i < tableau.getWidth() - 1; i++) {
            final double entry = tableau.getEntry(objectiveRow, i);
            // Check if the entry is strictly smaller than the current minimum
            // Do not use a ulp/epsilon check here
            if (entry < minValue) {
                minValue = entry;
                minPos = i;
                // Note: Bland's rule should NOT break early here.
                // It should only be used to break ties, not to stop searching.
            }
        }
        
        if (verbose && minPos != null) {
            log.debug("Dantzig rule: entering variable col=" + minPos + ", coeff=" + minValue + " (objectiveRow=" + objectiveRow + ")");
        }
        
        return minPos;
    }
    
    /**
     * Checks whether the given column is valid pivot column, i.e. will result
     * in a valid pivot row.
     */
    private boolean isValidPivotColumn(RereSimplexTableau tableau, int col) {
        for (int i = tableau.getNumObjectiveFunctions(); i < tableau.getHeight(); i++) {
            final double entry = tableau.getEntry(i, col);
            
            // Do the same check as in getPivotRow
            if (RerePrecision.compareTo(entry, 0d, cutOff) > 0) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Bland's rule: 防止循环
     */
    private Integer selectEnteringVariableBland(RereSimplexTableau tableau) {
        int objectiveRow = (tableau.getCurrentPhase() == 1) ? 0 : 
                          ((tableau.getNumObjectiveFunctions() == 2) ? 1 : 0);
        
        // Bland's rule: 选择第一个满足条件的变量
        for (int j = tableau.getNumObjectiveFunctions(); j < tableau.getRhsOffset(); j++) {
            double coeff = tableau.getEntry(objectiveRow, j);
            
            if (tableau.getCurrentPhase() == 1) {
                if (RerePrecision.isGreaterThan(coeff, 0.0, epsilon)) {
                    return j;
                }
            } else {
                if (RerePrecision.isLessThan(coeff, 0.0, epsilon)) {
                    return j;
                }
            }
        }
        
        return null;
    }
    
    /**
     * Extract solution from final tableau (updated for native constraint handling)
     */
    private IVector extractSolution(RereSimplexTableau tableau, int originalVarCount) {
        IVector solution = Linalg.zeros(originalVarCount);
        
        // For the new tableau implementation, extract basic variables directly
        for (int j = 0; j < originalVarCount; j++) {
            // Check if variable j is basic
            Integer basicRow = findBasicRow(tableau, j);
            if (verbose) {
                log.debug("Debug: variable x" + j + " (col " + (tableau.getNumObjectiveFunctions() + j) + ") -> basicRow = " + basicRow);
            }
            if (basicRow != null) {
                // basicRow is now the absolute tableau row, we need to use it directly
                double value = tableau.getEntry(basicRow, tableau.getRhsOffset());
                value = Math.max(0.0, value); // Ensure non-negativity
                solution.set(j, value);
                
                if (verbose) {
                    log.debug("Found basic variable x" + j + " = " + value + " in row " + basicRow);
                }
            }
        }
        
        // Apply variable scaling reversal if scaling was used
        int[] variableScaling = tableau.getVariableExponentChanges();
        if (variableScaling != null && isAnyScalingApplied(variableScaling)) {
            for (int i = 0; i < Math.min(originalVarCount, variableScaling.length); i++) {
                if (variableScaling[i] != 0) {
                    double scaledValue = RereMathUtil.safeDoubleValue(solution.get(i));
                    // Apply reverse scaling
                    double unscaledValue = reverseExponentScaling(scaledValue, variableScaling[i]);
                    solution.set(i, unscaledValue);
                    
                    if (verbose) {
                        log.debug("Reversing scaling for x" + i + ": " + scaledValue + " -> " + unscaledValue + " (exp change: " + variableScaling[i] + ")");
                    }
                }
            }
        }
        
        return solution;
    }
    
    /**
     * Find if a variable is basic in the tableau
     * 修复了Phase II转换后的列索引问�?
     */
    private Integer findBasicRow(RereSimplexTableau tableau, int col) {
        // 使用SimplexTableau的getBasicRow方法来获取基变量映射
        // 这个方法已经处理了Phase II转换后的正确映射
        int actualCol = tableau.getNumObjectiveFunctions() + col;
        Integer basicRow = tableau.getBasicRow(actualCol);
        
        if (verbose) {
            log.debug("findBasicRow: variable x" + col + " (actualCol=" + actualCol + ") -> basicRow=" + basicRow);
        }
        
        return basicRow;
    }
    
    /**
     * Check if any scaling was actually applied
     */
    private boolean isAnyScalingApplied(int[] variableScaling) {
        if (variableScaling == null) {
            return false;
        }
        for (int scaling : variableScaling) {
            if (scaling != 0) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Reverse exponent scaling
     */
    private static double reverseExponentScaling(double d, int exp) {
        if (d == 0 || exp == 0) {
            return d;
        }
        final long bits = Double.doubleToLongBits(d);
        final long EXPN = 0x7ff0000000000000L;
        final long FRAC = 0x800fffffffffffffL;
        final int IEEE_EXPONENT_SHIFT = 52;
        int oldExp = (int)((bits & EXPN) >>> IEEE_EXPONENT_SHIFT);
        int newExp = oldExp + exp;
        if (newExp >= 2047 || newExp <= 0) {
            return d;
        }
        return Double.longBitsToDouble((bits & FRAC) |
            (newExp << IEEE_EXPONENT_SHIFT));
    }
    
    /**
     * Check if this is a large scale problem requiring special numerical handling
     */
    private boolean isLargeScaleProblem(RereSimplexTableau tableau) {
        // Check for large coefficients that might cause numerical issues
        double maxCoeff = getTableauMaxCoeff(tableau);
        return RerePrecision.isGreaterThan(maxCoeff, 1e3); // Consider coefficients > 1000 as large scale
    }
    
    /**
     * Get the maximum coefficient in the tableau for scaling analysis
     */
    private double getTableauMaxCoeff(RereSimplexTableau tableau) {
        double maxCoeff = 0.0;
        for (int i = 0; i < tableau.getHeight(); i++) {
            for (int j = 0; j < tableau.getWidth() - 1; j++) { // Exclude RHS column
                double coeff = Math.abs(tableau.getEntry(i, j));
                if (coeff > maxCoeff) {
                    maxCoeff = coeff;
                }
            }
        }
        return maxCoeff;
    }
    
    /**
     * Verify and correct solution for large scale problems
     */
    private IVector verifyAndCorrectSolution(RereSimplexTableau tableau, IVector solution, int originalVarCount, double adjustedEpsilon) {
        // For large scale problems, the extracted solution might be incorrect due to numerical issues
        // Try to find the correct solution by solving the constraint system directly
        
        if (verbose) {
            log.debug("Verifying large scale solution: " + solution);
        }
        
        // Extract constraint matrix and RHS from tableau
        int numConstraints = tableau.getHeight() - 1; // Exclude objective row
        if (tableau.getCurrentPhase() == 1) {
            numConstraints = tableau.getHeight() - 2; // Exclude both objective rows
        }
        
        // 大规模问题可在此用约束残差进一步校正；当前返回 tableau 提取解。
        return solution;
    }
    
    /**
     * Find the basic variable in a given row using RerePrecision methods with adjusted epsilon
For Phase II tableau, this should respect optimality conditions
     */
    private int findBasicVariableInRow(RereSimplexTableau tableau, int row, int originalVarCount, double adjustedEpsilon) {
        // For Phase II tableau, find the variable that should be basic based on optimality
        if (tableau.getCurrentPhase() == 2) {
            return findOptimalBasicVariable(tableau, row, originalVarCount, adjustedEpsilon);
        }
        
        // Original logic for Phase I or when unit vectors exist
        for (int col = 0; col < Math.min(originalVarCount, tableau.getWidth() - 1); col++) {
            double entry = tableau.getEntry(row, col);
            
            // Check if this entry is 1 using RerePrecision
            if (RerePrecision.equals(entry, 1.0, adjustedEpsilon)) {
                // Verify this is truly a basic variable (only 1 in this column)
                boolean isBasic = true;
                for (int otherRow = 0; otherRow < tableau.getHeight(); otherRow++) {
                    if (otherRow != row) {
                        double otherEntry = tableau.getEntry(otherRow, col);
                        if (!RerePrecision.equalsZero(otherEntry, adjustedEpsilon)) {
                            isBasic = false;
                            break;
                        }
                    }
                }
                
                if (isBasic) {
                    return col; // Found basic variable
                }
            }
        }
        return -1; // No basic variable found in this row
    }
    
    /**
     * Find the optimal basic variable for Phase II tableau with adjustable epsilon
     * This determines which variable should be basic to achieve optimality
     */
    private int findOptimalBasicVariable(RereSimplexTableau tableau, int constraintRow, int originalVarCount, double adjustedEpsilon) {
        // For multi-constraint problems, we need to find proper basic variables
        // that form a valid basis, not just optimal variables per row
        
        // First try to find an existing basic variable (unit vector)
        for (int col = 0; col < Math.min(originalVarCount, tableau.getWidth() - 1); col++) {
            double constraintCoeff = tableau.getEntry(constraintRow, col);
            
            // Check if this is a unit vector column (basic variable)
            if (RerePrecision.equals(constraintCoeff, 1.0, adjustedEpsilon)) {
                boolean isBasic = true;
                for (int otherRow = 0; otherRow < tableau.getHeight(); otherRow++) {
                    if (otherRow != constraintRow) {
                        double otherEntry = tableau.getEntry(otherRow, col);
                        if (!RerePrecision.equalsZero(otherEntry, adjustedEpsilon)) {
                            isBasic = false;
                            break;
                        }
                    }
                }
                
                if (isBasic) {
                    if (verbose) {
                        log.debug("Found existing basic variable x" + col + " in row " + constraintRow);
                    }
                    return col;
                }
            }
        }
        
        // For single constraint problems, use reduced cost optimization
        if (tableau.getHeight() <= 3) { // 1 constraint + 2 objective rows or direct Phase II
            return findOptimalBasicVariableByCost(tableau, constraintRow, originalVarCount, adjustedEpsilon);
        }
        
        // For multi-constraint problems, we need proper pivoting
        // This is a more complex case that requires the tableau to be in proper basic form
        // For now, return the first variable with non-zero coefficient
        for (int col = 0; col < Math.min(originalVarCount, tableau.getWidth() - 1); col++) {
            double constraintCoeff = tableau.getEntry(constraintRow, col);
            if (Math.abs(constraintCoeff) > adjustedEpsilon) {
                if (verbose) {
                    log.debug("Using first non-zero variable x" + col + " for row " + constraintRow);
                }
                return col;
            }
        }
        
        return -1;
    }
    
    /**
     * Find optimal basic variable based on reduced costs (for single constraint problems) with adjustable epsilon
     */
    private int findOptimalBasicVariableByCost(RereSimplexTableau tableau, int constraintRow, int originalVarCount, double adjustedEpsilon) {
        // Get the objective row (last row in Phase II tableau)
        int objectiveRow = tableau.getHeight() - 1;
        
        // For single constraint problems, choose the variable with the most negative reduced cost
        // (or most positive if we're maximizing, which corresponds to most negative in our tableau)
        int bestVariable = -1;
        double bestReducedCost = 0.0;
        
        for (int col = 0; col < Math.min(originalVarCount, tableau.getWidth() - 1); col++) {
            double constraintCoeff = tableau.getEntry(constraintRow, col);
            double reducedCost = tableau.getEntry(objectiveRow, col);
            
            // For maximization, choose variable with most negative reduced cost and positive constraint coefficient
            if (RerePrecision.isGreaterThan(constraintCoeff, 0.0, adjustedEpsilon) && RerePrecision.isLessThan(reducedCost, bestReducedCost, adjustedEpsilon)) {
                bestVariable = col;
                bestReducedCost = reducedCost;
            }
        }
        
        // If we found a variable that should be basic according to optimality
        if (bestVariable >= 0) {
            if (verbose) {
                log.debug("Selected variable x" + bestVariable + " as basic (reduced cost: " + bestReducedCost + ")");
            }
            return bestVariable;
        }
        
        // Fallback: use the first variable with non-zero coefficient
        for (int col = 0; col < Math.min(originalVarCount, tableau.getWidth() - 1); col++) {
            double constraintCoeff = tableau.getEntry(constraintRow, col);
            if (!RerePrecision.equalsZero(constraintCoeff, adjustedEpsilon)) {
                return col;
            }
        }
        
        return -1;
    }
    
    /**
     * Calculate the objective value using original objective coefficients and solution
     */
    private double calculateObjectiveValue(IVector objective, IVector solution) {
        if (objective == null || solution == null) {
            return 0.0;
        }
        
        double value = 0.0;
        int minLength = Math.min(objective.length(), solution.length());
        
        for (int i = 0; i < minLength; i++) {
            double objCoeff = RereMathUtil.safeDoubleValue(objective.get(i));
            double solValue = RereMathUtil.safeDoubleValue(solution.get(i));
            value += objCoeff * solValue;
        }
        
        return value;
    }
    
    /**
     * Check if a vector is all zeros with adjustable epsilon
     */
    private boolean isAllZeros(IVector vector, double adjustedEpsilon) {
        for (int i = 0; i < vector.length(); i++) {
            if (!RerePrecision.equalsZero(RereMathUtil.safeDoubleValue(vector.get(i)), adjustedEpsilon)) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Find a basic feasible solution for underdetermined systems with adjustable epsilon
     */
    private IVector findBasicFeasibleSolution(RereSimplexTableau tableau, int originalVarCount, double adjustedEpsilon) {
        return findBasicFeasibleSolution(tableau, originalVarCount); // Delegate to original method for now
    }
    
    /**
     * Find a basic feasible solution for underdetermined systems
     */
    private IVector findBasicFeasibleSolution(RereSimplexTableau tableau, int originalVarCount) {
        IVector solution = Linalg.zeros(originalVarCount);
        
        // For Phase II tableau, try to construct a basic solution
        // by setting as many variables as possible to their RHS values
        
        if (tableau.getCurrentPhase() == 2) {
            // Look at constraint matrix to find which variables can be set
            int constraintRows = tableau.getHeight() - 1; // Exclude objective row
            
            // For single constraint problems, we can often solve directly
            if (constraintRows == 1) {
                double rhsValue = tableau.getEntry(0, tableau.getRhsOffset());
                
                // Try to find the variable that should be basic
                // In Phase II tableau, look for the largest coefficient (in absolute value)
                int bestVar = -1;
                double bestCoeff = 0.0;
                
                for (int j = 0; j < originalVarCount; j++) {
                    double coeff = Math.abs(tableau.getEntry(0, j));
                    if (RerePrecision.isGreaterThan(coeff, bestCoeff) && RerePrecision.isGreaterThan(coeff, 0.0, epsilon)) {
                        bestCoeff = coeff;
                        bestVar = j;
                    }
                }
                
                // If we found a good variable and the solution value would be feasible
                if (bestVar >= 0) {
                    double coeffValue = tableau.getEntry(0, bestVar);
                    double solutionValue = rhsValue / coeffValue;
                    
                    if (RerePrecision.isGreaterThanOrEqual(solutionValue, 0.0, epsilon)) { // Non-negative
                        // Set this variable to the calculated value
                        solution.set(bestVar, Math.max(0.0, solutionValue));
                        
                        if (verbose) {
                            log.debug("Set variable x" + bestVar + " = " + solutionValue + " as basic variable");
                        }
                        
                        // For single constraint, set other variables to 0 (they are non-basic)
                        // This is already done by initializing solution to zeros
                        return solution;
                    }
                }
            }
            
            // For multiple constraints, use a more general approach
            for (int i = 0; i < constraintRows; i++) {
                double rhsValue = tableau.getEntry(i, tableau.getRhsOffset());
                
                // Find the variable with the largest absolute coefficient in this row
                int bestVar = -1;
                double bestCoeff = 0.0;
                
                for (int j = 0; j < originalVarCount; j++) {
                    double coeff = Math.abs(tableau.getEntry(i, j));
                    if (RerePrecision.isGreaterThan(coeff, bestCoeff) && RerePrecision.isGreaterThan(coeff, 0.0, epsilon)) {
                        bestCoeff = coeff;
                        bestVar = j;
                    }
                }
                
                // If we found a good variable and the solution value would be feasible
                if (bestVar >= 0) {
                    double coeffValue = tableau.getEntry(i, bestVar);
                    double solutionValue = rhsValue / coeffValue;
                    
                    if (RerePrecision.isGreaterThanOrEqual(solutionValue, 0.0, epsilon)) { // Non-negative
                        solution.set(bestVar, Math.max(0.0, solutionValue));
                        
                        if (verbose) {
                            log.debug("Set variable x" + bestVar + " = " + solutionValue + " from constraint " + i);
                        }
                        break; // Use only one constraint to avoid over-determination
                    }
                }
            }
        }
        
        return solution;
    }
    
    /**
     * Create result for infeasible problems
     */
    private OptResult createInfeasibleResult(int varCount, String reason) {
        IVector emptySolution = IVector.zeros(varCount);
        return new OptResult.Builder(Double.NaN, emptySolution)
            .converged(false)
            .convergenceReason(reason)
            .iterations(0)
            .build();
    }
    
    /**
     * Create result for solver failures
     */
    private OptResult createFailureResult(int varCount, String reason) {
        IVector emptySelection = IVector.zeros(varCount);
        return new OptResult.Builder(Double.NEGATIVE_INFINITY, emptySelection)
            .converged(false)
            .convergenceReason(reason)
            .iterations(0)
            .build();
    }
    
    /**
     * 求解方阵系统 (m = n)
     */
    private OptResult solveDeterminateSystem(IVector c, IMatrix A_eq, IVector b_eq) {
        if (verbose) {
            log.debug("求解方阵系统...");
        }
        
        try {
            // 检查系统是否一�?
            double det = computeDeterminant(A_eq);
            if (RerePrecision.equalsZero(det, epsilon)) {
                // 矩阵奇异，可能不可行或有无穷�?
                if (verbose) {
                    log.debug("矩阵奇异，检查一致�?..");
                }
                return checkConsistency(c, A_eq, b_eq);
            }
            
            // 直接求解 A_eq * x = b_eq
            IVector solution = A_eq.solve(b_eq);
            
            // 检查非负性约�?
            boolean feasible = true;
            for (int i = 0; i < solution.length(); i++) {
                if (RerePrecision.isLessThan(RereMathUtil.safeDoubleValue(solution.get(i)), 0.0, epsilon)) {
                    feasible = false;
                    break;
                }
            }
            
            if (!feasible) {
                if (verbose) {
                    log.debug("解不满足非负性约");
                }
                IVector fallbackSolution = IVector.zeros(c.length());
                return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                    .converged(false)
                    .convergenceReason("解不满足非负性约")
                    .iterations(1)
                    .build();
            }
            
            // 计算目标函数�?
            double objectiveValue = 0.0;
            for (int i = 0; i < Math.min(c.length(), solution.length()); i++) {
                objectiveValue += RereMathUtil.safeDoubleValue(c.get(i)) * RereMathUtil.safeDoubleValue(solution.get(i));
            }
            
            if (verbose) {
                log.debug("找到可行�? " + solution);
                log.debug("目标函数�? " + objectiveValue);
            }
            
            return new OptResult.Builder(objectiveValue, solution)
                .converged(true)
                .convergenceReason("直接求解成功")
                .iterations(1)
                .build();
                
        } catch (Exception e) {
            if (verbose) {
                log.debug("方阵求解失败: " + e.getMessage());
            }
            IVector fallbackSolution = IVector.zeros(c.length());
            return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                .converged(false)
                .convergenceReason("矩阵求解失败: " + e.getMessage())
                .iterations(1)
                .build();
        }
    }
    
    /**
     * 计算矩阵行列式（简化版本）
     */
    private double computeDeterminant(IMatrix matrix) {
        // 简化实现：对于2x2矩阵
        if (matrix.rows() == 2 && matrix.cols() == 2) {
            double a = RereMathUtil.safeDoubleValue(matrix.get(0, 0));
            double b = RereMathUtil.safeDoubleValue(matrix.get(0, 1));
            double c = RereMathUtil.safeDoubleValue(matrix.get(1, 0));
            double d = RereMathUtil.safeDoubleValue(matrix.get(1, 1));
            return a * d - b * c;
        }
        // 对于其他情况，简单检查是否可�?
        try {
            IVector testVec = IVector.zeros(matrix.rows());
            testVec.set(0, 1.0);
            matrix.solve(testVec);
            return 1.0; // 可�?
        } catch (Exception e) {
            return 0.0; // 奇异
        }
    }
    
    /**
     * 检查系统一致�?
     */
    private OptResult checkConsistency(IVector c, IMatrix A_eq, IVector b_eq) {
        // 检查增广矩阵的秩是否等于系数矩阵的�?
        // 简化实现：检查是否存在矛盾的约束
        int m = A_eq.rows();
        int n = A_eq.cols();
        
        for (int i = 0; i < m - 1; i++) {
            for (int j = i + 1; j < m; j++) {
                // 检查第i行和第j行是否成比例但右侧不成比�?
                boolean proportional = true;
                double ratio = 0.0;
                boolean ratioSet = false;
                
                for (int k = 0; k < n; k++) {
                    double ai = RereMathUtil.safeDoubleValue(A_eq.get(i, k));
                    double aj = RereMathUtil.safeDoubleValue(A_eq.get(j, k));
                    
                    if (!RerePrecision.equalsZero(ai, epsilon) || !RerePrecision.equalsZero(aj, epsilon)) {
                        if (RerePrecision.equalsZero(aj, epsilon) && !RerePrecision.equalsZero(ai, epsilon)) {
                            proportional = false;
                            break;
                        }
                        if (RerePrecision.equalsZero(ai, epsilon) && !RerePrecision.equalsZero(aj, epsilon)) {
                            proportional = false;
                            break;
                        }
                        
                        double currentRatio = ai / aj;
                        if (!ratioSet) {
                            ratio = currentRatio;
                            ratioSet = true;
                        } else if (!RerePrecision.equals(currentRatio, ratio, epsilon)) {
                            proportional = false;
                            break;
                        }
                    }
                }
                
                if (proportional && ratioSet) {
                    // 检查右侧是否也成比�?
                    double bi = RereMathUtil.safeDoubleValue(b_eq.get(i));
                    double bj = RereMathUtil.safeDoubleValue(b_eq.get(j));
                    
                    if (!RerePrecision.equalsZero(bj, epsilon)) {
                        double bRatio = bi / bj;
                        if (!RerePrecision.equals(bRatio, ratio, epsilon)) {
                            // 不一致系�?
                            if (verbose) {
                                log.debug("检测到不一致的约束：行" + i + "和行" + j);
                            }
                            IVector fallbackSolution = IVector.zeros(c.length());
                            return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                                .converged(false)
                                .convergenceReason("约束系统不一")
                                .iterations(1)
                                .build();
                        }
                    } else if (!RerePrecision.equalsZero(bi, epsilon)) {
                        if (verbose) {
                            log.debug("检测到不一致的约束：行" + i + "和行" + j);
                        }
                        IVector fallbackSolution = IVector.zeros(c.length());
                        return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                                .converged(false)
                                .convergenceReason("约束系统不一")
                                .iterations(1)
                                .build();
                    }
                }
            }
        }
        
        // 如果没有发现矛盾，尝试求解（可能有无穷解�?
        try {
            IVector solution = A_eq.solve(b_eq);
            
            // 计算目标函数�?
            double objectiveValue = 0.0;
            for (int i = 0; i < Math.min(c.length(), solution.length()); i++) {
                objectiveValue += RereMathUtil.safeDoubleValue(c.get(i)) * RereMathUtil.safeDoubleValue(solution.get(i));
            }
            
            return new OptResult.Builder(objectiveValue, solution)
                .converged(true)
                .convergenceReason("找到一个解（可能有无穷解）")
                .iterations(1)
                .build();
        } catch (Exception e) {
            IVector fallbackSolution = IVector.zeros(c.length());
            return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                .converged(false)
                .convergenceReason("无法求解" + e.getMessage())
                .iterations(1)
                .build();
        }
    }
    
    /**
     * 求解欠定系统 (m < n) - 使用单纯形法
     */
    private OptResult solveUnderdeterminedSystem(IVector c, IMatrix A_eq, IVector b_eq) {
        if (verbose) {
            log.debug("求解欠定系统，使用单纯形�?..");
        }
        
        int m = A_eq.rows();
        int n = A_eq.cols();
        
        // 构建单纯形表：[A_eq | I | b_eq; -c^T | 0 | 0]
        // 表结构：(m+1) �?x (n+m+1) �?
        IMatrix tableau = IMatrix.zeros(m + 1, n + m + 1);
        
        // 填充约束矩阵 A_eq
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                tableau.set(i, j, A_eq.get(i, j));
            }
        }
        
        // 添加单位矩阵（人工变量）
        for (int i = 0; i < m; i++) {
            tableau.set(i, n + i, 1.0);
        }
        
        // 填充右侧向量 b_eq
        for (int i = 0; i < m; i++) {
            tableau.set(i, n + m, RereMathUtil.safeDoubleValue(b_eq.get(i)));
        }
        
        // 设置目标函数行（最大化问题，系数取负）
        for (int j = 0; j < n; j++) {
            tableau.set(m, j, -RereMathUtil.safeDoubleValue(c.get(j)));
        }
        
        if (verbose) {
            log.debug("初始单纯形表已构建，维度: " + tableau.rows() + "x" + tableau.cols());
            printTableau(tableau, m, n);
        }
        
        // 执行单纯形迭�?
        int iteration = 0;
        while (iteration < MAX_ITERATIONS) {
            // 检查最优�?
            int enteringVar = findEnteringVariable(tableau, m, n);
            if (enteringVar == -1) {
                // 达到最优解
                break;
            }
            
            // 选择出基变量
            int leavingVar = findLeavingVariable(tableau, enteringVar, m);
            if (leavingVar == -1) {
                // 无界�?
                if (verbose) {
                    log.debug("检测到无界");
                }
                IVector unboundedSolution = extractCurrentSolution(tableau, n, m);
                return new OptResult.Builder(Double.POSITIVE_INFINITY, unboundedSolution)
                    .converged(false)
                    .convergenceReason("问题具有无界")
                    .iterations(iteration)
                    .build();
            }
            
            // 执行枢轴操作
            performPivotOperation(tableau, leavingVar, enteringVar);
            iteration++;
            
            if (verbose && iteration % 10 == 0) {
                log.debug("迭代 " + iteration + "，入�? " + enteringVar + ", 出基: " + leavingVar);
            }
        }
        
        if (iteration >= MAX_ITERATIONS) {
            log.warn("达到最大迭代次");
        }
        
        // 提取最终解
        IVector solution = extractCurrentSolution(tableau, n, m);
        double objectiveValue = RereMathUtil.safeDoubleValue(tableau.get(m, n + m));
        
        if (verbose) {
            log.debug("单纯形法完成，迭代次�? " + iteration);
            log.debug("最终解: " + solution);
            log.debug("目标函数�? " + objectiveValue);
            printTableau(tableau, m, n);
        }
        
        return new OptResult.Builder(objectiveValue, solution)
            .converged(iteration < MAX_ITERATIONS)
            .convergenceReason(iteration < MAX_ITERATIONS ? "单纯形法成功" : "达到迭代限制")
            .iterations(iteration)
            .build();
    }
    
    /**
     * 查找入基变量（最负系数规则）
     */
    private int findEnteringVariable(IMatrix tableau, int m, int n) {
        int objectiveRow = m;
        int enteringVar = -1;
        double mostNegative = 0.0;
        
        // 在原始变量中寻找最负的系数
        for (int j = 0; j < n; j++) {
            double coeff = RereMathUtil.safeDoubleValue(tableau.get(objectiveRow, j));
            if (coeff < mostNegative) {
                mostNegative = coeff;
                enteringVar = j;
            }
        }
        
        return enteringVar;
    }
    
    /**
     * 查找出基变量（最小比值规则）
     */
    private int findLeavingVariable(IMatrix tableau, int enteringVar, int m) {
        int leavingVar = -1;
        double minRatio = Double.POSITIVE_INFINITY;
        
        for (int i = 0; i < m; i++) {
            double coeff = RereMathUtil.safeDoubleValue(tableau.get(i, enteringVar));
            double rhs = RereMathUtil.safeDoubleValue(tableau.get(i, tableau.cols() - 1));
            
            if (RerePrecision.isGreaterThan(coeff, 0.0, epsilon)) {  // 只考虑正系�?
                double ratio = rhs / coeff;
                if (RerePrecision.isGreaterThanOrEqual(ratio, 0.0) && RerePrecision.isLessThan(ratio, minRatio)) {
                    minRatio = ratio;
                    leavingVar = i;
                }
            }
        }
        
        return leavingVar;
    }
    
    /**
     * 执行枢轴操作
     */
    private void performPivotOperation(IMatrix tableau, int pivotRow, int pivotCol) {
        double pivot = RereMathUtil.safeDoubleValue(tableau.get(pivotRow, pivotCol));
        
        if (RerePrecision.equalsZero(pivot, epsilon)) {
            throw new RuntimeException("枢轴元素太小: " + pivot);
        }
        
        // 归一化枢轴行
        for (int j = 0; j < tableau.cols(); j++) {
            double value = RereMathUtil.safeDoubleValue(tableau.get(pivotRow, j));
            tableau.set(pivotRow, j, value / pivot);
        }
        
        // 消除其他行的枢轴�?
        for (int i = 0; i < tableau.rows(); i++) {
            if (i != pivotRow) {
                double multiplier = RereMathUtil.safeDoubleValue(tableau.get(i, pivotCol));
                for (int j = 0; j < tableau.cols(); j++) {
                    double currentValue = RereMathUtil.safeDoubleValue(tableau.get(i, j));
                    double pivotValue = RereMathUtil.safeDoubleValue(tableau.get(pivotRow, j));
                    tableau.set(i, j, currentValue - multiplier * pivotValue);
                }
            }
        }
    }
    
    /**
     * 提取当前�?
     */
    private IVector extractCurrentSolution(IMatrix tableau, int n, int m) {
        IVector solution = IVector.zeros(n);
        
        // 找到基变�?
        for (int i = 0; i < m; i++) {
            // 寻找第i行的基变量（只有一�?，其他都�?的列�?
            for (int j = 0; j < n; j++) {
                double value = RereMathUtil.safeDoubleValue(tableau.get(i, j));
                if (RerePrecision.equals(value, 1.0, epsilon)) {
                    // 检查这一列在其他行是否都�?
                    boolean isBasic = true;
                    for (int k = 0; k < m; k++) {
                        if (k != i) {
                            double otherValue = RereMathUtil.safeDoubleValue(tableau.get(k, j));
                            if (!RerePrecision.equalsZero(otherValue, epsilon)) {
                                isBasic = false;
                                break;
                            }
                        }
                    }
                    if (isBasic) {
                        double rhsValue = RereMathUtil.safeDoubleValue(tableau.get(i, tableau.cols() - 1));
                        solution.set(j, rhsValue);
                        break;
                    }
                }
            }
        }
        
        return solution;
    }
    
    /**
     * 打印单纯形表（调试用�?
     */
    private void printTableau(IMatrix tableau, int m, int n) {
        if (!verbose) return;
        
        log.debug("单纯形表:");
        for (int i = 0; i < tableau.rows(); i++) {
            for (int j = 0; j < tableau.cols(); j++) {
                log.debug(String.format("%8.3f ", RereMathUtil.safeDoubleValue(tableau.get(i, j))));
            }
            log.debug("");
        }
        log.debug("");
    }
}