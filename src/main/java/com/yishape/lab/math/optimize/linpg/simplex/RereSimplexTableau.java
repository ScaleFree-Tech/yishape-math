package com.yishape.lab.math.optimize.linpg.simplex;

import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.util.RerePrecision;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * Advanced Simplex Tableau implementation for industrial-strength linear programming
   RereSimplexTableau with native constraint type handling
 */
class RereSimplexTableau {
    
    // IEEE 754 bit manipulation constants for numerical scaling (like commons-math4)
    private static final long EXPN = 0x7ff0000000000000L;
    private static final long FRAC = 0x800fffffffffffffL;
    private static final int MAX_IEEE_EXP = 2047;
    private static final int MIN_IEEE_EXP = 0;
    private static final int OFFSET_IEEE_EXP = 1023;
    private static final int IEEE_EXPONENT_SHIFT = 52;
    
    /** Linear objective function */
    private final IVector f;
    
    /** Linear constraints */
    private final List<LinearConstraint> constraints;
    
    /** Whether to restrict the variables to non-negative values */
    private final boolean restrictToNonNegative;
    
    /** Number of decision variables */
    private final int numDecisionVariables;
    
    /** Number of slack variables */
    private final int numSlackVariables;
    
    /** Number of artificial variables */
    private int numArtificialVariables;
    
    /** Amount of error to accept when checking for optimality */
    private final double epsilon;
    
    /** Amount of error to accept in floating point comparisons */
    private final int maxUlps;
    
    /** Simple tableau */
    private IMatrix tableau;
    
    /** Maps basic variables to row they are basic in */
    private int[] basicVariables;
    
    /** Maps rows to their corresponding basic variables */
    private int[] basicRows;
    
    /** Changes in floating point exponent to scale the input */
    private int[] variableExpChange;
    
    /** Column labels for debugging */
    private final List<String> columnLabels = new ArrayList<>();
    
    /** Phase indicator: 1 or 2 */
    private int currentPhase = 1;
    
    /** Whether to use numerical scaling */
    private final boolean useNumericalScaling;
    
    /** Verbose debugging */
    private boolean verbose = false;
    
    /** Performance monitoring */
    private long constructionTime;
    private int pivotOperations = 0;

    /**
     * Main constructor
     * 
     * @param f Linear objective function
     * @param constraints Linear constraints
     * @param maximize true to maximize, false to minimize
     * @param restrictToNonNegative Whether to restrict the variables to non-negative values
     * @param epsilon Amount of error to accept when checking for optimality
     * @param maxUlps Amount of error to accept in floating point comparisons
     */
    RereSimplexTableau(final IVector f,
                   final Collection<LinearConstraint> constraints,
                   final boolean maximize,
                   final boolean restrictToNonNegative,
                   final double epsilon,
                   final int maxUlps) {
        this(f, constraints, maximize, restrictToNonNegative, epsilon, maxUlps, false, true);
    }
    
    /**
     * Extended constructor with verbose and scaling options
     */
    RereSimplexTableau(final IVector f,
                   final Collection<LinearConstraint> constraints,
                   final boolean maximize,
                   final boolean restrictToNonNegative,
                   final double epsilon,
                   final int maxUlps,
                   final boolean verbose,
                   final boolean useNumericalScaling) {
        long startTime = System.nanoTime();
        
        this.verbose = verbose;
        this.useNumericalScaling = useNumericalScaling;
        
        checkDimensions(f, constraints);
        this.f = f;
        this.constraints = normalizeConstraints(constraints);
        this.restrictToNonNegative = restrictToNonNegative;
        this.epsilon = epsilon;
        this.maxUlps = maxUlps;
        this.numDecisionVariables = f.length() + (restrictToNonNegative ? 0 : 1);
        this.numSlackVariables = getConstraintTypeCounts(ConstraintType.LEQ) +
                                 getConstraintTypeCounts(ConstraintType.GEQ);
        this.numArtificialVariables = getConstraintTypeCounts(ConstraintType.EQ) +
                                     getConstraintTypeCounts(ConstraintType.GEQ);
        
        this.tableau = createTableau(maximize);
        
        // 根据是否有人工变量来确定初始阶段
        // 只有LEQ约束时，松弛变量自然形成基本解，不需要Phase I
        this.currentPhase = (this.numArtificialVariables > 0) ? 1 : 2;
        
        if (this.currentPhase == 2) {
            // 纯不等式约束，直接进入Phase II，松弛变量作为初始基变量
            System.out.println("Pure inequality constraints detected, starting directly in Phase II");
        }
        // initialize the basic variables for phase 1:
        //   we know that only slack or artificial variables can be basic
        initializeBasicVariables(getSlackVariableOffset());
        initializeColumnLabels();
        
        this.constructionTime = System.nanoTime() - startTime;
    }
    
    /**
     * Convenience constructor for mixed constraints (our main interface)
     */
    public RereSimplexTableau(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
                         boolean restrictToNonNegative, double epsilon, int maxUlps) {
        this(c, createConstraints(A_ub, b_ub, A_eq, b_eq), true, restrictToNonNegative, epsilon, maxUlps, false, true);
    }
    
    /**
     * Enhanced constructor with scaling control and verbose output
     */
    public RereSimplexTableau(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
                         boolean restrictToNonNegative, double epsilon, int maxUlps, boolean useScaling, boolean verbose) {
        this(c, createConstraints(A_ub, b_ub, A_eq, b_eq), true, restrictToNonNegative, epsilon, maxUlps, verbose, useScaling);
    }
    
    /**
     * Enhanced constructor with scaling control
     */
    public RereSimplexTableau(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
                         boolean restrictToNonNegative, double epsilon, int maxUlps, boolean useScaling) {
        this(c, createConstraints(A_ub, b_ub, A_eq, b_eq), true, restrictToNonNegative, epsilon, maxUlps, false, useScaling);
    }
    
    /**
     * Legacy constructor for backward compatibility
     */
    public RereSimplexTableau(IVector c, IMatrix A_eq, IVector b_eq, 
                         boolean restrictToNonNegative, double epsilon, int maxUlps) {
        this(c, null, null, A_eq, b_eq, restrictToNonNegative, epsilon, maxUlps);
    }
    
    /**
     * Convert our matrix/vector format to LinearConstraint collection
     */
    private static Collection<LinearConstraint> createConstraints(IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq) {
        List<LinearConstraint> constraintList = new ArrayList<>();
        
        // Add inequality constraints
        if (A_ub != null && b_ub != null) {
            for (int i = 0; i < A_ub.rows(); i++) {
                IVector row = extractRow(A_ub, i);
                double rhs = RereMathUtil.safeDoubleValue(b_ub.get(i));
                constraintList.add(new LinearConstraint(row, ConstraintType.LEQ, rhs));
            }
        }
        
        // Add equality constraints
        if (A_eq != null && b_eq != null) {
            for (int i = 0; i < A_eq.rows(); i++) {
                IVector row = extractRow(A_eq, i);
                double rhs = RereMathUtil.safeDoubleValue(b_eq.get(i));
                constraintList.add(new LinearConstraint(row, ConstraintType.EQ, rhs));
            }
        }
        
        return constraintList;
    }
    
    /**
     * Extract a row from matrix as vector
     */
    private static IVector extractRow(IMatrix matrix, int row) {
        double[] rowData = new double[matrix.cols()];
        for (int j = 0; j < matrix.cols(); j++) {
            rowData[j] = RereMathUtil.safeDoubleValue(matrix.get(row, j));
        }
        return IVector.of(rowData);
    }

    /**
     * Checks that the dimensions of the objective function and the constraints match.
     * @param objectiveFunction the objective function
     * @param c the set of constraints
     */
    private void checkDimensions(final IVector objectiveFunction,
                                 final Collection<LinearConstraint> c) {
        final int dimension = objectiveFunction.length();
        for (final LinearConstraint constraint : c) {
            final int constraintDimension = constraint.getCoefficients().length();
            if (constraintDimension != dimension) {
                throw new IllegalArgumentException("Constraint dimension " + constraintDimension + 
                    " does not match objective function dimension " + dimension);
            }
        }
    }

    /**
     * Get new versions of the constraints which have positive right hand sides.
     */
    public List<LinearConstraint> normalizeConstraints(Collection<LinearConstraint> originalConstraints) {
        final List<LinearConstraint> normalized = new ArrayList<>(originalConstraints.size());
        for (LinearConstraint constraint : originalConstraints) {
            normalized.add(normalize(constraint));
        }
        return normalized;
    }

    /**
     * Get a new equation equivalent to this one with a positive right hand side.
     */
    private LinearConstraint normalize(final LinearConstraint constraint) {
        if (constraint.getValue() < 0) {
            IVector negatedCoeffs = constraint.getCoefficients().multiplyScalar(-1.0);
            return new LinearConstraint(negatedCoeffs,
                                      constraint.getRelationship().oppositeRelationship(),
                                      -1 * constraint.getValue());
        }
        return constraint;
    }

    /**
     * Get the number of objective functions in this tableau.
     * @return 2 for Phase 1.  1 for Phase 2.
     */
    public final int getNumObjectiveFunctions() {
        return this.numArtificialVariables > 0 ? 2 : 1;
    }

    /**
     * Get a count of constraints corresponding to a specified relationship.
     * @param relationship relationship to count
     * @return number of constraint with the specified relationship
     */
    private int getConstraintTypeCounts(final ConstraintType relationship) {
        int count = 0;
        for (final LinearConstraint constraint : constraints) {
            if (constraint.getRelationship() == relationship) {
                ++count;
            }
        }
        return count;
    }

    /**
     * Initialize the labels for the columns.
     */
    protected void initializeColumnLabels() {
        if (getNumObjectiveFunctions() == 2) {
            columnLabels.add("W");
        }
        columnLabels.add("Z");
        for (int i = 0; i < getOriginalNumDecisionVariables(); i++) {
            columnLabels.add("x" + i);
        }
        if (!restrictToNonNegative) {
            columnLabels.add("x-");
        }
        for (int i = 0; i < getNumSlackVariables(); i++) {
            columnLabels.add("s" + i);
        }
        for (int i = 0; i < getNumArtificialVariables(); i++) {
            columnLabels.add("a" + i);
        }
        columnLabels.add("RHS");
    }

    /**
     * Create the tableau by itself.
     * @param maximize if true, goal is to maximize the objective function
     * @return created tableau
     */
    protected IMatrix createTableau(final boolean maximize) {
        // create a matrix of the correct size
        int width = numDecisionVariables + numSlackVariables +
        numArtificialVariables + getNumObjectiveFunctions() + 1; // + 1 is for RHS
        int height = constraints.size() + getNumObjectiveFunctions();
        IMatrix matrix = IMatrix.zeros(height, width);

        // initialize the objective function rows
        // 只有当有人工变量时才创建Phase I目标函数
        if (getNumObjectiveFunctions() == 2) {
            matrix.set(0, 0, -1.0);  // Phase I 指示器
        }

        int zIndex = (getNumObjectiveFunctions() == 1) ? 0 : 1;
        matrix.set(zIndex, zIndex, maximize ? 1.0 : -1.0);

        double[][] scaled = new double[constraints.size() + 1][];

        // Objective coefficients (multiply by -1 for maximization in tableau)
        double[] objectiveCoefficients = new double[f.length()];
        for (int i = 0; i < f.length(); i++) {
            objectiveCoefficients[i] = maximize ? -RereMathUtil.safeDoubleValue(f.get(i)) : RereMathUtil.safeDoubleValue(f.get(i));
        }
        scaled[0] = objectiveCoefficients;
        double[] scaledRhs = new double[constraints.size() + 1];
        scaledRhs[0] = 0; // objective constant term

        // Constraint coefficients
        for (int i = 0; i < constraints.size(); i++) {
            LinearConstraint constraint = constraints.get(i);
            IVector coefficients = constraint.getCoefficients();
            double[] constraintCoeffs = new double[coefficients.length()];
            for (int j = 0; j < coefficients.length(); j++) {
                constraintCoeffs[j] = RereMathUtil.safeDoubleValue(coefficients.get(j));
            }
            scaled[i + 1] = constraintCoeffs;
            scaledRhs[i + 1] = constraint.getValue();
        }
        
        variableExpChange = new int[scaled[0].length];

        // Intelligent scaling control
        if (shouldApplyScaling(scaled, scaledRhs)) {
            if (verbose) {
                System.out.println("Applying numerical scaling based on coefficient analysis");
            }
            scale(scaled, scaledRhs);
        } else {
            if (verbose) {
                System.out.println("Skipping scaling - coefficients in safe numerical range");
            }
            // Initialize variableExpChange array to zeros when scaling is disabled
            Arrays.fill(variableExpChange, 0);
        }

        // Copy objective function to tableau
        copyArray(scaled[0], matrix, zIndex);
        matrix.set(zIndex, width - 1, scaledRhs[0]);

        if (!restrictToNonNegative) {
            matrix.set(zIndex, getSlackVariableOffset() - 1,
                            getInvertedCoefficientSum(scaled[0]));
        }

        // initialize the constraint rows
        int slackVar = 0;
        int artificialVar = 0;
        for (int i = 0; i < constraints.size(); i++) {
            final LinearConstraint constraint = constraints.get(i);
            final int row = getNumObjectiveFunctions() + i;

            // decision variable coefficients
            copyArray(scaled[i + 1], matrix, row);

            // x- (negative variable)
            if (!restrictToNonNegative) {
                matrix.set(row, getSlackVariableOffset() - 1,
                                getInvertedCoefficientSum(scaled[i + 1]));
            }

            // RHS
            matrix.set(row, width - 1, scaledRhs[i + 1]);

            // slack variables
            if (constraint.getRelationship() == ConstraintType.LEQ) {
                matrix.set(row, getSlackVariableOffset() + slackVar++, 1.0);  // slack variable
            } else if (constraint.getRelationship() == ConstraintType.GEQ) {
                matrix.set(row, getSlackVariableOffset() + slackVar++, -1.0); // surplus variable
            }

            // artificial variables (FOR EQ AND GEQ CONSTRAINTS)
            if (constraint.getRelationship() == ConstraintType.EQ ||
                constraint.getRelationship() == ConstraintType.GEQ) {
                // 只有当有Phase I目标函数时才设置人工变量系数
                if (getNumObjectiveFunctions() == 2) {
                    matrix.set(0, getArtificialVariableOffset() + artificialVar, 1.0);
                    // Eliminate artificial variable from phase I objective
                    subtractRow(matrix, 0, row);
                }
                matrix.set(row, getArtificialVariableOffset() + artificialVar++, 1.0);
            }
        }

        return matrix;
    }
    
    /**
     * Copy array to matrix row
     */
    private void copyArray(final double[] src, IMatrix dest, int row) {
        for (int i = 0; i < src.length; i++) {
            dest.set(row, getNumObjectiveFunctions() + i, src[i]);
        }
    }
    
    /**
     * Subtract one row from another
     */
    private void subtractRow(IMatrix matrix, int minuendRow, int subtrahendRow) {
        for (int i = 0; i < matrix.cols(); i++) {
            double minuend = RereMathUtil.safeDoubleValue(matrix.get(minuendRow, i));
            double subtrahend = RereMathUtil.safeDoubleValue(matrix.get(subtrahendRow, i));
            matrix.set(minuendRow, i, minuend - subtrahend);
        }
    }

    /**
     * Get the -1 times the sum of all coefficients in the given array.
     */
    private static double getInvertedCoefficientSum(final double[] coefficients) {
        double sum = 0;
        for (double coefficient : coefficients) {
            sum -= coefficient;
        }
        return sum;
    }

    /**
     * Intelligent scaling decision based on coefficient analysis
     * safety enhancements
     */
    private boolean shouldApplyScaling(double[][] scaled, double[] scaledRhs) {
        // Analysis
        double maxCoeff = 0.0;
        double minCoeff = Double.MAX_VALUE;
        int totalNonZero = 0;
        
        // Analyze coefficient distribution
        for (double[] row : scaled) {
            for (double coeff : row) {
                if (coeff != 0.0) {
                    double absCoeff = Math.abs(coeff);
                    maxCoeff = Math.max(maxCoeff, absCoeff);
                    minCoeff = Math.min(minCoeff, absCoeff);
                    totalNonZero++;
                }
            }
        }
        
        // Analyze RHS values
        for (double rhs : scaledRhs) {
            if (rhs != 0.0) {
                double absRhs = Math.abs(rhs);
                maxCoeff = Math.max(maxCoeff, absRhs);
                minCoeff = Math.min(minCoeff, absRhs);
                totalNonZero++;
            }
        }
        
        if (totalNonZero == 0) {
            return false; // No coefficients to scale
        }
        
        // Calculate coefficient range and distribution
        double coeffRange = maxCoeff / minCoeff;
        
        // Decision criteria based on commons-math4 experience:
        // 1. Avoid scaling for very large coefficient ranges (risk of precision loss)
        // 2. Avoid scaling when coefficients are already well-conditioned
        // 3. Apply scaling for moderate ranges that can benefit from normalization
        
        if (verbose) {
            System.out.println("Scaling analysis: maxCoeff=" + maxCoeff + 
                             ", minCoeff=" + minCoeff + 
                             ", range=" + coeffRange);
        }
        
        // Conservative scaling policy based on memory experience
        if (verbose) {
            System.out.println("Scaling decision logic:");
            System.out.println("  coeffRange > 1e6? " + (coeffRange > 1e6));
            System.out.println("  maxCoeff > 1e4? " + (maxCoeff > 1e4));
            System.out.println("  coeffRange < 10 && maxCoeff < 100? " + (coeffRange < 10 && maxCoeff < 100));
            System.out.println("  coeffRange > 100 && maxCoeff < 1000? " + (coeffRange > 100 && maxCoeff < 1000));
        }
        
        if (coeffRange > 1e6) {
            // Very large range - scaling may introduce precision errors
            if (verbose) {
                System.out.println("Scaling decision: DISABLED (very large coefficient range)");
            }
            return false;
        } else if (maxCoeff > 1e4) {
            // Large coefficients like testLargeNumbersProblem - risky for scaling
            if (verbose) {
                System.out.println("Scaling decision: DISABLED (large coefficients > 1e4)");
            }
            return false;
        } else if (coeffRange < 10 && maxCoeff < 100) {
            // Well-conditioned problem - scaling not necessary
            if (verbose) {
                System.out.println("Scaling decision: DISABLED (well-conditioned problem)");
            }
            return false;
        } else if (coeffRange > 100 && maxCoeff < 1000) {
            // Moderate range that can benefit from scaling
            if (verbose) {
                System.out.println("Scaling decision: ENABLED (moderate range benefits from scaling)");
            }
            return useNumericalScaling;
        }
        
        // Default: use user preference for borderline cases
        if (verbose) {
            System.out.println("Scaling decision: USER_PREFERENCE (borderline case, useNumericalScaling=" + useNumericalScaling + ")");
        }
        return useNumericalScaling;
    }

    // Scaling methods
    private void scale(double[][] scaled, double[] scaledRhs) {
        // Row scaling
        for (int i = 0; i < scaled.length; i++) {
            int minExp = MAX_IEEE_EXP + 1;
            int maxExp = MIN_IEEE_EXP - 1;
            for (double d: scaled[i]) {
                if (d != 0) {
                    int e = exponent(d);
                    if (e < minExp) {
                        minExp = e;
                    }
                    if (e > maxExp) {
                        maxExp = e;
                    }
                }
            }
            if (scaledRhs[i] != 0) {
                final int e = exponent(scaledRhs[i]);
                if (e < minExp) {
                    minExp = e;
                }
                if (e > maxExp) {
                    maxExp = e;
                }
            }
            final int expChange = computeExpChange(minExp, maxExp);
            if (expChange != 0) {
                scaledRhs[i] = updateExponent(scaledRhs[i], expChange);
                updateExponent(scaled[i], expChange);
            }
        }

        // Column scaling
        for (int i = 0; i < variableExpChange.length; i++) {
            int minExp = MAX_IEEE_EXP + 1;
            int maxExp = MIN_IEEE_EXP - 1;

            for (double[] coefficients : scaled) {
                final double d = coefficients[i];
                if (d != 0) {
                    int e = exponent(d);
                    if (e < minExp) {
                        minExp = e;
                    }
                    if (e > maxExp) {
                        maxExp = e;
                    }
                }
            }
            final int expChange = computeExpChange(minExp, maxExp);
            variableExpChange[i] = expChange;
            if (expChange != 0) {
                for (double[] coefficients : scaled) {
                     coefficients[i] = updateExponent(coefficients[i], expChange);
                }
            }
        }
    }

    private int computeExpChange(int minExp, int maxExp) {
        int expChange = 0;
        if (minExp <= MAX_IEEE_EXP &&
            minExp > OFFSET_IEEE_EXP) {
            expChange = OFFSET_IEEE_EXP - minExp;
        } else if (maxExp >= MIN_IEEE_EXP &&
                   maxExp < OFFSET_IEEE_EXP) {
            expChange = OFFSET_IEEE_EXP - maxExp;
        }
        return expChange;
    }

    private static void updateExponent(double[] dar, int exp) {
        for (int i = 0; i < dar.length; i++) {
            dar[i] = updateExponent(dar[i], exp);
        }
    }

    private static int exponent(double d) {
        final long bits = Double.doubleToLongBits(d);
        return (int) ((bits & EXPN) >>> IEEE_EXPONENT_SHIFT);
    }

    private static double updateExponent(double d, int exp) {
        if (d == 0 ||
            exp == 0) {
            return d;
        }
        final long bits = Double.doubleToLongBits(d);
        return Double.longBitsToDouble((bits & FRAC) | ((((bits & EXPN) >>> IEEE_EXPONENT_SHIFT) + exp) << IEEE_EXPONENT_SHIFT));
    }

    /**
     * Initialize basic variable mappings
     * 根据约束类型初始化基变量
     */
    private void initializeBasicVariables(final int startColumn) {
        basicVariables = new int[getWidth() - 1];
        basicRows = new int[getHeight() - getNumObjectiveFunctions()]; // 只考虑约束行

        Arrays.fill(basicVariables, -1);
        Arrays.fill(basicRows, -1);

        // 从约束行开始查找基变量
        int constraintRowStart = getNumObjectiveFunctions();
        
        if (verbose) {
            System.out.println("Initializing basic variables, constraintRowStart=" + constraintRowStart + ", getWidth()=" + getWidth() + ", getHeight()=" + getHeight());
        }
        
        for (int i = startColumn; i < getWidth() - 1; i++) {
            Integer row = findBasicRow(i);
            if (verbose) {
                System.out.println("Checking column " + i + " for basic variable, found row: " + row);
            }
            if (row != null && row >= constraintRowStart) {
                int adjustedRow = row - constraintRowStart;
                if (adjustedRow < basicRows.length) {
                    basicVariables[i] = row;  // Store absolute row index
                    basicRows[adjustedRow] = i;
                    if (verbose) {
                        System.out.println("Set basicVariables[" + i + "] = " + row + ", basicRows[" + adjustedRow + "] = " + i);
                    }
                }
            }
        }
        
        if (currentPhase == 2 && numSlackVariables > 0) {
            // 对于纯LEQ约束，松弛变量应该是初始基变量
            System.out.println("Initializing slack variables as basic variables for Phase II");
        }
        
        if (verbose) {
            System.out.println("Basic variables mapping completed:");
            for (int i = 0; i < basicVariables.length; i++) {
                if (basicVariables[i] != -1) {
                    System.out.println("Column " + i + " -> Row " + basicVariables[i]);
                }
            }
        }
    }

    private Integer findBasicRow(final int col) {
        Integer row = null;
        if (verbose) {
            System.out.println("findBasicRow checking column " + col + ", height=" + getHeight());
        }
        for (int i = 0; i < getHeight(); i++) {
            final double entry = getEntry(i, col);
            if (verbose) {
                System.out.println("  Entry at (" + i + ", " + col + ") = " + entry);
            }
            if (RerePrecision.equals(entry, 1d, maxUlps) && row == null) {
                row = i;
                if (verbose) {
                    System.out.println("  Found 1 at row " + i);
                }
            } else if (!RerePrecision.equals(entry, 0d, maxUlps)) {
                if (verbose) {
                    System.out.println("  Found non-zero at row " + i + ", returning null");
                }
                return null;
            }
        }
        if (verbose) {
            System.out.println("  Returning row: " + row);
        }
        return row;
    }

    /**
     * Gets the basic row for a given column.
     * @param col the column to get basic row for
     * @return the basic row or null if column is not basic
     */
    protected Integer getBasicRow(final int col) {
        if (col < 0 || col >= basicVariables.length) {
            return null;
        }
        final int row = basicVariables[col];
        return row == -1 ? null : row;
    }
    
    /**
     * Returns the variable that is basic in this row.
     * @param row the index of the row to check
     * @return the variable that is basic for this row.
     */
    protected int getBasicVariable(final int row) {
        if (row >= 0 && row < basicRows.length) {
            return basicRows[row];
        }
        return -1;
    }
    
    /**
     * Sets the basic variable mapping for a column.
     * @param col the column index
     * @param row the row index, or -1 if not basic
     */
    protected void setBasicVariable(final int col, final int row) {
        if (col >= 0 && col < basicVariables.length) {
            basicVariables[col] = row;
        }
    }
    
    /**
     * Sets the basic row mapping for a row.
     * @param row the row index (relative to constraint rows)
     * @param col the column index, or -1 if no basic variable
     */
    protected void setBasicRow(final int row, final int col) {
        // 在Phase II中，basicRows数组只包含约束行，不包括目标函数行
        // 所以row索引是相对于约束行的
        int constraintRowIndex = row - getNumObjectiveFunctions();
        if (constraintRowIndex >= 0 && constraintRowIndex < basicRows.length) {
            basicRows[constraintRowIndex] = col;
        }
    }

    // Getter methods
    public int getWidth() { return tableau.cols(); }
    public int getHeight() { return tableau.rows(); }
    public double getEntry(int row, int col) { return RereMathUtil.safeDoubleValue(tableau.get(row, col)); }
    public void setEntry(int row, int col, double value) { tableau.set(row, col, value); }
    public int getSlackVariableOffset() { return getNumObjectiveFunctions() + numDecisionVariables; }
    public int getArtificialVariableOffset() { return getSlackVariableOffset() + numSlackVariables; }
    public int getRhsOffset() { return getWidth() - 1; }
    public int getNumDecisionVariables() { return numDecisionVariables; }
    public int getOriginalNumDecisionVariables() { return f.length(); }
    public int getNumSlackVariables() { return numSlackVariables; }
    public int getNumArtificialVariables() { return numArtificialVariables; }
    public int getCurrentPhase() { return currentPhase; }
    public long getConstructionTime() { return constructionTime; }
    public int getPivotOperations() { return pivotOperations; }
    public void incrementPivotOperations() { pivotOperations++; }
    public int[] getVariableExponentChanges() { return variableExpChange; }
    
    /**
     * Set verbose mode
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
    
    /**
     * Check if the tableau is optimal
     * 根据记忆中的经验，需要检查所有非基变量（包括松弛和人工变量）
     */
    public boolean isOptimal() {
        int objectiveRow = (currentPhase == 1) ? 0 : ((getNumObjectiveFunctions() == 2) ? 1 : 0);
        
        // 检查所有变量列（除了RHS）
        for (int i = getNumObjectiveFunctions(); i < getRhsOffset(); i++) {
            final double entry = getEntry(objectiveRow, i);
            
            if (currentPhase == 1) {
                // Phase I: 最小化人工变量之和，寻找负系数（可以改进的方向）
                // 负系数表示增加该变量可以减少人工变量
                if (i < getArtificialVariableOffset() && RerePrecision.compareTo(entry, 0d, epsilon) < 0) {
                    return false;
                }
            } else {
                // Phase II: 最大化目标函数，找最小的负系数
                if (RerePrecision.compareTo(entry, 0d, epsilon) < 0) {
                    return false;
                }
            }
        }
        return true;
    }
    
    /**
     * Transition from Phase I to Phase II
     * 根据记忆，需要验证所有人工变量已从基中移除
     */
    public void transitionToPhaseII() {
        if (currentPhase != 1) {
            return; // 已经在Phase II了
        }
        
        // 验证Phase I目标值为零（所有人工变量都为0）
        double phaseIObjective = getEntry(0, getRhsOffset());
        if (Math.abs(phaseIObjective) > epsilon) {
            throw new RuntimeException("Cannot transition to Phase II: artificial variables still in solution (objective = " + phaseIObjective + ")");
        }
        
        // 检查是否有人工变量仍在基中
        for (int i = 0; i < getNumArtificialVariables(); i++) {
            int col = i + getArtificialVariableOffset();
            Integer basicRow = findBasicRow(col);
            if (basicRow != null) {
                System.out.println("Warning: Artificial variable a" + i + " still basic in row " + basicRow);
                // 根据记忆中的经验，这里需要强制移除人工变量
                // 通过新的pivot操作来移除人工变量
                removeArtificialVariableFromBasis(col, basicRow);
            }
        }
        
        currentPhase = 2;
        // Remove Phase I objective and artificial variables
        dropPhase1Objective();
    }
    
    /**
     * 强制移除人工变量从基中
     */
    private void removeArtificialVariableFromBasis(int artificialCol, int basicRow) {
        // 在Phase I中，如果人工变量仍在基中且目标值为0，
        // 这意味着存在退化解，需要找到一个非人工变量来替换它
        
        // 查找可以与人工变量交换的非人工变量
        for (int col = getNumObjectiveFunctions(); col < getArtificialVariableOffset(); col++) {
            double entry = getEntry(basicRow, col);
            if (!RerePrecision.equals(entry, 0.0, maxUlps)) {
                // 找到了非零元素，可以进行pivot操作
                System.out.println("Removing artificial variable col=" + artificialCol + 
                                 " by pivoting with col=" + col + " in row=" + basicRow);
                
                // 执行pivot操作，使非人工变量成为基变量
                performPivot(basicRow, col);
                return;
            }
        }
        
        // 如果没有找到合适的非人工变量，这可能是一个冗余约束
        System.out.println("Warning: Cannot remove artificial variable col=" + artificialCol + 
                         ", may be a redundant constraint");
    }
    
    /**
     * 执行pivot操作
     */
    private void performPivot(int pivotRow, int pivotCol) {
        double pivotElement = getEntry(pivotRow, pivotCol);
        
        if (RerePrecision.equals(pivotElement, 0.0, maxUlps)) {
            throw new RuntimeException("Cannot pivot on zero element at (" + pivotRow + ", " + pivotCol + ")");
        }
        
        // 对pivot行进行归一化
        for (int j = 0; j < getWidth(); j++) {
            setEntry(pivotRow, j, getEntry(pivotRow, j) / pivotElement);
        }
        
        // 消除其他行在pivot列上的元素
        for (int i = 0; i < getHeight(); i++) {
            if (i != pivotRow) {
                double multiplier = getEntry(i, pivotCol);
                if (!RerePrecision.equals(multiplier, 0.0, maxUlps)) {
                    for (int j = 0; j < getWidth(); j++) {
                        double newValue = getEntry(i, j) - multiplier * getEntry(pivotRow, j);
                        setEntry(i, j, newValue);
                    }
                }
            }
        }
        
        incrementPivotOperations();
    }
    
    /**
     * Remove Phase I objective and artificial variables
     */
    private void dropPhase1Objective() {
        if (getNumObjectiveFunctions() == 1) {
            return;
        }
        
        // 确定需要删除的列
        final Set<Integer> columnsToDrop = new TreeSet<>();
        columnsToDrop.add(0); // Phase I目标函数列
        
        // 添加所有人工变量列到删除列表
        for (int i = 0; i < getNumArtificialVariables(); i++) {
            int col = i + getArtificialVariableOffset();
            columnsToDrop.add(col);
        }
        
        // 在重构tableau之前，保存当前的基变量信息
        // 建立从旧列索引到新列索引的映射
        int[] oldToNewColumnMapping = new int[getWidth()];
        Arrays.fill(oldToNewColumnMapping, -1);
        
        int newColIndex = 0;
        for (int oldColIndex = 0; oldColIndex < getWidth(); oldColIndex++) {
            if (!columnsToDrop.contains(oldColIndex)) {
                oldToNewColumnMapping[oldColIndex] = newColIndex++;
            }
        }
        
        // 创建新的tableau，移除指定的行和列
        int newHeight = getHeight() - 1; // 移除Phase I目标行
        int newWidth = getWidth() - columnsToDrop.size(); // 移除删除的列数
        IMatrix newTableau = IMatrix.zeros(newHeight, newWidth);
        
        // 复制数据（跳过第一行Phase I目标和要删除的列）
        for (int i = 1; i < getHeight(); i++) {
            int newJ = 0;
            for (int j = 0; j < getWidth(); j++) {
                if (!columnsToDrop.contains(j)) {
                    newTableau.set(i - 1, newJ, getEntry(i, j));
                    newJ++;
                }
            }
        }
        
        // 更新标签
        Integer[] drop = columnsToDrop.toArray(new Integer[0]);
        for (int i = drop.length - 1; i >= 0; i--) {
            if (drop[i] < columnLabels.size()) {
                columnLabels.remove((int) drop[i]);
            }
        }
        
        this.tableau = newTableau;
        this.numArtificialVariables = 0;
        
        // 重新初始化基变量映射，现在Phase II只有一个目标函数行
        basicVariables = new int[newWidth - 1]; // 不包括RHS列
        basicRows = new int[newHeight - 1]; // 只考虑约束行，不包括目标函数行
        Arrays.fill(basicVariables, -1);
        Arrays.fill(basicRows, -1);
        
        // 在新的tableau中重新查找基变量
        // 对每一列检查是否为基变量
        for (int col = 1; col < newWidth - 1; col++) { // 跳过目标函数列和RHS列
            // 查找该列是否为单位向量（基变量）
            Integer basicRow = null;
            boolean isUnitVector = true;
            int onesCount = 0;
            
            // 从约束行开始检查（跳过目标函数行）
            for (int row = 1; row < newHeight; row++) {
                double entry = newTableau.get(row, col).doubleValue();
                if (RerePrecision.equals(entry, 1.0, maxUlps)) {
                    onesCount++;
                    if (onesCount == 1) {
                        basicRow = row;
                    } else {
                        // 多个1，不是单位向量
                        isUnitVector = false;
                        basicRow = null;
                        break;
                    }
                } else if (!RerePrecision.equals(entry, 0.0, maxUlps)) {
                    // 非0非1，不是单位向量
                    isUnitVector = false;
                    basicRow = null;
                    break;
                }
            }
            
            // 只有当恰好有一个1且其他都是0时，才是基变量
            if (isUnitVector && onesCount == 1 && basicRow != null) {
                // 找到基变量，注意basicRow是相对于新tableau的索引
                // basicRows数组只考虑约束行，所以需要减去目标函数行偏移
                int constraintRow = basicRow - 1;
                if (constraintRow >= 0 && constraintRow < basicRows.length) {
                    // 确保这一行还没有被其他变量占用
                    if (basicRows[constraintRow] == -1) {
                        basicVariables[col] = basicRow; // 存储绝对行索引（相对于新tableau）
                        basicRows[constraintRow] = col; // 存储约束行索引到列的映射
                        if (verbose) {
                            System.out.println("Phase II转换: 找到基变量 col=" + col + " -> row=" + basicRow);
                        }
                    }
                }
            }
        }
        
        if (verbose) {
            System.out.println("Phase II基变量映射重建完成:");
            for (int i = 0; i < basicRows.length; i++) {
                if (basicRows[i] != -1) {
                    System.out.println("约束行 " + i + " -> 列 " + basicRows[i]);
                }
            }
        }
    }
}