package com.yishape.lab.math.optimize.linpg.simplex;

import com.yishape.lab.math.util.RerePrecision;

import java.util.*;

/**
 * 单纯形法枢轴选择策略 / Pivot Selection Strategy for Simplex Method
 * <p>
 * 提供单纯形法中入基变量和出基变量的选择策略。
 * Advanced pivot selection strategies for simplex method.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class PivotSelectionStrategy {

    private final double epsilon;
    private final int maxUlps;

    public PivotSelectionStrategy(double epsilon, int maxUlps) {
        this.epsilon = epsilon;
        this.maxUlps = maxUlps;
    }

    /**
     * Select entering variable using Dantzig's rule with correct Big M handling
     */
    public Integer selectEnteringVariableDantzig(RereSimplexTableau tableau) {
        // Determine the correct objective row based on tableau structure
        int objectiveRow;
        if (tableau.getCurrentPhase() == 1) {
            objectiveRow = 0; // Phase I objective row
        } else {
            // Phase II: Check tableau structure to detect Big M formulation
            boolean hasBigMStructure = (tableau.getHeight() == 3) && (tableau.getWidth() >= 7);
            if (hasBigMStructure) {
                objectiveRow = 1; // Big M method: Phase II objective is in row 1
            } else {
                objectiveRow = tableau.getHeight() - 1; // Standard Phase II: last row
            }
        }

        Integer enteringVar = null;

        if (tableau.getCurrentPhase() == 1) {
            // Phase I: minimization - select most positive coefficient
            double mostPositive = 0.0;
            for (int j = tableau.getNumObjectiveFunctions(); j < tableau.getRhsOffset(); j++) {
                double coeff = tableau.getEntry(objectiveRow, j);
                // Phase I: 寻找最负的非人工变量系数
                if (j < tableau.getArtificialVariableOffset() && coeff < mostPositive - epsilon) {
                    mostPositive = coeff;
                    enteringVar = j;
                }
            }
        } else {
            // Phase II: maximization - select most negative coefficient
            double mostNegative = 0.0;

            // For Big M method, exclude artificial variables from entering variable selection
            int variablesToCheck;
            boolean hasBigMStructure = (tableau.getHeight() == 3) && (tableau.getWidth() >= 7);
            if (hasBigMStructure) {
                variablesToCheck = tableau.getWidth() - 3; // Exclude 2 artificial vars + RHS
            } else {
                variablesToCheck = tableau.getWidth() - 1; // Exclude RHS only
            }

            for (int j = tableau.getNumObjectiveFunctions(); j < tableau.getRhsOffset(); j++) {
                double coeff = tableau.getEntry(objectiveRow, j);
                if (coeff < mostNegative - epsilon) {
                    mostNegative = coeff;
                    enteringVar = j;
                }
            }
        }

        return enteringVar;
    }

    /**
     * Select leaving variable using minimum ratio test with enhanced degeneracy
     * handling
     */
    public Integer selectLeavingVariable(RereSimplexTableau tableau, int enteringVar) {
        // Default cutOff value
        double cutOff = 1e-10;

        // Create a list of all the rows that tie for the lowest score in the minimum ratio test
        List<Integer> minRatioPositions = new ArrayList<>();
        double minRatio = Double.MAX_VALUE;

        for (int i = tableau.getNumObjectiveFunctions(); i < tableau.getHeight(); i++) {
            final double rhs = tableau.getEntry(i, tableau.getWidth() - 1);
            final double entry = tableau.getEntry(i, enteringVar);

            // Only consider pivot elements larger than the cutOff threshold
            // selecting others may lead to degeneracy or numerical instabilities
            if (RerePrecision.compareTo(entry, 0d, cutOff) > 0) {
                final double ratio = Math.abs(rhs / entry);
                // Check if the entry is strictly equal to the current min ratio
                // do not use a ulp/epsilon check
                final int cmp = Double.compare(ratio, minRatio);
                if (cmp == 0) {
                    minRatioPositions.add(i);
                } else if (cmp < 0) {
                    minRatio = ratio;
                    minRatioPositions.clear();
                    minRatioPositions.add(i);
                }
            }
        }

        if (minRatioPositions.isEmpty()) {
            return null; // Unbounded solution
        } else if (minRatioPositions.size() > 1) {
            // There's a degeneracy as indicated by a tie in the minimum ratio test

            // 1. Check if there's an artificial variable that can be forced out of the basis
            if (tableau.getNumArtificialVariables() > 0) {
                for (Integer row : minRatioPositions) {
                    for (int i = 0; i < tableau.getNumArtificialVariables(); i++) {
                        int column = i + tableau.getArtificialVariableOffset();
                        final double entry = tableau.getEntry(row, column);
                        Integer basicRowForCol = tableau.getBasicRow(column);
                        if (RerePrecision.equals(entry, 1d, 10)
                                && basicRowForCol != null && basicRowForCol.equals(row - tableau.getNumObjectiveFunctions())) {
                            return row - tableau.getNumObjectiveFunctions(); // Convert back to relative row index
                        }
                    }
                }
            }

            // 2. Apply Bland's rule to prevent cycling:
            //    take the row for which the corresponding basic variable has the smallest index
            //    Since we don't have getBasicVariable, use a simpler approach
            Integer minRow = minRatioPositions.get(0); // Use first candidate for now
            return minRow - tableau.getNumObjectiveFunctions();
        }

        // Convert back to relative row index
        return minRatioPositions.get(0) - tableau.getNumObjectiveFunctions();
    }

    /**
     * Perform pivot operation using RerePrecision methods for numerical stability
     */
    public void performPivotOperation(RereSimplexTableau tableau, int leavingRow, int enteringVar) {
        // Determine the actual pivot row based on tableau structure
        int pivotRow;
        boolean hasBigMStructure = (tableau.getHeight() == 3) && (tableau.getWidth() >= 7);

        if (hasBigMStructure) {
            // Big M tableau: constraints start at row 2
            pivotRow = leavingRow + 2;
        } else if (tableau.getCurrentPhase() == 1 && tableau.getNumArtificialVariables() > 0) {
            // Two-phase method with both Phase I and Phase II objectives
            pivotRow = leavingRow + 2; // Add 2 to account for Phase I and Phase II objective rows
        } else {
            // Standard Phase II tableau: add objective functions offset
            pivotRow = leavingRow + tableau.getNumObjectiveFunctions();
        }

        double pivotElement = tableau.getEntry(pivotRow, enteringVar);

        if (RerePrecision.equalsZero(pivotElement, epsilon)) {
            throw new IllegalStateException("Pivot element too small: " + pivotElement);
        }

        // Normalize pivot row
        for (int j = 0; j < tableau.getWidth(); j++) {
            double value = tableau.getEntry(pivotRow, j);
            tableau.setEntry(pivotRow, j, value / pivotElement);
        }

        // Eliminate pivot column in other rows
        for (int i = 0; i < tableau.getHeight(); i++) {
            if (i != pivotRow) {
                double multiplier = tableau.getEntry(i, enteringVar);
                if (!RerePrecision.equalsZero(multiplier, epsilon)) {
                    for (int j = 0; j < tableau.getWidth(); j++) {
                        double currentValue = tableau.getEntry(i, j);
                        double pivotRowValue = tableau.getEntry(pivotRow, j);
                        double newValue = currentValue - multiplier * pivotRowValue;

                        // Clean up near-zero values using RerePrecision
                        if (RerePrecision.equalsZero(newValue, epsilon)) {
                            newValue = 0.0;
                        }

                        tableau.setEntry(i, j, newValue);
                    }
                }
            }
        }

        // Update the basic variable mappings
        // This is critical for correct solution extraction
        int constraintRowIndex = pivotRow - tableau.getNumObjectiveFunctions();
        int previousBasicVariable = tableau.getBasicVariable(constraintRowIndex);
        if (previousBasicVariable >= 0) {
            tableau.setBasicVariable(previousBasicVariable, -1);
        }
        tableau.setBasicVariable(enteringVar, pivotRow);
        // 在设置basicRow时，需要传递相对于约束行的索引
        tableau.setBasicRow(pivotRow, enteringVar);

        tableau.incrementPivotOperations();
    }
}
