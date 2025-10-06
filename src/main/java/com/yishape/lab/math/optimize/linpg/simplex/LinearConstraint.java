package com.yishape.lab.math.optimize.linpg.simplex;

import com.yishape.lab.math.linalg.IVector;

/**
 * Linear constraint for simplex method
 * Based on commons-math4 LinearConstraint
 */
public class LinearConstraint {
    
    /** Coefficients of the constraint (left hand side). */
    private final IVector coefficients;
    
    /** Relationship between left and right hand sides */
    private final ConstraintType relationship;
    
    /** Value of the constraint (right hand side). */
    private final double value;

    /**
     * Build a constraint involving a single linear equation.
     *
     * @param coefficients The coefficients of the constraint (left hand side)
     * @param relationship The type of (in)equality used in the constraint
     * @param value The value of the constraint (right hand side)
     */
    public LinearConstraint(final IVector coefficients,
                           final ConstraintType relationship,
                           final double value) {
        this.coefficients = coefficients;
        this.relationship = relationship;
        this.value = value;
    }

    /**
     * Gets the coefficients of the constraint (left hand side).
     *
     * @return the coefficients of the constraint (left hand side).
     */
    public IVector getCoefficients() {
        return coefficients;
    }

    /**
     * Gets the relationship between left and right hand sides.
     *
     * @return the relationship between left and right hand sides.
     */
    public ConstraintType getRelationship() {
        return relationship;
    }

    /**
     * Gets the value of the constraint (right hand side).
     *
     * @return the value of the constraint (right hand side).
     */
    public double getValue() {
        return value;
    }
}