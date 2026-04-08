package com.yishape.lab.math.optimize.linpg.simplex;

/**
 * Types of relationships between constraint coefficients and values
 */
public enum ConstraintType {
    /** Equality relationship: ax = b */
    EQ("="),
    /** Less than or equal relationship: ax &lt;= b */
    LEQ("<="),
    /** Greater than or equal relationship: ax &gt;= b */
    GEQ(">=");

    /** Display string for the relationship. */
    private final String stringValue;

    /**
     * Constructor
     *
     * @param stringValue Display string for the relationship.
     */
    ConstraintType(String stringValue) {
        this.stringValue = stringValue;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return stringValue;
    }

    /**
     * Gets the relationship obtained when multiplying all coefficients by -1.
     *
     * @return the opposite relationship.
     */
    public ConstraintType oppositeRelationship() {
        switch (this) {
        case LEQ :
            return GEQ;
        case GEQ :
            return LEQ;
        default :
            return EQ;
        }
    }
}