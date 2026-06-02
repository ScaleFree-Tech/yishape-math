package com.yishape.lab.math.linalg;

/**
 * A lightweight zero-copy view of a sub-rectangle of a parent {@link IMatrix}.
 * All get/set calls delegate to the parent's backing data with offset
 * translation. No data is copied.
 *
 * <p><b>Mutability:</b> mutations to the view mutate the parent.
 * Views are intended for internal use in decomposition algorithms
 * and should not be held beyond the scope of the algorithm.</p>
 *
 * @param <T> element type (Double / Float)
 */
public interface IMatrixView<T extends Number> extends IMatrix<T> {

    /** The parent matrix this view looks into. */
    IMatrix<T> parent();

    /** Row offset within parent (0-based). */
    int rowOffset();

    /** Column offset within parent (0-based). */
    int colOffset();

    /** Create a sub-view of this view (zero-copy). */
    IMatrixView<T> subView(int startRow, int endRow, int startCol, int endCol);

    @Override
    default boolean isView() {
        return true;
    }
}
