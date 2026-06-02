package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;

/**
 * 分解求解器的行主序工作区：对 Rere 的 {@link IMatrix#toDoubleArray()} 已为独立副本，
 * 此处直接返回以供就地消元；其它 {@link IMatrix} 实现亦应保证返回值可安全改写。
 */
final class DecompositionRhsCopy {

    private DecompositionRhsCopy() {
    }

    /**
     * @return 与 {@code b} 数值相同的行主序数组，可安全地被求解器就地修改
     */
    static double[][] mutableRowMajorCopy(IMatrix<Double> b) {
        return b.toDoubleArray();
    }
}
