package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DebugEmptySliceTest5 {

    @Test
    public void debugEmptyMatrixCreation() {
        // 测试创建空矩阵
        IMatrix<Float> emptyMatrix = IFloatMatrix.of(new float[0][2]);
        
        System.out.println("Empty matrix shape: " + emptyMatrix.rows() + "x" + emptyMatrix.cols());
        
        // 检查结果
        assertEquals(0, emptyMatrix.rows(), "Expected 0 rows");
        assertEquals(2, emptyMatrix.cols(), "Expected 2 columns");
    }
}