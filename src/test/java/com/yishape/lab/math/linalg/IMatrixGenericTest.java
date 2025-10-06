package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for IMatrix interface and its generic factory methods
 测试IMatrixGeneric接口及其泛型工厂方法的测试类
 */
public class IMatrixGenericTest {
    
    private IDoubleMatrix floatMatrix;
    
    @BeforeEach
    public void setUp() {
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        };
        floatMatrix = IDoubleMatrix.of(data);
    }
    
    @Test
    public void testGenericFactoryMethods() {
        // Test generic ones matrix for Float type
        IMatrix<Float> onesFloat = IMatrix.ones(2, 3, Float.class);
        assertNotNull(onesFloat);
        assertEquals(2, onesFloat.getRowNum());
        assertEquals(3, onesFloat.getColNum());
        assertEquals(Float.valueOf(1.0f), onesFloat.get(0, 0));
        assertEquals(Float.valueOf(1.0f), onesFloat.get(1, 2));
        
        // Test generic zeros matrix for Float type
        IMatrix<Float> zerosFloat = IMatrix.zeros(3, 2, Float.class);
        assertNotNull(zerosFloat);
        assertEquals(3, zerosFloat.getRowNum());
        assertEquals(2, zerosFloat.getColNum());
        assertEquals(Float.valueOf(0.0f), zerosFloat.get(0, 0));
        assertEquals(Float.valueOf(0.0f), zerosFloat.get(2, 1));
        
        // Test generic identity matrix for Float type
        IMatrix<Float> eyeFloat = IMatrix.eye(3, Float.class);
        assertNotNull(eyeFloat);
        assertEquals(3, eyeFloat.getRowNum());
        assertEquals(3, eyeFloat.getColNum());
        assertEquals(Float.valueOf(1.0f), eyeFloat.get(0, 0));
        assertEquals(Float.valueOf(1.0f), eyeFloat.get(1, 1));
        assertEquals(Float.valueOf(1.0f), eyeFloat.get(2, 2));
        assertEquals(Float.valueOf(0.0f), eyeFloat.get(0, 1));
        assertEquals(Float.valueOf(0.0f), eyeFloat.get(1, 0));
        
        // Test generic random matrix for Float type
        IMatrix<Float> randFloat = IMatrix.rand(2, 2, Float.class);
        assertNotNull(randFloat);
        assertEquals(2, randFloat.getRowNum());
        assertEquals(2, randFloat.getColNum());
    }
    
    @Test
    public void testGenericFactoryMethodsWithUnsupportedType() {
        // Test unsupported type
        assertThrows(UnsupportedOperationException.class, () -> {
            IMatrix.ones(2, 2, Integer.class);
        });
        
        assertThrows(UnsupportedOperationException.class, () -> {
            IMatrix.zeros(2, 2, Integer.class);
        });
        
        assertThrows(UnsupportedOperationException.class, () -> {
            IMatrix.eye(2, Integer.class);
        });
        
        assertThrows(UnsupportedOperationException.class, () -> {
            IMatrix.rand(2, 2, Integer.class);
        });
    }
    
    @Test
    public void testInheritedGenericMethods() {
        // Test shape method from generic interface
        int[] shape = floatMatrix.shape();
        assertNotNull(shape);
        assertEquals(2, shape.length);
        assertEquals(2, shape[0]); // rows
        assertEquals(3, shape[1]); // columns
        
        // Test alias methods from generic interface
        assertEquals(2, floatMatrix.rows());
        assertEquals(3, floatMatrix.cols());
        
        // Test transpose shorthand method
        IMatrix<Double> transposed = floatMatrix.t();
        assertNotNull(transposed);
        assertEquals(3, transposed.getRowNum());
        assertEquals(2, transposed.getColNum());
        assertEquals(Double.valueOf(1.0), transposed.get(0, 0));
        assertEquals(Double.valueOf(4.0), transposed.get(0, 1));
    }
    
    @Test
    public void testBasicMatrixOperations() {
        // Test basic operations from generic interface
        double[][] data2 = {
            {1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0}
        };
        IDoubleMatrix matrix2 = IDoubleMatrix.of(data2);
        
        // Test addition
        IMatrix<Double> sum = floatMatrix.add(matrix2);
        assertNotNull(sum);
        assertEquals(Double.valueOf(2.0), sum.get(0, 0));
        assertEquals(Double.valueOf(3.0), sum.get(0, 1));
        assertEquals(Double.valueOf(5.0), sum.get(1, 0));
        
        // Test subtraction
        IMatrix<Double> diff = floatMatrix.sub(matrix2);
        assertNotNull(diff);
        assertEquals(Double.valueOf(0.0), diff.get(0, 0));
        assertEquals(Double.valueOf(1.0), diff.get(0, 1));
        assertEquals(Double.valueOf(3.0), diff.get(1, 0));
        
        // Test scalar multiplication
        IMatrix<Double> scaled = floatMatrix.multiplyScalar(2.0);
        assertNotNull(scaled);
        assertEquals(Double.valueOf(2.0), scaled.get(0, 0));
        assertEquals(Double.valueOf(4.0), scaled.get(0, 1));
        assertEquals(Double.valueOf(8.0), scaled.get(1, 0));
    }
    
    @Test
    public void testStatisticalOperations() {
        // Test sum operation
        Double sum = floatMatrix.sum();
        assertEquals(Double.valueOf(21.0), sum); // 1+2+3+4+5+6 = 21
        
        // Test mean operation
        Double mean = floatMatrix.mean();
        assertEquals(Double.valueOf(3.5), mean); // 21/6 = 3.5
        
        // Test max operation
        Double max = floatMatrix.max();
        assertEquals(Double.valueOf(6.0), max);
        
        // Test min operation
        Double min = floatMatrix.min();
        assertEquals(Double.valueOf(1.0), min);
    }
    
    @Test
    public void testCopyAndToArray() {
        // Test copy operation
        IMatrix<Double> copy = floatMatrix.copy();
        assertNotNull(copy);
        assertNotSame(floatMatrix, copy);
        assertEquals(floatMatrix.get(0, 0), copy.get(0, 0));
        assertEquals(floatMatrix.get(1, 2), copy.get(1, 2));
        
        // Test toArray operation
        double[] array = floatMatrix.flatten().toDoubleArray();
        assertNotNull(array);
        assertEquals(6, array.length); // 2*3 = 6 elements
        assertEquals(1.0, array[0]);
        assertEquals(2.0, array[1]);
        assertEquals(6.0, array[5]);
    }
}