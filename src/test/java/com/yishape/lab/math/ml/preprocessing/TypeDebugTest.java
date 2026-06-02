package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;
import com.yishape.lab.math.linalg.IDoubleMatrix;

public class TypeDebugTest {
    public static void main(String[] args) {
        float[][] fdata = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        IMatrix<Float> Xf = IMatrix.of(fdata);

        System.out.println("Xf.getClass(): " + Xf.getClass().getName());
        System.out.println("Xf instanceof IFloatMatrix: " + (Xf instanceof IFloatMatrix));
        System.out.println("Xf instanceof IMatrix: " + (Xf instanceof IMatrix));

        double[][] ddata = {{1.0, 2.0}, {3.0, 4.0}};
        IMatrix<Double> Xd = IMatrix.of(ddata);
        System.out.println("Xd.getClass(): " + Xd.getClass().getName());
        System.out.println("Xd instanceof IDoubleMatrix: " + (Xd instanceof IDoubleMatrix));

        RereMinMaxScaler scaler = new RereMinMaxScaler();
        scaler.fit(Xf);
        IMatrix<?> result = scaler.transform(Xf);
        System.out.println("result.getClass(): " + result.getClass().getName());
        Object val = result.get(0, 0);
        System.out.println("result.get(0,0): " + val + " type: " + val.getClass().getName());
    }
}