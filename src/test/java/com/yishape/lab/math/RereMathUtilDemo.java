package com.yishape.lab.math;

import com.yishape.lab.math.linalg.IDoubleVector;

public class RereMathUtilDemo {
    public static void main(String[] args) {
        System.out.println("Testing RereMathUtil with null values...");
        
        // Test Double array with null values
        Double[] doubleArrayWithNulls = {1.0, null, 3.0, null, 5.0};
        System.out.println("Original array: " + java.util.Arrays.toString(doubleArrayWithNulls));
        
        try {
            double[] primitiveDoubleArray = RereMathUtil.toPrimitive(doubleArrayWithNulls);
            System.out.println("Converted array: " + java.util.Arrays.toString(primitiveDoubleArray));
            System.out.println("Success: No NullPointerException thrown!");
        } catch (NullPointerException e) {
            System.out.println("Failed: NullPointerException thrown: " + e.getMessage());
        }
        
        // Test IDoubleVector creation with null values (this was failing before our fix)
        try {
            IDoubleVector vector = IDoubleVector.of(doubleArrayWithNulls);
            System.out.println("Vector created successfully with length: " + vector.length());
            System.out.println("Vector values: [" + vector.get(0) + ", " + vector.get(1) + ", " + vector.get(2) + ", " + vector.get(3) + ", " + vector.get(4) + "]");
            System.out.println("Success: IDoubleVector creation with null values works!");
        } catch (Exception e) {
            System.out.println("Failed: Exception thrown during IDoubleVector creation: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("Test completed.");
    }
}