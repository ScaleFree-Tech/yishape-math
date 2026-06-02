package com.yishape.lab.math.linalg;

public class TestEmptyArray {
    public static void main(String[] args) {
        // Test what happens with empty 2D arrays
        float[][] emptyArray = new float[0][2];
        System.out.println("Array length (rows): " + emptyArray.length);
        
        if (emptyArray.length > 0) {
            System.out.println("Columns: " + emptyArray[0].length);
        } else {
            System.out.println("No rows to access, so can't get column count");
        }
        
        // Test what happens when we try to access emptyArray[0]
        try {
            System.out.println("Trying to access emptyArray[0].length: " + emptyArray[0].length);
        } catch (Exception e) {
            System.out.println("Exception: " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }
}