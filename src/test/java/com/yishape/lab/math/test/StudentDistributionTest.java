package com.yishape.lab.math.test;

import com.yishape.lab.math.stats.distribution.StudentDistribution;

public class StudentDistributionTest {
    
    public static void main(String[] args) {
        System.out.println("Testing StudentDistribution with location-scale parameters");
        
        // Test standard t-distribution (dof=10)
        StudentDistribution standardT = new StudentDistribution(10.0);
        System.out.println("Standard t-distribution: " + standardT);
        System.out.println("Mean: " + standardT.mean());
        System.out.println("Median: " + standardT.median());
        System.out.println("Mode: " + standardT.mode());
        System.out.println("Variance: " + standardT.var());
        System.out.println("PDF at x=0: " + standardT.pdf(0.0));
        System.out.println("CDF at x=0: " + standardT.cdf(0.0));
        System.out.println("PPF at p=0.5: " + standardT.ppf(0.5));
        System.out.println();
        
        // Test location-scale t-distribution (dof=10, location=2, scale=3)
        StudentDistribution locationScaleT = new StudentDistribution(10.0, 2.0, 3.0);
        System.out.println("Location-scale t-distribution: " + locationScaleT);
        System.out.println("Mean: " + locationScaleT.mean());
        System.out.println("Median: " + locationScaleT.median());
        System.out.println("Mode: " + locationScaleT.mode());
        System.out.println("Variance: " + locationScaleT.var());
        System.out.println("PDF at x=2: " + locationScaleT.pdf(2.0)); // Should be same as PDF at 0 for standard
        System.out.println("CDF at x=2: " + locationScaleT.cdf(2.0)); // Should be same as CDF at 0 for standard
        System.out.println("PPF at p=0.5: " + locationScaleT.ppf(0.5)); // Should be location=2
        System.out.println();
        
        // Test some sample values
        double[] samples = locationScaleT.sample(5);
        System.out.println("Sample values:");
        for (int i = 0; i < samples.length; i++) {
            System.out.println("  Sample " + (i+1) + ": " + samples[i]);
        }
    }
}