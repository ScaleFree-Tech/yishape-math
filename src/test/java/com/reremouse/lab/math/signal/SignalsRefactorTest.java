package com.reremouse.lab.math.signal;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

public class SignalsRefactorTest {
    public static void main(String[] args) {
        try {
            // Test signal generation
            System.out.println("Testing signal generation...");
            IVector<Double> sineWave = Signals.sineWave(100, 10, 1000, 1.0, 0);
            System.out.println("Generated sine wave with " + sineWave.length() + " samples");
            
            IVector<Double> cosineWave = Signals.cosineWave(100, 10, 1000, 1.0, 0);
            System.out.println("Generated cosine wave with " + cosineWave.length() + " samples");
            
            IVector<Double> squareWave = Signals.squareWave(100, 10, 1000, 1.0, 0.5);
            System.out.println("Generated square wave with " + squareWave.length() + " samples");
            
            IVector<Double> noise = Signals.whiteNoise(100, 0.1);
            System.out.println("Generated white noise with " + noise.length() + " samples");
            
            // Test signal filtering
            System.out.println("\nTesting signal filtering...");
            IVector<Double> filtered = Signals.butterworthLowPass(sineWave, 50, 1000, 4);
            System.out.println("Applied Butterworth filter");
            
            // Test signal analysis
            System.out.println("\nTesting signal analysis...");
            IVector<Double> autocorr = Signals.autocorrelation(sineWave);
            System.out.println("Calculated autocorrelation with " + autocorr.length() + " samples");
            
            System.out.println("\nAll tests passed!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}