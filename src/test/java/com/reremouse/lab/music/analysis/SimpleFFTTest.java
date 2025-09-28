package com.reremouse.lab.music.analysis;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.signal.core.Complex;
import com.reremouse.lab.math.signal.core.RereFFT;

/**
 * Simple test to verify FFT improvements
 */
public class SimpleFFTTest {

    public static void main(String[] args) {
        try {
            System.out.println("Testing FFT Improvements...");
            
            // Create a test signal with a clear pitch (A4 = 440Hz)
            double sampleRate = 44100.0;
            int durationSeconds = 1;
            int numSamples = (int) (sampleRate * durationSeconds);
            double[] samples = new double[numSamples];
            
            // Generate a pure sine wave at 440Hz
            for (int i = 0; i < numSamples; i++) {
                double t = i / sampleRate;
                samples[i] = Math.sin(2 * Math.PI * 440 * t);
            }
            
            System.out.println("Generated sine wave with " + numSamples + " samples");
            
            // Convert to IVector
            IVector<Double> sampleVector = Linalg.vector(samples);
            
            // Test windowing
            int windowSize = Math.min(4096, sampleVector.length());
            System.out.println("Window size: " + windowSize);
            
            // Apply window function manually for testing
            IVector<Double> windowed = Linalg.zeros(windowSize);
            for (int i = 0; i < windowSize; i++) {
                double window = 0.54 - 0.46 * Math.cos(2.0 * Math.PI * i / (windowSize - 1));
                windowed.set(i, sampleVector.get(i) * window);
            }
            
            // Convert to Complex array
            Complex[] input = new Complex[windowSize];
            for (int i = 0; i < windowSize; i++) {
                input[i] = new Complex(windowed.get(i), 0.0);
            }
            
            System.out.println("Converted to complex array of length: " + input.length);
            
            // Zero-pad to power of 2
            Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(input);
            System.out.println("Zero-padded to length: " + paddedInput.length);
            
            // Compute FFT
            System.out.println("Computing FFT...");
            Complex[] spectrum = RereFFT.fft(paddedInput);
            System.out.println("FFT computed successfully with " + spectrum.length + " elements");
            
            // Check some spectrum values
            System.out.println("First 10 spectrum magnitudes:");
            for (int i = 0; i < Math.min(10, spectrum.length); i++) {
                System.out.println("  [" + i + "]: " + String.format("%.6f", spectrum[i].magnitude()));
            }
            
            // Find peak frequency
            double maxMagnitude = 0.0;
            int peakIndex = 0;
            for (int i = 0; i < spectrum.length/2; i++) {
                double magnitude = spectrum[i].magnitude();
                if (magnitude > maxMagnitude) {
                    maxMagnitude = magnitude;
                    peakIndex = i;
                }
            }
            
            double peakFrequency = (peakIndex * sampleRate) / paddedInput.length;
            System.out.println("Peak frequency: " + String.format("%.2f", peakFrequency) + " Hz (index: " + peakIndex + ", magnitude: " + String.format("%.6f", maxMagnitude) + ")");
            
            if (Math.abs(peakFrequency - 440) < 10) {
                System.out.println("SUCCESS: Peak frequency is close to expected 440Hz");
            } else {
                System.out.println("WARNING: Peak frequency is not close to expected 440Hz");
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}