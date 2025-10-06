package com.yishape.lab.audio;

import com.yishape.lab.audio.analysis.FFTProcessor;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple2;

/**
 * Test for FFTProcessor to verify the fixes
 */
public class FFTProcessorTest {

    public static void main(String[] args) {
        // Create a simple test signal (sine wave)
        int sampleRate = 44100;
        int duration = 1; // 1 second
        int numSamples = sampleRate * duration;
        
        double[] samples = new double[numSamples];
        double frequency = 440; // A4 note
        
        for (int i = 0; i < numSamples; i++) {
            samples[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
        }
        
        // Create AudioData from the samples
        IVector<Double> sampleVector = Linalg.vector(samples);
        AudioData audioData = new AudioData(sampleVector, sampleRate, 1, 16, AudioFormat.WAV);
        
        try {
            System.out.println("Testing FFT processing...");
            long startTime = System.currentTimeMillis();
            
            // Process FFT
            FFTProcessor processor = FFTProcessor.getInstance();
            Tuple2<IVector<Double>, IVector<Double>> spectrum = processor.processFFT(audioData);
            
            long endTime = System.currentTimeMillis();
            System.out.println("FFT processed in " + (endTime - startTime) + " ms");
            
            IVector<Double> frequencies = spectrum.getFirst();
            IVector<Double> magnitudes = spectrum.getSecond();
            
            System.out.println("Frequency vector length: " + frequencies.length());
            System.out.println("Magnitude vector length: " + magnitudes.length());
            
            // Check that we have non-zero magnitudes
            double maxMagnitude = 0;
            int maxIndex = 0;
            for (int i = 0; i < magnitudes.length(); i++) {
                if (magnitudes.get(i) > maxMagnitude) {
                    maxMagnitude = magnitudes.get(i);
                    maxIndex = i;
                }
            }
            
            System.out.println("Maximum magnitude: " + maxMagnitude);
            System.out.println("Frequency at maximum magnitude: " + frequencies.get(maxIndex) + " Hz");
            
            if (maxMagnitude > 0) {
                System.out.println("SUCCESS: Non-zero magnitudes found");
            } else {
                System.out.println("ERROR: All magnitudes are zero");
            }
            
        } catch (AudioProcessingException e) {
            e.printStackTrace();
        }
    }
}