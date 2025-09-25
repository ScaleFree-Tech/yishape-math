package com.reremouse.lab.audio.core;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

/**
 * Audio Data Class
 * <p>
 * Encapsulates basic audio information including sample rate, channels, bit depth, duration, etc.
 * Uses IVector interface to store audio sample data, ensuring compatibility with existing codebase.
 * Provides comprehensive methods for audio data manipulation and analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioData {
    
    /** Audio sample data */
    private final IVector<Double> samples;
    
    /** Sample rate (Hz) */
    private final double sampleRate;
    
    /** Number of channels */
    private final int channels;
    
    /** Bit depth (bits) */
    private final int bitDepth;
    
    /** Duration (seconds) */
    private final double duration;
    
    /** Audio format */
    private final AudioFormat format;
    
    /**
     * Constructor
     *
     * @param samples Audio sample data
     * @param sampleRate Sample rate
     * @param channels Number of channels
     * @param bitDepth Bit depth
     * @param format Audio format
     */
    public AudioData(IVector<Double> samples, double sampleRate, int channels, int bitDepth, AudioFormat format) {
        if (samples == null) {
            throw new IllegalArgumentException("Samples cannot be null");
        }
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("Sample rate must be positive");
        }
        if (channels <= 0) {
            throw new IllegalArgumentException("Number of channels must be positive");
        }
        if (bitDepth <= 0) {
            throw new IllegalArgumentException("Bit depth must be positive");
        }
        if (format == null) {
            throw new IllegalArgumentException("Audio format cannot be null");
        }
        
        this.samples = samples;
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.bitDepth = bitDepth;
        this.format = format;
        this.duration = samples.length() / (sampleRate * channels);
    }
    
    /**
     * Get audio sample data
     *
     * @return Audio sample vector
     */
    public IVector<Double> getSamples() {
        return samples;
    }
    
    /**
     * Get sample rate
     *
     * @return Sample rate (Hz)
     */
    public double getSampleRate() {
        return sampleRate;
    }
    
    /**
     * Get number of channels
     *
     * @return Number of channels
     */
    public int getChannels() {
        return channels;
    }
    
    /**
     * Get bit depth
     *
     * @return Bit depth (bits)
     */
    public int getBitDepth() {
        return bitDepth;
    }
    
    /**
     * Get bits per sample
     *
     * @return Bits per sample (bits)
     */
    public int getBitsPerSample() {
        return bitDepth;
    }
    
    /**
     * Get duration
     *
     * @return Duration (seconds)
     */
    public double getDuration() {
        return duration;
    }
    
    /**
     * Get audio format
     *
     * @return Audio format
     */
    public AudioFormat getFormat() {
        return format;
    }
    
    /**
     * Get data for specific channel
     *
     * @param channel Channel index (0-based)
     * @return Sample data for specified channel
     * @throws IllegalArgumentException If channel index is invalid
     */
    public IVector<Double> getChannel(int channel) {
        if (channel < 0 || channel >= channels) {
            throw new IllegalArgumentException("Invalid channel index: " + channel);
        }
        
        int samplesPerChannel = samples.length() / channels;
        IVector<Double> channelSamples = Linalg.zeros(samplesPerChannel);
        
        for (int i = 0; i < samplesPerChannel; i++) {
            channelSamples.set(i, samples.get(i * channels + channel));
        }
        
        return channelSamples;
    }
    
    /**
     * Get data for all channels
     *
     * @return Channel data matrix, each row represents a channel
     */
    public IVector<Double>[] getAllChannels() {
        @SuppressWarnings("unchecked")
        IVector<Double>[] channelData = new IVector[channels];
        
        for (int ch = 0; ch < channels; ch++) {
            channelData[ch] = getChannel(ch);
        }
        
        return channelData;
    }
    
    /**
     * Get audio length (number of samples)
     *
     * @return Total number of samples
     */
    public int getLength() {
        return samples.length();
    }
    
    /**
     * Get number of samples per channel
     *
     * @return Number of samples per channel
     */
    public int getSamplesPerChannel() {
        return samples.length() / channels;
    }
    
    /**
     * Check if mono
     *
     * @return True if mono
     */
    public boolean isMono() {
        return channels == 1;
    }
    
    /**
     * Check if stereo
     *
     * @return True if stereo
     */
    public boolean isStereo() {
        return channels == 2;
    }
    
    /**
     * Extract audio segment
     *
     * @param startTime Start time (seconds)
     * @param endTime End time (seconds)
     * @return Audio segment
     */
    public AudioData extractSegment(double startTime, double endTime) {
        if (startTime < 0 || endTime > duration || startTime >= endTime) {
            throw new IllegalArgumentException("Invalid time range: start=" + startTime + ", end=" + endTime + ", duration=" + duration);
        }
        
        int startSample = (int) (startTime * sampleRate * channels);
        int endSample = (int) (endTime * sampleRate * channels);
        
        // Ensure indices are within valid range
        startSample = Math.max(0, Math.min(startSample, samples.length() - 1));
        endSample = Math.max(startSample + 1, Math.min(endSample, samples.length()));
        
        // Extract segment data
        int segmentLength = endSample - startSample;
        double[] segmentArray = new double[segmentLength];
        
        for (int i = 0; i < segmentLength; i++) {
            segmentArray[i] = samples.get(startSample + i);
        }
        
        IVector<Double> segmentSamples = Linalg.vector(segmentArray);
        
        return new AudioData(segmentSamples, sampleRate, channels, bitDepth, format);
    }
    
    /**
     * Get audio amplitude at specific time
     *
     * @param time Time in seconds
     * @return Amplitude value
     */
    public double getAmplitudeAtTime(double time) {
        if (time < 0 || time >= duration) {
            throw new IllegalArgumentException("Time out of range: " + time);
        }
        
        int sampleIndex = (int) (time * sampleRate * channels);
        sampleIndex = Math.max(0, Math.min(sampleIndex, samples.length() - 1));
        
        return samples.get(sampleIndex);
    }
    
    /**
     * Get maximum amplitude
     *
     * @return Maximum amplitude value
     */
    public double getMaxAmplitude() {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < samples.length(); i++) {
            double value = Math.abs(samples.get(i));
            if (value > max) {
                max = value;
            }
        }
        return max;
    }
    
    /**
     * Get minimum amplitude
     *
     * @return Minimum amplitude value
     */
    public double getMinAmplitude() {
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < samples.length(); i++) {
            double value = samples.get(i);
            if (value < min) {
                min = value;
            }
        }
        return min;
    }
    
    /**
     * Get RMS (Root Mean Square) amplitude
     *
     * @return RMS amplitude
     */
    public double getRMSAmplitude() {
        double sum = 0.0;
        for (int i = 0; i < samples.length(); i++) {
            double value = samples.get(i);
            sum += value * value;
        }
        return Math.sqrt(sum / samples.length());
    }
    
    /**
     * Get average amplitude
     *
     * @return Average amplitude
     */
    public double getAverageAmplitude() {
        double sum = 0.0;
        for (int i = 0; i < samples.length(); i++) {
            sum += Math.abs(samples.get(i));
        }
        return sum / samples.length();
    }
    
    /**
     * Normalize audio data to specified range
     *
     * @param targetMax Target maximum amplitude
     * @return Normalized audio data
     */
    public AudioData normalize(double targetMax) {
        if (targetMax <= 0) {
            throw new IllegalArgumentException("Target maximum must be positive");
        }
        
        double currentMax = getMaxAmplitude();
        if (currentMax == 0) {
            return this; // Avoid division by zero
        }
        
        double scaleFactor = targetMax / currentMax;
        double[] normalizedArray = new double[samples.length()];
        
        for (int i = 0; i < samples.length(); i++) {
            normalizedArray[i] = samples.get(i) * scaleFactor;
        }
        
        IVector<Double> normalizedSamples = Linalg.vector(normalizedArray);
        return new AudioData(normalizedSamples, sampleRate, channels, bitDepth, format);
    }
    
    /**
     * Apply fade in effect
     *
     * @param fadeTime Fade time in seconds
     * @return Audio data with fade in effect
     */
    public AudioData fadeIn(double fadeTime) {
        if (fadeTime <= 0 || fadeTime >= duration) {
            throw new IllegalArgumentException("Invalid fade time: " + fadeTime);
        }
        
        int fadeSamples = (int) (fadeTime * sampleRate * channels);
        double[] fadedArray = new double[samples.length()];
        
        for (int i = 0; i < samples.length(); i++) {
            double multiplier = 1.0;
            if (i < fadeSamples) {
                multiplier = (double) i / fadeSamples;
            }
            fadedArray[i] = samples.get(i) * multiplier;
        }
        
        IVector<Double> fadedSamples = Linalg.vector(fadedArray);
        return new AudioData(fadedSamples, sampleRate, channels, bitDepth, format);
    }
    
    /**
     * Apply fade out effect
     *
     * @param fadeTime Fade time in seconds
     * @return Audio data with fade out effect
     */
    public AudioData fadeOut(double fadeTime) {
        if (fadeTime <= 0 || fadeTime >= duration) {
            throw new IllegalArgumentException("Invalid fade time: " + fadeTime);
        }
        
        int fadeSamples = (int) (fadeTime * sampleRate * channels);
        double[] fadedArray = new double[samples.length()];
        
        for (int i = 0; i < samples.length(); i++) {
            double multiplier = 1.0;
            if (i >= samples.length() - fadeSamples) {
                multiplier = (double) (samples.length() - i) / fadeSamples;
            }
            fadedArray[i] = samples.get(i) * multiplier;
        }
        
        IVector<Double> fadedSamples = Linalg.vector(fadedArray);
        return new AudioData(fadedSamples, sampleRate, channels, bitDepth, format);
    }
    
    /**
     * Mix with another audio data
     *
     * @param other Other audio data to mix with
     * @param ratio Mix ratio (0.0 = only this, 1.0 = only other)
     * @return Mixed audio data
     */
    public AudioData mixWith(AudioData other, double ratio) {
        if (other == null) {
            throw new IllegalArgumentException("Other audio data cannot be null");
        }
        if (ratio < 0.0 || ratio > 1.0) {
            throw new IllegalArgumentException("Mix ratio must be between 0.0 and 1.0");
        }
        if (this.sampleRate != other.sampleRate || this.channels != other.channels) {
            throw new IllegalArgumentException("Audio data must have same sample rate and channels");
        }
        
        int minLength = Math.min(this.samples.length(), other.samples.length());
        double[] mixedArray = new double[minLength];
        
        for (int i = 0; i < minLength; i++) {
            mixedArray[i] = this.samples.get(i) * (1.0 - ratio) + other.samples.get(i) * ratio;
        }
        
        IVector<Double> mixedSamples = Linalg.vector(mixedArray);
        return new AudioData(mixedSamples, sampleRate, channels, bitDepth, format);
    }
    
    /**
     * Get basic audio statistics
     *
     * @return Audio statistics
     */
    public AudioStatistics getStatistics() {
        return new AudioStatistics(samples);
    }
    
    /**
     * Convert to mono by averaging channels
     *
     * @return Mono audio data
     */
    public AudioData toMono() {
        if (isMono()) {
            return this;
        }
        
        int samplesPerChannel = getSamplesPerChannel();
        double[] monoArray = new double[samplesPerChannel];
        
        for (int i = 0; i < samplesPerChannel; i++) {
            double sum = 0.0;
            for (int ch = 0; ch < channels; ch++) {
                sum += samples.get(i * channels + ch);
            }
            monoArray[i] = sum / channels;
        }
        
        IVector<Double> monoSamples = Linalg.vector(monoArray);
        return new AudioData(monoSamples, sampleRate, 1, bitDepth, format);
    }
    
    /**
     * Clone audio data
     *
     * @return Cloned audio data
     */
    public AudioData clone() {
        double[] clonedArray = new double[samples.length()];
        for (int i = 0; i < samples.length(); i++) {
            clonedArray[i] = samples.get(i);
        }
        
        IVector<Double> clonedSamples = Linalg.vector(clonedArray);
        return new AudioData(clonedSamples, sampleRate, channels, bitDepth, format);
    }
    
    /**
     * Check if audio data is valid
     *
     * @return True if valid
     */
    public boolean isValid() {
        return samples != null && samples.length() > 0 && 
               sampleRate > 0 && channels > 0 && bitDepth > 0 && 
               format != null && duration >= 0;
    }
    
    /**
     * Get file size estimation in bytes
     *
     * @return Estimated file size
     */
    public long getEstimatedFileSize() {
        return (long) (samples.length() * bitDepth / 8);
    }
    
    @Override
    public String toString() {
        return String.format("AudioData{sampleRate=%.1fHz, channels=%d, bitDepth=%d, duration=%.2fs, samples=%d, format=%s}",
                sampleRate, channels, bitDepth, duration, samples.length(), format);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        AudioData audioData = (AudioData) obj;
        // Check if all elements in the boolean array returned by samples.equals() are true
        boolean[] samplesEqual = samples.equals(audioData.samples);
        boolean allSamplesEqual = true;
        for (boolean b : samplesEqual) {
            if (!b) {
                allSamplesEqual = false;
                break;
            }
        }
        return Double.compare(audioData.sampleRate, sampleRate) == 0 &&
               channels == audioData.channels &&
               bitDepth == audioData.bitDepth &&
               Double.compare(audioData.duration, duration) == 0 &&
               allSamplesEqual &&
               format.equals(audioData.format);
    }
    
    @Override
    public int hashCode() {
        int result = samples.hashCode();
        result = 31 * result + Double.hashCode(sampleRate);
        result = 31 * result + channels;
        result = 31 * result + bitDepth;
        result = 31 * result + Double.hashCode(duration);
        result = 31 * result + format.hashCode();
        return result;
    }
}