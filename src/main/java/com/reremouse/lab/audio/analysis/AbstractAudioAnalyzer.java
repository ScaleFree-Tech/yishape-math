package com.reremouse.lab.audio.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Collections;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Standardized Audio Analyzer Abstract Base Class
 * <p>
 * Provides basic implementation of audio analyzer interface, including parameter management, validation, and cloning.
 * All concrete audio analyzer implementations should extend this class.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public abstract class AbstractAudioAnalyzer implements IAudioAnalyzer {

    protected String name;
    protected String description;
    protected Map<String, Object> defaultParameters;
    protected Set<String> supportedParameters;
    // Add parameter storage for the setParameter/getParameter methods
    protected final Map<String, Object> parameters = new ConcurrentHashMap<>();

    /**
     * Constructor
     *
     * @param name Analyzer name
     * @param description Analyzer description
     */
    public AbstractAudioAnalyzer(String name, String description) {
        this.name = name;
        this.description = description;
        this.defaultParameters = new HashMap<>();
        this.supportedParameters = new HashSet<>();
        initializeDefaultParameters();
    }

    @Override
    public IVector<Double> extractFeatures(AudioData input) throws AudioProcessingException {
        return extractFeatures(input, getDefaultParameters());
    }

    @Override
    public IVector<Double> extractFeatures(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        // Validate input
        if (!validateInput(input)) {
            throw new AudioProcessingException("Invalid input audio data");
        }

        // Validate parameters
        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters");
        }

        // Perform feature extraction
        return doExtractFeatures(input, parameters);
    }

    @Override
    public Tuple2<IVector<Double>, IVector<Double>> calculateSpectrum(AudioData input) throws AudioProcessingException {
        return calculateSpectrum(input, getDefaultParameters());
    }

    @Override
    public Tuple2<IVector<Double>, IVector<Double>> calculateSpectrum(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        // Validate input
        if (!validateInput(input)) {
            throw new AudioProcessingException("Invalid input audio data");
        }

        // Validate parameters
        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters");
        }

        // Perform spectrum calculation
        return doCalculateSpectrum(input, parameters);
    }

    /**
     * Perform actual feature extraction operation
     * <p>
     * Subclasses must implement this method to provide specific feature extraction logic.
     * </p>
     *
     * @param input Input audio
     * @param parameters Analysis parameters
     * @return Feature vector
     * @throws AudioProcessingException Error occurred during analysis
     */
    protected abstract IVector<Double> doExtractFeatures(AudioData input, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * Perform actual spectrum calculation operation
     * <p>
     * Subclasses must implement this method to provide specific spectrum calculation logic.
     * </p>
     *
     * @param input Input audio
     * @param parameters Analysis parameters
     * @return Tuple of frequencies and magnitudes
     * @throws AudioProcessingException Error occurred during analysis
     */
    protected abstract Tuple2<IVector<Double>, IVector<Double>> doCalculateSpectrum(AudioData input, Map<String, Object> parameters) throws AudioProcessingException;

    @Override
    public String getName() {
        return name;
    }

    @Override
    public String getDescription() {
        return description;
    }

    @Override
    public Set<String> getSupportedParameters() {
        return Collections.unmodifiableSet(supportedParameters);
    }

    @Override
    public Map<String, Object> getDefaultParameters() {
        return Collections.unmodifiableMap(defaultParameters);
    }

    @Override
    public boolean validateInput(AudioData input) {
        // Basic validation
        return input != null && input.getSamples() != null && input.getSamples().length() > 0;
    }

    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        // Basic validation
        if (parameters == null) {
            return true; // null parameters are treated as using default parameters
        }

        // Check if all parameters are in the supported parameter list
        for (String param : parameters.keySet()) {
            if (!supportedParameters.contains(param)) {
                return false;
            }
        }

        return true;
    }

    @Override
    public IAudioAnalyzer clone() {
        try {
            AbstractAudioAnalyzer cloned = (AbstractAudioAnalyzer) super.clone();
            // Deep copy parameter maps
            cloned.defaultParameters = new HashMap<>(this.defaultParameters);
            cloned.supportedParameters = new HashSet<>(this.supportedParameters);
            return cloned;
        } catch (CloneNotSupportedException e) {
            // This shouldn't happen since we implement Cloneable
            throw new RuntimeException("Cloning not supported", e);
        }
    }

    /**
     * Add supported parameter
     *
     * @param paramName Parameter name
     * @param defaultValue Default value
     */
    protected void addSupportedParameter(String paramName, Object defaultValue) {
        supportedParameters.add(paramName);
        if (defaultValue != null) {
            defaultParameters.put(paramName, defaultValue);
        }
    }

    /**
     * Set default parameter
     *
     * @param paramName Parameter name
     * @param defaultValue Default value
     */
    protected void setDefaultParameter(String paramName, Object defaultValue) {
        defaultParameters.put(paramName, defaultValue);
    }

    // Add the missing methods from IAudioAnalyzer

    /**
     * Initialize default parameters - subclasses can override
     */
    protected void initializeDefaultParameters() {
        parameters.put("window_size", 1024);
        parameters.put("hop_size", 512);
        parameters.put("sample_rate", 44100.0);
    }

    @Override
    public void setParameter(String key, Object value) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }

        // Validate parameter
        validateParameter(key, value);

        parameters.put(key, value);
    }

    /**
     * Validate parameter - subclasses can override
     *
     * @param key Parameter key
     * @param value Parameter value
     * @throws IllegalArgumentException Parameter is invalid
     */
    protected void validateParameter(String key, Object value) throws IllegalArgumentException {
        switch (key) {
            case "window_size":
                if (!(value instanceof Integer) || (Integer) value <= 0) {
                    throw new IllegalArgumentException("window_size must be a positive integer");
                }
                break;
            case "hop_size":
                if (!(value instanceof Integer) || (Integer) value <= 0) {
                    throw new IllegalArgumentException("hop_size must be a positive integer");
                }
                break;
            case "sample_rate":
                if (!(value instanceof Number) || ((Number) value).doubleValue() <= 0) {
                    throw new IllegalArgumentException("sample_rate must be a positive number");
                }
                break;
            default:
                // Subclasses can handle other parameters
                break;
        }
    }

    @Override
    public Object getParameter(String key) throws IllegalArgumentException {
        if (!parameters.containsKey(key)) {
            throw new IllegalArgumentException("Unknown parameter: " + key);
        }
        return parameters.get(key);
    }

    @Override
    public void reset() {
        parameters.clear();
        initializeDefaultParameters();
    }

    @Override
    public String[] getSupportedFeatureTypes() {
        // Default implementation - subclasses should override
        return new String[0];
    }

    @Override
    public boolean supportsFeatureType(String featureType) {
        if (featureType == null) {
            return false;
        }

        for (String supportedType : getSupportedFeatureTypes()) {
            if (supportedType.equalsIgnoreCase(featureType)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public int getFeatureDimension(String featureType) {
        if (!supportsFeatureType(featureType)) {
            throw new IllegalArgumentException("Unsupported feature type: " + featureType);
        }

        // Subclasses should override this method to provide accurate feature dimension
        return getDefaultFeatureDimension(featureType);
    }

    /**
     * Get default feature dimension - subclasses should override
     *
     * @param featureType Feature type
     * @return Feature dimension
     */
    protected int getDefaultFeatureDimension(String featureType) {
        // Provide default dimensions for common features
        switch (featureType.toLowerCase()) {
            case "mfcc":
                return 13;
            case "chroma":
                return 12;
            case "spectral_centroid":
                return 1;
            case "spectral_bandwidth":
                return 1;
            case "spectral_rolloff":
                return 1;
            case "zero_crossing_rate":
                return 1;
            case "spectral_contrast":
                return 6;
            default:
                return 1; // Default dimension
        }
    }

    @Override
    public String toString() {
        return String.format("%s{name='%s', description='%s'}",
                getClass().getSimpleName(), name, description);
    }
}