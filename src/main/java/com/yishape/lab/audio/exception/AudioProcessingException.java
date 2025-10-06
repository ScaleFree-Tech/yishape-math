package com.yishape.lab.audio.exception;

/**
 * Audio Processing Exception Class
 * <p>
 * Exception thrown when errors occur during audio processing.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioProcessingException extends Exception {

    private static final long serialVersionUID = 1L;

    /**
     * Default constructor
     */
    public AudioProcessingException() {
        super();
    }

    /**
     * Constructor with error message
     *
     * @param message Error message
     */
    public AudioProcessingException(String message) {
        super(message);
    }

    /**
     * Constructor with error message and cause
     *
     * @param message Error message
     * @param cause Cause
     */
    public AudioProcessingException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * Constructor with cause
     *
     * @param cause Cause
     */
    public AudioProcessingException(Throwable cause) {
        super(cause);
    }
}