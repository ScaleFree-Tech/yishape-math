package com.reremouse.lab.audio.core;

import com.reremouse.lab.audio.AudioData;

/**
 * 音频监听器接口 / Audio Listener Interface
 * <p>
 * 定义音频事件监听器的基本操作，用于实现观察者模式。
 * 监听器可以接收音频处理过程中的各种事件通知。
 * </p>
 * <p>
 * Defines basic operations for audio event listeners, used to implement observer pattern.
 * Listeners can receive various event notifications during audio processing.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IAudioListener {
    
    /**
     * 音频事件类型枚举 / Audio Event Type Enum
     */
    enum AudioEventType {
        PROCESSING_STARTED("处理开始", "Processing Started"),
        PROCESSING_COMPLETED("处理完成", "Processing Completed"),
        PROCESSING_FAILED("处理失败", "Processing Failed"),
        PROGRESS_UPDATE("进度更新", "Progress Update"),
        PARAMETER_CHANGED("参数改变", "Parameter Changed"),
        FORMAT_CHANGED("格式改变", "Format Changed"),
        BUFFER_OVERFLOW("缓冲区溢出", "Buffer Overflow"),
        BUFFER_UNDERFLOW("缓冲区下溢", "Buffer Underflow"),
        CLIPPING_DETECTED("削波检测", "Clipping Detected"),
        SILENCE_DETECTED("静音检测", "Silence Detected");
        
        private final String chineseName;
        private final String englishName;
        
        AudioEventType(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
        
        @Override
        public String toString() {
            return chineseName + " / " + englishName;
        }
    }
    
    /**
     * 音频事件类 / Audio Event Class
     */
    class AudioEvent {
        private final AudioEventType eventType;
        private final Object source;
        private final AudioData audioData;
        private final String message;
        private final long timestamp;
        private final Object additionalData;
        
        public AudioEvent(AudioEventType eventType, Object source, AudioData audioData, String message) {
            this(eventType, source, audioData, message, null);
        }
        
        public AudioEvent(AudioEventType eventType, Object source, AudioData audioData, String message, Object additionalData) {
            this.eventType = eventType;
            this.source = source;
            this.audioData = audioData;
            this.message = message;
            this.additionalData = additionalData;
            this.timestamp = System.currentTimeMillis();
        }
        
        public AudioEventType getEventType() { return eventType; }
        public Object getSource() { return source; }
        public AudioData getAudioData() { return audioData; }
        public String getMessage() { return message; }
        public long getTimestamp() { return timestamp; }
        public Object getAdditionalData() { return additionalData; }
        
        @Override
        public String toString() {
            return String.format("AudioEvent{type=%s, source=%s, message='%s', timestamp=%d}",
                               eventType, source.getClass().getSimpleName(), message, timestamp);
        }
    }
    
    /**
     * 处理音频事件 / Handle audio event
     * <p>
     * 当音频事件发生时被调用。
     * Called when audio event occurs.
     * </p>
     *
     * @param event 音频事件 / Audio event
     */
    void onAudioEvent(AudioEvent event);
    
    /**
     * 处理处理开始事件 / Handle processing started event
     * <p>
     * 当音频处理开始时被调用。
     * Called when audio processing starts.
     * </p>
     *
     * @param source 事件源 / Event source
     * @param audioData 音频数据 / Audio data
     */
    default void onProcessingStarted(Object source, AudioData audioData) {
        onAudioEvent(new AudioEvent(AudioEventType.PROCESSING_STARTED, source, audioData, "Processing started"));
    }
    
    /**
     * 处理处理完成事件 / Handle processing completed event
     * <p>
     * 当音频处理完成时被调用。
     * Called when audio processing completes.
     * </p>
     *
     * @param source 事件源 / Event source
     * @param audioData 音频数据 / Audio data
     */
    default void onProcessingCompleted(Object source, AudioData audioData) {
        onAudioEvent(new AudioEvent(AudioEventType.PROCESSING_COMPLETED, source, audioData, "Processing completed"));
    }
    
    /**
     * 处理处理失败事件 / Handle processing failed event
     * <p>
     * 当音频处理失败时被调用。
     * Called when audio processing fails.
     * </p>
     *
     * @param source 事件源 / Event source
     * @param audioData 音频数据 / Audio data
     * @param error 错误信息 / Error information
     */
    default void onProcessingFailed(Object source, AudioData audioData, Throwable error) {
        onAudioEvent(new AudioEvent(AudioEventType.PROCESSING_FAILED, source, audioData, 
                                  "Processing failed: " + error.getMessage(), error));
    }
    
    /**
     * 处理进度更新事件 / Handle progress update event
     * <p>
     * 当处理进度更新时被调用。
     * Called when processing progress updates.
     * </p>
     *
     * @param source 事件源 / Event source
     * @param audioData 音频数据 / Audio data
     * @param progress 进度百分比 (0-100) / Progress percentage (0-100)
     */
    default void onProgressUpdate(Object source, AudioData audioData, double progress) {
        onAudioEvent(new AudioEvent(AudioEventType.PROGRESS_UPDATE, source, audioData, 
                                  String.format("Progress: %.1f%%", progress), progress));
    }
    
    /**
     * 处理参数改变事件 / Handle parameter changed event
     * <p>
     * 当处理器参数改变时被调用。
     * Called when processor parameter changes.
     * </p>
     *
     * @param source 事件源 / Event source
     * @param parameterName 参数名称 / Parameter name
     * @param oldValue 旧值 / Old value
     * @param newValue 新值 / New value
     */
    default void onParameterChanged(Object source, String parameterName, Object oldValue, Object newValue) {
        onAudioEvent(new AudioEvent(AudioEventType.PARAMETER_CHANGED, source, null, 
                                  String.format("Parameter '%s' changed from %s to %s", parameterName, oldValue, newValue)));
    }
    
    /**
     * 处理削波检测事件 / Handle clipping detected event
     * <p>
     * 当检测到音频削波时被调用。
     * Called when audio clipping is detected.
     * </p>
     *
     * @param source 事件源 / Event source
     * @param audioData 音频数据 / Audio data
     * @param clippingLevel 削波程度 / Clipping level
     */
    default void onClippingDetected(Object source, AudioData audioData, double clippingLevel) {
        onAudioEvent(new AudioEvent(AudioEventType.CLIPPING_DETECTED, source, audioData, 
                                  String.format("Clipping detected: %.2f", clippingLevel), clippingLevel));
    }
}