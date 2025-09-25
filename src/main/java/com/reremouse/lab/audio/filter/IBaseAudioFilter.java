package com.reremouse.lab.audio.filter;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.audio.processing.IAudioProcessor;
import java.util.Map;
import java.util.Set;

/**
 * 基础音频滤波器接口 / Base Audio Filter Interface
 * <p>
 * 定义音频滤波器的核心操作，结合了显式参数方法和基于Map的参数管理。
 * Defines core operations for audio filters, combining explicit parameter methods with Map-based parameter management.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public interface IBaseAudioFilter extends IAudioProcessor {
    
    /**
     * 滤波类型枚举 / Filter Type Enum
     */
    enum FilterType {
        LOW_PASS("低通", "Low Pass"),
        HIGH_PASS("高通", "High Pass"),
        BAND_PASS("带通", "Band Pass"),
        BAND_STOP("带阻", "Band Stop"),
        NOTCH("陷波", "Notch"),
        ALL_PASS("全通", "All Pass");
        
        private final String chineseName;
        private final String englishName;
        
        FilterType(String chineseName, String englishName) {
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
     * 滤波音频 / Filter Audio
     * <p>
     * 对输入音频执行滤波操作。
     * Performs filtering operation on input audio.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @return 滤波后的音频 / Filtered audio
     * @throws AudioProcessingException 滤波过程中发生错误 / Error occurred during filtering
     */
    AudioData filter(AudioData input) throws AudioProcessingException;
    
    /**
     * 使用参数滤波音频 / Filter Audio with Parameters
     * <p>
     * 对输入音频执行滤波操作，使用指定的参数。
     * Performs filtering operation on input audio with specified parameters.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @param parameters 滤波参数 / Filtering parameters
     * @return 滤波后的音频 / Filtered audio
     * @throws AudioProcessingException 滤波过程中发生错误 / Error occurred during filtering
     */
    AudioData filter(AudioData input, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 获取滤波器类型 / Get Filter Type
     * 
     * @return 滤波器类型 / Filter type
     */
    FilterType getFilterType();
    
    /**
     * 设置滤波器类型 / Set Filter Type
     * 
     * @param filterType 滤波器类型 / Filter type
     */
    void setFilterType(FilterType filterType);
    
    /**
     * 获取截止频率 / Get Cutoff Frequency
     * 
     * @return 截止频率 / Cutoff frequency
     */
    double getCutoffFrequency();
    
    /**
     * 设置截止频率 / Set Cutoff Frequency
     * 
     * @param cutoffFrequency 截止频率 / Cutoff frequency
     */
    void setCutoffFrequency(double cutoffFrequency);
    
    /**
     * 获取滤波器阶数 / Get Filter Order
     * 
     * @return 滤波器阶数 / Filter order
     */
    int getOrder();
    
    /**
     * 设置滤波器阶数 / Set Filter Order
     * 
     * @param order 滤波器阶数 / Filter order
     */
    void setOrder(int order);
    
    /**
     * 获取滤波器名称 / Get Filter Name
     * 
     * @return 滤波器名称 / Filter name
     */
    String getName();
    
    /**
     * 获取滤波器描述 / Get Filter Description
     * 
     * @return 滤波器描述 / Filter description
     */
    String getDescription();
    
    /**
     * 获取支持的参数 / Get Supported Parameters
     * 
     * @return 支持的参数名称列表 / List of supported parameter names
     */
    Set<String> getSupportedParameters();
    
    /**
     * 获取默认参数 / Get Default Parameters
     * 
     * @return 默认参数映射 / Default parameter mapping
     */
    Map<String, Object> getDefaultParameters();
    
    /**
     * 验证输入音频 / Validate Input Audio
     * <p>
     * 验证输入音频是否符合滤波器的要求。
     * Validates if input audio meets filter requirements.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @return 验证结果 / Validation result
     */
    boolean validateInput(AudioData input);
    
    /**
     * 验证参数 / Validate Parameters
     * <p>
     * 验证参数是否有效。
     * Validates if parameters are valid.
     * </p>
     * 
     * @param parameters 参数映射 / Parameter mapping
     * @return 验证结果 / Validation result
     */
    boolean validateParameters(Map<String, Object> parameters);
    
    /**
     * 克隆滤波器 / Clone Filter
     * <p>
     * 创建滤波器的副本。
     * Creates a copy of filter.
     * </p>
     * 
     * @return 滤波器副本 / Filter copy
     */
    IBaseAudioFilter clone();
}