package com.reremouse.lab.music;

import com.reremouse.lab.music.analysis.basic.BeatDetectionResult;
import com.reremouse.lab.music.analysis.basic.ChordDetectionResult;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.music.analysis.IMusicAnalyzer;
import com.reremouse.lab.music.analysis.AdvancedMusicAnalyzer;
import com.reremouse.lab.music.analysis.BasicMusicAnalyzer;
import com.reremouse.lab.music.analysis.ComprehensiveMusicAnalyzer;
import com.reremouse.lab.music.analysis.UnifiedMusicAnalysisResult;
import com.reremouse.lab.music.analysis.MusicDetectionResult;
import com.reremouse.lab.music.analysis.advanced.IAdvancedAnalyzer;
import com.reremouse.lab.music.analysis.advanced.EmotionAnalyzer;
import com.reremouse.lab.music.analysis.advanced.GenreAnalyzer;
import com.reremouse.lab.music.analysis.advanced.ComplexityAnalyzer;
import com.reremouse.lab.music.analysis.basic.IBeatAnalyzer;
import com.reremouse.lab.music.analysis.basic.BeatAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.IKeyAnalyzer;
import com.reremouse.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.IChordAnalyzer;
import com.reremouse.lab.music.analysis.basic.ChordAnalyzerImpl;
import com.reremouse.lab.music.analysis.feature.IFeatureExtractor;
import com.reremouse.lab.music.analysis.feature.FeatureExtractorImpl;
import com.reremouse.lab.music.analysis.feature.MusicFeatureResult;
import com.reremouse.lab.music.analysis.feature.RhythmFeatureResult;
import com.reremouse.lab.music.analysis.feature.TonalFeatureResult;
import com.reremouse.lab.music.analysis.feature.StructureFeatureResult;
import com.reremouse.lab.music.analysis.feature.ExpressivenessFeatureResult;
import com.reremouse.lab.music.processing.IMusicProcessor;
import com.reremouse.lab.music.processing.MusicTheoryProcessor;
import com.reremouse.lab.music.processing.Harmonizer;
import com.reremouse.lab.music.processing.Quantizer;
import com.reremouse.lab.music.processing.Transposer;
import com.reremouse.lab.music.generation.IMusicGenerator;
import com.reremouse.lab.music.generation.ChordGenerator;
import com.reremouse.lab.music.generation.IntervalGenerator;
import com.reremouse.lab.music.generation.ScaleGenerator;
import com.reremouse.lab.music.factory.MusicComponentFactory;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.music.filter.IMusicFilter;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

/**
 * 音乐处理的静态工厂入口类 / Music Processing Entry Factory Class
 * <p>
 * 提供统一的音乐处理接口，封装了音乐分析、特征提取等核心功能。
 * Provides a unified music processing interface that encapsulates core functions such as music analysis and feature extraction.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class Musics {
    
    // ========== 音乐分析器创建方法 / Music Analyzer Creation Methods ==========
    
    /**
     * 创建基础音乐分析器 / Create basic music analyzer
     */
    public static BasicMusicAnalyzer createBasicMusicAnalyzer() {
        try {
            return (BasicMusicAnalyzer) MusicComponentFactory.getInstance().createMusicAnalyzer("basic");
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create basic music analyzer", e);
        }
    }
    
    /**
     * 创建综合音乐分析器 / Create comprehensive music analyzer
     */
    public static ComprehensiveMusicAnalyzer createComprehensiveMusicAnalyzer() {
        try {
            return (ComprehensiveMusicAnalyzer) MusicComponentFactory.getInstance().createMusicAnalyzer("comprehensive");
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create comprehensive music analyzer", e);
        }
    }
    
    /**
     * 创建高级音乐分析器 / Create advanced music analyzer
     */
    public static AdvancedMusicAnalyzer createAdvancedMusicAnalyzer() {
        try {
            return (AdvancedMusicAnalyzer) MusicComponentFactory.getInstance().createAdvancedMusicAnalyzer();
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create advanced music analyzer", e);
        }
    }
    
    /**
     * 创建节拍分析器 / Create beat analyzer
     */
    public static IBeatAnalyzer createBeatAnalyzer() {
        try {
            return (IBeatAnalyzer) MusicComponentFactory.getInstance().createBasicAnalyzer("beat");
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create beat analyzer", e);
        }
    }
    
    /**
     * 创建调性分析器 / Create key analyzer
     */
    public static IKeyAnalyzer createKeyAnalyzer() {
        try {
            return (IKeyAnalyzer) MusicComponentFactory.getInstance().createBasicAnalyzer("key");
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create key analyzer", e);
        }
    }
    
    /**
     * 创建和弦分析器 / Create chord analyzer
     */
    public static IChordAnalyzer createChordAnalyzer() {
        try {
            return (IChordAnalyzer) MusicComponentFactory.getInstance().createBasicAnalyzer("chord");
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create chord analyzer", e);
        }
    }
    
    /**
     * 创建情感分析器 / Create emotion analyzer
     */
    public static IAdvancedAnalyzer createEmotionAnalyzer() {
        try {
            return MusicComponentFactory.getInstance().createAdvancedAnalyzer("emotion");
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create emotion analyzer", e);
        }
    }
    
    /**
     * 创建风格分析器 / Create genre analyzer
     */
    public static IAdvancedAnalyzer createGenreAnalyzer() {
        try {
            return MusicComponentFactory.getInstance().createAdvancedAnalyzer("genre");
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create genre analyzer", e);
        }
    }
    
    /**
     * 创建复杂度分析器 / Create complexity analyzer
     */
    public static IAdvancedAnalyzer createComplexityAnalyzer() {
        try {
            return MusicComponentFactory.getInstance().createAdvancedAnalyzer("complexity");
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create complexity analyzer", e);
        }
    }
    
    /**
     * 创建特征提取器 / Create feature extractor
     */
    public static IFeatureExtractor createFeatureExtractor() {
        try {
            return MusicComponentFactory.getInstance().createFeatureExtractor();
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create feature extractor", e);
        }
    }
    
    // ========== 音乐处理器创建方法 / Music Processor Creation Methods ==========
    
    /**
     * 创建音乐处理器 / Create music processor
     *
     * @param processorType 处理器类型 / Processor type
     * @return 音乐处理器实例 / Music processor instance
     */
    public static IMusicProcessor createProcessor(String processorType) {
        try {
            return MusicComponentFactory.getInstance().createProcessor(processorType);
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create processor: " + processorType, e);
        }
    }
    
    /**
     * 创建和声化处理器 / Create harmonizer processor
     */
    public static IMusicProcessor createHarmonizer() {
        return createProcessor("harmonizer");
    }
    
    /**
     * 创建量化处理器 / Create quantizer processor
     */
    public static IMusicProcessor createQuantizer() {
        return createProcessor("quantizer");
    }
    
    /**
     * 创建转调处理器 / Create transposer processor
     */
    public static IMusicProcessor createTransposer() {
        return createProcessor("transposer");
    }
    
    /**
     * 创建音乐理论处理器 / Create music theory processor
     */
    public static IMusicProcessor createMusicTheoryProcessor() {
        return createProcessor("theory_processor");
    }
    
    // ========== 音乐生成器创建方法 / Music Generator Creation Methods ==========
    
    /**
     * 创建音乐生成器 / Create music generator
     *
     * @param generatorType 生成器类型 / Generator type
     * @return 音乐生成器实例 / Music generator instance
     */
    public static Object createGenerator(String generatorType) {
        try {
            // Note: The generator classes don't implement IMusicGenerator interface directly
            // We'll return the actual generator objects
            return MusicComponentFactory.getInstance().createGenerator(generatorType);
        } catch (AudioProcessingException e) {
            // If the factory method fails, we'll try to create the generator directly
            switch (generatorType.toLowerCase()) {
                case "chord_generator":
                    return new com.reremouse.lab.music.generation.ChordGenerator();
                case "interval_generator":
                    return new com.reremouse.lab.music.generation.IntervalGenerator();
                case "scale_generator":
                    return new com.reremouse.lab.music.generation.ScaleGenerator();
                default:
                    throw new RuntimeException("Failed to create generator: " + generatorType, e);
            }
        } catch (Exception e) {
            // If all else fails, throw the exception
            throw new RuntimeException("Failed to create generator: " + generatorType, e);
        }
    }
    
    /**
     * 创建和弦生成器 / Create chord generator
     */
    public static com.reremouse.lab.music.generation.ChordGenerator createChordGenerator() {
        return (com.reremouse.lab.music.generation.ChordGenerator) createGenerator("chord_generator");
    }
    
    /**
     * 创建音程生成器 / Create interval generator
     */
    public static com.reremouse.lab.music.generation.IntervalGenerator createIntervalGenerator() {
        return (com.reremouse.lab.music.generation.IntervalGenerator) createGenerator("interval_generator");
    }
    
    /**
     * 创建音阶生成器 / Create scale generator
     */
    public static com.reremouse.lab.music.generation.ScaleGenerator createScaleGenerator() {
        return (com.reremouse.lab.music.generation.ScaleGenerator) createGenerator("scale_generator");
    }
    
    // ========== 音乐分析方法 / Music Analysis Methods ==========
    
    /**
     * 基础音乐分析 / Basic Music Analysis
     */
    public static UnifiedMusicAnalysisResult basicAnalysis(AudioData audio) {
        try {
            IMusicAnalyzer analyzer = MusicComponentFactory.getInstance().createMusicAnalyzer("basic");
            return (UnifiedMusicAnalysisResult) analyzer.analyzeMusic(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to perform basic music analysis", e);
        }
    }
    
    /**
     * 基础音乐分析（带参数） / Basic Music Analysis (with parameters)
     */
    public static UnifiedMusicAnalysisResult basicAnalysis(AudioData audio, Map<String, Object> parameters) {
        try {
            IMusicAnalyzer analyzer = MusicComponentFactory.getInstance().createMusicAnalyzer("basic");
            return (UnifiedMusicAnalysisResult) analyzer.analyzeMusic(audio, parameters);
        } catch (Exception e) {
            throw new RuntimeException("Failed to perform basic music analysis", e);
        }
    }
    
    /**
     * 音乐特征提取 / Extract Music Features
     */
    public static Map<String, Object> extractMusicFeatures(AudioData audio) {
        try {
            IFeatureExtractor extractor = MusicComponentFactory.getInstance().createFeatureExtractor();
            MusicFeatureResult result = extractor.extractMusicFeatures(audio);
            
            // 将MusicFeatureResult转换为Map
            Map<String, Object> features = new HashMap<>();
            features.put("rhythm", convertRhythmToMap(result.getRhythmFeatures()));
            features.put("tonal", convertTonalToMap(result.getTonalFeatures()));
            features.put("structure", convertStructureToMap(result.getStructureFeatures()));
            features.put("expressiveness", convertExpressivenessToMap(result.getExpressivenessFeatures()));
            
            return features;
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to extract music features", e);
        }
    }
    
    /**
     * 音乐特征提取（带参数） / Extract Music Features (with parameters)
     */
    public static Map<String, Object> extractMusicFeatures(AudioData audio, Map<String, Object> parameters) {
        try {
            IFeatureExtractor extractor = MusicComponentFactory.getInstance().createFeatureExtractor();
            MusicFeatureResult result = extractor.extractMusicFeatures(audio, parameters);
            
            // 将MusicFeatureResult转换为Map
            Map<String, Object> features = new HashMap<>();
            features.put("rhythm", convertRhythmToMap(result.getRhythmFeatures()));
            features.put("tonal", convertTonalToMap(result.getTonalFeatures()));
            features.put("structure", convertStructureToMap(result.getStructureFeatures()));
            features.put("expressiveness", convertExpressivenessToMap(result.getExpressivenessFeatures()));
            
            return features;
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract music features", e);
        }
    }
    
    /**
     * 提取指定音乐特征 / Extract Specific Music Features
     */
    public static Map<String, Object> extractMusicFeatures(AudioData audio, String[] featureNames) {
        try {
            IFeatureExtractor extractor = MusicComponentFactory.getInstance().createFeatureExtractor();
            MusicFeatureResult result = extractor.extractMusicFeatures(audio);
            
            // 将MusicFeatureResult转换为Map，只返回请求的特征
            Map<String, Object> features = new HashMap<>();
            
            for (String featureName : featureNames) {
                switch (featureName.toLowerCase()) {
                    case "rhythm":
                        features.put("rhythm", convertRhythmToMap(result.getRhythmFeatures()));
                        break;
                    case "tonal":
                        features.put("tonal", convertTonalToMap(result.getTonalFeatures()));
                        break;
                    case "structure":
                        features.put("structure", convertStructureToMap(result.getStructureFeatures()));
                        break;
                    case "expressiveness":
                        features.put("expressiveness", convertExpressivenessToMap(result.getExpressivenessFeatures()));
                        break;
                }
            }
            
            return features;
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract music features", e);
        }
    }
    
    /**
     * 提取指定音乐特征（带参数） / Extract Specific Music Features (with parameters)
     */
    public static Map<String, Object> extractMusicFeatures(AudioData audio, String[] featureNames, Map<String, Object> parameters) {
        try {
            IFeatureExtractor extractor = MusicComponentFactory.getInstance().createFeatureExtractor();
            MusicFeatureResult result = extractor.extractMusicFeatures(audio, parameters);
            
            // 将MusicFeatureResult转换为Map，只返回请求的特征
            Map<String, Object> features = new HashMap<>();
            
            for (String featureName : featureNames) {
                switch (featureName.toLowerCase()) {
                    case "rhythm":
                        if (result.getRhythmFeatures() != null) {
                            features.put("rhythm", convertRhythmToMap(result.getRhythmFeatures()));
                        }
                        break;
                    case "tonal":
                        if (result.getTonalFeatures() != null) {
                            features.put("tonal", convertTonalToMap(result.getTonalFeatures()));
                        }
                        break;
                    case "structure":
                        if (result.getStructureFeatures() != null) {
                            features.put("structure", convertStructureToMap(result.getStructureFeatures()));
                        }
                        break;
                    case "expressiveness":
                        if (result.getExpressivenessFeatures() != null) {
                            features.put("expressiveness", convertExpressivenessToMap(result.getExpressivenessFeatures()));
                        }
                        break;
                }
            }
            
            return features;
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract music features", e);
        }
    }
    
    // ========== 音乐特征提取辅助方法 / Music Feature Extraction Helper Methods ==========
    
    private static Map<String, Object> convertRhythmToMap(RhythmFeatureResult rhythmFeatures) {
        Map<String, Object> map = new HashMap<>();
        if (rhythmFeatures != null) {
            // 核心节拍特征 / Core rhythm features
            map.put("节拍速度", rhythmFeatures.getTempo());                    // Tempo (BPM)
            map.put("节拍强度", rhythmFeatures.getBeatStrength());             // Beat Strength (0.0-1.0)
            map.put("节奏规律性", rhythmFeatures.getRhythmRegularity());        // Rhythm Regularity (0.0-1.0)
            map.put("切分音程度", rhythmFeatures.getSyncopation());             // Syncopation Level (0.0-1.0)
            
            // 节拍分类特征 / Beat classification features
            map.put("快节拍", rhythmFeatures.isFastTempo());                   // Fast Tempo (boolean)
            map.put("慢节拍", rhythmFeatures.isSlowTempo());                   // Slow Tempo (boolean)
            
            // 数组特征（仅保留长度信息，避免过大的数据） / Array features (length only)
            map.put("节奏模式复杂度", rhythmFeatures.getRhythmPattern().length); // Rhythm Pattern Complexity
            map.put("起始点数量", rhythmFeatures.getOnsetTimes().length);        // Number of Onsets
            
            // 元数据 / Metadata
            map.put("置信度", rhythmFeatures.getConfidence());                 // Confidence (0.0-1.0)
            
            // 计算的衍生特征 / Derived features
            double tempo = rhythmFeatures.getTempo();
            if (tempo > 0) {
                if (tempo < 60) {
                    map.put("节拍类型", "极慢");      // Very Slow
                } else if (tempo < 90) {
                    map.put("节拍类型", "慢");        // Slow
                } else if (tempo < 120) {
                    map.put("节拍类型", "中等");      // Moderate
                } else if (tempo < 150) {
                    map.put("节拍类型", "快");        // Fast
                } else {
                    map.put("节拍类型", "极快");      // Very Fast
                }
            } else {
                map.put("节拍类型", "未知");          // Unknown
            }
        }
        return map;
    }
    
    private static Map<String, Object> convertTonalToMap(TonalFeatureResult tonalFeatures) {
        Map<String, Object> map = new HashMap<>();
        if (tonalFeatures != null) {
            // 核心调性特征 / Core tonal features
            map.put("调性", tonalFeatures.getKey());                          // Key (e.g., "C", "F#")
            map.put("调式", tonalFeatures.getMode());                         // Mode (major/minor)
            map.put("完整调性", tonalFeatures.getFullKey());                   // Full Key (e.g., "C major")
            map.put("调性强度", tonalFeatures.getKeyStrength());               // Key Strength (0.0-1.0)
            map.put("调性稳定性", tonalFeatures.getTonalStability());          // Tonal Stability (0.0-1.0)
            map.put("和声复杂度", tonalFeatures.getHarmonicComplexity());       // Harmonic Complexity (0.0-1.0)
            
            // 调性分类特征 / Tonal classification features
            map.put("大调", tonalFeatures.isMajorKey());                      // Is Major Key (boolean)
            map.put("小调", tonalFeatures.isMinorKey());                      // Is Minor Key (boolean)
            map.put("强调性", tonalFeatures.isStrongKey());                   // Strong Key (boolean)
            map.put("稳定调性", tonalFeatures.isStableTonality());            // Stable Tonality (boolean)
            
            // 和弦相关特征 / Chord-related features
            map.put("检测和弦数量", tonalFeatures.getDetectedChords().length);  // Number of Detected Chords
            map.put("和弦进行复杂度", tonalFeatures.getChordProgression().length); // Chord Progression Complexity
            
            // 高级特征（仅保留维度信息） / Advanced features (dimensions only)
            map.put("色度向量维度", tonalFeatures.getChromaVector().length);     // Chroma Vector Dimension
            map.put("Tonnetz特征维度", tonalFeatures.getTonnetzFeatures().length); // Tonnetz Features Dimension
            map.put("HPCP特征维度", tonalFeatures.getHpcpFeatures().length);    // HPCP Features Dimension
            map.put("音调稳定性特征维度", tonalFeatures.getPitchStabilityFeatures().length); // Pitch Stability Features Dimension
            
            // 元数据 / Metadata
            map.put("置信度", tonalFeatures.getConfidence());                 // Confidence (0.0-1.0)
            
            // 衍生特征 / Derived features
            String key = tonalFeatures.getKey();
            if (key != null && !key.isEmpty()) {
                // 判断调性类型
                if (key.contains("#") || key.contains("b")) {
                    map.put("变化音调性", true);                              // Chromatic Key
                } else {
                    map.put("自然音调性", true);                              // Natural Key
                }
                
                // 调性复杂度评估
                double complexity = tonalFeatures.getHarmonicComplexity();
                if (complexity < 0.3) {
                    map.put("和声复杂度等级", "简单");                         // Simple
                } else if (complexity < 0.7) {
                    map.put("和声复杂度等级", "中等");                         // Moderate
                } else {
                    map.put("和声复杂度等级", "复杂");                         // Complex
                }
            }
        }
        return map;
    }
    
    private static Map<String, Object> convertStructureToMap(StructureFeatureResult structureFeatures) {
        Map<String, Object> map = new HashMap<>();
        if (structureFeatures != null) {
            // 核心结构特征 / Core structural features
            map.put("结构复杂度", structureFeatures.getStructuralComplexity());    // Structural Complexity (0.0-1.0)
            map.put("重复性", structureFeatures.getRepetitiveness());             // Repetitiveness (0.0-1.0)
            map.put("平均段落长度", structureFeatures.getAverageSegmentLength());   // Average Segment Length (seconds)
            
            // 段落分析特征 / Segment analysis features
            map.put("段落数量", structureFeatures.getSegments().size());          // Number of Segments
            map.put("估计章节数量", structureFeatures.getEstimatedSections()); // Number of Estimated Sections
            
            // 结构分类特征 / Structural classification features
            map.put("简单结构", structureFeatures.isSimpleStructure());           // Simple Structure (boolean)
            map.put("复杂结构", structureFeatures.isComplexStructure());          // Complex Structure (boolean)
            map.put("高重复性", structureFeatures.isHighlyRepetitive());          // Highly Repetitive (boolean)
            map.put("低重复性", structureFeatures.isLowRepetitive());             // Low Repetitive (boolean)
            
            // 段落类型统计 / Segment type statistics
            java.util.List<StructureFeatureResult.MusicSegment> segments = structureFeatures.getSegments();
            if (segments != null && segments.size() > 0) {
                int introCount = 0, verseCount = 0, chorusCount = 0, bridgeCount = 0, outroCount = 0, otherCount = 0;
                
                for (StructureFeatureResult.MusicSegment segment : segments) {
                    switch (segment.getType()) {
                        case INTRO:
                            introCount++;
                            break;
                        case VERSE:
                            verseCount++;
                            break;
                        case CHORUS:
                            chorusCount++;
                            break;
                        case BRIDGE:
                            bridgeCount++;
                            break;
                        case OUTRO:
                            outroCount++;
                            break;
                        default:
                            otherCount++;
                            break;
                    }
                }
                
                map.put("前奏段数", introCount);                                  // Number of Intro Segments
                map.put("主歌段数", verseCount);                                  // Number of Verse Segments
                map.put("副歌段数", chorusCount);                                 // Number of Chorus Segments
                map.put("桥段数", bridgeCount);                                   // Number of Bridge Segments
                map.put("尾奏段数", outroCount);                                  // Number of Outro Segments
                map.put("其他段数", otherCount);                                  // Number of Other Segments
                
                // 结构完整性评估
                boolean hasIntro = introCount > 0;
                boolean hasVerse = verseCount > 0;
                boolean hasChorus = chorusCount > 0;
                boolean hasOutro = outroCount > 0;
                
                map.put("有前奏", hasIntro);                                      // Has Intro
                map.put("有主歌", hasVerse);                                      // Has Verse
                map.put("有副歌", hasChorus);                                     // Has Chorus
                map.put("有尾奏", hasOutro);                                      // Has Outro
                map.put("结构完整", hasIntro && hasVerse && hasChorus && hasOutro); // Complete Structure
            }
            
            // 高级特征（仅保留维度信息） / Advanced features (dimensions only)
            map.put("新颖性函数维度", structureFeatures.getNoveltyFunction().length); // Novelty Function Dimension
            map.put("自相似矩阵维度", structureFeatures.getSelfSimilarityMatrix().length); // Self-Similarity Matrix Dimension
            
            // 元数据 / Metadata
            map.put("置信度", structureFeatures.getConfidence());               // Confidence (0.0-1.0)
            
            // 衍生特征 / Derived features
            double complexity = structureFeatures.getStructuralComplexity();
            if (complexity < 0.3) {
                map.put("结构复杂度等级", "简单");                                // Simple
            } else if (complexity < 0.7) {
                map.put("结构复杂度等级", "中等");                                // Moderate
            } else {
                map.put("结构复杂度等级", "复杂");                                // Complex
            }
            
            double repetitiveness = structureFeatures.getRepetitiveness();
            if (repetitiveness < 0.3) {
                map.put("重复性等级", "低");                                      // Low
            } else if (repetitiveness < 0.7) {
                map.put("重复性等级", "中");                                      // Medium
            } else {
                map.put("重复性等级", "高");                                      // High
            }
        }
        return map;
    }
    
    private static Map<String, Object> convertExpressivenessToMap(ExpressivenessFeatureResult expressivenessFeatures) {
        Map<String, Object> map = new HashMap<>();
        if (expressivenessFeatures != null) {
            // 核心音乐表现力特征 / Core musical expressiveness features
            map.put("音乐能量", expressivenessFeatures.getEnergy());              // Musical Energy (0.0-1.0)
            map.put("可舞性", expressivenessFeatures.getDanceability());          // Danceability (0.0-1.0)
            map.put("情感强度", expressivenessFeatures.getEmotionalIntensity());   // Emotional Intensity (0.0-1.0)
            map.put("原声性", expressivenessFeatures.getAcousticness());          // Acousticness (0.0-1.0)
            
            // 情感维度特征 / Emotional dimension features
            map.put("效价", expressivenessFeatures.getValence());                // Valence (-1.0 to 1.0)
            map.put("唤醒度", expressivenessFeatures.getArousal());              // Arousal (0.0-1.0)
            map.put("支配度", expressivenessFeatures.getDominance());            // Dominance (0.0-1.0)
            
            // 音乐风格特征 / Musical style features
            map.put("器乐性", expressivenessFeatures.getInstrumentalness());     // Instrumentalness (0.0-1.0)
            map.put("现场感", expressivenessFeatures.getLiveness());             // Liveness (0.0-1.0)
            map.put("语音性", expressivenessFeatures.getSpeechiness());          // Speechiness (0.0-1.0)
            
            // 综合评分和元数据 / Overall scores and metadata
            map.put("综合表现力评分", expressivenessFeatures.getOverallExpressivenessScore());
            map.put("预测情绪", expressivenessFeatures.getPredictedMood());
            map.put("情感象限", expressivenessFeatures.getEmotionalQuadrant());
            map.put("置信度", expressivenessFeatures.getConfidence());
            
            // 布尔判断特征 / Boolean judgment features
            map.put("正面情感", expressivenessFeatures.isPositiveEmotion());
            map.put("高能量音乐", expressivenessFeatures.isHighEnergy());
            map.put("适合跳舞", expressivenessFeatures.isSuitableForDancing());
            map.put("原声音乐", expressivenessFeatures.isAcousticMusic());
            map.put("器乐音乐", expressivenessFeatures.isInstrumentalMusic());
        }
        return map;
    }
    
    /**
     * 提取MFCC特征 / Extract MFCC Features
     */
    public static Map<String, Object> extractMFCCFeatures(AudioData audio) {
        try {
            IFeatureExtractor extractor = MusicComponentFactory.getInstance().createFeatureExtractor();
            MusicFeatureResult result = extractor.extractMusicFeatures(audio);
            
            // Return MFCC features from the result
            Map<String, Object> features = new HashMap<>();
            if (result.getExpressivenessFeatures() != null) {
                features.put("mfcc", result.getExpressivenessFeatures().getTimbreVariation());
            }
            return features;
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract MFCC features", e);
        }
    }
    
    /**
     * 提取MFCC特征（带参数） / Extract MFCC Features (with parameters)
     */
    public static Map<String, Object> extractMFCCFeatures(AudioData audio, Map<String, Object> parameters) {
        try {
            IFeatureExtractor extractor = MusicComponentFactory.getInstance().createFeatureExtractor();
            MusicFeatureResult result = extractor.extractMusicFeatures(audio, parameters);
            
            // Return MFCC features from the result
            Map<String, Object> features = new HashMap<>();
            if (result.getExpressivenessFeatures() != null) {
                features.put("mfcc", result.getExpressivenessFeatures().getTimbreVariation());
            }
            return features;
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract MFCC features", e);
        }
    }
    
    /**
     * 提取时域特征 / Extract Time Domain Features
     */
    public static Map<String, Object> extractTimeDomainFeatures(AudioData audio) {
        try {
            IFeatureExtractor extractor = MusicComponentFactory.getInstance().createFeatureExtractor();
            MusicFeatureResult result = extractor.extractMusicFeatures(audio);
            
            // Return time domain features from the result
            Map<String, Object> features = new HashMap<>();
            if (result.getRhythmFeatures() != null) {
                features.put("beatStrength", result.getRhythmFeatures().getBeatStrength());
                features.put("rhythmRegularity", result.getRhythmFeatures().getRhythmRegularity());
                // Add other time domain features as needed
            }
            return features;
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract time domain features", e);
        }
    }
    
    /**
     * 提取频域特征 / Extract Frequency Domain Features
     */
    public static Map<String, Object> extractFrequencyDomainFeatures(AudioData audio) {
        try {
            IFeatureExtractor extractor = MusicComponentFactory.getInstance().createFeatureExtractor();
            MusicFeatureResult result = extractor.extractMusicFeatures(audio);
            
            // Return frequency domain features from the result
            Map<String, Object> features = new HashMap<>();
            if (result.getTonalFeatures() != null) {
                features.put("tonalCentroid", result.getTonalFeatures().getTonalCentroid());
                features.put("harmonicComplexity", result.getTonalFeatures().getHarmonicComplexity());
                // Add other frequency domain features as needed
            }
            return features;
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract frequency domain features", e);
        }
    }
    
    /**
     * 提取色度特征 / Extract Chroma Features
     */
    public static Map<String, Object> extractChromaFeatures(AudioData audio) {
        try {
            IFeatureExtractor extractor = MusicComponentFactory.getInstance().createFeatureExtractor();
            MusicFeatureResult result = extractor.extractMusicFeatures(audio);
            
            // Return chroma features from the result
            Map<String, Object> features = new HashMap<>();
            if (result.getTonalFeatures() != null) {
                features.put("chromaVector", result.getTonalFeatures().getChromaVector());
                features.put("chordProgression", result.getTonalFeatures().getChordProgression());
                // Add other chroma features as needed
            }
            return features;
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract chroma features", e);
        }
    }
    
    /**
     * 综合音乐分析 / Comprehensive Music Analysis
     */
    public static UnifiedMusicAnalysisResult comprehensiveAnalysis(AudioData audio) {
        try {
            IMusicAnalyzer analyzer = MusicComponentFactory.getInstance().createMusicAnalyzer("comprehensive");
            return (UnifiedMusicAnalysisResult) analyzer.analyzeMusic(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to perform comprehensive music analysis", e);
        }
    }
    
    /**
     * 综合音乐分析（带参数） / Comprehensive Music Analysis (with parameters)
     */
    public static UnifiedMusicAnalysisResult comprehensiveAnalysis(AudioData audio, Map<String, Object> parameters) {
        try {
            IMusicAnalyzer analyzer = MusicComponentFactory.getInstance().createMusicAnalyzer("comprehensive");
            return (UnifiedMusicAnalysisResult) analyzer.analyzeMusic(audio, parameters);
        } catch (Exception e) {
            throw new RuntimeException("Failed to perform comprehensive music analysis", e);
        }
    }
    
    /**
     * 高级音乐分析 / Advanced Music Analysis
     */
    public static UnifiedMusicAnalysisResult advancedAnalysis(AudioData audio) {
        try {
            IAdvancedAnalyzer analyzer = MusicComponentFactory.getInstance().createAdvancedMusicAnalyzer();
            return (UnifiedMusicAnalysisResult) analyzer.analyzeAdvancedMusic(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to perform advanced music analysis", e);
        }
    }
    
    /**
     * 高级音乐分析（带参数） / Advanced Music Analysis (with parameters)
     */
    public static UnifiedMusicAnalysisResult advancedAnalysis(AudioData audio, Map<String, Object> parameters) {
        try {
            IAdvancedAnalyzer analyzer = MusicComponentFactory.getInstance().createAdvancedMusicAnalyzer();
            return (UnifiedMusicAnalysisResult) analyzer.analyzeAdvancedMusic(audio, parameters);
        } catch (Exception e) {
            throw new RuntimeException("Failed to perform advanced music analysis", e);
        }
    }
    
    /**
     * 情感分析 / Emotion Analysis
     */
    public static MusicDetectionResult analyzeEmotion(AudioData audio) {
        try {
            IAdvancedAnalyzer analyzer = MusicComponentFactory.getInstance().createAdvancedAnalyzer("emotion");
            return analyzer.analyzeAdvancedMusic(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to analyze emotion", e);
        }
    }
    
    /**
     * 风格分析 / Genre Analysis
     */
    public static MusicDetectionResult analyzeGenre(AudioData audio) {
        try {
            IAdvancedAnalyzer analyzer = MusicComponentFactory.getInstance().createAdvancedAnalyzer("genre");
            return analyzer.analyzeAdvancedMusic(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to analyze genre", e);
        }
    }
    
    /**
     * 复杂度分析 / Complexity Analysis
     */
    public static MusicDetectionResult analyzeComplexity(AudioData audio) {
        try {
            IAdvancedAnalyzer analyzer = MusicComponentFactory.getInstance().createAdvancedAnalyzer("complexity");
            return analyzer.analyzeAdvancedMusic(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to analyze complexity", e);
        }
    }
    
    /**
     * 生成和弦 / Generate Chords
     */
    public static AudioData generateChords(String[] chordProgression, int key, int octave, 
                                         double duration) {
        try {
            ChordGenerator generator = new ChordGenerator();
            // For simplicity, we'll generate the first chord in the progression
            if (chordProgression != null && chordProgression.length > 0) {
                String firstChord = chordProgression[0];
                // Parse chord type from string
                ChordGenerator.ChordType chordType = parseChordType(firstChord);
                return generator.generateChord(key, chordType, octave, duration, 44100.0);
            }
            // Fallback to simple sine wave
            double sampleRate = 44100;
            int numSamples = (int) (duration * sampleRate);
            double[] samples = new double[numSamples];
            for (int i = 0; i < numSamples; i++) {
                samples[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate); // 440Hz sine wave
            }
            // Use Linalg to create IVector from double array
            IVector<Double> sampleVector = Linalg.vector(samples);
            return new AudioData(sampleVector, sampleRate, 1, 16, AudioFormat.WAV);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate chords", e);
        }
    }
    
    /**
     * 生成音程 / Generate Intervals
     */
    public static AudioData generateIntervals(String[] intervals, int rootNote, int octave,
                                            double duration) {
        try {
            IntervalGenerator generator = new IntervalGenerator();
            // For simplicity, we'll generate the first interval
            if (intervals != null && intervals.length > 0) {
                String firstInterval = intervals[0];
                // Parse interval type from string
                IntervalGenerator.IntervalType intervalType = parseIntervalType(firstInterval);
                IntervalGenerator.PlayMode playMode = IntervalGenerator.PlayMode.HARMONIC;
                return generator.generateInterval(rootNote, intervalType, octave, duration, 44100.0, playMode);
            }
            // Fallback to simple sine wave
            double sampleRate = 44100;
            int numSamples = (int) (duration * sampleRate);
            double[] samples = new double[numSamples];
            for (int i = 0; i < numSamples; i++) {
                samples[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate); // 440Hz sine wave
            }
            // Use Linalg to create IVector from double array
            IVector<Double> sampleVector = Linalg.vector(samples);
            return new AudioData(sampleVector, sampleRate, 1, 16, AudioFormat.WAV);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate intervals", e);
        }
    }
    
    /**
     * 生成音阶 / Generate Scales
     */
    public static AudioData generateScales(String[] scales, int rootNote, int octave,
                                         double duration) {
        try {
            ScaleGenerator generator = new ScaleGenerator();
            // For simplicity, we'll generate the first scale
            if (scales != null && scales.length > 0) {
                String firstScale = scales[0];
                // Parse scale type from string
                ScaleGenerator.ScaleType scaleType = parseScaleType(firstScale);
                ScaleGenerator.PlayingPattern pattern = ScaleGenerator.PlayingPattern.ASCENDING;
                return generator.generateScale(rootNote, scaleType, octave, duration, 44100.0, pattern);
            }
            // Fallback to simple sine wave
            double sampleRate = 44100;
            int numSamples = (int) (duration * sampleRate);
            double[] samples = new double[numSamples];
            for (int i = 0; i < numSamples; i++) {
                samples[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate); // 440Hz sine wave
            }
            // Use Linalg to create IVector from double array
            IVector<Double> sampleVector = Linalg.vector(samples);
            return new AudioData(sampleVector, sampleRate, 1, 16, AudioFormat.WAV);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate scales", e);
        }
    }
    
    /**
     * 生成和弦进行 / Generate Chord Progression
     */
    public static AudioData generateChordProgression(String[] chordProgression, int key, int octave, 
                                                   double chordDuration, double sampleRate) {
        try {
            ChordGenerator generator = new ChordGenerator();
            // Convert string array to ChordType array
            ChordGenerator.ChordType[] chordTypes = new ChordGenerator.ChordType[chordProgression.length];
            for (int i = 0; i < chordProgression.length; i++) {
                chordTypes[i] = parseChordType(chordProgression[i]);
            }
            return generator.generateChordProgression(chordTypes, key, octave, chordDuration, sampleRate);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate chord progression", e);
        }
    }
    
    /**
     * 生成音程序列 / Generate Interval Sequence
     */
    public static AudioData generateIntervalSequence(String[] intervals, int rootNote, int octave,
                                                   double noteDuration, double sampleRate) {
        try {
            IntervalGenerator generator = new IntervalGenerator();
            // Convert string array to IntervalType array
            IntervalGenerator.IntervalType[] intervalTypes = new IntervalGenerator.IntervalType[intervals.length];
            for (int i = 0; i < intervals.length; i++) {
                intervalTypes[i] = parseIntervalType(intervals[i]);
            }
            IntervalGenerator.PlayMode playMode = IntervalGenerator.PlayMode.MELODIC_ASCENDING;
            return generator.generateIntervalSequence(rootNote, intervalTypes, octave, noteDuration, sampleRate, playMode);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate interval sequence", e);
        }
    }
    
    /**
     * 生成音阶 / Generate Scale
     */
    public static AudioData generateScale(String scaleType, int rootNote, int octave,
                                        double noteDuration, double sampleRate) {
        try {
            ScaleGenerator generator = new ScaleGenerator();
            ScaleGenerator.ScaleType scaleTypeEnum = parseScaleType(scaleType);
            ScaleGenerator.PlayingPattern pattern = ScaleGenerator.PlayingPattern.ASCENDING;
            return generator.generateScale(rootNote, scaleTypeEnum, octave, noteDuration, sampleRate, pattern);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate scale", e);
        }
    }
    
    /**
     * 处理音乐理论 / Process Music Theory
     */
    public static MusicDetectionResult processMusicTheory(AudioData audio) {
        try {
            // Create a simple music detection result
            MusicDetectionResult result = new MusicDetectionResult() {
                @Override
                public String getDescription() {
                    return "Music theory processing result";
                }
            };
            return result;
        } catch (Exception e) {
            throw new RuntimeException("Failed to process music theory", e);
        }
    }
    
    /**
     * 和声化处理 / Harmonize Processing
     */
    public static AudioData harmonize(AudioData audio, int[] chordProgression, double[] timing) {
        try {
            Harmonizer processor = new Harmonizer();
            // Create parameters for harmonization
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("chordProgression", chordProgression);
            parameters.put("timing", timing);
            return processor.process(audio, parameters);
        } catch (Exception e) {
            throw new RuntimeException("Failed to harmonize audio", e);
        }
    }
    
    /**
     * 量化处理 / Quantize Processing
     */
    public static AudioData quantize(AudioData audio, double gridResolution) {
        try {
            Quantizer processor = new Quantizer();
            return processor.quantize(audio, gridResolution);
        } catch (Exception e) {
            throw new RuntimeException("Failed to quantize audio", e);
        }
    }
    
    /**
     * 转调处理 / Transpose Processing
     */
    public static AudioData transpose(AudioData audio, int semitones) {
        try {
            Transposer processor = new Transposer();
            return processor.transpose(audio, semitones);
        } catch (Exception e) {
            throw new RuntimeException("Failed to transpose audio", e);
        }
    }
    
    /**
     * 音乐理论处理 / Music Theory Processing
     */
    public static AudioData applyMusicTheory(AudioData audio, Map<String, Object> parameters) {
        try {
            MusicTheoryProcessor processor = new MusicTheoryProcessor();
            return processor.process(audio, parameters);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply music theory", e);
        }
    }
    
    /**
     * 检测音频中的节拍 / Detect beats in audio
     */
    public static BeatDetectionResult detectBeats(AudioData audio) {
        try {
            // Use the createBasicAnalyzer method instead
            IBeatAnalyzer analyzer = (IBeatAnalyzer) MusicComponentFactory.getInstance().createBasicAnalyzer("beat");
            return analyzer.detectBeats(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to detect beats", e);
        }
    }
    
    /**
     * 检测音频中的调性 / Detect key in audio
     */
    public static KeyDetectionResult detectKey(AudioData audio) {
        try {
            // Use the createBasicAnalyzer method instead
            IKeyAnalyzer analyzer = (IKeyAnalyzer) MusicComponentFactory.getInstance().createBasicAnalyzer("key");
            return analyzer.detectKey(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to detect key", e);
        }
    }
    
    /**
     * 检测音频中的和弦 / Detect chords in audio
     */
    public static ChordDetectionResult detectChords(AudioData audio) {
        try {
            // Use the createBasicAnalyzer method instead
            IChordAnalyzer analyzer = (IChordAnalyzer) MusicComponentFactory.getInstance().createBasicAnalyzer("chord");
            List<ChordDetectionResult> results = analyzer.detectChords(audio);
            return results.isEmpty() ? null : results.get(0);
        } catch (Exception e) {
            throw new RuntimeException("Failed to detect chords", e);
        }
    }
    
    /**
     * 检测音频中的调性名称 / Detect key name in audio
     */
    public static String detectKeyName(AudioData audio) {
        try {
            KeyDetectionResult result = detectKey(audio);
            return result != null ? result.getKeyName() : "Unknown";
        } catch (Exception e) {
            throw new RuntimeException("Failed to detect key name", e);
        }
    }
    
    /**
     * 解析和弦类型 / Parse Chord Type
     */
    private static ChordGenerator.ChordType parseChordType(String chord) {
        if (chord == null || chord.isEmpty()) {
            return ChordGenerator.ChordType.MAJOR;
        }
        
        String upperChord = chord.toUpperCase();
        switch (upperChord) {
            case "MAJOR":
            case "M":
            case "":
                return ChordGenerator.ChordType.MAJOR;
            case "MINOR":
            case "M-":
            case "MIN":
                return ChordGenerator.ChordType.MINOR;
            case "DIMINISHED":
            case "DIM":
                return ChordGenerator.ChordType.DIMINISHED;
            case "AUGMENTED":
            case "AUG":
                return ChordGenerator.ChordType.AUGMENTED;
            case "MAJOR7":
            case "MAJ7":
                return ChordGenerator.ChordType.MAJOR_SEVENTH;
            case "MINOR7":
            case "MIN7":
                return ChordGenerator.ChordType.MINOR_SEVENTH;
            case "DOMINANT7":
            case "DOM7":
                return ChordGenerator.ChordType.DOMINANT_SEVENTH;
            case "DIMINISHED7":
            case "DIM7":
                return ChordGenerator.ChordType.DIMINISHED_SEVENTH;
            case "HALF_DIMINISHED":
            case "HALF-DIM":
                return ChordGenerator.ChordType.HALF_DIMINISHED_SEVENTH;
            case "MAJOR9":
            case "MAJ9":
                return ChordGenerator.ChordType.MAJOR_NINTH;
            case "MINOR9":
            case "MIN9":
                return ChordGenerator.ChordType.MINOR_NINTH;
            case "SUS2":
                return ChordGenerator.ChordType.SUSPENDED_SECOND;
            case "SUS4":
                return ChordGenerator.ChordType.SUSPENDED_FOURTH;
            default:
                return ChordGenerator.ChordType.MAJOR;
        }
    }
    
    /**
     * 解析音程类型 / Parse Interval Type
     */
    private static IntervalGenerator.IntervalType parseIntervalType(String interval) {
        if (interval == null || interval.isEmpty()) {
            return IntervalGenerator.IntervalType.PERFECT_FIFTH;
        }
        
        String upperInterval = interval.toUpperCase();
        switch (upperInterval) {
            case "UNISON":
            case "P1":
                return IntervalGenerator.IntervalType.UNISON;
            case "MINOR_SECOND":
            case "M2":
                return IntervalGenerator.IntervalType.MINOR_SECOND;
            case "MAJOR_SECOND":
            case "MAJ2":
                return IntervalGenerator.IntervalType.MAJOR_SECOND;
            case "MINOR_THIRD":
            case "MI3":
                return IntervalGenerator.IntervalType.MINOR_THIRD;
            case "MAJOR_THIRD":
            case "MAJ3":
                return IntervalGenerator.IntervalType.MAJOR_THIRD;
            case "PERFECT_FOURTH":
            case "P4":
                return IntervalGenerator.IntervalType.PERFECT_FOURTH;
            case "TRITONE":
            case "A4":
                return IntervalGenerator.IntervalType.TRITONE;
            case "PERFECT_FIFTH":
            case "P5":
                return IntervalGenerator.IntervalType.PERFECT_FIFTH;
            case "MINOR_SIXTH":
            case "MI6":
                return IntervalGenerator.IntervalType.MINOR_SIXTH;
            case "MAJOR_SIXTH":
            case "MAJ6":
                return IntervalGenerator.IntervalType.MAJOR_SIXTH;
            case "MINOR_SEVENTH":
            case "MI7":
                return IntervalGenerator.IntervalType.MINOR_SEVENTH;
            case "MAJOR_SEVENTH":
            case "MAJ7":
                return IntervalGenerator.IntervalType.MAJOR_SEVENTH;
            case "OCTAVE":
            case "P8":
                return IntervalGenerator.IntervalType.OCTAVE;
            default:
                return IntervalGenerator.IntervalType.PERFECT_FIFTH;
        }
    }
    
    /**
     * 解析音阶类型 / Parse Scale Type
     */
    private static ScaleGenerator.ScaleType parseScaleType(String scale) {
        if (scale == null || scale.isEmpty()) {
            return ScaleGenerator.ScaleType.MAJOR;
        }
        
        String upperScale = scale.toUpperCase();
        switch (upperScale) {
            case "MAJOR":
                return ScaleGenerator.ScaleType.MAJOR;
            case "NATURAL_MINOR":
            case "MINOR":
                return ScaleGenerator.ScaleType.NATURAL_MINOR;
            case "HARMONIC_MINOR":
                return ScaleGenerator.ScaleType.HARMONIC_MINOR;
            case "MELODIC_MINOR":
                return ScaleGenerator.ScaleType.MELODIC_MINOR;
            case "DORIAN":
                return ScaleGenerator.ScaleType.DORIAN;
            case "PHRYGIAN":
                return ScaleGenerator.ScaleType.PHRYGIAN;
            case "LYDIAN":
                return ScaleGenerator.ScaleType.LYDIAN;
            case "MIXOLYDIAN":
                return ScaleGenerator.ScaleType.MIXOLYDIAN;
            case "LOCRIAN":
                return ScaleGenerator.ScaleType.LOCRIAN;
            case "PENTATONIC_MAJOR":
                return ScaleGenerator.ScaleType.PENTATONIC_MAJOR;
            case "PENTATONIC_MINOR":
                return ScaleGenerator.ScaleType.PENTATONIC_MINOR;
            case "BLUES":
                return ScaleGenerator.ScaleType.BLUES;
            case "WHOLE_TONE":
                return ScaleGenerator.ScaleType.WHOLE_TONE;
            case "CHROMATIC":
                return ScaleGenerator.ScaleType.CHROMATIC;
            case "DIMINISHED":
                return ScaleGenerator.ScaleType.DIMINISHED;
            case "AUGMENTED":
                return ScaleGenerator.ScaleType.AUGMENTED;
            case "HUNGARIAN_MINOR":
                return ScaleGenerator.ScaleType.HUNGARIAN_MINOR;
            case "GYPSY":
                return ScaleGenerator.ScaleType.GYPSY;
            case "ARABIC":
                return ScaleGenerator.ScaleType.ARABIC;
            case "JAPANESE":
                return ScaleGenerator.ScaleType.JAPANESE;
            default:
                return ScaleGenerator.ScaleType.MAJOR;
        }
    }
    
    // ========== 音乐滤波器创建方法 / Music Filter Creation Methods ==========
    
    /**
     * 创建音乐滤波器 / Create music filter
     *
     * @param filterType 滤波器类型 / Filter type
     * @return 音乐滤波器实例 / Music filter instance
     */
    public static IMusicFilter createFilter(String filterType) {
        try {
            return MusicComponentFactory.getInstance().createFilter(filterType);
        } catch (AudioProcessingException e) {
            throw new RuntimeException("Failed to create filter: " + filterType, e);
        }
    }
}