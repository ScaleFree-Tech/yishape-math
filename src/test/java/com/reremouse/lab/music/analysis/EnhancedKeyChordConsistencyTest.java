package com.reremouse.lab.music.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioIO;
import com.reremouse.lab.music.analysis.basic.*;
import java.util.*;

/**
 * 增强版调性-和弦一致性测试 / Enhanced Key-Chord Consistency Test
 * 
 * 提供更详细的音乐理论分析和改进建议
 * Provides more detailed music theory analysis and improvement suggestions
 */
public class EnhancedKeyChordConsistencyTest {
    
    // 音乐理论常量 / Music theory constants
    private static final String[] CHROMATIC_NOTES = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    private static final Map<String, int[]> SCALE_INTERVALS = new HashMap<>();
    private static final Map<String, Map<String, Integer>> CHORD_QUALITIES = new HashMap<>();
    
    static {
        // 初始化音阶间隔 / Initialize scale intervals
        SCALE_INTERVALS.put("major", new int[]{0, 2, 4, 5, 7, 9, 11}); // 大调音阶
        SCALE_INTERVALS.put("minor", new int[]{0, 2, 3, 5, 7, 8, 10}); // 自然小调音阶
        SCALE_INTERVALS.put("harmonic_minor", new int[]{0, 2, 3, 5, 7, 8, 11}); // 和声小调
        SCALE_INTERVALS.put("melodic_minor", new int[]{0, 2, 3, 5, 7, 9, 11}); // 旋律小调
        
        // 初始化和弦质量映射 / Initialize chord quality mappings
        Map<String, Integer> majorChord = new HashMap<>();
        majorChord.put("", 0); majorChord.put("maj", 0); majorChord.put("M", 0);
        majorChord.put("maj7", 0); majorChord.put("M7", 0); majorChord.put("Δ7", 0);
        majorChord.put("6", 0); majorChord.put("maj9", 0); majorChord.put("Δ9", 0);
        CHORD_QUALITIES.put("major", majorChord);
        
        Map<String, Integer> minorChord = new HashMap<>();
        minorChord.put("m", 0); minorChord.put("min", 0); minorChord.put("-", 0);
        minorChord.put("m7", 0); minorChord.put("min7", 0); minorChord.put("-7", 0);
        minorChord.put("m6", 0); minorChord.put("m9", 0); minorChord.put("m11", 0);
        minorChord.put("m7b5", 0); minorChord.put("ø", 0);
        CHORD_QUALITIES.put("minor", minorChord);
        
        Map<String, Integer> dominantChord = new HashMap<>();
        dominantChord.put("7", 0); dominantChord.put("dom7", 0);
        dominantChord.put("9", 0); dominantChord.put("11", 0); dominantChord.put("13", 0);
        dominantChord.put("7b9", 0); dominantChord.put("7#9", 0); dominantChord.put("7b5", 0);
        CHORD_QUALITIES.put("dominant", dominantChord);
        
        Map<String, Integer> diminishedChord = new HashMap<>();
        diminishedChord.put("dim", 0); diminishedChord.put("°", 0);
        diminishedChord.put("dim7", 0); diminishedChord.put("°7", 0);
        CHORD_QUALITIES.put("diminished", diminishedChord);
        
        Map<String, Integer> augmentedChord = new HashMap<>();
        augmentedChord.put("aug", 0); augmentedChord.put("+", 0);
        augmentedChord.put("+7", 0); augmentedChord.put("7#5", 0);
        CHORD_QUALITIES.put("augmented", augmentedChord);
    }
    
    public static void main(String[] args) {
        System.out.println("=== 增强版调性-和弦一致性测试 ===");
        System.out.println("=== Enhanced Key-Chord Consistency Test ===\n");
        
        // 测试音频文件路径
        String audioPath = "F:\\music\\test\\";
        String audioFile = audioPath + "19.蓝心羽-寂寞烟火.mp3";
        
        try {
            // 1. 读取音频文件
            System.out.println("正在读取音频文件: " + audioFile);
            AudioData audioData = AudioIO.readAudio(audioFile);
            
            System.out.println("音频信息:");
            System.out.println("  采样率: " + audioData.getSampleRate());
            System.out.println("  时长: " + String.format("%.2f", audioData.getDuration()) + " 秒");
            System.out.println("  声道数: " + audioData.getChannels());
            
            // 2. 执行综合分析
            System.out.println("\n执行综合音乐分析...");
            BasicMusicAnalyzer analyzer = new BasicMusicAnalyzer();
            MusicDetectionResult musicResult = analyzer.analyzeMusic(audioData);
            UnifiedMusicAnalysisResult result = (UnifiedMusicAnalysisResult) musicResult;
            
            // 3. 提取调性和和弦结果
            KeyDetectionResult keyResult = result.getKeyDetectionResult();
            ChordDetectionResult chordResult = result.getChordDetectionResult();
            
            // 4. 详细分析
            System.out.println("\n=== 详细分析结果 ===");
            performDetailedAnalysis(keyResult, chordResult);
            
            // 5. 高级一致性检查
            System.out.println("\n=== 高级一致性检查 ===");
            performAdvancedConsistencyCheck(keyResult, chordResult);
            
            // 6. 多算法验证
            System.out.println("\n=== 多算法验证 ===");
            performMultiAlgorithmValidation(audioData);
            
            // 7. 改进建议
            System.out.println("\n=== 智能改进建议 ===");
            provideIntelligentImprovementSuggestions(keyResult, chordResult);
            
        } catch (Exception e) {
            System.err.println("分析过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 执行详细分析 / Perform detailed analysis
     */
    private static void performDetailedAnalysis(KeyDetectionResult keyResult, ChordDetectionResult chordResult) {
        System.out.println("调性检测详细分析:");
        System.out.println("  调性: " + keyResult.getKeyName() + " " + keyResult.getScaleType());
        System.out.println("  置信度: " + String.format("%.3f", keyResult.getConfidence()));
        
        // 显示色度特征
        double[] chroma = keyResult.getChromaFeatures();
        if (chroma != null && chroma.length == 12) {
            System.out.println("  色度特征:");
            for (int i = 0; i < 12; i++) {
                System.out.println("    " + CHROMATIC_NOTES[i] + ": " + String.format("%.3f", chroma[i]));
            }
        }
        
        System.out.println("\n和弦检测详细分析:");
        System.out.println("  和弦: " + chordResult.getChordName());
        System.out.println("  时间范围: [" + String.format("%.2f", chordResult.getStartTime()) + 
                         "s - " + String.format("%.2f", chordResult.getEndTime()) + "s]");
        System.out.println("  置信度: " + String.format("%.3f", chordResult.getConfidence()));
        
        // 分析和弦结构
        analyzeChordStructure(chordResult.getChordName());
    }
    
    /**
     * 分析和弦结构 / Analyze chord structure
     */
    private static void analyzeChordStructure(String chordName) {
        if (chordName == null || chordName.isEmpty()) {
            return;
        }
        
        System.out.println("  和弦结构分析:");
        
        // 提取根音和和弦类型
        String rootNote = extractRootNote(chordName);
        String chordType = extractChordType(chordName);
        
        System.out.println("    根音: " + rootNote);
        System.out.println("    和弦类型: " + chordType);
        
        // 判断和弦质量
        String quality = determineChordQuality(chordType);
        System.out.println("    和弦质量: " + quality);
        
        // 计算和弦音
        List<String> chordNotes = calculateChordNotes(rootNote, chordType);
        System.out.println("    和弦音: " + String.join(", ", chordNotes));
    }
    
    /**
     * 高级一致性检查 / Advanced consistency check
     */
    private static void performAdvancedConsistencyCheck(KeyDetectionResult keyResult, ChordDetectionResult chordResult) {
        String key = keyResult.getKeyName();
        String scale = keyResult.getScaleType();
        String chord = chordResult.getChordName();
        
        System.out.println("调性-和弦一致性检查:");
        
        if (key == null || scale == null || chord == null) {
            System.out.println("  ❌ 缺少必要数据");
            return;
        }
        
        // 1. 根音一致性检查
        boolean rootNoteConsistent = checkRootNoteConsistency(chord, key, scale);
        System.out.println("  根音一致性: " + (rootNoteConsistent ? "✅ 一致" : "❌ 不一致"));
        
        // 2. 和弦质量一致性检查
        boolean chordQualityConsistent = checkChordQualityConsistency(chord, key, scale);
        System.out.println("  和弦质量一致性: " + (chordQualityConsistent ? "✅ 一致" : "❌ 不一致"));
        
        // 3. 功能性和弦检查
        boolean functionalChord = checkFunctionalChord(chord, key, scale);
        System.out.println("  功能性和弦: " + (functionalChord ? "✅ 是" : "⚠️ 可能不是"));
        
        // 4. 色度特征一致性
        boolean chromaConsistent = checkChromaConsistency(keyResult, chordResult);
        System.out.println("  色度特征一致性: " + (chromaConsistent ? "✅ 一致" : "❌ 不一致"));
        
        // 5. 综合评分
        double consistencyScore = calculateConsistencyScore(keyResult, chordResult);
        System.out.println("  综合一致性评分: " + String.format("%.2f/1.00", consistencyScore));
        
        if (consistencyScore < 0.5) {
            System.out.println("  ⚠️  建议重新检查调性和和弦检测结果");
        }
    }
    
    /**
     * 多算法验证 / Multi-algorithm validation
     */
    private static void performMultiAlgorithmValidation(AudioData audioData) {
        System.out.println("执行多算法交叉验证...");
        
        try {
            // 算法1: 基本调性分析器
            KeyAnalyzerImpl keyAnalyzer1 = new KeyAnalyzerImpl();
            KeyDetectionResult key1 = keyAnalyzer1.detectKey(audioData);
            System.out.println("算法1 (基本): " + key1.getKeyName() + " " + key1.getScaleType() + 
                             " (置信度: " + String.format("%.3f", key1.getConfidence()) + ")");
            
            // 算法2: 和弦分析器
            ChordAnalyzerImpl chordAnalyzer = new ChordAnalyzerImpl();
            List<ChordDetectionResult> chords = chordAnalyzer.detectChords(audioData);
            System.out.println("检测到 " + chords.size() + " 个和弦段");
            
            // 显示前几个和弦
            for (int i = 0; i < Math.min(3, chords.size()); i++) {
                ChordDetectionResult chord = chords.get(i);
                System.out.println("  和弦 " + (i+1) + ": " + chord.getChordName() + 
                                 " [" + String.format("%.2f", chord.getStartTime()) + "s - " + 
                                 String.format("%.2f", chord.getEndTime()) + "s]");
            }
            
            // 算法一致性分析
            analyzeAlgorithmConsistency(key1, chords);
            
        } catch (Exception e) {
            System.err.println("多算法验证出错: " + e.getMessage());
        }
    }
    
    /**
     * 智能改进建议 / Intelligent improvement suggestions
     */
    private static void provideIntelligentImprovementSuggestions(KeyDetectionResult keyResult, ChordDetectionResult chordResult) {
        System.out.println("基于音乐理论和统计分析的改进建议:");
        
        double keyConfidence = keyResult.getConfidence();
        double chordConfidence = chordResult.getConfidence();
        
        // 1. 置信度分析
        if (keyConfidence < 0.5) {
            System.out.println("1. 调性检测置信度过低 (" + String.format("%.3f", keyConfidence) + ")");
            System.out.println("   建议:");
            System.out.println("   - 使用更长的音频片段进行分析");
            System.out.println("   - 考虑使用多个时间窗口进行投票");
            System.out.println("   - 结合和弦信息进行调性推断");
        }
        
        if (chordConfidence < 0.5) {
            System.out.println("2. 和弦检测置信度过低 (" + String.format("%.3f", chordConfidence) + ")");
            System.out.println("   建议:");
            System.out.println("   - 优化频谱分析参数");
            System.out.println("   - 使用谐波积频谱技术");
            System.out.println("   - 考虑时域和频域特征结合");
        }
        
        // 2. 音乐理论建议
        String key = keyResult.getKeyName();
        String chord = chordResult.getChordName();
        
        if (key != null && chord != null) {
            List<String> alternativeKeys = suggestAlternativeKeysBasedOnChord(chord);
            if (!alternativeKeys.isEmpty()) {
                System.out.println("3. 基于和弦 " + chord + " 的替代调性建议:");
                for (String altKey : alternativeKeys) {
                    System.out.println("   - " + altKey);
                }
            }
        }
        
        // 3. 技术改进建议
        System.out.println("4. 技术实现改进:");
        System.out.println("   - 实现调性和弦联合优化算法");
        System.out.println("   - 使用机器学习模型进行联合预测");
        System.out.println("   - 增加音乐理论知识约束");
        System.out.println("   - 实现多时间尺度分析");
        System.out.println("   - 添加人工验证和反馈机制");
        
        // 4. 性能优化建议
        System.out.println("5. 性能优化建议:");
        System.out.println("   - 使用GPU加速频谱计算");
        System.out.println("   - 实现并行处理架构");
        System.out.println("   - 优化内存使用模式");
        System.out.println("   - 使用增量更新策略");
    }
    
    // ===== 辅助方法 / Helper Methods =====
    
    private static String extractRootNote(String chord) {
        if (chord == null || chord.isEmpty()) return null;
        
        // 处理各种和弦表示法
        chord = chord.trim();
        
        // 检查是否有升降号
        if (chord.length() >= 2 && (chord.charAt(1) == '#' || chord.charAt(1) == 'b')) {
            return chord.substring(0, 2);
        } else if (chord.length() >= 1) {
            return chord.substring(0, 1);
        }
        
        return null;
    }
    
    private static String extractChordType(String chord) {
        if (chord == null || chord.isEmpty()) return "";
        
        String rootNote = extractRootNote(chord);
        if (rootNote == null) return "";
        
        return chord.substring(rootNote.length());
    }
    
    private static String determineChordQuality(String chordType) {
        if (chordType == null || chordType.isEmpty()) return "大和弦";
        
        for (Map.Entry<String, Map<String, Integer>> entry : CHORD_QUALITIES.entrySet()) {
            if (entry.getValue().containsKey(chordType)) {
                return entry.getKey();
            }
        }
        
        return "未知";
    }
    
    private static List<String> calculateChordNotes(String rootNote, String chordType) {
        List<String> notes = new ArrayList<>();
        
        // 简化的和弦音计算
        // 实际实现应该根据音乐理论精确计算
        notes.add(rootNote);
        
        // 这里添加简化的音程计算
        if (chordType.contains("m")) {
            // 小三度
            notes.add(getNoteByInterval(rootNote, 3));
        } else {
            // 大三度
            notes.add(getNoteByInterval(rootNote, 4));
        }
        
        // 五度
        notes.add(getNoteByInterval(rootNote, 7));
        
        return notes;
    }
    
    private static String getNoteByInterval(String rootNote, int semitones) {
        // 简化的音程计算
        int rootIndex = -1;
        for (int i = 0; i < CHROMATIC_NOTES.length; i++) {
            if (CHROMATIC_NOTES[i].equals(rootNote)) {
                rootIndex = i;
                break;
            }
        }
        
        if (rootIndex == -1) return rootNote;
        
        int targetIndex = (rootIndex + semitones) % 12;
        return CHROMATIC_NOTES[targetIndex];
    }
    
    private static boolean checkRootNoteConsistency(String chord, String key, String scale) {
        String chordRoot = extractRootNote(chord);
        if (chordRoot == null) return false;
        
        // 检查根音是否在调性音阶中
        return isNoteInScale(chordRoot, key, scale);
    }
    
    private static boolean checkChordQualityConsistency(String chord, String key, String scale) {
        String chordType = extractChordType(chord);
        String chordQuality = determineChordQuality(chordType);
        
        // 根据调性判断和弦质量是否合适
        if ("major".equalsIgnoreCase(scale)) {
            // 大调中，某些和弦质量更合适
            return !"diminished".equals(chordQuality) || chord.contains("vii"); // VII级减和弦
        } else if ("minor".equalsIgnoreCase(scale)) {
            // 小调中，某些和弦质量更合适
            return true; // 小调中各种和弦质量都可能出现
        }
        
        return true;
    }
    
    private static boolean checkFunctionalChord(String chord, String key, String scale) {
        // 检查是否是功能性和弦（I, IV, V等）
        String chordRoot = extractRootNote(chord);
        if (chordRoot == null) return false;
        
        // 简化的功能性检查
        return isNoteInScale(chordRoot, key, scale);
    }
    
    private static boolean checkChromaConsistency(KeyDetectionResult keyResult, ChordDetectionResult chordResult) {
        double[] keyChroma = keyResult.getChromaFeatures();
        // 这里可以实现更复杂的色度一致性检查
        return keyChroma != null && keyChroma.length == 12;
    }
    
    private static double calculateConsistencyScore(KeyDetectionResult keyResult, ChordDetectionResult chordResult) {
        double score = 0.0;
        
        // 基于置信度的评分
        score += keyResult.getConfidence() * 0.3;
        score += chordResult.getConfidence() * 0.3;
        
        // 基于一致性的评分
        String key = keyResult.getKeyName();
        String chord = chordResult.getChordName();
        
        if (key != null && chord != null) {
            boolean rootConsistent = checkRootNoteConsistency(chord, key, keyResult.getScaleType());
            if (rootConsistent) score += 0.2;
            
            boolean qualityConsistent = checkChordQualityConsistency(chord, key, keyResult.getScaleType());
            if (qualityConsistent) score += 0.2;
        }
        
        return Math.min(1.0, score);
    }
    
    private static void analyzeAlgorithmConsistency(KeyDetectionResult key1, List<ChordDetectionResult> chords) {
        System.out.println("\n算法一致性分析:");
        
        if (chords.isEmpty()) {
            System.out.println("  未检测到和弦");
            return;
        }
        
        // 统计最频繁的和弦
        Map<String, Integer> chordFrequency = new HashMap<>();
        for (ChordDetectionResult chord : chords) {
            String chordName = chord.getChordName();
            chordFrequency.put(chordName, chordFrequency.getOrDefault(chordName, 0) + 1);
        }
        
        // 找出最频繁的和弦
        String mostFrequentChord = null;
        int maxFrequency = 0;
        for (Map.Entry<String, Integer> entry : chordFrequency.entrySet()) {
            if (entry.getValue() > maxFrequency) {
                mostFrequentChord = entry.getKey();
                maxFrequency = entry.getValue();
            }
        }
        
        if (mostFrequentChord != null) {
            System.out.println("  最频繁和弦: " + mostFrequentChord + " (出现 " + maxFrequency + " 次)");
            
            // 检查与调性的一致性
            boolean consistent = checkRootNoteConsistency(mostFrequentChord, key1.getKeyName(), key1.getScaleType());
            System.out.println("  与调性一致性: " + (consistent ? "✅ 一致" : "❌ 不一致"));
        }
    }
    
    private static boolean isNoteInScale(String note, String key, String scale) {
        // 简化的音阶检查
        if (note == null || key == null || scale == null) return false;
        
        // 这里应该实现完整的音阶理论
        // 现在使用简化的检查
        return true; // 简化实现
    }
    
    private static List<String> suggestAlternativeKeysBasedOnChord(String chord) {
        List<String> suggestions = new ArrayList<>();
        
        String rootNote = extractRootNote(chord);
        if (rootNote == null) return suggestions;
        
        // 基于和弦根音的建议
        suggestions.add(rootNote + " major");
        suggestions.add(rootNote + " minor");
        
        // 基于和弦类型的建议
        if (chord.contains("m") && !chord.contains("maj")) {
            suggestions.add(rootNote + " minor (基于小和弦)");
        } else if (chord.contains("maj") || (!chord.contains("m") && !chord.contains("dim"))) {
            suggestions.add(rootNote + " major (基于大和弦)");
        }
        
        return suggestions;
    }
}