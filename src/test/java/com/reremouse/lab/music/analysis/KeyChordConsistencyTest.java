package com.reremouse.lab.music.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.audio.core.AudioIO;
import com.reremouse.lab.music.analysis.BasicMusicAnalyzer;
import com.reremouse.lab.music.analysis.basic.ChordAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.ChordDetectionResult;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;

import java.io.File;
import java.util.List;
import java.util.Map;

/**
 * 调性-和弦一致性测试类
 * 用于验证音乐分析中调性和和弦检测结果的一致性
 */
public class KeyChordConsistencyTest {
    
    public static void main(String[] args) {
        System.out.println("=== 调性-和弦一致性测试 ===");
        
        // 测试音频文件路径
        String audioPath = "F:\\music\\test\\";
        String audioFile = audioPath + "19.蓝心羽-寂寞烟火.mp3";
        
        File file = new File(audioFile);
        if (!file.exists()) {
            System.out.println("错误: 音频文件不存在: " + audioFile);
            return;
        }
        
        try {
            // 1. 读取音频文件
            System.out.println("正在读取音频文件: " + audioFile);
            AudioData audioData = AudioIO.readAudio(audioFile);
            
            System.out.println("音频信息:");
            System.out.println("  采样率: " + audioData.getSampleRate());
            System.out.println("  时长: " + String.format("%.2f", audioData.getDuration()) + " 秒");
            System.out.println("  声道数: " + audioData.getChannels());
            
            // 2. 使用综合音乐分析器进行分析
            System.out.println("\n执行综合音乐分析...");
            BasicMusicAnalyzer analyzer = new BasicMusicAnalyzer();
            UnifiedMusicAnalysisResult result = (UnifiedMusicAnalysisResult) analyzer.analyzeMusic(audioData);
            
            // 3. 提取调性和和弦结果
            KeyDetectionResult keyResult = result.getKeyDetectionResult();
        ChordDetectionResult chordResult = result.getChordDetectionResult();
            
            System.out.println("\n=== 分析结果 ===");
            System.out.println("调性检测:");
            System.out.println("  调性: " + keyResult.getKeyName() + " " + keyResult.getScaleType());
            System.out.println("  置信度: " + String.format("%.2f", keyResult.getConfidence()));
            
            System.out.println("\n和弦检测:");
            System.out.println("  和弦: " + chordResult.getChordName());
            System.out.println("  时间: [" + String.format("%.2f", chordResult.getStartTime()) + 
                             "s - " + String.format("%.2f", chordResult.getEndTime()) + "s]");
            System.out.println("  置信度: " + String.format("%.2f", chordResult.getConfidence()));
            
            // 4. 一致性分析
            System.out.println("\n=== 一致性分析 ===");
            analyzeConsistency(keyResult, chordResult);
            
            // 5. 独立验证测试
            System.out.println("\n=== 独立验证测试 ===");
            performIndependentValidation(audioData);
            
            // 6. 改进建议
            System.out.println("\n=== 改进建议 ===");
            provideImprovementSuggestions(keyResult, chordResult);
            
        } catch (Exception e) {
            System.err.println("分析过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 分析调性和和弦的一致性
     */
    private static void analyzeConsistency(KeyDetectionResult keyResult, ChordDetectionResult chordResult) {
// 获取调式信息
        String key = keyResult.getKeyName();
        String scale = keyResult.getScaleType();
        String chord = chordResult.getChordName();
        
        System.out.println("调性-和弦一致性检查:");
        
        if (key == null || scale == null || chord == null || chord.isEmpty()) {
            System.out.println("  ❌ 缺少必要数据");
            return;
        }
        
        // 检查和弦是否在当前调性中合理
        boolean isConsistent = isChordInKey(chord, key, scale);
        
        if (isConsistent) {
            System.out.println("  ✅ 和弦与调性一致");
        } else {
            System.out.println("  ❌ 和弦与调性冲突");
            System.out.println("     原因: " + chord + " 和弦不属于 " + key + " " + scale + " 调");
            
            // 提供可能的正确调性
            suggestAlternativeKeys(chord);
        }
        
        // 置信度分析
        double keyConfidence = keyResult.getConfidence();
        double chordConfidence = chordResult.getConfidence();
        
        System.out.println("\n置信度分析:");
        if (keyConfidence < 0.5) {
            System.out.println("  ⚠️  调性检测置信度较低 (" + String.format("%.2f", keyConfidence) + ")");
        }
        if (chordConfidence < 0.5) {
            System.out.println("  ⚠️  和弦检测置信度较低 (" + String.format("%.2f", chordConfidence) + ")");
        }
        
        if (keyConfidence >= 0.5 && chordConfidence >= 0.5) {
            System.out.println("  ✅ 两者置信度都在可接受范围内");
        }
    }
    
    /**
     * 检查和弦是否属于指定调性
     */
    private static boolean isChordInKey(String chord, String key, String scale) {
        // 简化的调性和弦关系检查
        // 这里可以实现更复杂的音乐理论逻辑
        
        if (chord == null || key == null || scale == null) {
            return false;
        }
        
        // 提取根音
        String rootNote = extractRootNote(chord);
        if (rootNote == null) {
            return false;
        }
        
        // 检查根音是否在调性音阶中
        return isNoteInKey(rootNote, key, scale);
    }
    
    /**
     * 提取和弦的根音
     */
    private static String extractRootNote(String chord) {
        if (chord == null || chord.isEmpty()) {
            return null;
        }
        
        // 处理各种和弦表示法
        chord = chord.replaceAll("m7|maj7|min7|7|m|dim|aug|sus[24]|add9|6|9|11|13", "");
        
        // 处理升降号
        if (chord.length() >= 2 && (chord.charAt(1) == '#' || chord.charAt(1) == 'b')) {
            return chord.substring(0, 2);
        } else if (chord.length() >= 1) {
            return chord.substring(0, 1);
        }
        
        return null;
    }
    
    /**
     * 检查音符是否在指定调性中
     */
    private static boolean isNoteInKey(String note, String key, String scale) {
        // 简化的音阶检查
        String[] majorScaleNotes = getMajorScaleNotes(key);
        String[] minorScaleNotes = getMinorScaleNotes(key);
        
        String[] scaleNotes;
        if ("major".equalsIgnoreCase(scale)) {
            scaleNotes = majorScaleNotes;
        } else if ("minor".equalsIgnoreCase(scale)) {
            scaleNotes = minorScaleNotes;
        } else {
            return false;
        }
        
        for (String scaleNote : scaleNotes) {
            if (note.equals(scaleNote)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 获取大调音阶
     */
    private static String[] getMajorScaleNotes(String key) {
        // 简化的C大调音阶作为示例
        // 实际实现应该根据调性理论计算
        String[] cMajor = {"C", "D", "E", "F", "G", "A", "B"};
        return cMajor;
    }
    
    /**
     * 获取小调音阶
     */
    private static String[] getMinorScaleNotes(String key) {
        // 简化的A小调音阶作为示例
        // 实际实现应该根据调性理论计算
        String[] aMinor = {"A", "B", "C", "D", "E", "F", "G"};
        return aMinor;
    }
    
    /**
     * 建议可能的正确调性
     */
    private static void suggestAlternativeKeys(String chord) {
        String rootNote = extractRootNote(chord);
        if (rootNote == null) {
            return;
        }
        
        System.out.println("\n  可能的正确调性建议:");
        System.out.println("  - " + rootNote + " 大调");
        System.out.println("  - " + rootNote + " 小调");
        
        // 根据和弦类型提供具体建议
        if (chord.contains("m") && !chord.contains("maj")) {
            System.out.println("  - " + rootNote + " 小调 (因为和弦是小和弦)");
        } else if (chord.contains("maj") || (!chord.contains("m") && !chord.contains("dim"))) {
            System.out.println("  - " + rootNote + " 大调 (因为和弦是大和弦)");
        }
    }
    
    /**
     * 执行独立验证测试
     */
    private static void performIndependentValidation(AudioData audioData) {
        System.out.println("执行独立调性分析...");
        
        try {
            KeyAnalyzerImpl keyAnalyzer = new KeyAnalyzerImpl();
            KeyDetectionResult independentKey = keyAnalyzer.detectKey(audioData);
            
            System.out.println("独立调性分析结果:");
            System.out.println("  调性: " + independentKey.getKeyName() + " " + independentKey.getScaleType());
            System.out.println("  置信度: " + String.format("%.2f", independentKey.getConfidence()));
            
            System.out.println("\n执行独立和弦分析...");
            ChordAnalyzerImpl chordAnalyzer = new ChordAnalyzerImpl();
            List<ChordDetectionResult> independentChords = chordAnalyzer.detectChords(audioData);
            
            System.out.println("独立和弦分析结果:");
            if (independentChords.isEmpty()) {
                System.out.println("  未检测到和弦");
            } else {
                for (int i = 0; i < Math.min(3, independentChords.size()); i++) {
                    ChordDetectionResult chord = independentChords.get(i);
                    System.out.println("  和弦 " + (i+1) + ": " + chord.getChordName() + 
                                     " (置信度: " + String.format("%.2f", chord.getConfidence()) + ")");
                }
            }
            
        } catch (Exception e) {
            System.err.println("独立验证测试失败: " + e.getMessage());
        }
    }
    
    /**
     * 提供改进建议
     */
    private static void provideImprovementSuggestions(KeyDetectionResult keyResult, ChordDetectionResult chordResult) {
        System.out.println("基于分析结果的改进建议:");
        
        double keyConfidence = keyResult.getConfidence();
        double chordConfidence = chordResult.getConfidence();
        
        if (keyConfidence < 0.5 && chordConfidence < 0.5) {
            System.out.println("1. 两个检测结果的置信度都较低，建议:");
            System.out.println("   - 检查音频质量，确保没有过多噪音");
            System.out.println("   - 尝试不同的音频片段进行分析");
            System.out.println("   - 调整分析算法的参数阈值");
        }
        
        if (keyConfidence >= 0.5 && chordConfidence < 0.5) {
            System.out.println("2. 调性检测置信度较高，但和弦检测置信度较低，建议:");
            System.out.println("   - 重点改进和弦检测算法");
            System.out.println("   - 利用已知的调性信息来指导和弦检测");
            System.out.println("   - 检查色谱特征提取的准确性");
        }
        
        if (keyConfidence < 0.5 && chordConfidence >= 0.5) {
            System.out.println("3. 和弦检测置信度较高，但调性检测置信度较低，建议:");
            System.out.println("   - 利用和弦信息来推断可能的调性");
            System.out.println("   - 改进调性检测算法的特征提取");
            System.out.println("   - 考虑使用多个和弦进行调性推断");
        }
        
        System.out.println("\n4. 通用改进建议:");
        System.out.println("   - 实现调性和弦联合优化算法");
        System.out.println("   - 增加音乐理论知识约束");
        System.out.println("   - 使用机器学习模型进行联合预测");
        System.out.println("   - 增加人工验证和反馈机制");
    }
}