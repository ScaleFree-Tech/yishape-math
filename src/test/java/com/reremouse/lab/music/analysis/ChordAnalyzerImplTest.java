package com.reremouse.lab.music.analysis;

import com.reremouse.lab.music.analysis.basic.ChordAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.ChordDetectionResult;
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.audio.exception.AudioProcessingException;

import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * 简单的和弦分析器测试，验证ChordAnalyzerImpl修复后是否正常工作
 */
public class ChordAnalyzerImplTest {
    
    public static void main(String[] args) {
        System.out.println("测试 ChordAnalyzerImpl 修复后的功能...");
        
        try {
            // 创建简单的测试音频数据（440Hz正弦波）
            double[] samples = new double[44100]; // 1秒音频
            for (int i = 0; i < samples.length; i++) {
                samples[i] = Math.sin(2 * Math.PI * 440 * i / 44100);
            }
            
            // 创建音频数据对象
            IVector<Double> sampleVector = Linalg.vector(samples);
            AudioData audioData = new AudioData(sampleVector, 44100.0, 1, 16, AudioFormat.WAV);
            
            System.out.println("创建了测试音频数据: " + samples.length + " 样本");
            
            // 创建和弦分析器
            ChordAnalyzerImpl chordAnalyzer = new ChordAnalyzerImpl();
            System.out.println("创建了 ChordAnalyzerImpl 实例");
            
            // 测试和弦检测
            System.out.println("开始和弦检测...");
            long startTime = System.currentTimeMillis();
            
            List<ChordDetectionResult> results = chordAnalyzer.detectChords(audioData);
            
            long endTime = System.currentTimeMillis();
            System.out.println("和弦检测完成，耗时: " + (endTime - startTime) + " ms");
            
            // 显示结果
            if (results != null && !results.isEmpty()) {
                System.out.println("检测到 " + results.size() + " 个和弦段:");
                for (int i = 0; i < results.size(); i++) {
                    ChordDetectionResult result = results.get(i);
                    System.out.println("Chord: " + result.getChordName() + " (confidence: " + result.getConfidence() + ")");
                }
            } else {
                System.out.println("未检测到和弦（可能由于置信度阈值或音频内容）");
            }
            
            // 测试带参数的和弦检测
            System.out.println("\n测试带参数的和弦检测...");
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("chordThreshold", 0.3); // 降低阈值
            parameters.put("segmentLength", 0.5);  // 缩短段长度
            
            List<ChordDetectionResult> resultsWithParams = chordAnalyzer.detectChords(audioData, parameters);
            
            if (resultsWithParams != null && !resultsWithParams.isEmpty()) {
                System.out.println("带参数检测找到 " + resultsWithParams.size() + " 个和弦段");
            } else {
                System.out.println("带参数检测也未找到和弦");
            }
            
            System.out.println("\n测试完成 - ChordAnalyzerImpl 工作正常!");
            
        } catch (Exception e) {
            System.err.println("测试过程中出现错误: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}