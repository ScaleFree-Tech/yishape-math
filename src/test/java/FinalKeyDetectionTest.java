import java.util.Arrays;

public class FinalKeyDetectionTest {
    
    // 模拟改进的色度增强方法
    private static double[] enhanceDominantNotes(double[] chroma) {
        double[] enhanced = chroma.clone();
        
        // 找到最大值用于归一化
        double maxValue = 0.0;
        for (double value : enhanced) {
            maxValue = Math.max(maxValue, value);
        }
        
        if (maxValue <= 0) {
            return enhanced;
        }
        
        // 计算平均值和阈值
        double sum = 0.0;
        for (double value : enhanced) {
            sum += value;
        }
        double average = sum / enhanced.length;
        double threshold = average + (maxValue - average) * 0.3; // 30%阈值
        
        // 增强处理
        for (int i = 0; i < enhanced.length; i++) {
            double normalizedValue = enhanced[i] / maxValue;
            
            if (enhanced[i] > threshold) {
                // 增强峰值：使用平方根来增强对比度，并增加幅度
                enhanced[i] = Math.pow(normalizedValue, 0.5) * maxValue * 1.2;
            } else if (enhanced[i] < average) {
                // 轻微抑制低于平均值的部分
                enhanced[i] = enhanced[i] * 0.8;
            }
            // 中等值保持不变
        }
        
        return enhanced;
    }
    
    // L2归一化
    private static double[] normalizeL2(double[] vector) {
        double norm = 0.0;
        for (double value : vector) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        
        if (norm < 1e-10) {
            return vector.clone();
        }
        
        double[] normalized = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        return normalized;
    }
    
    // 模板匹配
    private static double calculateTemplateMatch(double[] chroma, double[] template) {
        double dotProduct = 0.0;
        double chromaNorm = 0.0;
        double templateNorm = 0.0;
        
        for (int i = 0; i < chroma.length; i++) {
            dotProduct += chroma[i] * template[i];
            chromaNorm += chroma[i] * chroma[i];
            templateNorm += template[i] * template[i];
        }
        
        chromaNorm = Math.sqrt(chromaNorm);
        templateNorm = Math.sqrt(templateNorm);
        
        if (chromaNorm < 1e-10 || templateNorm < 1e-10) {
            return 0.0;
        }
        
        return dotProduct / (chromaNorm * templateNorm);
    }
    
    // 改进的置信度计算
    private static double calculateImprovedConfidence(double[] chromaFeatures, double bestScore) {
        double totalEnergy = 0.0;
        double maxValue = 0.0;
        double minValue = Double.MAX_VALUE;
        
        for (double value : chromaFeatures) {
            totalEnergy += value;
            maxValue = Math.max(maxValue, value);
            minValue = Math.min(minValue, value);
        }
        
        if (totalEnergy < 1e-10) {
            return 0.0;
        }
        
        double average = totalEnergy / 12;
        double variance = 0.0;
        
        for (double value : chromaFeatures) {
            variance += (value - average) * (value - average);
        }
        variance /= 12;
        
        double dynamicRange = maxValue - minValue;
        double rangeFactor = Math.min(1.0, dynamicRange * 2.0);
        double varianceFactor = Math.min(1.0, variance * 50.0);
        
        double[] sortedValues = chromaFeatures.clone();
        Arrays.sort(sortedValues);
        double topThree = sortedValues[11] + sortedValues[10] + sortedValues[9];
        double bottomNine = totalEnergy - topThree;
        double peakProminence = bottomNine > 0 ? topThree / bottomNine : 3.0;
        double prominenceFactor = Math.min(1.0, peakProminence / 3.0);
        
        double featureQuality = (rangeFactor * 0.4 + varianceFactor * 0.4 + prominenceFactor * 0.2);
        double confidence = bestScore * (0.7 + featureQuality * 0.3);
        
        return Math.max(0.0, Math.min(1.0, confidence));
    }
    
    public static void main(String[] args) {
        System.out.println("=== 最终键检测测试 ===\n");
        
        // 创建主要模板
        double[] cMajorTemplate = {1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}; // C大调
        double[] eBluesTemplate = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0}; // E布鲁斯
        
        // 归一化模板
        cMajorTemplate = normalizeL2(cMajorTemplate);
        eBluesTemplate = normalizeL2(eBluesTemplate);
        
        // 测试1：原始问题色度特征（过于均匀）
        double[] originalChroma = {0.3240, 0.3346, 0.3346, 0.3050, 0.2440, 0.2316, 0.2407, 0.2532, 0.2684, 0.2831, 0.3045, 0.3130};
        
        System.out.println("测试1：原始问题色度特征");
        System.out.println("原始色度: " + Arrays.toString(originalChroma));
        
        double originalCMajorScore = calculateTemplateMatch(originalChroma, cMajorTemplate);
        double originalEBluesScore = calculateTemplateMatch(originalChroma, eBluesTemplate);
        double originalBestScore = Math.max(originalCMajorScore, originalEBluesScore);
        String originalBestKey = originalCMajorScore > originalEBluesScore ? "C Major" : "E Blues";
        double originalConfidence = calculateImprovedConfidence(originalChroma, originalBestScore);
        
        System.out.printf("C Major得分: %.6f%n", originalCMajorScore);
        System.out.printf("E Blues得分: %.6f%n", originalEBluesScore);
        System.out.printf("检测结果: %s (置信度: %.6f)%n%n", originalBestKey, originalConfidence);
        
        // 测试2：应用色度增强后
        double[] enhancedChroma = enhanceDominantNotes(originalChroma);
        enhancedChroma = normalizeL2(enhancedChroma);
        
        System.out.println("测试2：应用色度增强后");
        System.out.println("增强色度: " + Arrays.toString(enhancedChroma));
        
        double enhancedCMajorScore = calculateTemplateMatch(enhancedChroma, cMajorTemplate);
        double enhancedEBluesScore = calculateTemplateMatch(enhancedChroma, eBluesTemplate);
        double enhancedBestScore = Math.max(enhancedCMajorScore, enhancedEBluesScore);
        String enhancedBestKey = enhancedCMajorScore > enhancedEBluesScore ? "C Major" : "E Blues";
        double enhancedConfidence = calculateImprovedConfidence(enhancedChroma, enhancedBestScore);
        
        System.out.printf("C Major得分: %.6f%n", enhancedCMajorScore);
        System.out.printf("E Blues得分: %.6f%n", enhancedEBluesScore);
        System.out.printf("检测结果: %s (置信度: %.6f)%n%n", enhancedBestKey, enhancedConfidence);
        
        // 测试3：理想C大调和弦
        double[] idealCMajor = new double[12];
        idealCMajor[0] = 0.5;  // C
        idealCMajor[4] = 0.4;  // E
        idealCMajor[7] = 0.3;  // G
        for (int i = 0; i < 12; i++) {
            if (i != 0 && i != 4 && i != 7) {
                idealCMajor[i] = 0.05;
            }
        }
        idealCMajor = normalizeL2(idealCMajor);
        
        System.out.println("测试3：理想C大调和弦");
        System.out.println("理想色度: " + Arrays.toString(idealCMajor));
        
        double idealCMajorScore = calculateTemplateMatch(idealCMajor, cMajorTemplate);
        double idealEBluesScore = calculateTemplateMatch(idealCMajor, eBluesTemplate);
        double idealBestScore = Math.max(idealCMajorScore, idealEBluesScore);
        String idealBestKey = idealCMajorScore > idealEBluesScore ? "C Major" : "E Blues";
        double idealConfidence = calculateImprovedConfidence(idealCMajor, idealBestScore);
        
        System.out.printf("C Major得分: %.6f%n", idealCMajorScore);
        System.out.printf("E Blues得分: %.6f%n", idealEBluesScore);
        System.out.printf("检测结果: %s (置信度: %.6f)%n%n", idealBestKey, idealConfidence);
        
        // 总结改进效果
        System.out.println("=== 改进效果总结 ===");
        System.out.printf("原始检测: %s (置信度: %.3f)%n", originalBestKey, originalConfidence);
        System.out.printf("增强检测: %s (置信度: %.3f)%n", enhancedBestKey, enhancedConfidence);
        System.out.printf("理想检测: %s (置信度: %.3f)%n", idealBestKey, idealConfidence);
        
        if (enhancedConfidence > originalConfidence) {
            System.out.printf("置信度提升: %.3f -> %.3f (提升%.1f%%)%n", 
                             originalConfidence, enhancedConfidence, 
                             (enhancedConfidence - originalConfidence) / originalConfidence * 100);
        }
    }
}