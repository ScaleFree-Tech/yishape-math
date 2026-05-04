package com.yishape.lab.music.analysis.feature;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple2;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * 音乐特征提取结果 / Music Feature Extraction Result
 * <p>
 * 封装音乐特征提取的完整结果，包含各类音乐特征数据。
 * Encapsulates complete music feature extraction results, containing various music feature data.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 2.0
 */
public class MusicFeatureResult {

    /** 节拍特征结果 / Rhythm feature result */
    private final RhythmFeatureResult rhythmFeatures;
    /** 调性特征结果 / Tonal feature result */
    private final TonalFeatureResult tonalFeatures;
    /** 结构特征结果 / Structure feature result */
    private final StructureFeatureResult structureFeatures;
    /** 表现力特征结果 / Expressiveness feature result */
    private final ExpressivenessFeatureResult expressivenessFeatures;
    /** 元数据 / Metadata */
    private final Map<String, Object> metadata;
    
    /**
     * 构造函数 / Constructor
     *
     * @param rhythmFeatures 节拍特征结果 / Rhythm feature result
     * @param tonalFeatures 调性特征结果 / Tonal feature result
     * @param structureFeatures 结构特征结果 / Structure feature result
     * @param expressivenessFeatures 表现力特征结果 / Expressiveness feature result
     */
    public MusicFeatureResult(RhythmFeatureResult rhythmFeatures,
                             TonalFeatureResult tonalFeatures,
                             StructureFeatureResult structureFeatures,
                             ExpressivenessFeatureResult expressivenessFeatures) {
        this.rhythmFeatures = rhythmFeatures;
        this.tonalFeatures = tonalFeatures;
        this.structureFeatures = structureFeatures;
        this.expressivenessFeatures = expressivenessFeatures;
        this.metadata = new HashMap<>();
    }
    
    /**
     * 获取节拍特征 / Get rhythm features
     */
    public RhythmFeatureResult getRhythmFeatures() {
        return rhythmFeatures;
    }
    
    /**
     * 获取调性特征 / Get tonal features
     */
    public TonalFeatureResult getTonalFeatures() {
        return tonalFeatures;
    }
    
    /**
     * 获取结构特征 / Get structure features
     */
    public StructureFeatureResult getStructureFeatures() {
        return structureFeatures;
    }
    
    /**
     * 获取表现力特征 / Get expressiveness features
     */
    public ExpressivenessFeatureResult getExpressivenessFeatures() {
        return expressivenessFeatures;
    }
    
    /**
     * 获取元数据 / Get metadata
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    /**
     * 添加元数据 / Add metadata
     *
     * @param key 元数据键 / Metadata key
     * @param value 元数据值 / Metadata value
     */
    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }
    
    /**
     * 计算特征完整性 / Calculate feature completeness
     */
    public double getFeatureCompleteness() {
        int totalFeatures = 4;
        int availableFeatures = 0;
        
        if (rhythmFeatures != null) availableFeatures++;
        if (tonalFeatures != null) availableFeatures++;
        if (structureFeatures != null) availableFeatures++;
        if (expressivenessFeatures != null) availableFeatures++;
        
        return (double) availableFeatures / totalFeatures;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("MusicFeatureResult{");
        sb.append("completeness=").append(String.format("%.2f", getFeatureCompleteness()));
        sb.append(", metadata=").append(metadata.size()).append(" items");
        
        // Add detailed rhythm features
        if (rhythmFeatures != null) {
            sb.append(", rhythm={");
            sb.append("tempo=").append(String.format("%.1f", rhythmFeatures.getTempo()));
            sb.append(", type=").append(rhythmFeatures.getTempoType());
            sb.append(", beatStrength=").append(String.format("%.3f", rhythmFeatures.getBeatStrength()));
            sb.append(", regularity=").append(String.format("%.3f", rhythmFeatures.getRhythmRegularity()));
            sb.append(", syncopation=").append(String.format("%.3f", rhythmFeatures.getSyncopation()));
            sb.append(", confidence=").append(String.format("%.3f", rhythmFeatures.getConfidence()));
            if (rhythmFeatures.getOnsetTimes().length > 0) {
                sb.append(", onsets=").append(rhythmFeatures.getOnsetTimes().length);
            }
            sb.append("}");
        }
        
        // Add detailed tonal features
        if (tonalFeatures != null) {
            sb.append(", tonal={");
            sb.append("key=").append(tonalFeatures.getFullKey());
            sb.append(", keyStrength=").append(String.format("%.3f", tonalFeatures.getKeyStrength()));
            sb.append(", complexity=").append(String.format("%.3f", tonalFeatures.getHarmonicComplexity()));
            sb.append(", stability=").append(String.format("%.3f", tonalFeatures.getTonalStability()));
            sb.append(", confidence=").append(String.format("%.3f", tonalFeatures.getConfidence()));
            if (tonalFeatures.getDetectedChords().length > 0) {
                sb.append(", chords=").append(tonalFeatures.getDetectedChords().length);
            }
            sb.append("}");
        }
        
        // Add detailed structure features
        if (structureFeatures != null) {
            sb.append(", structure={");
            sb.append("sections=").append(structureFeatures.getEstimatedSections());
            sb.append(", complexity=").append(structureFeatures.getComplexityLevel());
            sb.append(", repetitiveness=").append(String.format("%.3f", structureFeatures.getRepetitiveness()));
            sb.append(", avgLength=").append(String.format("%.1fs", structureFeatures.getAverageSegmentLength()));
            sb.append(", confidence=").append(String.format("%.3f", structureFeatures.getConfidence()));
            if (!structureFeatures.getSegments().isEmpty()) {
                sb.append(", segments=").append(structureFeatures.getSegments().size());
            }
            sb.append("}");
        }
        
        // Add detailed expressiveness features
        if (expressivenessFeatures != null) {
            sb.append(", expressiveness={");
            sb.append("mood=").append(expressivenessFeatures.getPredictedMood());
            sb.append(", quadrant=").append(expressivenessFeatures.getEmotionalQuadrant());
            sb.append(", energy=").append(expressivenessFeatures.getEnergy());
            sb.append(", valence=").append(String.format("%.3f", expressivenessFeatures.getValence()));
            sb.append(", arousal=").append(String.format("%.3f", expressivenessFeatures.getArousal()));
            sb.append(", confidence=").append(String.format("%.3f", expressivenessFeatures.getConfidence()));
            sb.append("}");
        }
        
        sb.append('}');
        return sb.toString();
    }
    
    /**
     * 将提取的特征全部转换为数值特征，其中List<String>中记录了特征名称，IVector<Double>中记录了特征的值。
     * Converts all extracted features to numerical features.
     *
     * @return 特征名称和特征值的元组 / Tuple of feature names and feature values
     */
    public Tuple2<List<String>, IVector<Double>> toNumericalFeatures() {
        // 创建特征名称列表
        List<String> featureNames = new ArrayList<>();
        
        // 创建特征值列表
        List<Double> featureValues = new ArrayList<>();
        
        // 添加节拍特征
        if (rhythmFeatures != null) {
            featureNames.add("rhythm.tempo");
            featureValues.add(rhythmFeatures.getTempo());
            
            featureNames.add("rhythm.beatStrength");
            featureValues.add(rhythmFeatures.getBeatStrength());
            
            featureNames.add("rhythm.rhythmRegularity");
            featureValues.add(rhythmFeatures.getRhythmRegularity());
            
            featureNames.add("rhythm.syncopation");
            featureValues.add(rhythmFeatures.getSyncopation());
            
            featureNames.add("rhythm.confidence");
            featureValues.add(rhythmFeatures.getConfidence());
        }
        
        // 添加调性特征
        if (tonalFeatures != null) {
            featureNames.add("tonal.keyStrength");
            featureValues.add(tonalFeatures.getKeyStrength());
            
            featureNames.add("tonal.tonalCentroid");
            featureValues.add(tonalFeatures.getTonalCentroid());
            
            featureNames.add("tonal.harmonicComplexity");
            featureValues.add(tonalFeatures.getHarmonicComplexity());
            
            featureNames.add("tonal.tonalStability");
            featureValues.add(tonalFeatures.getTonalStability());
            
            featureNames.add("tonal.confidence");
            featureValues.add(tonalFeatures.getConfidence());
            
            // 添加色度向量特征 (12维)
            double[] chromaVector = tonalFeatures.getChromaVector();
            for (int i = 0; i < chromaVector.length && i < 12; i++) {
                featureNames.add("tonal.chroma_" + i);
                featureValues.add(chromaVector[i]);
            }
            
            // 添加Tonnetz特征 (6维)
            double[] tonnetzFeatures = tonalFeatures.getTonnetzFeatures();
            for (int i = 0; i < tonnetzFeatures.length && i < 6; i++) {
                featureNames.add("tonal.tonnetz_" + i);
                featureValues.add(tonnetzFeatures[i]);
            }
            
            // 添加HPCP特征 (12维)
            double[] hpcpFeatures = tonalFeatures.getHpcpFeatures();
            for (int i = 0; i < hpcpFeatures.length && i < 12; i++) {
                featureNames.add("tonal.hpcp_" + i);
                featureValues.add(hpcpFeatures[i]);
            }
            
            // 添加音调稳定性特征 (4维)
            double[] pitchStabilityFeatures = tonalFeatures.getPitchStabilityFeatures();
            for (int i = 0; i < pitchStabilityFeatures.length && i < 4; i++) {
                featureNames.add("tonal.pitchStability_" + i);
                featureValues.add(pitchStabilityFeatures[i]);
            }
        }
        
        // 添加结构特征
        if (structureFeatures != null) {
            featureNames.add("structure.structuralComplexity");
            featureValues.add(structureFeatures.getStructuralComplexity());
            
            featureNames.add("structure.repetitiveness");
            featureValues.add(structureFeatures.getRepetitiveness());
            
            featureNames.add("structure.estimatedSections");
            featureValues.add((double) structureFeatures.getEstimatedSections());
            
            featureNames.add("structure.averageSegmentLength");
            featureValues.add(structureFeatures.getAverageSegmentLength());
            
            featureNames.add("structure.confidence");
            featureValues.add(structureFeatures.getConfidence());
        }
        
        // 添加表现力特征
        if (expressivenessFeatures != null) {
            featureNames.add("expressiveness.valence");
            featureValues.add(expressivenessFeatures.getValence());
            
            featureNames.add("expressiveness.arousal");
            featureValues.add(expressivenessFeatures.getArousal());
            
            featureNames.add("expressiveness.dominance");
            featureValues.add(expressivenessFeatures.getDominance());
            
            featureNames.add("expressiveness.energy");
            featureValues.add(expressivenessFeatures.getEnergy());
            
            featureNames.add("expressiveness.danceability");
            featureValues.add(expressivenessFeatures.getDanceability());
            
            featureNames.add("expressiveness.acousticness");
            featureValues.add(expressivenessFeatures.getAcousticness());
            
            featureNames.add("expressiveness.instrumentalness");
            featureValues.add(expressivenessFeatures.getInstrumentalness());
            
            featureNames.add("expressiveness.liveness");
            featureValues.add(expressivenessFeatures.getLiveness());
            
            featureNames.add("expressiveness.speechiness");
            featureValues.add(expressivenessFeatures.getSpeechiness());
            
            featureNames.add("expressiveness.emotionalIntensity");
            featureValues.add(expressivenessFeatures.getEmotionalIntensity());
            
            featureNames.add("expressiveness.confidence");
            featureValues.add(expressivenessFeatures.getConfidence());
            
            // 添加动态范围特征 (取平均值)
            double[] dynamicRange = expressivenessFeatures.getDynamicRange();
            if (dynamicRange.length > 0) {
                double dynamicRangeAvg = 0;
                for (double value : dynamicRange) {
                    dynamicRangeAvg += value;
                }
                dynamicRangeAvg /= dynamicRange.length;
                featureNames.add("expressiveness.dynamicRangeAvg");
                featureValues.add(dynamicRangeAvg);
            }
            
            // 添加频谱重心演化特征 (取平均值)
            double[] spectralCentroidEvolution = expressivenessFeatures.getSpectralCentroidEvolution();
            if (spectralCentroidEvolution.length > 0) {
                double spectralCentroidAvg = 0;
                for (double value : spectralCentroidEvolution) {
                    spectralCentroidAvg += value;
                }
                spectralCentroidAvg /= spectralCentroidEvolution.length;
                featureNames.add("expressiveness.spectralCentroidAvg");
                featureValues.add(spectralCentroidAvg);
            }
            
            // 添加计算特征
            featureNames.add("expressiveness.timbreVariation");
            featureValues.add(expressivenessFeatures.getTimbreVariation());
            
            featureNames.add("expressiveness.articulationClarity");
            featureValues.add(expressivenessFeatures.getArticulationClarity());
            
            featureNames.add("expressiveness.expressiveIntensity");
            featureValues.add(expressivenessFeatures.getExpressiveIntensity());
            
            featureNames.add("expressiveness.microtiming");
            featureValues.add(expressivenessFeatures.getMicrotiming());
            
            featureNames.add("expressiveness.vibrato");
            featureValues.add(expressivenessFeatures.getVibrato());
            
            featureNames.add("expressiveness.rubato");
            featureValues.add(expressivenessFeatures.getRubato());
            
            featureNames.add("expressiveness.overallScore");
            featureValues.add(expressivenessFeatures.getOverallExpressivenessScore());
        }
        
        // 转换为数组
        double[] values = new double[featureValues.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = featureValues.get(i);
        }
        
        // 创建向量
        IVector<Double> featureVector = Linalg.vector(values);
        
        return new Tuple2<>(featureNames, featureVector);
    }
}