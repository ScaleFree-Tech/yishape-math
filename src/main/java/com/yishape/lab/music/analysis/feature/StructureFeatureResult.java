package com.yishape.lab.music.analysis.feature;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

/**
 * 结构特征结果 / Structure Feature Result
 * <p>
 * 封装音乐的结构、形式相关特征数据。
 * Encapsulates structural and formal feature data of music.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 2.0
 */
public class StructureFeatureResult {

    /** 音乐段落列表 / List of music segments */
    private final List<MusicSegment> segments;
    /** 新颖性函数 / Novelty function */
    private final double[] noveltyFunction;
    /** 自相似矩阵 (压缩表示) / Self-similarity matrix (compressed) */
    private final double[] selfSimilarityMatrix;
    /** 结构复杂度 / Structural complexity */
    private final double structuralComplexity;
    /** 重复性 / Repetitiveness */
    private final double repetitiveness;
    /** 估计段落数 / Estimated sections */
    private final int estimatedSections;
    /** 平均段落长度 / Average segment length */
    private final double averageSegmentLength;
    /** 置信度 / Confidence */
    private final double confidence;
    
    
    /**
     * 音乐段落类型 / Music segment type
     */
    public enum SegmentType {
        INTRO, VERSE, CHORUS, BRIDGE, OUTRO, INSTRUMENTAL, UNKNOWN
    }

    /**
     * 音乐段落 / Music segment
     *
     * 表示音乐中的一个段落，包含时间范围和类型信息。
     * Represents a segment in music with time range and type information.
     */
    public static class MusicSegment {
        /** 起始时间 (秒) / Start time (seconds) */
        private final double startTime;
        /** 结束时间 (秒) / End time (seconds) */
        private final double endTime;
        /** 段落类型 / Segment type */
        private final SegmentType type;
        /** 置信度 / Confidence */
        private final double confidence;

        /**
         * 构造函数 / Constructor
         *
         * @param startTime 起始时间 (秒) / Start time (seconds)
         * @param endTime 结束时间 (秒) / End time (seconds)
         * @param type 段落类型 / Segment type
         * @param confidence 置信度 / Confidence
         */
        public MusicSegment(double startTime, double endTime, SegmentType type, double confidence) {
            this.startTime = startTime;
            this.endTime = endTime;
            this.type = type;
            this.confidence = confidence;
        }

        /**
         * 获取起始时间 / Get start time
         *
         * @return 起始时间 (秒) / Start time (seconds)
         */
        public double getStartTime() { return startTime; }

        /**
         * 获取结束时间 / Get end time
         *
         * @return 结束时间 (秒) / End time (seconds)
         */
        public double getEndTime() { return endTime; }

        /**
         * 获取持续时间 / Get duration
         *
         * @return 持续时间 (秒) / Duration (seconds)
         */
        public double getDuration() { return endTime - startTime; }

        /**
         * 获取段落类型 / Get segment type
         *
         * @return 段落类型 / Segment type
         */
        public SegmentType getType() { return type; }

        /**
         * 获取置信度 / Get confidence
         *
         * @return 置信度 / Confidence
         */
        public double getConfidence() { return confidence; }

        @Override
        public String toString() {
            return String.format("%s[%.1fs-%.1fs](%.3f)", type, startTime, endTime, confidence);
        }
    }


    /**
     * 构造函数 / Constructor
     *
     * @param segments 音乐段落列表 / List of music segments
     * @param noveltyFunction 新颖性函数 / Novelty function
     * @param selfSimilarityMatrix 自相似矩阵 / Self-similarity matrix
     * @param structuralComplexity 结构复杂度 / Structural complexity
     * @param repetitiveness 重复性 / Repetitiveness
     * @param estimatedSections 估计段落数 / Estimated sections
     * @param averageSegmentLength 平均段落长度 / Average segment length
     * @param confidence 置信度 / Confidence
     */
    public StructureFeatureResult(List<MusicSegment> segments, double[] noveltyFunction,
                                 double[] selfSimilarityMatrix, double structuralComplexity,
                                 double repetitiveness, int estimatedSections,
                                 double averageSegmentLength, double confidence) {
        this.segments = segments != null ? new ArrayList<>(segments) : new ArrayList<>();
        this.noveltyFunction = noveltyFunction != null ? noveltyFunction.clone() : new double[0];
        this.selfSimilarityMatrix = selfSimilarityMatrix != null ? selfSimilarityMatrix.clone() : new double[0];
        this.structuralComplexity = structuralComplexity;
        this.repetitiveness = repetitiveness;
        this.estimatedSections = estimatedSections;
        this.averageSegmentLength = averageSegmentLength;
        this.confidence = confidence;
    }

    /**
     * 获取音乐段落 / Get music segments
     */
    public List<MusicSegment> getSegments() {
        return new ArrayList<>(segments);
    }

    /**
     * 获取新颖性函数 / Get novelty function
     */
    public double[] getNoveltyFunction() {
        return noveltyFunction.clone();
    }

    /**
     * 获取自相似矩阵 / Get self-similarity matrix
     */
    public double[] getSelfSimilarityMatrix() {
        return selfSimilarityMatrix.clone();
    }

    /**
     * 获取结构复杂度 / Get structural complexity
     */
    public double getStructuralComplexity() {
        return structuralComplexity;
    }

    /**
     * 获取重复性 / Get repetitiveness
     */
    public double getRepetitiveness() {
        return repetitiveness;
    }

    /**
     * 获取估计段落数 / Get estimated sections
     */
    public int getEstimatedSections() {
        return estimatedSections;
    }

    /**
     * 获取平均段落长度 / Get average segment length
     */
    public double getAverageSegmentLength() {
        return averageSegmentLength;
    }

    /**
     * 获取置信度 / Get confidence
     */
    public double getConfidence() {
        return confidence;
    }

    /**
     * 获取特定类型的段落 / Get segments of specific type
     *
     * @param type 段落类型 / Segment type
     * @return 指定类型的段落列表 / List of segments of the specified type
     */
    public List<MusicSegment> getSegmentsByType(SegmentType type) {
        List<MusicSegment> result = new ArrayList<>();
        for (MusicSegment segment : segments) {
            if (segment.getType() == type) {
                result.add(segment);
            }
        }
        return result;
    }

    /**
     * 获取总时长 / Get total duration
     */
    public double getTotalDuration() {
        if (segments.isEmpty()) return 0.0;
        return segments.get(segments.size() - 1).getEndTime();
    }

    /**
     * 判断是否有明确的结构 / Check if has clear structure
     *
     * @return 置信度大于0.7且段落数至少为2时返回true / True if confidence > 0.7 and estimated sections >= 2
     */
    public boolean hasClearStructure() {
        return confidence > 0.7 && estimatedSections >= 2;
    }

    /**
     * 判断是否高度重复 / Check if highly repetitive
     *
     * @return 重复性大于0.8时返回true / True if repetitiveness > 0.8
     */
    public boolean isHighlyRepetitive() {
        return repetitiveness > 0.8;
    }

    /**
     * 获取结构复杂度等级 / Get structural complexity level
     */
    public String getComplexityLevel() {
        if (structuralComplexity < 0.3) return "Simple";
        else if (structuralComplexity < 0.6) return "Moderate";
        else if (structuralComplexity < 0.8) return "Complex";
        else return "Very Complex";
    }

    /**
     * 判断是否为简单结构 / Check if has simple structure
     *
     * @return 结构复杂度小于 0.3 时返回 true / True if structural complexity is below 0.3
     */
    public boolean isSimpleStructure() {
        return structuralComplexity < 0.3;
    }

    /**
     * 判断是否为复杂结构 / Check if has complex structure
     *
     * @return 结构复杂度大于0.6时返回true / True if structural complexity > 0.6
     */
    public boolean isComplexStructure() {
        return structuralComplexity > 0.6;
    }

    /**
     * 判断是否低重复性 / Check if has low repetitiveness
     *
     * @return 重复性小于 0.4 时返回 true / True if repetitiveness is below 0.4
     */
    public boolean isLowRepetitive() {
        return repetitiveness < 0.4;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("StructureFeatureResult{");
        sb.append("sections=").append(estimatedSections);
        sb.append(", complexity=").append(getComplexityLevel());
        sb.append(", repetitiveness=").append(String.format("%.3f", repetitiveness));
        sb.append(", avgSegmentLength=").append(String.format("%.1f", averageSegmentLength)).append("s");
        sb.append(", confidence=").append(String.format("%.3f", confidence));
        if (!segments.isEmpty()) {
            sb.append(", segments=").append(segments.size());
            sb.append(", duration=").append(String.format("%.1f", getTotalDuration())).append("s");
            // Count different segment types
            int intros = getSegmentsByType(SegmentType.INTRO).size();
            int verses = getSegmentsByType(SegmentType.VERSE).size();
            int choruses = getSegmentsByType(SegmentType.CHORUS).size();
            int bridges = getSegmentsByType(SegmentType.BRIDGE).size();
            if (intros > 0) sb.append(", intros=").append(intros);
            if (verses > 0) sb.append(", verses=").append(verses);
            if (choruses > 0) sb.append(", choruses=").append(choruses);
            if (bridges > 0) sb.append(", bridges=").append(bridges);
        }
        if (noveltyFunction.length > 0) {
            sb.append(", noveltyLength=").append(noveltyFunction.length);
        }
        sb.append('}');
        return sb.toString();
    }
}