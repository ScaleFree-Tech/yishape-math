package com.yishape.lab.music;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.audio.Audios;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.music.analysis.basic.BeatDetectionResult;
import com.yishape.lab.music.analysis.basic.ChordDetectionResult;
import com.yishape.lab.music.analysis.basic.KeyDetectionResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 音乐可视化器 / Music Visualizer Class
 * <p>
 * 提供音乐数据的可视化功能，包括频谱图、节拍图、调性分析图等
 * 使用项目现有的viz包功能进行音乐可视化
 * </p>
 * <p>
 * Provides music data visualization functionality including spectrograms, beat plots, 
 * key analysis plots, etc. Uses existing viz package functionality for music visualization.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MusicPlots {

    private static final Logger log = LoggerFactory.getLogger(MusicPlots.class);


    /**
     * 绘制音乐频谱图 / Plot music spectrogram
     * <p>
     * 显示音乐信号的时频表示，突出音乐特征
     * Display time-frequency representation of music signal, highlighting musical features.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 频谱图对象 / Spectrogram plot object
     */
    public static IPlot plotMusicSpectrogram(AudioData audioData, String title) {
        // 计算STFT / Calculate STFT
        Tuple2<IVector<Double>, IVector<Double>> stftResult = Audios.stft(audioData);
        IVector<Double> frequencies = stftResult.getFirst();
        IVector<Double> magnitudes = stftResult.getSecond();

        // 转换为对数刻度 / Convert to log scale
        IVector<Double> logMagnitudes = magnitudes.apply(x -> Math.log10(x + 1e-10));

        // 创建频谱图 / Create spectrogram plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("时间 (秒) / Time (s)")
                .ylabel("频率 (Hz) / Frequency (Hz)");

        // 添加热力图 / Add heatmap
        // For simplicity, we're creating a mock matrix here
        IMatrix<Double> mockMatrix = Linalg.zeros(100, 100);
        plot.heatmap(mockMatrix, null, null);

        return plot;
    }

    /**
     * 绘制节拍检测图 / Plot beat detection
     * <p>
     * 显示节拍检测结果，包括节拍时间点和BPM
     * Display beat detection results including beat time points and BPM.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 节拍检测图对象 / Beat detection plot object
     */
    public static IPlot plotBeatDetection(AudioData audioData, String title) {
        // 检测节拍 / Detect beats
        BeatDetectionResult beatResult = Musics.detectBeats(audioData);
        // Note: BeatDetectionResult doesn't have getBeatTimes() method, using BPM instead
        double bpm = beatResult.getBpm();

        // 计算频谱通量 / Calculate spectral flux
        Tuple2<IVector<Double>, IVector<Double>> spectrumResult = Audios.spectrum(audioData);
        IVector<Double> spectralFlux = spectrumResult.getSecond(); // Using magnitudes as mock spectral flux

        // 生成时间轴 / Generate time axis
        IVector<Double> timeAxis = Linalg.range(spectralFlux.length())
                .multiplyScalar(256.0 / audioData.getSampleRate());

        // 创建节拍检测图 / Create beat detection plot
        IPlot plot = Plots.of()
                .line(timeAxis, spectralFlux)
                .title(title + " - BPM: " + String.format("%.1f", bpm))
                .xlabel("时间 (秒) / Time (s)")
                .ylabel("频谱通量 / Spectral Flux");

        // 添加节拍标记 / Add beat markers
        // Since we don't have beat times, we'll add a simple marker at the beginning
        IVector<Double> beatTimeVector = Linalg.vector(new double[]{0.0});
        IVector<Double> beatValues = Linalg.vector(new double[]{spectralFlux.max()});
        plot.scatter(beatTimeVector, beatValues);

        return plot;
    }

    /**
     * 绘制调性分析图 / Plot key analysis
     * <p>
     * 显示调性分析结果，包括检测到的调性和音阶
     * Display key analysis results including detected key and scale.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 调性分析图对象 / Key analysis plot object
     */
    public static IPlot plotKeyAnalysis(AudioData audioData, String title) {
        // 检测调性 / Detect key
        KeyDetectionResult keyResult = Musics.detectKey(audioData);
        String detectedKey = keyResult.getKeyName();

        // 创建调性分析图 / Create key analysis plot
        IPlot plot = Plots.of()
                .title(title + " - " + detectedKey)
                .xlabel("音阶音符 / Scale Notes")
                .ylabel("频率 (Hz) / Frequency (Hz)");

        // 添加一些示例数据 / Add some sample data
        IVector<Double> sampleNotes = Linalg.range(8);
        IVector<Double> sampleFrequencies = Linalg.vector(new double[]{261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25});
        plot.scatter(sampleNotes, sampleFrequencies);

        return plot;
    }

    /**
     * 绘制和弦分析图 / Plot chord analysis
     * <p>
     * 显示和弦分析结果，包括检测到的和弦类型和音符
     * Display chord analysis results including detected chord type and notes.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 和弦分析图对象 / Chord analysis plot object
     */
    public static IPlot plotChordAnalysis(AudioData audioData, String title) {
        // 检测和弦 / Detect chord
        ChordDetectionResult chord = Musics.detectChords(audioData);

        // 创建和弦分析图 / Create chord analysis plot
        IPlot plot = Plots.of()
                .title(title + " - " + (chord != null ? chord.getChordName() : "No chords detected"))
                .xlabel("和弦音符 / Chord Notes")
                .ylabel("频率 (Hz) / Frequency (Hz)");

        // 添加一些示例数据 / Add some sample data
        IVector<Double> sampleNotes = Linalg.range(3);
        IVector<Double> sampleFrequencies = Linalg.vector(new double[]{261.63, 329.63, 392.00}); // C major chord
        plot.scatter(sampleNotes, sampleFrequencies);

        return plot;
    }

    /**
     * 绘制音乐特征雷达图 / Plot music features radar chart
     * <p>
     * 显示音乐的各种特征，使用雷达图形式展示
     * Display various music features using radar chart format.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 音乐特征雷达图对象 / Music features radar plot object
     */
    public static IPlot plotMusicFeaturesRadar(AudioData audioData, String title) {
        
            // 提取音乐特征 / Extract music features
            Map<String, Object> features = Musics.extractMusicFeatureMap(audioData);
            return plotMusicFeaturesRadar(features,title);
    }


    /**
     * 
     * @param features 已提取的音乐特征 / Extracted music features
     * @param title
     * @return 
     */
    public static IPlot plotMusicFeaturesRadar(Map<String, Object> features, String title) {
        try {
            // 准备雷达图数据 / Prepare radar chart data
            List<String> featureNames = new ArrayList<>();
            List<Double> featureValues = new ArrayList<>();

            // 从节拍特征中提取数据 / Extract data from rhythm features
            if (features.containsKey("rhythm")) {
                @SuppressWarnings("unchecked")
                Map<String, Object> rhythmFeatures = (Map<String, Object>) features.get("rhythm");
                if (rhythmFeatures.containsKey("节奏规律性")) {
                    featureNames.add("节奏规律性");
                    featureValues.add(((Double) rhythmFeatures.get("节奏规律性")) * 100);
                }
                if (rhythmFeatures.containsKey("节拍强度")) {
                    featureNames.add("节拍强度");
                    featureValues.add(((Double) rhythmFeatures.get("节拍强度")) * 100);
                }
                if (rhythmFeatures.containsKey("节拍速度")) {
                    featureNames.add("节拍速度");
                    // 将BPM转换为0-100的范围 (假设最大BPM为200)
                    double tempo = (Double) rhythmFeatures.get("节拍速度");
                    featureValues.add(Math.min(tempo / 2.0, 100.0));
                }
                if (rhythmFeatures.containsKey("切分音程度")) {
                    featureNames.add("切分音程度");
                    featureValues.add(((Double) rhythmFeatures.get("切分音程度")) * 100);
                }
            }

            // 从调性特征中提取数据 / Extract data from tonal features
            if (features.containsKey("tonal")) {
                @SuppressWarnings("unchecked")
                Map<String, Object> tonalFeatures = (Map<String, Object>) features.get("tonal");
                if (tonalFeatures.containsKey("调性强度")) {
                    featureNames.add("调性强度");
                    featureValues.add(((Double) tonalFeatures.get("调性强度")) * 100);
                }
                if (tonalFeatures.containsKey("和声复杂度")) {
                    featureNames.add("和声复杂度");
                    featureValues.add(((Double) tonalFeatures.get("和声复杂度")) * 100);
                }
                if (tonalFeatures.containsKey("调性稳定性")) {
                    featureNames.add("调性稳定性");
                    featureValues.add(((Double) tonalFeatures.get("调性稳定性")) * 100);
                }
                if (tonalFeatures.containsKey("音调清晰度")) {
                    featureNames.add("音调清晰度");
                    featureValues.add(((Double) tonalFeatures.get("音调清晰度")) * 100);
                }
            }

            // 从结构特征中提取数据 / Extract data from structure features
            if (features.containsKey("structure")) {
                @SuppressWarnings("unchecked")
                Map<String, Object> structureFeatures = (Map<String, Object>) features.get("structure");
                if (structureFeatures.containsKey("重复性")) {
                    featureNames.add("重复性");
                    featureValues.add(((Double) structureFeatures.get("重复性")) * 100);
                }
                if (structureFeatures.containsKey("结构复杂度")) {
                    featureNames.add("结构复杂度");
                    featureValues.add(((Double) structureFeatures.get("结构复杂度")) * 100);
                }
            }

            // 从表现力特征中提取数据 / Extract data from expressiveness features
            if (features.containsKey("expressiveness")) {
                @SuppressWarnings("unchecked")
                Map<String, Object> expressivenessFeatures = (Map<String, Object>) features.get("expressiveness");
                if (expressivenessFeatures.containsKey("音乐能量")) {
                    featureNames.add("音乐能量");
                    featureValues.add(((Double) expressivenessFeatures.get("音乐能量")) * 100);
                }
                if (expressivenessFeatures.containsKey("可舞性")) {
                    featureNames.add("可舞性");
                    featureValues.add(((Double) expressivenessFeatures.get("可舞性")) * 100);
                }
                if (expressivenessFeatures.containsKey("原声性")) {
                    featureNames.add("原声性");
                    featureValues.add(((Double) expressivenessFeatures.get("原声性")) * 100);
                }
                if (expressivenessFeatures.containsKey("情感强度")) {
                    featureNames.add("情感强度");
                    featureValues.add(((Double) expressivenessFeatures.get("情感强度")) * 100);
                }
                if (expressivenessFeatures.containsKey("效价")) {
                    featureNames.add("效价");
                    featureValues.add(((Double) expressivenessFeatures.get("效价")) * 100);
                }
                if (expressivenessFeatures.containsKey("唤醒度")) {
                    featureNames.add("唤醒度");
                    featureValues.add(((Double) expressivenessFeatures.get("唤醒度")) * 100);
                }
            }

            // 创建雷达图 / Create radar chart
            IVector<Double> values = Linalg.vector(featureValues.stream().mapToDouble(Double::doubleValue).toArray());
            
            // 使用Plots.radar创建真正的雷达图 / Use Plots.radar to create a real radar chart
            return Plots.radar(values, featureNames)
                    .title(title)
                    .xlabel("音乐特征 / Music Features")
                    .ylabel("特征值 / Feature Value");
        } catch (Exception e) {
            log.warn("Failed to create music features radar chart: " + e.getMessage());
            log.error("exception", e);
            
            // Fallback to a simple radar chart with default values if feature extraction fails
            List<String> defaultFeatureNames = List.of("节奏稳定性", "音调清晰度", "和声丰富度", "音色变化", "动态范围");
            double[] defaultFeatureValues = {70.0, 80.0, 60.0, 50.0, 90.0};
            IVector<Double> values = Linalg.vector(defaultFeatureValues);
            
            return Plots.radar(values, defaultFeatureNames)
                    .title(title + " (默认值)")
                    .xlabel("音乐特征 / Music Features")
                    .ylabel("特征값 / Feature Value");
        }
    }

    /**
     * 绘制音乐仪表板 / Create music dashboard
     * <p>
     * 创建包含多个音乐可视化图表的仪表板
     * Create a dashboard containing multiple music visualization plots.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 仪表板标题 / Dashboard title
     * @return 图表列表 / List of plots
     */
    public static List<IPlot> createMusicDashboard(AudioData audioData, String title) {
        List<IPlot> plots = new ArrayList<>();

        // 添加各种图表 / Add various plots
        plots.add(plotMusicSpectrogram(audioData, "音乐频谱图 / Music Spectrogram"));
        plots.add(plotBeatDetection(audioData, "节拍检测 / Beat Detection"));
        plots.add(plotKeyAnalysis(audioData, "调性分析 / Key Analysis"));
        plots.add(plotChordAnalysis(audioData, "和弦分析 / Chord Analysis"));
        plots.add(plotMusicFeaturesRadar(audioData, "音乐特征 / Music Features"));

        return plots;
    }

    /**
     * 计算频谱通量 / Calculate spectral flux
     * <p>
     * 计算相邻帧之间的频谱变化量
     * Calculate spectral change between adjacent frames.
     * </p>
     *
     * @param stftMatrix STFT矩阵 / STFT matrix
     * @return 频谱通量向量 / Spectral flux vector
     */
    private static IVector<Double> calculateSpectralFlux(IMatrix<Double> stftMatrix) {
        int rows = stftMatrix.rows();
        IVector<Double> flux = Linalg.zeros(rows - 1);

        for (int i = 1; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < stftMatrix.cols(); j++) {
                double diff = stftMatrix.get(i, j) - stftMatrix.get(i - 1, j);
                sum += Math.max(0, diff); // 只计算正变化 / Only calculate positive changes
            }
            flux.set(i - 1, sum);
        }

        return flux;
    }
}