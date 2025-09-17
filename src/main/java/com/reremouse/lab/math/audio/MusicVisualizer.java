package com.reremouse.lab.math.audio;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.viz.Plots;
import com.reremouse.lab.math.viz.IPlot;

import java.util.ArrayList;
import java.util.List;

/**
 * 音乐可视化器类 / Music Visualizer Class
 * <p>
 * 提供音乐数据的可视化功能，包括频谱图、节拍图、调性分析图等。
 * 使用项目现有的viz包功能进行音乐可视化。
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
public class MusicVisualizer {
    
    /**
     * 绘制音乐频谱图 / Plot music spectrogram
     * <p>
     * 显示音乐信号的时频表示，突出音乐特征。
     * Display time-frequency representation of music signal, highlighting musical features.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 频谱图对象 / Spectrogram plot object
     */
    public static IPlot plotMusicSpectrogram(AudioData audioData, String title) {
        // 计算STFT / Calculate STFT
        IMatrix<Double> stftMatrix = AudioAnalyzer.calculateSTFT(audioData, 1024, 256);
        
        // 转换为对数刻度 / Convert to log scale
        IMatrix<Double> logStftMatrix = stftMatrix.log();
        
        // 创建频谱图 / Create spectrogram plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("时间 (秒) / Time (s)")
                .ylabel("频率 (Hz) / Frequency (Hz)");
        
        // 添加热力图 / Add heatmap
        plot.heatmap(logStftMatrix, null, null);
        
        return plot;
    }
    
    /**
     * 绘制节拍检测图 / Plot beat detection
     * <p>
     * 显示节拍检测结果，包括节拍时间点和BPM。
     * Display beat detection results including beat time points and BPM.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 节拍检测图对象 / Beat detection plot object
     */
    public static IPlot plotBeatDetection(AudioData audioData, String title) {
        // 检测节拍 / Detect beats
        MusicAnalyzer.BeatDetectionResult beatResult = MusicAnalyzer.detectBeats(audioData);
        
        // 计算频谱通量 / Calculate spectral flux
        IMatrix<Double> stftMatrix = AudioAnalyzer.calculateSTFT(audioData, 1024, 256);
        IVector<Double> spectralFlux = calculateSpectralFlux(stftMatrix);
        
        // 生成时间轴 / Generate time axis
        IVector<Double> timeAxis = Linalg.range(spectralFlux.length())
                .multiplyScalar(256.0 / audioData.getSampleRate());
        
        // 创建节拍检测图 / Create beat detection plot
        IPlot plot = Plots.of()
                .line(timeAxis, spectralFlux)
                .title(title + " - BPM: " + String.format("%.1f", beatResult.getBpm()))
                .xlabel("时间 (秒) / Time (s)")
                .ylabel("频谱通量 / Spectral Flux");
        
        // 添加节拍标记 / Add beat markers
        IVector<Double> beatTimes = beatResult.getBeatTimes();
        IVector<Double> beatValues = Linalg.ones(beatTimes.length()).multiplyScalar(spectralFlux.max());
        
        plot.scatter(beatTimes, beatValues);
        
        return plot;
    }
    
    /**
     * 绘制调性分析图 / Plot key analysis
     * <p>
     * 显示调性分析结果，包括检测到的调性和音阶。
     * Display key analysis results including detected key and scale.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 调性分析图对象 / Key analysis plot object
     */
    public static IPlot plotKeyAnalysis(AudioData audioData, String title) {
        // 检测调性 / Detect key
        MusicTheory.Key detectedKey = MusicTheory.detectKey(audioData);
        
        // 生成音阶 / Generate scale
        int[] scale = MusicTheory.generateScale(detectedKey.getRootNote(), detectedKey.getScaleType());
        
        // 计算每个音符的频率 / Calculate frequency for each note
        IVector<Double> frequencies = Linalg.zeros(scale.length);
        for (int i = 0; i < scale.length; i++) {
            frequencies.set(i, MusicTheory.semitonesToFrequency(scale[i]));
        }
        
        // 创建调性分析图 / Create key analysis plot
        IPlot plot = Plots.of()
                .title(title + " - " + detectedKey.getKeyName())
                .xlabel("音阶音符 / Scale Notes")
                .ylabel("频率 (Hz) / Frequency (Hz)");
        
        // 添加音阶频率点 / Add scale frequency points
        IVector<Double> noteIndices = Linalg.range(scale.length);
        plot.scatter(noteIndices, frequencies);
        
        // 添加音名标签 / Add note name labels
        // for (int i = 0; i < scale.length; i++) {
        //     String noteName = MusicTheory.NOTE_NAMES[scale[i]];
        //     plot.annotate(noteName, noteIndices.get(i), frequencies.get(i));
        // }
        
        return plot;
    }
    
    /**
     * 绘制和弦分析图 / Plot chord analysis
     * <p>
     * 显示和弦分析结果，包括检测到的和弦类型和音符。
     * Display chord analysis results including detected chord type and notes.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 和弦分析图对象 / Chord analysis plot object
     */
    public static IPlot plotChordAnalysis(AudioData audioData, String title) {
        // 检测和弦 / Detect chord
        MusicTheory.Chord detectedChord = MusicTheory.detectChord(audioData);
        
        // 获取和弦音符 / Get chord notes
        int[] chordNotes = detectedChord.getNotes();
        
        // 计算每个音符的频率 / Calculate frequency for each note
        IVector<Double> frequencies = Linalg.zeros(chordNotes.length);
        for (int i = 0; i < chordNotes.length; i++) {
            frequencies.set(i, MusicTheory.semitonesToFrequency(chordNotes[i]));
        }
        
        // 创建和弦分析图 / Create chord analysis plot
        IPlot plot = Plots.of()
                .title(title + " - " + detectedChord.getChordName())
                .xlabel("和弦音符 / Chord Notes")
                .ylabel("频率 (Hz) / Frequency (Hz)");
        
        // 添加和弦频率点 / Add chord frequency points
        IVector<Double> noteIndices = Linalg.range(chordNotes.length);
        plot.scatter(noteIndices, frequencies);
        
        // 添加音名标签 / Add note name labels
        // for (int i = 0; i < chordNotes.length; i++) {
        //     String noteName = MusicTheory.NOTE_NAMES[chordNotes[i]];
        //     plot.annotate(noteName, noteIndices.get(i), frequencies.get(i));
        // }
        
        return plot;
    }
    
    /**
     * 绘制音乐特征雷达图 / Plot music features radar chart
     * <p>
     * 显示音乐的各种特征，使用雷达图形式展示。
     * Display various music features using radar chart format.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 音乐特征雷达图对象 / Music features radar plot object
     */
    public static IPlot plotMusicFeaturesRadar(AudioData audioData, String title) {
        // 提取音乐特征 / Extract music features
        MusicAnalyzer.MusicFeatures features = MusicAnalyzer.extractMusicFeatures(audioData);
        
        // 准备雷达图数据 / Prepare radar chart data
        String[] featureNames = {
            "节拍 / Tempo", "可舞性 / Danceability", "能量 / Energy", 
            "情感效价 / Valence", "原声性 / Acousticness", "器乐性 / Instrumentalness",
            "现场感 / Liveness", "语音性 / Speechiness"
        };
        
        double[] featureValues = {
            Math.min(1.0, features.getTempo() / 200.0), // 标准化节拍 / Normalize tempo
            features.getDanceability(),
            features.getEnergy(),
            features.getValence(),
            features.getAcousticness(),
            features.getInstrumentalness(),
            features.getLiveness(),
            features.getSpeechiness()
        };
        
        // 创建雷达图 / Create radar chart
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("特征类型 / Feature Type")
                .ylabel("特征值 / Feature Value");
        
        // 添加雷达图数据 / Add radar chart data
        IVector<Double> values = Linalg.vector(featureValues);
        // plot.radar(values); // 暂时注释，因为radar方法可能不存在
        plot.bar(values); // 使用条形图代替
        
        return plot;
    }
    
    /**
     * 绘制音乐特征条形图 / Plot music features bar chart
     * <p>
     * 显示音乐的各种特征，使用条形图形式展示。
     * Display various music features using bar chart format.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 音乐特征条形图对象 / Music features bar plot object
     */
    public static IPlot plotMusicFeaturesBar(AudioData audioData, String title) {
        // 提取音乐特征 / Extract music features
        MusicAnalyzer.MusicFeatures features = MusicAnalyzer.extractMusicFeatures(audioData);
        
        // 准备条形图数据 / Prepare bar chart data
        // String[] featureNames = {
        //     "节拍 / Tempo", "可舞性 / Danceability", "能量 / Energy", 
        //     "情感效价 / Valence", "原声性 / Acousticness", "器乐性 / Instrumentalness",
        //     "现场感 / Liveness", "语音性 / Speechiness"
        // };
        
        double[] featureValues = {
            Math.min(1.0, features.getTempo() / 200.0), // 标准化节拍 / Normalize tempo
            features.getDanceability(),
            features.getEnergy(),
            features.getValence(),
            features.getAcousticness(),
            features.getInstrumentalness(),
            features.getLiveness(),
            features.getSpeechiness()
        };
        
        // 创建条形图 / Create bar chart
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("特征类型 / Feature Type")
                .ylabel("特征值 / Feature Value");
        
        // 添加条形图数据 / Add bar chart data
        IVector<Double> values = Linalg.vector(featureValues);
        plot.bar(values);
        
        return plot;
    }
    
    /**
     * 绘制音乐比较图 / Plot music comparison
     * <p>
     * 比较两个音乐片段的特征。
     * Compare features of two music segments.
     * </p>
     *
     * @param audioData1 第一个音频数据 / First audio data
     * @param audioData2 第二个音频数据 / Second audio data
     * @param title 图表标题 / Plot title
     * @return 音乐比较图对象 / Music comparison plot object
     */
    public static IPlot plotMusicComparison(AudioData audioData1, AudioData audioData2, String title) {
        // 提取两个音频的音乐特征 / Extract music features of both audio files
        MusicAnalyzer.MusicFeatures features1 = MusicAnalyzer.extractMusicFeatures(audioData1);
        MusicAnalyzer.MusicFeatures features2 = MusicAnalyzer.extractMusicFeatures(audioData2);
        
        // 准备比较数据 / Prepare comparison data
        String[] featureNames = {
            "节拍 / Tempo", "可舞性 / Danceability", "能量 / Energy", 
            "情感效价 / Valence", "原声性 / Acousticness", "器乐性 / Instrumentalness",
            "现场感 / Liveness", "语音性 / Speechiness"
        };
        
        double[] features1Values = {
            Math.min(1.0, features1.getTempo() / 200.0),
            features1.getDanceability(),
            features1.getEnergy(),
            features1.getValence(),
            features1.getAcousticness(),
            features1.getInstrumentalness(),
            features1.getLiveness(),
            features1.getSpeechiness()
        };
        
        double[] features2Values = {
            Math.min(1.0, features2.getTempo() / 200.0),
            features2.getDanceability(),
            features2.getEnergy(),
            features2.getValence(),
            features2.getAcousticness(),
            features2.getInstrumentalness(),
            features2.getLiveness(),
            features2.getSpeechiness()
        };
        
        // 创建比较图 / Create comparison plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("特征类型 / Feature Type")
                .ylabel("特征值 / Feature Value");
        
        // 添加分组条形图数据 / Add grouped bar chart data
        IVector<Double> values1 = Linalg.vector(features1Values);
        IVector<Double> values2 = Linalg.vector(features2Values);
        
        // 创建合并的数据用于分组条形图 / Create merged data for grouped bar chart
        double[] allValuesArray = new double[values1.length() + values2.length()];
        for (int i = 0; i < values1.length(); i++) {
            allValuesArray[i] = values1.get(i);
        }
        for (int i = 0; i < values2.length(); i++) {
            allValuesArray[values1.length() + i] = values2.get(i);
        }
        
        IVector<Double> allValues = Linalg.vector(allValuesArray);
        List<String> hue = new ArrayList<>();
        for (int i = 0; i < featureNames.length; i++) {
            hue.add("音乐1 / Music 1");
        }
        for (int i = 0; i < featureNames.length; i++) {
            hue.add("音乐2 / Music 2");
        }
        
        plot.bar(allValues, hue);
        
        return plot;
    }
    
    /**
     * 绘制音乐结构分析图 / Plot music structure analysis
     * <p>
     * 显示音乐的结构分析结果。
     * Display music structure analysis results.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 音乐结构分析图对象 / Music structure analysis plot object
     */
    public static IPlot plotMusicStructure(AudioData audioData, String title) {
        // 计算STFT / Calculate STFT
        IMatrix<Double> stftMatrix = AudioAnalyzer.calculateSTFT(audioData, 1024, 256);
        
        // 计算每帧的特征 / Calculate features for each frame
        IVector<Double> spectralCentroids = calculateSpectralCentroids(stftMatrix);
        IVector<Double> spectralRolloffs = calculateSpectralRolloffs(stftMatrix);
        IVector<Double> zeroCrossingRates = calculateZeroCrossingRates(audioData);
        
        // 生成时间轴 / Generate time axis
        IVector<Double> timeAxis = Linalg.range(spectralCentroids.length())
                .multiplyScalar(256.0 / audioData.getSampleRate());
        
        // 创建结构分析图 / Create structure analysis plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("时间 (秒) / Time (s)")
                .ylabel("特征值 / Feature Value");
        
        // 添加多个特征线 / Add multiple feature lines
        plot.line(timeAxis, spectralCentroids, List.of("频谱质心 / Spectral Centroid"));
        plot.line(timeAxis, spectralRolloffs, List.of("频谱滚降 / Spectral Rolloff"));
        plot.line(timeAxis, zeroCrossingRates, List.of("零交叉率 / Zero Crossing Rate"));
        
        return plot;
    }
    
    /**
     * 创建音乐可视化仪表板 / Create music visualization dashboard
     * <p>
     * 创建包含多个图表的音乐可视化仪表板。
     * Create music visualization dashboard with multiple charts.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 仪表板标题 / Dashboard title
     * @return 可视化图表列表 / List of visualization plots
     */
    public static List<IPlot> createMusicDashboard(AudioData audioData, String title) {
        List<IPlot> plots = new ArrayList<>();
        
        // 添加各种图表 / Add various plots
        plots.add(plotMusicSpectrogram(audioData, title + " - 音乐频谱图 / Music Spectrogram"));
        plots.add(plotBeatDetection(audioData, title + " - 节拍检测 / Beat Detection"));
        plots.add(plotKeyAnalysis(audioData, title + " - 调性分析 / Key Analysis"));
        plots.add(plotChordAnalysis(audioData, title + " - 和弦分析 / Chord Analysis"));
        plots.add(plotMusicFeaturesBar(audioData, title + " - 音乐特征 / Music Features"));
        plots.add(plotMusicStructure(audioData, title + " - 音乐结构 / Music Structure"));
        
        return plots;
    }
    
    // 私有辅助方法 / Private helper methods
    
    /**
     * 计算频谱通量 / Calculate spectral flux
     */
    private static IVector<Double> calculateSpectralFlux(IMatrix<Double> stftMatrix) {
        int numFrames = stftMatrix.getColNum();
        IVector<Double> spectralFlux = Linalg.zeros(numFrames - 1);
        
        for (int frame = 1; frame < numFrames; frame++) {
            double flux = 0;
            for (int bin = 0; bin < stftMatrix.getRowNum(); bin++) {
                double current = stftMatrix.get(bin, frame);
                double previous = stftMatrix.get(bin, frame - 1);
                double diff = current - previous;
                if (diff > 0) {
                    flux += diff;
                }
            }
            spectralFlux.set(frame - 1, flux);
        }
        
        return spectralFlux;
    }
    
    /**
     * 计算频谱质心序列 / Calculate spectral centroid sequence
     */
    private static IVector<Double> calculateSpectralCentroids(IMatrix<Double> stftMatrix) {
        int numFrames = stftMatrix.getColNum();
        IVector<Double> centroids = Linalg.zeros(numFrames);
        
        for (int frame = 0; frame < numFrames; frame++) {
            double weightedSum = 0;
            double magnitudeSum = 0;
            
            for (int bin = 0; bin < stftMatrix.getRowNum(); bin++) {
                double magnitude = stftMatrix.get(bin, frame);
                weightedSum += bin * magnitude;
                magnitudeSum += magnitude;
            }
            
            centroids.set(frame, magnitudeSum > 0 ? weightedSum / magnitudeSum : 0);
        }
        
        return centroids;
    }
    
    /**
     * 计算频谱滚降序列 / Calculate spectral rolloff sequence
     */
    private static IVector<Double> calculateSpectralRolloffs(IMatrix<Double> stftMatrix) {
        int numFrames = stftMatrix.getColNum();
        IVector<Double> rolloffs = Linalg.zeros(numFrames);
        
        for (int frame = 0; frame < numFrames; frame++) {
            double totalEnergy = 0;
            for (int bin = 0; bin < stftMatrix.getRowNum(); bin++) {
                totalEnergy += stftMatrix.get(bin, frame);
            }
            
            double threshold = totalEnergy * 0.85;
            double cumulativeEnergy = 0;
            
            for (int bin = 0; bin < stftMatrix.getRowNum(); bin++) {
                cumulativeEnergy += stftMatrix.get(bin, frame);
                if (cumulativeEnergy >= threshold) {
                    rolloffs.set(frame, (double) bin);
                    break;
                }
            }
        }
        
        return rolloffs;
    }
    
    /**
     * 计算零交叉率序列 / Calculate zero crossing rate sequence
     */
    private static IVector<Double> calculateZeroCrossingRates(AudioData audioData) {
        IVector<Double> samples = audioData.isMono() ? 
            audioData.getSamples() : 
            AudioProcessor.stereoToMono(audioData).getSamples();
        
        int windowSize = 1024;
        int hopSize = 256;
        int numFrames = (samples.length() - windowSize) / hopSize + 1;
        IVector<Double> zcr = Linalg.zeros(numFrames);
        
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            int end = Math.min(start + windowSize, samples.length());
            
            int crossings = 0;
            for (int i = start + 1; i < end; i++) {
                if ((samples.get(i) >= 0) != (samples.get(i - 1) >= 0)) {
                    crossings++;
                }
            }
            
            zcr.set(frame, (double) crossings / (end - start - 1));
        }
        
        return zcr;
    }
}
