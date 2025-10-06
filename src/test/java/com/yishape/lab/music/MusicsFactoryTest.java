package com.yishape.lab.music;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.music.analysis.BasicMusicAnalyzer;
import com.yishape.lab.music.analysis.ComprehensiveMusicAnalyzer;
import com.yishape.lab.music.analysis.AdvancedMusicAnalyzer;
import com.yishape.lab.music.analysis.basic.IBeatAnalyzer;
import com.yishape.lab.music.analysis.basic.IKeyAnalyzer;
import com.yishape.lab.music.analysis.basic.IChordAnalyzer;
import com.yishape.lab.music.analysis.advanced.IAdvancedAnalyzer;
import com.yishape.lab.music.analysis.feature.IFeatureExtractor;
import com.yishape.lab.music.processing.IMusicProcessor;
import com.yishape.lab.music.generation.ChordGenerator;
import com.yishape.lab.music.generation.IntervalGenerator;
import com.yishape.lab.music.generation.ScaleGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试Musics工厂方法 / Test Musics factory methods
 */
public class MusicsFactoryTest {
    
    private AudioData testAudioData;
    
    @BeforeEach
    public void setUp() {
        // 创建测试音频数据 / Create test audio data
        double[] samples = new double[44100]; // 1秒的音频数据 / 1 second of audio data
        for (int i = 0; i < samples.length; i++) {
            samples[i] = Math.sin(2 * Math.PI * 440 * i / 44100); // 440Hz 正弦波 / 440Hz sine wave
        }
        IVector<Double> sampleVector = Linalg.vector(samples);
        testAudioData = new AudioData(sampleVector, 44100, 1, 16, AudioFormat.WAV);
    }
    
    @Test
    public void testCreateBasicMusicAnalyzer() {
        BasicMusicAnalyzer analyzer = Musics.createBasicMusicAnalyzer();
        assertNotNull(analyzer, "BasicMusicAnalyzer should be created");
    }
    
    @Test
    public void testCreateComprehensiveMusicAnalyzer() {
        ComprehensiveMusicAnalyzer analyzer = Musics.createComprehensiveMusicAnalyzer();
        assertNotNull(analyzer, "ComprehensiveMusicAnalyzer should be created");
    }
    
    @Test
    public void testCreateAdvancedMusicAnalyzer() {
        AdvancedMusicAnalyzer analyzer = Musics.createAdvancedMusicAnalyzer();
        assertNotNull(analyzer, "AdvancedMusicAnalyzer should be created");
    }
    
    @Test
    public void testCreateBeatAnalyzer() {
        IBeatAnalyzer analyzer = Musics.createBeatAnalyzer();
        assertNotNull(analyzer, "IBeatAnalyzer should be created");
    }
    
    @Test
    public void testCreateKeyAnalyzer() {
        IKeyAnalyzer analyzer = Musics.createKeyAnalyzer();
        assertNotNull(analyzer, "IKeyAnalyzer should be created");
    }
    
    @Test
    public void testCreateChordAnalyzer() {
        IChordAnalyzer analyzer = Musics.createChordAnalyzer();
        assertNotNull(analyzer, "IChordAnalyzer should be created");
    }
    
    @Test
    public void testCreateEmotionAnalyzer() {
        IAdvancedAnalyzer analyzer = Musics.createEmotionAnalyzer();
        assertNotNull(analyzer, "Emotion analyzer should be created");
    }
    
    @Test
    public void testCreateGenreAnalyzer() {
        IAdvancedAnalyzer analyzer = Musics.createGenreAnalyzer();
        assertNotNull(analyzer, "Genre analyzer should be created");
    }
    
    @Test
    public void testCreateComplexityAnalyzer() {
        IAdvancedAnalyzer analyzer = Musics.createComplexityAnalyzer();
        assertNotNull(analyzer, "Complexity analyzer should be created");
    }
    
    @Test
    public void testCreateFeatureExtractor() {
        IFeatureExtractor extractor = Musics.createFeatureExtractor();
        assertNotNull(extractor, "Feature extractor should be created");
    }
    
    @Test
    public void testCreateProcessor() {
        IMusicProcessor processor = Musics.createProcessor("harmonizer");
        assertNotNull(processor, "Harmonizer processor should be created");
    }
    
    @Test
    public void testCreateHarmonizer() {
        IMusicProcessor processor = Musics.createHarmonizer();
        assertNotNull(processor, "Harmonizer should be created");
    }
    
    @Test
    public void testCreateQuantizer() {
        IMusicProcessor processor = Musics.createQuantizer();
        assertNotNull(processor, "Quantizer should be created");
    }
    
    @Test
    public void testCreateTransposer() {
        IMusicProcessor processor = Musics.createTransposer();
        assertNotNull(processor, "Transposer should be created");
    }
    
    @Test
    public void testCreateMusicTheoryProcessor() {
        IMusicProcessor processor = Musics.createMusicTheoryProcessor();
        assertNotNull(processor, "Music theory processor should be created");
    }
    
    @Test
    public void testCreateChordGenerator() {
        ChordGenerator generator = Musics.createChordGenerator();
        assertNotNull(generator, "Chord generator should be created");
    }
    
    @Test
    public void testCreateIntervalGenerator() {
        IntervalGenerator generator = Musics.createIntervalGenerator();
        assertNotNull(generator, "Interval generator should be created");
    }
    
    @Test
    public void testCreateScaleGenerator() {
        ScaleGenerator generator = Musics.createScaleGenerator();
        assertNotNull(generator, "Scale generator should be created");
    }
}