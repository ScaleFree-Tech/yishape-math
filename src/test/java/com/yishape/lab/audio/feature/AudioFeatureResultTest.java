package com.yishape.lab.audio.feature;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * AudioFeatureResult测试类
 */
public class AudioFeatureResultTest {

    @Test
    public void testToNumericalFeatures() {
        // 创建测试数据
        double spectralCentroid = 1000.0;
        double spectralBandwidth = 2000.0;
        double spectralRolloff = 3000.0;
        double zeroCrossingRate = 0.5;
        double[] mfcc = {0.1, 0.2, 0.3, 0.4, 0.5};
        double[] spectralContrast = {0.6, 0.7, 0.8};
        double sampleRate = 44100.0;

        // 创建AudioFeatureResult实例
        AudioFeatureResult result = new AudioFeatureResult(
                spectralCentroid,
                spectralBandwidth,
                spectralRolloff,
                zeroCrossingRate,
                mfcc,
                spectralContrast,
                sampleRate
        );

        // 调用toNumericalFeatures方法
        Tuple2<List<String>, IVector<Double>> numericalFeatures = result.toNumericalFeatures();

        // 验证结果
        assertNotNull(numericalFeatures);
        assertNotNull(numericalFeatures._1); // 特征名称列表
        assertNotNull(numericalFeatures._2); // 特征值向量

        List<String> featureNames = numericalFeatures._1;
        IVector<Double> featureVector = numericalFeatures._2;

        // 验证特征数量
        int expectedFeatureCount = 4 + mfcc.length + spectralContrast.length; // 4个基本特征 + 5个MFCC + 3个频谱对比度
        assertEquals(expectedFeatureCount, featureNames.size());
        assertEquals(expectedFeatureCount, featureVector.length());

        // 验证特征名称
        assertEquals("spectralCentroid", featureNames.get(0));
        assertEquals("spectralBandwidth", featureNames.get(1));
        assertEquals("spectralRolloff", featureNames.get(2));
        assertEquals("zeroCrossingRate", featureNames.get(3));
        assertEquals("mfccMeans_0", featureNames.get(4));
        assertEquals("mfccMeans_1", featureNames.get(5));
        assertEquals("mfccMeans_2", featureNames.get(6));
        assertEquals("mfccMeans_3", featureNames.get(7));
        assertEquals("mfccMeans_4", featureNames.get(8));
        assertEquals("spectralContrast_0", featureNames.get(9));
        assertEquals("spectralContrast_1", featureNames.get(10));
        assertEquals("spectralContrast_2", featureNames.get(11));

        // 验证特征值
        assertEquals(spectralCentroid, featureVector.get(0), 1e-10);
        assertEquals(spectralBandwidth, featureVector.get(1), 1e-10);
        assertEquals(spectralRolloff, featureVector.get(2), 1e-10);
        assertEquals(zeroCrossingRate, featureVector.get(3), 1e-10);
        assertEquals(0.1, featureVector.get(4), 1e-10);
        assertEquals(0.2, featureVector.get(5), 1e-10);
        assertEquals(0.3, featureVector.get(6), 1e-10);
        assertEquals(0.4, featureVector.get(7), 1e-10);
        assertEquals(0.5, featureVector.get(8), 1e-10);
        assertEquals(0.6, featureVector.get(9), 1e-10);
        assertEquals(0.7, featureVector.get(10), 1e-10);
        assertEquals(0.8, featureVector.get(11), 1e-10);
    }

    @Test
    public void testToNumericalFeaturesWithEmptyArrays() {
        // 创建测试数据，MFCC和频谱对比度数组为空
        double spectralCentroid = 1000.0;
        double spectralBandwidth = 2000.0;
        double spectralRolloff = 3000.0;
        double zeroCrossingRate = 0.5;
        double[] mfcc = {};
        double[] spectralContrast = {};
        double sampleRate = 44100.0;

        // 创建AudioFeatureResult实例
        AudioFeatureResult result = new AudioFeatureResult(
                spectralCentroid,
                spectralBandwidth,
                spectralRolloff,
                zeroCrossingRate,
                mfcc,
                spectralContrast,
                sampleRate
        );

        // 调用toNumericalFeatures方法
        Tuple2<List<String>, IVector<Double>> numericalFeatures = result.toNumericalFeatures();

        // 验证结果
        assertNotNull(numericalFeatures);
        assertNotNull(numericalFeatures._1); // 特征名称列表
        assertNotNull(numericalFeatures._2); // 特征值向量

        List<String> featureNames = numericalFeatures._1;
        IVector<Double> featureVector = numericalFeatures._2;

        // 验证特征数量（只有4个基本特征）
        int expectedFeatureCount = 4;
        assertEquals(expectedFeatureCount, featureNames.size());
        assertEquals(expectedFeatureCount, featureVector.length());

        // 验证特征名称
        assertEquals("spectralCentroid", featureNames.get(0));
        assertEquals("spectralBandwidth", featureNames.get(1));
        assertEquals("spectralRolloff", featureNames.get(2));
        assertEquals("zeroCrossingRate", featureNames.get(3));

        // 验证特征值
        assertEquals(spectralCentroid, featureVector.get(0), 1e-10);
        assertEquals(spectralBandwidth, featureVector.get(1), 1e-10);
        assertEquals(spectralRolloff, featureVector.get(2), 1e-10);
        assertEquals(zeroCrossingRate, featureVector.get(3), 1e-10);
    }
}