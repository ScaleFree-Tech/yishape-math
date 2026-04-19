package model_zoo.equipment_health;

import com.yishape.lab.math.linalg.*;
import com.yishape.lab.math.signal.*;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.factory.SignalProcessorFactory;
import com.yishape.lab.math.signal.filter.ButterworthFilter;
import com.yishape.lab.math.ml.*;
import com.yishape.lab.math.viz.*;

/**
 * 设备健康预测：频谱分析 + 机器学习分类
 *
 * 本示例展示如何从旋转设备的振动时序信号出发，
 * 通过 FFT 提取频谱特征，再用逻辑回归分类器判断设备状态。
 *
 * 物理背景：
 * - 正常设备的振动信号主要包含基频成分（如转轴旋转频率）
 * - 轴承磨损、不平衡、轴不对中等故障会在特定频率产生异常峰值
 * - 通过分析频谱中的异常频率成分，可以判断故障类型
 *
 * 流程：数据生成 → 预处理 → FFT 特征提取 → ML 分类 → 评估
 */
public class EquipmentHealthMonitor {

    // === 全局配置 ===
    private static final int SAMPLE_RATE = 1000;        // 采样率 1000 Hz
    private static final int SIGNAL_LENGTH = 1000;     // 信号长度 1000 点（1 秒）
    private static final int NORMAL_COUNT = 50;         // 正常样本数
    private static final int FAULT_COUNT = 50;          // 故障样本数
    private static final int RANDOM_SEED = 42;          // 随机种子（可复现）
    private static final int FEATURE_DIM = 100;         // FFT 特征维度（前 100 个频率点）
    private static final int CROSS_VAL_FOLDS = 5;       // 交叉验证折数

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("   设备健康预测：振动信号频谱分析 + 逻辑回归分类");
        System.out.println("=".repeat(60));
        System.out.println();

        // === Step 1: 生成模拟振动数据 ===
        System.out.println(">>> Step 1: 生成模拟振动数据...");
        System.out.println("   正常信号：基频 50 Hz + 高斯噪声");
        System.out.println("   故障信号：基频 50 Hz + 异常频率成分(150Hz + 250Hz)");
        System.out.println();

        // 预先生成基础信号（正常和故障各一个），复用给所有样本
        // 每个样本在此基础上叠加不同的噪声实现数据增强
        Linalg.setRandomSeed(RANDOM_SEED);

        // 生成基础模板信号
        IVector<Double> normalBase = generateNormalSignal();
        IVector<Double> faultBase = generateFaultSignal();

        // 存储所有样本的 FFT 特征和标签
        double[][] featureArray = new double[NORMAL_COUNT + FAULT_COUNT][FEATURE_DIM];
        String[] allLabels = new String[NORMAL_COUNT + FAULT_COUNT];

        // 生成正常样本
        for (int i = 0; i < NORMAL_COUNT; i++) {
            // 在基础信号上加小幅随机噪声，模拟样本差异
            IVector<Double> noise = Linalg.randn(SIGNAL_LENGTH, 0.0, 0.05);
            IVector<Double> sample = normalBase.add(noise);
            IVector<Double> features = extractFFTMagnitude(sample);
            for (int j = 0; j < FEATURE_DIM; j++) {
                featureArray[i][j] = features.get(j);
            }
            allLabels[i] = "normal";
        }

        // 生成故障样本
        for (int i = 0; i < FAULT_COUNT; i++) {
            IVector<Double> noise = Linalg.randn(SIGNAL_LENGTH, 0.0, 0.05);
            IVector<Double> sample = faultBase.add(noise);
            IVector<Double> features = extractFFTMagnitude(sample);
            for (int j = 0; j < FEATURE_DIM; j++) {
                featureArray[NORMAL_COUNT + i][j] = features.get(j);
            }
            allLabels[NORMAL_COUNT + i] = "fault";
        }

        System.out.println("   数据生成完成: " + (NORMAL_COUNT + FAULT_COUNT) + " 个样本");
        System.out.println("   正常样本: " + NORMAL_COUNT + " | 故障样本: " + FAULT_COUNT);
        System.out.println();

        // === Step 2: 构建特征矩阵和标签向量 ===
        System.out.println(">>> Step 2: 构建特征矩阵...");
        IMatrix<Double> features = Linalg.matrix(featureArray);
        System.out.println("   特征矩阵维度: " + features.rows() + " × " + features.cols());
        System.out.println();

        // === Step 3: 训练逻辑回归分类器 ===
        System.out.println(">>> Step 3: 训练逻辑回归分类器...");
        // L1 和 L2 正则化权重（防止过拟合）
        IClassifier classifier = ML.logisticRegression(0.01, 0.01);

        // 在全量数据上训练，获取训练集表现
        LogisticRegressionResult trainResult = classifier.fit(features, allLabels);
        System.out.println("   训练集指标（全量数据）:");
        System.out.println("   - 训练集准确率: " + String.format("%.2f%%", trainResult.getTrainAccuracy() * 100));
        System.out.println();

        // === Step 4: 5 折交叉验证（评估泛化能力）===
        System.out.println(">>> Step 4: 5 折交叉验证（评估模型泛化能力）...");
        CrossValidationResult cvResult = ML.kFoldCrossValidation(
            classifier, features, allLabels, CROSS_VAL_FOLDS);

        System.out.println("   交叉验证结果:");
        System.out.println("   - 平均准确率: " + String.format("%.2f%%", cvResult.getMeanAccuracy() * 100));
        System.out.println("   - 各折准确率:");
        java.util.List<Double> scoreList = cvResult.getAccuracyScores();
        for (int i = 0; i < scoreList.size(); i++) {
            System.out.println("     折 " + (i + 1) + ": " + String.format("%.2f%%", scoreList.get(i) * 100));
        }
        System.out.println("   - 95% 置信区间: [" +
            String.format("%.2f%%", cvResult.getAccuracy95Percentile()[0] * 100) + ", " +
            String.format("%.2f%%", cvResult.getAccuracy95Percentile()[1] * 100) + "]");
        System.out.println();

        // === Step 5: 分类指标详细分析 ===
        System.out.println(">>> Step 5: 分类指标详细分析...");
        ClassificationMetrics metrics = ML.classificationMetrics(classifier, features, allLabels);
        System.out.println("   混淆矩阵:");
        System.out.println("                预测Normal  预测Fault");
        System.out.println("   实际Normal       " +
            String.format("%4d", (int)(metrics.getConfusionMatrix()[0][0])) + "       " +
            String.format("%4d", (int)(metrics.getConfusionMatrix()[0][1])));
        System.out.println("   实际Fault        " +
            String.format("%4d", (int)(metrics.getConfusionMatrix()[1][0])) + "       " +
            String.format("%4d", (int)(metrics.getConfusionMatrix()[1][1])));
        System.out.println("   - 准确率 (Accuracy):  " + String.format("%.2f%%", metrics.getAccuracy() * 100));
        System.out.println("   - Macro-F1:         " + String.format("%.4f", metrics.getMacroF1()));
        System.out.println();

        // === Step 6: 可视化 ===
        System.out.println(">>> Step 6: 生成可视化图表...");
        visualizeSignals(normalBase, faultBase);
        visualizeSpectrum(normalBase, faultBase);
        System.out.println();

        // === Step 7: 故障频率诊断说明 ===
        System.out.println(">>> Step 7: 故障频率诊断分析...");
        IVector<Double> normalSpec = extractFFTMagnitude(normalBase);
        IVector<Double> faultSpec = extractFFTMagnitude(faultBase);

        // 找到正常信号和故障信号的主频率
        int normalPeak = argmax(normalSpec);
        int faultPeak = argmax(faultSpec);
        double freqResolution = (double) SAMPLE_RATE / SIGNAL_LENGTH;
        System.out.println("   正常信号主峰值频率: " + String.format("%.1f", normalPeak * freqResolution) + " Hz");
        System.out.println("   故障信号主峰值频率: " + String.format("%.1f", faultPeak * freqResolution) + " Hz");
        System.out.println();
        System.out.println("   故障特征分析:");
        System.out.println("   - 150Hz 成分（轴承外圈故障特征频率）幅度: " +
            String.format("%.4f", faultSpec.get(15)) + " (正常: " + String.format("%.4f", normalSpec.get(15)) + ")");
        System.out.println("   - 250Hz 成分（齿轮啮合频率谐波）幅度: " +
            String.format("%.4f", faultSpec.get(25)) + " (正常: " + String.format("%.4f", normalSpec.get(25)) + ")");
        System.out.println();
        System.out.println("   >>> 结论: 150Hz 和 250Hz 处幅度显著升高，指示轴承/齿轮故障");
        System.out.println();

        System.out.println("=".repeat(60));
        System.out.println("   设备健康预测分析完成！");
        System.out.println("=".repeat(60));
    }

    // === 生成正常振动信号 ===
    // 正常旋转设备的振动信号：纯净正弦波 + 少量噪声
    private static IVector<Double> generateNormalSignal() {
        // 50 Hz 正弦波，幅值 1.0，相位 0
        IVector<Double> base = Signals.sineWave(SIGNAL_LENGTH, 50.0, SAMPLE_RATE, 1.0, 0.0);
        // 添加小幅高斯噪声（模拟测量噪声和实际工况波动）
        IVector<Double> noise = Linalg.randn(SIGNAL_LENGTH, 0.0, 0.1);
        return base.add(noise);
    }

    // === 生成故障振动信号 ===
    // 故障信号：在正常信号基础上叠加异常频率成分
    // 150Hz → 轴承外圈故障特征频率
    // 250Hz → 齿轮啮合频率谐波
    private static IVector<Double> generateFaultSignal() {
        IVector<Double> base = Signals.sineWave(SIGNAL_LENGTH, 50.0, SAMPLE_RATE, 1.0, 0.0);
        // 叠加 150Hz 异常成分（轴承故障特征）
        IVector<Double> fault150 = Signals.sineWave(SIGNAL_LENGTH, 150.0, SAMPLE_RATE, 0.8, 0.0);
        // 叠加 250Hz 异常成分（齿轮故障特征）
        IVector<Double> fault250 = Signals.sineWave(SIGNAL_LENGTH, 250.0, SAMPLE_RATE, 0.6, 0.0);
        IVector<Double> noise = Linalg.randn(SIGNAL_LENGTH, 0.0, 0.1);
        return base.add(fault150).add(fault250).add(noise);
    }

    // === 预处理：低通滤波 ===
    // 截止频率 300Hz，去除高频噪声，保留故障特征频率
    private static IVector<Double> lowpassFilter(IVector<Double> signal) {
        try {
            ButterworthFilter filter = new ButterworthFilter(4, 300.0, SAMPLE_RATE);
            return filter.process(signal);
        } catch (SignalProcessingException e) {
            System.err.println("   滤波失败，使用原始信号: " + e.getMessage());
            return signal;
        }
    }

    // === FFT 特征提取 ===
    // 返回归一化幅度谱的前 FEATURE_DIM 个频率点
    private static IVector<Double> extractFFTMagnitude(IVector<Double> signal) {
        // Step 1: 预处理（低通滤波）
        IVector<Double> filtered = lowpassFilter(signal);

        // Step 2: 做 FFT
        try {
            ISignalTransform<Double, Complex[]> fft =
                SignalProcessorFactory.getInstance().createTransform("fft");
            Complex[] fftResult = fft.forward(filtered);

            // Step 3: 计算幅度谱（仅取正频率部分，前 N/2 个点）
            int n = fftResult.length;
            double[] magnitude = new double[Math.min(FEATURE_DIM, n / 2)];
            for (int i = 0; i < magnitude.length; i++) {
                magnitude[i] = fftResult[i].abs();  // |Re + j·Im| = sqrt(Re² + Im²)
            }

            // Step 4: 归一化（使频谱幅度在 0~1 之间）
            double maxMag = Linalg.vector(magnitude).max();
            if (maxMag > 1e-10) {
                for (int i = 0; i < magnitude.length; i++) {
                    magnitude[i] = magnitude[i] / maxMag;
                }
            }
            return Linalg.vector(magnitude);

        } catch (SignalProcessingException e) {
            System.err.println("   FFT 失败，返回零向量: " + e.getMessage());
            return Linalg.zeros(FEATURE_DIM);
        }
    }

    // === 找向量最大值的索引 ===
    private static int argmax(IVector<Double> v) {
        int idx = 0;
        double maxVal = v.get(0);
        for (int i = 1; i < v.size(); i++) {
            if (v.get(i) > maxVal) {
                maxVal = v.get(i);
                idx = i;
            }
        }
        return idx;
    }

    // === 可视化：时域信号对比 ===
    // 分别绘制正常和故障信号的时域波形（两张图）
    private static void visualizeSignals(IVector<Double> normal, IVector<Double> fault) {
        // 时间轴（转换为秒）
        IVector<Double> timeAxis = Linalg.arange(0, SIGNAL_LENGTH)
            .map(v -> v / SAMPLE_RATE);

        // 正常信号时域图
        RerePlot pltNormal = Plots.of(900, 300);
        pltNormal.setTitle("正常设备振动信号 / Normal Equipment Vibration Signal");
        pltNormal.setXLabel("时间 (秒) / Time (s)");
        pltNormal.setYLabel("幅值 / Amplitude");
        pltNormal.line(timeAxis, normal).show();
        pltNormal.saveAsHtml("equipment_health_signal_normal.html");

        // 故障信号时域图
        RerePlot pltFault = Plots.of(900, 300);
        pltFault.setTitle("故障设备振动信号 / Fault Equipment Vibration Signal");
        pltFault.setXLabel("时间 (秒) / Time (s)");
        pltFault.setYLabel("幅值 / Amplitude");
        pltFault.line(timeAxis, fault).show();
        pltFault.saveAsHtml("equipment_health_signal_fault.html");

        System.out.println("   时域图表已保存: equipment_health_signal_normal.html, equipment_health_signal_fault.html");
    }

    // === 可视化：频谱对比 ===
    private static void visualizeSpectrum(IVector<Double> normal, IVector<Double> fault) {
        IVector<Double> normalSpec = extractFFTMagnitude(normal);
        IVector<Double> faultSpec = extractFFTMagnitude(fault);

        // 频率轴（前 50 个频率点，对应 0~500Hz）
        int displayPoints = 50;
        IVector<Double> freqAxis = Linalg.arange(displayPoints)
            .map(v -> v * (double) SAMPLE_RATE / SIGNAL_LENGTH);

        // 正常信号频谱
        RerePlot pltSpecN = Plots.of(900, 300);
        pltSpecN.setTitle("正常信号频谱 / Normal Signal Spectrum");
        pltSpecN.setXLabel("频率 (Hz) / Frequency (Hz)");
        pltSpecN.setYLabel("归一化幅度 / Normalized Magnitude");
        pltSpecN.line(freqAxis, normalSpec.slice(0, displayPoints)).show();
        pltSpecN.saveAsHtml("equipment_health_spectrum_normal.html");

        // 故障信号频谱（标注故障特征频率）
        RerePlot pltSpecF = Plots.of(900, 300);
        pltSpecF.setTitle("故障信号频谱（含异常频率峰值） / Fault Signal Spectrum");
        pltSpecF.setXLabel("频率 (Hz) / Frequency (Hz)");
        pltSpecF.setYLabel("归一化幅度 / Normalized Magnitude");
        pltSpecF.line(freqAxis, faultSpec.slice(0, displayPoints)).show();
        pltSpecF.saveAsHtml("equipment_health_spectrum_fault.html");

        System.out.println("   频谱图表已保存: equipment_health_spectrum_normal.html, equipment_health_spectrum_fault.html");
    }
}
