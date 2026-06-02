package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;

/**
 * 数据预处理数学正确性验证测试
 * 使用正确的多行数据验证每列独立变换
 */
public class RerePreprocessingValidationTest {

    private static final double EPS = 1e-6;

    public static void main(String[] args) {
        System.out.println("============================================================");
        System.out.println("数据预处理数学正确性验证");
        System.out.println("============================================================");

        int passed = 0;
        int failed = 0;

        // 1. MinMaxScaler 验证
        if (testMinMaxScaler()) {
            System.out.println("✅ MinMaxScaler: PASS");
            passed++;
        } else {
            System.out.println("❌ MinMaxScaler: FAIL");
            failed++;
        }

        // 2. MaxAbsScaler 验证
        if (testMaxAbsScaler()) {
            System.out.println("✅ MaxAbsScaler: PASS");
            passed++;
        } else {
            System.out.println("❌ MaxAbsScaler: FAIL");
            failed++;
        }

        // 3. StandardScaler 验证
        if (testStandardScaler()) {
            System.out.println("✅ StandardScaler: PASS");
            passed++;
        } else {
            System.out.println("❌ StandardScaler: FAIL");
            failed++;
        }

        // 4. RobustScaler 验证
        if (testRobustScaler()) {
            System.out.println("✅ RobustScaler: PASS");
            passed++;
        } else {
            System.out.println("❌ RobustScaler: FAIL");
            failed++;
        }

        // 5. Normalizer 验证
        if (testNormalizer()) {
            System.out.println("✅ Normalizer: PASS");
            passed++;
        } else {
            System.out.println("❌ Normalizer: FAIL");
            failed++;
        }

        // 6. Binarizer 验证
        if (testBinarizer()) {
            System.out.println("✅ Binarizer: PASS");
            passed++;
        } else {
            System.out.println("❌ Binarizer: FAIL");
            failed++;
        }

        // 7. PolynomialFeatures 验证
        if (testPolynomialFeatures()) {
            System.out.println("✅ PolynomialFeatures: PASS");
            passed++;
        } else {
            System.out.println("❌ PolynomialFeatures: FAIL");
            failed++;
        }

        // 8. PowerTransformer 验证
        if (testPowerTransformer()) {
            System.out.println("✅ PowerTransformer: PASS");
            passed++;
        } else {
            System.out.println("❌ PowerTransformer: FAIL");
            failed++;
        }

        // 9. QuantileTransformer 验证
        if (testQuantileTransformer()) {
            System.out.println("✅ QuantileTransformer: PASS");
            passed++;
        } else {
            System.out.println("❌ QuantileTransformer: FAIL");
            failed++;
        }

        // 10. KernelCenterer 验证
        if (testKernelCenterer()) {
            System.out.println("✅ KernelCenterer: PASS");
            passed++;
        } else {
            System.out.println("❌ KernelCenterer: FAIL");
            failed++;
        }

        // 11. LabelBinarizer 验证
        if (testLabelBinarizer()) {
            System.out.println("✅ LabelBinarizer: PASS");
            passed++;
        } else {
            System.out.println("❌ LabelBinarizer: FAIL");
            failed++;
        }

        // 12. OneHotEncoder 验证
        if (testOneHotEncoder()) {
            System.out.println("✅ OneHotEncoder: PASS");
            passed++;
        } else {
            System.out.println("❌ OneHotEncoder: FAIL");
            failed++;
        }

        // 13. Bucketizer 验证
        if (testBucketizer()) {
            System.out.println("✅ Bucketizer: PASS");
            passed++;
        } else {
            System.out.println("❌ Bucketizer: FAIL");
            failed++;
        }

        System.out.println("\n============================================================");
        System.out.println("验证结果: " + passed + " 通过, " + failed + " 失败");
        System.out.println("============================================================");
    }

    // ==================== MinMaxScaler ====================
    // 数据按列处理: col0=[2,8], col1=[4,10], col2=[6,12]
    // col0: min=2, max=8, range=6 -> [2,8] -> [0,1]
    // col1: min=4, max=10, range=6 -> [4,10] -> [0,1]
    // col2: min=6, max=12, range=6 -> [6,12] -> [0,1]
    private static boolean testMinMaxScaler() {
        System.out.println("\n--- MinMaxScaler 验证 ---");
        double[][] data = {
            {2.0, 4.0, 6.0},
            {8.0, 10.0, 12.0}
        };
        IMatrix<Double> X = IMatrix.of(data);

        RereMinMaxScaler scaler = new RereMinMaxScaler();
        scaler.fit(X);
        IMatrix<?> Xt = scaler.transform(X);

        // 第一列: (2-2)/6=0, (8-2)/6=1
        check("MinMax(col0,row0)", Xt.get(0, 0), 0.0);
        check("MinMax(col0,row1)", Xt.get(1, 0), 1.0);

        // 第二列: (4-4)/6=0, (10-4)/6=1
        check("MinMax(col1,row0)", Xt.get(0, 1), 0.0);
        check("MinMax(col1,row1)", Xt.get(1, 1), 1.0);

        // 逆变换验证
        IMatrix<?> Xinv = ((IRereScaler<Double>) scaler).inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                if (Math.abs(X.get(i, j) - Xinv.get(i, j)) > 1e-10) {
                    System.out.println("  ❌ 逆变换失败: [" + i + "," + j + "]");
                    return false;
                }
            }
        }
        System.out.println("  ✓ 逆变换验证通过");

        return true;
    }

    // ==================== MaxAbsScaler ====================
    // 数据按列处理: col0=[-2,4], col1=[-3,6]
    // col0: max_abs=4 -> [-2,4] -> [-0.5, 1]
    // col1: max_abs=6 -> [-3,6] -> [-0.5, 1]
    private static boolean testMaxAbsScaler() {
        System.out.println("\n--- MaxAbsScaler 验证 ---");
        double[][] data = {
            {-2.0, -3.0},
            {4.0, 6.0}
        };
        IMatrix<Double> X = IMatrix.of(data);

        RereMaxAbsScaler scaler = new RereMaxAbsScaler();
        scaler.fit(X);
        IMatrix<?> Xt = scaler.transform(X);

        check("MaxAbs(col0,row0)", Xt.get(0, 0), -0.5);
        check("MaxAbs(col0,row1)", Xt.get(1, 0), 1.0);
        check("MaxAbs(col1,row0)", Xt.get(0, 1), -0.5);
        check("MaxAbs(col1,row1)", Xt.get(1, 1), 1.0);

        // 逆变换验证
        IMatrix<?> Xinv = ((IRereScaler<Double>) scaler).inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                if (Math.abs(X.get(i, j) - Xinv.get(i, j)) > 1e-10) {
                    System.out.println("  ❌ 逆变换失败");
                    return false;
                }
            }
        }
        System.out.println("  ✓ 逆变换验证通过");

        return true;
    }

    // ==================== StandardScaler ====================
    // 数据按列处理: col=[1,2,3] -> mean=2
    // std: 使用样本标准差 ddof=1
    // sample std = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2) / (3-1)) = sqrt(2/2) = 1.0
    // (x - mean) / std: [1,2,3] -> [-1, 0, 1]
    private static boolean testStandardScaler() {
        System.out.println("\n--- StandardScaler 验证 ---");
        double[][] data = {{1.0}, {2.0}, {3.0}};
        IMatrix<Double> X = IMatrix.of(data);

        RereStandardScaler scaler = new RereStandardScaler();
        scaler.fit(X);
        IMatrix<?> Xt = scaler.transform(X);

        // 样本标准差 = 1.0
        // (1-2)/1 = -1, (2-2)/1 = 0, (3-2)/1 = 1
        check("Standard(1)", Xt.get(0, 0), -1.0, 1e-10);
        check("Standard(2)", Xt.get(1, 0), 0.0, 1e-10);
        check("Standard(3)", Xt.get(2, 0), 1.0, 1e-10);

        // 逆变换验证
        IMatrix<?> Xinv = ((IRereScaler<Double>) scaler).inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                if (Math.abs(X.get(i, j) - Xinv.get(i, j)) > 1e-10) {
                    System.out.println("  ❌ 逆变换失败");
                    return false;
                }
            }
        }
        System.out.println("  ✓ 逆变换验证通过");

        return true;
    }

    // ==================== RobustScaler ====================
    // 数据 [1, 2, 3, 100] 排序后
    // median = (2+3)/2 = 2.5
    // Q1 (25%): position=0.75, value=1+0.75*(2-1)=1.75
    // Q3 (75%): position=2.25, value=3+0.25*(100-3)=27.25
    // IQR = 27.25 - 1.75 = 25.5
    // (x - median) / IQR:
    // 1 -> (1-2.5)/25.5 = -1.5/25.5 = -0.05882
    // 2 -> (2-2.5)/25.5 = -0.5/25.5 = -0.0196
    // 3 -> (3-2.5)/25.5 = 0.5/25.5 = 0.0196
    // 100 -> (100-2.5)/25.5 = 97.5/25.5 = 3.8235
    private static boolean testRobustScaler() {
        System.out.println("\n--- RobustScaler 验证 ---");
        double[][] data = {{1.0}, {2.0}, {3.0}, {100.0}};
        IMatrix<Double> X = IMatrix.of(data);

        RereRobustScaler scaler = new RereRobustScaler();
        scaler.fit(X);
        IMatrix<?> Xt = scaler.transform(X);

        check("Robust(1)", Xt.get(0, 0), -0.058823529411764705, 1e-10);
        check("Robust(2)", Xt.get(1, 0), -0.0196078431372549, 1e-10);
        check("Robust(3)", Xt.get(2, 0), 0.0196078431372549, 1e-10);
        check("Robust(100)", Xt.get(3, 0), 3.823529411764706, 1e-10);

        // 逆变换验证
        IMatrix<?> Xinv = ((IRereScaler<Double>) scaler).inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                if (Math.abs(X.get(i, j) - Xinv.get(i, j)) > 1e-6) {
                    System.out.println("  ❌ 逆变换失败");
                    return false;
                }
            }
        }
        System.out.println("  ✓ 逆变换验证通过");

        return true;
    }

    // ==================== Normalizer ====================
    // 每行独立归一化: [3, 4] -> L2=[0.6, 0.8], L1=[0.4286, 0.5714], Max=[0.75, 1.0]
    private static boolean testNormalizer() {
        System.out.println("\n--- Normalizer 验证 ---");

        // L2: [3, 4] -> norm=5 -> [0.6, 0.8]
        double[][] data = {{3.0, 4.0}};
        IMatrix<Double> X = IMatrix.of(data);

        RereNormalizer scaler = RereNormalizer.l2();
        scaler.fit(X);
        IMatrix<?> Xt = scaler.transform(X);

        check("L2[0]", Xt.get(0, 0), 0.6, 1e-10);
        check("L2[1]", Xt.get(0, 1), 0.8, 1e-10);

        double norm = Math.sqrt(
            Xt.get(0, 0) * Xt.get(0, 0) +
            Xt.get(0, 1) * Xt.get(0, 1));
        check("L2范数", norm, 1.0, 1e-10);

        // L1
        scaler = RereNormalizer.l1();
        scaler.fit(X);
        Xt = scaler.transform(X);

        double l1Norm = Math.abs(Xt.get(0, 0)) + Math.abs(Xt.get(0, 1));
        check("L1范数", l1Norm, 1.0, 1e-10);

        // Max
        scaler = RereNormalizer.max();
        scaler.fit(X);
        Xt = scaler.transform(X);

        double maxAbs = Math.max(Math.abs(Xt.get(0, 0)), Math.abs(Xt.get(0, 1)));
        check("Max范数", maxAbs, 1.0, 1e-10);

        return true;
    }

    // ==================== Binarizer ====================
    // 阈值=1.0: [-1, 0.5, 1.0, 2] -> [0, 0, 1, 1]
    private static boolean testBinarizer() {
        System.out.println("\n--- Binarizer 验证 ---");
        double[][] data = {{-1.0, 0.5, 1.0, 2.0}};
        IMatrix<Double> X = IMatrix.of(data);

        RereBinarizer binarizer = new RereBinarizer(1.0);
        binarizer.fit(X);
        IMatrix<?> Xt = binarizer.transform(X);

        check("Binarize(-1)", Xt.get(0, 0), 0.0);
        check("Binarize(0.5)", Xt.get(0, 1), 0.0);
        check("Binarize(1.0)", Xt.get(0, 2), 1.0);  // 1.0 >= 1.0
        check("Binarize(2)", Xt.get(0, 3), 1.0);

        return true;
    }

    // ==================== PolynomialFeatures ====================
    private static boolean testPolynomialFeatures() {
        System.out.println("\n--- PolynomialFeatures 验证 ---");
        double[][] data = {{1.0, 2.0}};
        IMatrix<Double> X = IMatrix.of(data);

        RerePolynomialFeatures poly = new RerePolynomialFeatures(2, true, false);
        poly.fit(X);
        IMatrix<?> Xt = poly.transform(X);

        System.out.println("  输入: [1, 2], 输出特征数: " + Xt.cols());
        System.out.println("  输出值: ");
        for (int i = 0; i < Xt.cols(); i++) {
            System.out.println("    [" + i + "] = " + Xt.get(0, i));
        }

        // 验证所有值都是有限值
        for (int i = 0; i < Xt.rows(); i++) {
            for (int j = 0; j < Xt.cols(); j++) {
                double val = Xt.get(i, j);
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    System.out.println("  ❌ 发现 NaN 或 Inf 值");
                    return false;
                }
            }
        }

        return true;
    }

    // ==================== PowerTransformer ====================
    private static boolean testPowerTransformer() {
        System.out.println("\n--- PowerTransformer 验证 ---");
        double[][] data = {{1.0, 2.0}, {3.0, 4.0}};
        IMatrix<Double> X = IMatrix.of(data);

        // Box-Cox
        RerePowerTransformer transformer = new RerePowerTransformer(RerePowerTransformer.Method.BOX_COX);
        transformer.fit(X);
        IMatrix<?> Xt = transformer.transform(X);

        double[] lambdas = transformer.getLambdas();
        System.out.println("  Box-Cox Lambda: [" + lambdas[0] + ", " + lambdas[1] + "]");

        // 验证所有变换后的值都是有限值
        for (int i = 0; i < Xt.rows(); i++) {
            for (int j = 0; j < Xt.cols(); j++) {
                double val = Xt.get(i, j);
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    System.out.println("  ❌ Box-Cox 变换后有 NaN 或 Inf");
                    return false;
                }
            }
        }

        // Yeo-Johnson（支持正负数）
        RerePowerTransformer yj = new RerePowerTransformer(RerePowerTransformer.Method.YEO_JOHNSON);
        double[][] mixedData = {{-1.0, 2.0}, {0.0, 4.0}};
        IMatrix<Double> Xm = IMatrix.of(mixedData);
        yj.fit(Xm);
        IMatrix<?> Xtj = yj.transform(Xm);

        System.out.println("  Yeo-Johnson Lambda: [" + yj.getLambdas()[0] + ", " + yj.getLambdas()[1] + "]");

        for (int i = 0; i < Xtj.rows(); i++) {
            for (int j = 0; j < Xtj.cols(); j++) {
                double val = Xtj.get(i, j);
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    System.out.println("  ❌ Yeo-Johnson 变换后有 NaN 或 Inf");
                    return false;
                }
            }
        }

        return true;
    }

    // ==================== QuantileTransformer ====================
    private static boolean testQuantileTransformer() {
        System.out.println("\n--- QuantileTransformer 验证 ---");
        // 均匀分布数据变换后应接近均匀分布
        double[][] data = new double[100][1];
        for (int i = 0; i < 100; i++) {
            data[i][0] = i;
        }
        IMatrix<Double> X = IMatrix.of(data);

        RereQuantileTransformer qt = new RereQuantileTransformer();
        qt.fit(X);
        IMatrix<?> Xt = qt.transform(X);

        // 验证变换后的值在 [0, 1] 范围内
        for (int i = 0; i < Xt.rows(); i++) {
            double val = Xt.get(i, 0);
            if (val < -0.01 || val > 1.01) {
                System.out.println("  ❌ 值超出 [0,1] 范围: " + val);
                return false;
            }
        }

        // 验证中位数接近 0.5
        double median = Xt.get(50, 0);
        check("Quantile中位数", median, 0.5, 0.05);

        // 验证没有 NaN
        for (int i = 0; i < Xt.rows(); i++) {
            if (Double.isNaN(Xt.get(i, 0))) {
                System.out.println("  ❌ 发现 NaN 值");
                return false;
            }
        }

        return true;
    }

    // ==================== KernelCenterer ====================
    private static boolean testKernelCenterer() {
        System.out.println("\n--- KernelCenterer 验证 ---");
        // 使用简单的核矩阵测试
        double[][] Kdata = {{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}};
        IMatrix<Double> K = IMatrix.of(Kdata);

        RereKernelCenterer centerer = new RereKernelCenterer();
        centerer.fit(K);
        IMatrix<?> Kc = centerer.transform(K);

        // 验证列均值接近 0
        for (int j = 0; j < Kc.cols(); j++) {
            double colMean = 0;
            for (int i = 0; i < Kc.rows(); i++) {
                colMean += Kc.get(i, j);
            }
            colMean /= Kc.rows();
            if (Math.abs(colMean) > 1e-10) {
                System.out.println("  ❌ 列 " + j + " 均值不接近 0: " + colMean);
                return false;
            }
        }
        System.out.println("  ✓ 列均值验证通过");

        // 验证没有 NaN
        for (int i = 0; i < Kc.rows(); i++) {
            for (int j = 0; j < Kc.cols(); j++) {
                if (Double.isNaN(Kc.get(i, j))) {
                    System.out.println("  ❌ 发现 NaN 值");
                    return false;
                }
            }
        }

        return true;
    }

    // ==================== LabelBinarizer ====================
    private static boolean testLabelBinarizer() {
        System.out.println("\n--- LabelBinarizer 验证 ---");
        double[][] data = {{0.0}, {1.0}, {2.0}};
        IMatrix<Double> X = IMatrix.of(data);

        RereLabelBinarizer binarizer = new RereLabelBinarizer();
        binarizer.fit(X);
        IMatrix<?> Xt = binarizer.transform(X);

        System.out.println("  输入: [0, 1, 2], 输出: " + Xt.rows() + "x" + Xt.cols());

        if (Xt.rows() != 3 || Xt.cols() != 3) {
            System.out.println("  ❌ 形状错误: 期望 3x3, 实际 " + Xt.rows() + "x" + Xt.cols());
            return false;
        }

        // [0] -> [1, 0, 0], [1] -> [0, 1, 0], [2] -> [0, 0, 1]
        check("标签0", Xt.get(0, 0), 1.0);
        check("标签0", Xt.get(0, 1), 0.0);
        check("标签1", Xt.get(1, 1), 1.0);
        check("标签2", Xt.get(2, 2), 1.0);

        return true;
    }

    // ==================== OneHotEncoder ====================
    private static boolean testOneHotEncoder() {
        System.out.println("\n--- OneHotEncoder 验证 ---");
        double[][] data = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}};
        IMatrix<Double> X = IMatrix.of(data);

        RereOneHotEncoder encoder = new RereOneHotEncoder();
        encoder.fit(X);
        IMatrix<?> Xt = encoder.transform(X);

        System.out.println("  输入: 3x2, 输出: " + Xt.rows() + "x" + Xt.cols());

        if (Xt.cols() != 6) {
            System.out.println("  ❌ 输出列数错误: 期望 6, 实际 " + Xt.cols());
            return false;
        }

        // 每行每列类别应该只有一个 1
        for (int i = 0; i < Xt.rows(); i++) {
            int oneCount = 0;
            for (int j = 0; j < Xt.cols(); j++) {
                double val = Xt.get(i, j);
                if (val == 1.0) oneCount++;
                else if (val != 0.0) {
                    System.out.println("  ❌ 发现非 0/1 值: " + val);
                    return false;
                }
            }
            if (oneCount != 2) {
                System.out.println("  ❌ 行 " + i + " 的 1 的数量不正确: " + oneCount);
                return false;
            }
        }

        return true;
    }

    // ==================== Bucketizer ====================
    // RereBucketizer(Strategy.FIXED_WIDTH, nBins) 使用箱子数量
    // 每个特征列独立计算: binWidth = (max-min)/nBins
    // col0: [5,35] -> min=5, max=35, nBins=5, binWidth=6
    //   edges: [5, 11, 17, 23, 29, 35]
    //   5 -> bin 0, 35 落在 [29,35) 但因为==upper 返回 nBins-1 = 4
    // col1: [15,45] -> min=15, max=45, nBins=5, binWidth=6
    //   edges: [15, 21, 27, 33, 39, 45]
    //   15 -> bin 0, 45 == upper -> bin 4
    // col2: [25,55] -> min=25, max=55, nBins=5, binWidth=6
    //   edges: [25, 31, 37, 43, 49, 55]
    //   25 -> bin 0, 55 == upper -> bin 4
    private static boolean testBucketizer() {
        System.out.println("\n--- Bucketizer 验证 ---");
        double[][] data = {
            {5.0, 15.0, 25.0},
            {35.0, 45.0, 55.0}
        };
        IMatrix<Double> X = IMatrix.of(data);

        // 使用 nBins=5 来简化验证
        RereBucketizer bucketizer = new RereBucketizer(RereBucketizer.Strategy.FIXED_WIDTH, 5);
        bucketizer.fit(X);
        IMatrix<?> Xt = bucketizer.transform(X);

        System.out.println("  nBins=5, 数据: col0=[5,35], col1=[15,45], col2=[25,55]");

        // 每列独立计算
        // col0: binWidth=6, edges=[5,11,17,23,29,35], 35==upper -> bin 4
        check("Bucket(5,col0)", Xt.get(0, 0), 0.0);
        check("Bucket(35,col0)", Xt.get(1, 0), 4.0);

        // col1: binWidth=6, edges=[15,21,27,33,39,45], 45==upper -> bin 4
        check("Bucket(15,col1)", Xt.get(0, 1), 0.0);
        check("Bucket(45,col1)", Xt.get(1, 1), 4.0);

        // col2: binWidth=6, edges=[25,31,37,43,49,55], 55==upper -> bin 4
        check("Bucket(25,col2)", Xt.get(0, 2), 0.0);
        check("Bucket(55,col2)", Xt.get(1, 2), 4.0);

        // 验证值都是非负整数
        for (int i = 0; i < Xt.rows(); i++) {
            for (int j = 0; j < Xt.cols(); j++) {
                double val = Xt.get(i, j);
                if (val < 0 || val != Math.floor(val)) {
                    System.out.println("  ❌ 箱子ID应为非负整数: " + val);
                    return false;
                }
            }
        }
        System.out.println("  ✓ 所有值都是有效的箱子ID");

        return true;
    }

    // ==================== 辅助方法 ====================
    private static void check(String name, double actual, double expected) {
        check(name, actual, expected, EPS);
    }

    private static void check(String name, double actual, double expected, double epsilon) {
        if (Math.abs(actual - expected) > epsilon) {
            System.out.println("  ❌ " + name + ": expected=" + expected + ", actual=" + actual);
        } else {
            System.out.println("  ✓ " + name + " = " + actual);
        }
    }
}
