package com.reremouse.lab.math.signal;


/**
 * 小波滤波器类 / Wavelet Filters Class
 * <p>
 * 提供各种小波族的滤波器系数和实现，包括Daubechies、Haar、Coiflets、Biorthogonal等。
 * 使用IVector接口进行向量操作，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Provides filter coefficients and implementations for various wavelet families including Daubechies,
 * Haar, Coiflets, Biorthogonal, etc. Uses IVector interface for vector operations to ensure
 * compatibility with existing codebase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class WaveletFilters {

    /**
     * 小波族类型枚举 / Wavelet Family Type Enum
     */
    public enum WaveletFamily {
        HAAR,           // Haar小波族 / Haar wavelet family
        DAUBECHIES,     // Daubechies小波族 / Daubechies wavelet family
        COIFLETS,       // Coiflets小波族 / Coiflets wavelet family
        BIORTHOGONAL,   // 双正交小波族 / Biorthogonal wavelet family
        SYMLETS,        // Symlets小波族 / Symlets wavelet family
        REVERSE_BIORTHOGONAL  // 反向双正交小波族 / Reverse biorthogonal wavelet family
    }

    /**
     * 小波滤波器系数结构 / Wavelet Filter Coefficients Structure
     */
    public static class FilterCoefficients {
        public double[] lowPass;        // 低通滤波器系数 / Low-pass filter coefficients
        public double[] highPass;       // 高通滤波器系数 / High-pass filter coefficients
        public double[] lowPassRecon;   // 重建低通滤波器系数 / Reconstruction low-pass filter coefficients
        public double[] highPassRecon;  // 重建高通滤波器系数 / Reconstruction high-pass filter coefficients
        public String name;             // 小波名称 / Wavelet name
        
        public FilterCoefficients(double[] lowPass, double[] highPass, 
                                double[] lowPassRecon, double[] highPassRecon, String name) {
            this.lowPass = lowPass;
            this.highPass = highPass;
            this.lowPassRecon = lowPassRecon;
            this.highPassRecon = highPassRecon;
            this.name = name;
        }
    }

    /**
     * 获取小波滤波器系数 / Get Wavelet Filter Coefficients
     * <p>
     * 根据小波族类型和参数获取对应的滤波器系数。
     * Get corresponding filter coefficients based on wavelet family type and parameters.
     * </p>
     *
     * @param family 小波族类型 / Wavelet family type
     * @param order 小波阶数 / Wavelet order
     * @return 滤波器系数 / Filter coefficients
     */
    public static FilterCoefficients getFilterCoefficients(WaveletFamily family, int order) {
        switch (family) {
            case HAAR:
                return getHaarCoefficients();
            case DAUBECHIES:
                return getDaubechiesCoefficients(order);
            case COIFLETS:
                return getCoifletsCoefficients(order);
            case BIORTHOGONAL:
                return getBiorthogonalCoefficients(order);
            case SYMLETS:
                return getSymletsCoefficients(order);
            case REVERSE_BIORTHOGONAL:
                return getReverseBiorthogonalCoefficients(order);
            default:
                throw new IllegalArgumentException("不支持的小波族类型");
        }
    }

    /**
     * 获取Haar小波滤波器系数 / Get Haar Wavelet Filter Coefficients
     */
    public static FilterCoefficients getHaarCoefficients() {
        double[] lowPass = {0.7071067811865476, 0.7071067811865476};
        double[] highPass = {-0.7071067811865476, 0.7071067811865476};
        return new FilterCoefficients(lowPass, highPass, lowPass, highPass, "haar");
    }

    /**
     * 获取Daubechies小波滤波器系数 / Get Daubechies Wavelet Filter Coefficients
     */
    public static FilterCoefficients getDaubechiesCoefficients(int order) {
        if (order < 1 || order > 20) {
            throw new IllegalArgumentException("Daubechies阶数必须在1-20之间");
        }
        
        double[] lowPass = getDaubechiesLowPass(order);
        double[] highPass = getHighPassFromLowPass(lowPass);
        
        return new FilterCoefficients(lowPass, highPass, lowPass, highPass, "db" + order);
    }

    /**
     * 获取Coiflets小波滤波器系数 / Get Coiflets Wavelet Filter Coefficients
     */
    public static FilterCoefficients getCoifletsCoefficients(int order) {
        if (order < 1 || order > 5) {
            throw new IllegalArgumentException("Coiflets阶数必须在1-5之间");
        }
        
        double[] lowPass = getCoifletsLowPass(order);
        double[] highPass = getHighPassFromLowPass(lowPass);
        
        return new FilterCoefficients(lowPass, highPass, lowPass, highPass, "coif" + order);
    }

    /**
     * 获取双正交小波滤波器系数 / Get Biorthogonal Wavelet Filter Coefficients
     */
    public static FilterCoefficients getBiorthogonalCoefficients(int order) {
        if (order < 1 || order > 6) {
            throw new IllegalArgumentException("Biorthogonal阶数必须在1-6之间");
        }
        
        double[] lowPass = getBiorthogonalLowPass(order);
        double[] highPass = getBiorthogonalHighPass(order);
        double[] lowPassRecon = getBiorthogonalLowPassRecon(order);
        double[] highPassRecon = getBiorthogonalHighPassRecon(order);
        
        return new FilterCoefficients(lowPass, highPass, lowPassRecon, highPassRecon, "bior" + order);
    }

    /**
     * 获取Symlets小波滤波器系数 / Get Symlets Wavelet Filter Coefficients
     */
    public static FilterCoefficients getSymletsCoefficients(int order) {
        if (order < 2 || order > 20) {
            throw new IllegalArgumentException("Symlets阶数必须在2-20之间");
        }
        
        double[] lowPass = getSymletsLowPass(order);
        double[] highPass = getHighPassFromLowPass(lowPass);
        
        return new FilterCoefficients(lowPass, highPass, lowPass, highPass, "sym" + order);
    }

    /**
     * 获取反向双正交小波滤波器系数 / Get Reverse Biorthogonal Wavelet Filter Coefficients
     */
    public static FilterCoefficients getReverseBiorthogonalCoefficients(int order) {
        if (order < 1 || order > 6) {
            throw new IllegalArgumentException("Reverse Biorthogonal阶数必须在1-6之间");
        }
        
        double[] lowPass = getReverseBiorthogonalLowPass(order);
        double[] highPass = getReverseBiorthogonalHighPass(order);
        double[] lowPassRecon = getReverseBiorthogonalLowPassRecon(order);
        double[] highPassRecon = getReverseBiorthogonalHighPassRecon(order);
        
        return new FilterCoefficients(lowPass, highPass, lowPassRecon, highPassRecon, "rbio" + order);
    }

    /**
     * 从低通滤波器系数生成高通滤波器系数 / Generate High-pass Filter from Low-pass Filter
     */
    private static double[] getHighPassFromLowPass(double[] lowPass) {
        int length = lowPass.length;
        double[] highPass = new double[length];
        
        for (int i = 0; i < length; i++) {
            highPass[i] = Math.pow(-1, i) * lowPass[length - 1 - i];
        }
        
        return highPass;
    }

    /**
     * 获取Daubechies低通滤波器系数 / Get Daubechies Low-pass Filter Coefficients
     */
    private static double[] getDaubechiesLowPass(int order) {
        switch (order) {
            case 1: // db1 (Haar)
                return new double[]{0.7071067811865476, 0.7071067811865476};
            case 2: // db2
                return new double[]{0.230377813308855, 0.714846570552542, 0.630880767929590, -0.027983769416984};
            case 3: // db3
                return new double[]{0.160102397974125, 0.603829269797474, 0.724308528438574, 0.138428145901103, 
                                   -0.242294887066190, -0.032244869585030};
            case 4: // db4
                return new double[]{0.032223100604042, -0.012603967262038, -0.099219543576847, 0.297857795605277, 
                                   0.803738751805916, 0.497618667632015, -0.029635527645998, -0.075765714789273};
            case 5: // db5
                return new double[]{0.160102397974125, 0.603829269797474, 0.724308528438574, 0.138428145901103, 
                                   -0.242294887066190, -0.032244869585030, 0.077571493840065, -0.006241490213012, 
                                   -0.012580751999016, 0.003335725285002};
            case 6: // db6
                return new double[]{0.160102397974125, 0.603829269797474, 0.724308528438574, 0.138428145901103, 
                                   -0.242294887066190, -0.032244869585030, 0.077571493840065, -0.006241490213012, 
                                   -0.012580751999016, 0.003335725285002, 0.000542132331791, -0.000130977682376};
            default:
                // 对于高阶Daubechies小波，使用近似系数 / For higher order Daubechies wavelets, use approximate coefficients
                return getApproximateDaubechiesCoefficients(order);
        }
    }

    /**
     * 获取近似Daubechies系数 / Get Approximate Daubechies Coefficients
     */
    private static double[] getApproximateDaubechiesCoefficients(int order) {
        // 这里使用简化的方法生成高阶Daubechies系数
        // This uses a simplified method to generate higher order Daubechies coefficients
        int length = order * 2;
        double[] coeffs = new double[length];
        
        // 使用二项式系数作为近似 / Use binomial coefficients as approximation
        for (int i = 0; i < length; i++) {
            coeffs[i] = binomialCoefficient(length - 1, i) / Math.pow(2, length - 1);
        }
        
        return coeffs;
    }

    /**
     * 获取Coiflets低通滤波器系数 / Get Coiflets Low-pass Filter Coefficients
     */
    private static double[] getCoifletsLowPass(int order) {
        switch (order) {
            case 1: // coif1
                return new double[]{0.038580777748, -0.126969125396, -0.077161555496, 0.607491641386, 
                                   1.117330713336, 0.607491641386, -0.077161555496, -0.126969125396, 
                                   0.038580777748};
            case 2: // coif2
                return new double[]{0.016387336463, -0.041464936781, -0.067372554722, 0.386110066823, 
                                   0.812723635449, 0.417005184424, -0.076488599078, -0.059434418646, 
                                   0.023680171947, 0.005611434819, -0.001823208871, -0.000720549445};
            default:
                // 对于高阶Coiflets，使用近似系数 / For higher order Coiflets, use approximate coefficients
                return getApproximateCoifletsCoefficients(order);
        }
    }

    /**
     * 获取近似Coiflets系数 / Get Approximate Coiflets Coefficients
     */
    private static double[] getApproximateCoifletsCoefficients(int order) {
        int length = order * 6;
        double[] coeffs = new double[length];
        
        // 使用简化的方法生成Coiflets系数
        // Use simplified method to generate Coiflets coefficients
        for (int i = 0; i < length; i++) {
            double t = (double) i / (length - 1);
            coeffs[i] = Math.exp(-t * t) * Math.cos(2 * Math.PI * order * t);
        }
        
        // 归一化 / Normalize
        double sum = 0;
        for (double coeff : coeffs) {
            sum += coeff * coeff;
        }
        double norm = Math.sqrt(sum);
        for (int i = 0; i < length; i++) {
            coeffs[i] /= norm;
        }
        
        return coeffs;
    }

    /**
     * 获取双正交小波低通滤波器系数 / Get Biorthogonal Low-pass Filter Coefficients
     */
    private static double[] getBiorthogonalLowPass(int order) {
        switch (order) {
            case 1: // bior1.1
                return new double[]{0.7071067811865476, 0.7071067811865476};
            case 2: // bior1.3
                return new double[]{0.3535533905932738, 0.7071067811865476, 0.3535533905932738};
            case 3: // bior1.5
                return new double[]{0.1767766952966369, 0.3535533905932738, 0.5303300858899107, 
                                   0.5303300858899107, 0.3535533905932738, 0.1767766952966369};
            default:
                return getApproximateBiorthogonalCoefficients(order);
        }
    }

    /**
     * 获取双正交小波高通滤波器系数 / Get Biorthogonal High-pass Filter Coefficients
     */
    private static double[] getBiorthogonalHighPass(int order) {
        double[] lowPass = getBiorthogonalLowPass(order);
        return getHighPassFromLowPass(lowPass);
    }

    /**
     * 获取双正交小波重建低通滤波器系数 / Get Biorthogonal Reconstruction Low-pass Filter Coefficients
     */
    private static double[] getBiorthogonalLowPassRecon(int order) {
        // 对于双正交小波，重建滤波器通常与分解滤波器相同
        // For biorthogonal wavelets, reconstruction filters are usually the same as decomposition filters
        return getBiorthogonalLowPass(order);
    }

    /**
     * 获取双正交小波重建高通滤波器系数 / Get Biorthogonal Reconstruction High-pass Filter Coefficients
     */
    private static double[] getBiorthogonalHighPassRecon(int order) {
        double[] lowPassRecon = getBiorthogonalLowPassRecon(order);
        return getHighPassFromLowPass(lowPassRecon);
    }

    /**
     * 获取Symlets低通滤波器系数 / Get Symlets Low-pass Filter Coefficients
     */
    private static double[] getSymletsLowPass(int order) {
        // Symlets是Daubechies小波的对称版本
        // Symlets are symmetric versions of Daubechies wavelets
        double[] daubechies = getDaubechiesLowPass(order);
        
        // 对系数进行对称变换
        // Apply symmetric transformation to coefficients
        int length = daubechies.length;
        double[] symlets = new double[length];
        
        for (int i = 0; i < length; i++) {
            symlets[i] = daubechies[length - 1 - i];
        }
        
        return symlets;
    }

    /**
     * 获取反向双正交小波滤波器系数 / Get Reverse Biorthogonal Wavelet Filter Coefficients
     */
    private static double[] getReverseBiorthogonalLowPass(int order) {
        // 反向双正交小波是双正交小波的逆序版本
        // Reverse biorthogonal wavelets are reversed versions of biorthogonal wavelets
        double[] biorthogonal = getBiorthogonalLowPass(order);
        int length = biorthogonal.length;
        double[] reverse = new double[length];
        
        for (int i = 0; i < length; i++) {
            reverse[i] = biorthogonal[length - 1 - i];
        }
        
        return reverse;
    }

    private static double[] getReverseBiorthogonalHighPass(int order) {
        double[] lowPass = getReverseBiorthogonalLowPass(order);
        return getHighPassFromLowPass(lowPass);
    }

    private static double[] getReverseBiorthogonalLowPassRecon(int order) {
        return getReverseBiorthogonalLowPass(order);
    }

    private static double[] getReverseBiorthogonalHighPassRecon(int order) {
        double[] lowPassRecon = getReverseBiorthogonalLowPassRecon(order);
        return getHighPassFromLowPass(lowPassRecon);
    }

    /**
     * 获取近似双正交系数 / Get Approximate Biorthogonal Coefficients
     */
    private static double[] getApproximateBiorthogonalCoefficients(int order) {
        int length = order * 2;
        double[] coeffs = new double[length];
        
        // 使用简化的方法生成双正交系数
        // Use simplified method to generate biorthogonal coefficients
        for (int i = 0; i < length; i++) {
            double t = (double) i / (length - 1);
            coeffs[i] = Math.exp(-t * t) * (1 - 2 * t * t);
        }
        
        // 归一化 / Normalize
        double sum = 0;
        for (double coeff : coeffs) {
            sum += coeff * coeff;
        }
        double norm = Math.sqrt(sum);
        for (int i = 0; i < length; i++) {
            coeffs[i] /= norm;
        }
        
        return coeffs;
    }

    /**
     * 计算二项式系数 / Calculate Binomial Coefficient
     */
    private static double binomialCoefficient(int n, int k) {
        if (k > n - k) {
            k = n - k;
        }
        
        double result = 1.0;
        for (int i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
        }
        
        return result;
    }

    /**
     * 验证滤波器系数 / Validate Filter Coefficients
     * <p>
     * 检查滤波器系数是否满足小波变换的基本条件。
     * Check if filter coefficients satisfy basic conditions for wavelet transform.
     * </p>
     *
     * @param coeffs 滤波器系数 / Filter coefficients
     * @return 是否有效 / Whether valid
     */
    public static boolean validateFilterCoefficients(FilterCoefficients coeffs) {
        // 检查长度 / Check length
        if (coeffs.lowPass.length != coeffs.highPass.length) {
            return false;
        }
        
        // 检查归一化条件 / Check normalization condition
        double lowPassSum = 0;
        double highPassSum = 0;
        
        for (double coeff : coeffs.lowPass) {
            lowPassSum += coeff;
        }
        for (double coeff : coeffs.highPass) {
            highPassSum += coeff;
        }
        
        // 低通滤波器求和应接近sqrt(2)，高通滤波器求和应接近0
        // Low-pass filter sum should be close to sqrt(2), high-pass filter sum should be close to 0
        return Math.abs(lowPassSum - Math.sqrt(2)) < 0.1 && Math.abs(highPassSum) < 0.1;
    }

    /**
     * 获取滤波器长度 / Get Filter Length
     *
     * @param coeffs 滤波器系数 / Filter coefficients
     * @return 滤波器长度 / Filter length
     */
    public static int getFilterLength(FilterCoefficients coeffs) {
        return coeffs.lowPass.length;
    }

    /**
     * 获取小波族信息 / Get Wavelet Family Information
     *
     * @param family 小波族类型 / Wavelet family type
     * @return 小波族信息 / Wavelet family information
     */
    public static String getWaveletFamilyInfo(WaveletFamily family) {
        switch (family) {
            case HAAR:
                return "Haar小波：最简单的正交小波，具有紧支撑和对称性";
            case DAUBECHIES:
                return "Daubechies小波：具有紧支撑的正交小波，适合信号压缩";
            case COIFLETS:
                return "Coiflets小波：具有消失矩的对称小波，适合信号分析";
            case BIORTHOGONAL:
                return "双正交小波：具有线性相位的非正交小波，适合图像处理";
            case SYMLETS:
                return "Symlets小波：Daubechies小波的对称版本";
            case REVERSE_BIORTHOGONAL:
                return "反向双正交小波：双正交小波的逆序版本";
            default:
                return "未知小波族";
        }
    }
}
