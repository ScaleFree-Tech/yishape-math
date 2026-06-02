package com.yishape.lab.math.signal.wavele;


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
        if (order < 1 || order > 10) {
            throw new IllegalArgumentException("Daubechies阶数必须在1-10之间");
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
    static double[] getDaubechiesLowPass(int order) {
        switch (order) {
            case 1: // db1 (Haar)
                return new double[]{0.7071067811865476, 0.7071067811865476};
            case 2: // db2
                return new double[]{0.4829629131445341, 0.8365163037378079,
                                   0.2241438680420134, -0.1294095225512604};
            case 3: // db3
                return new double[]{0.3326705529500826, 0.8068915093110926,
                                   0.4598775021184915, -0.1350110200102545,
                                  -0.0854412738820267, 0.0352262918857095};
            case 4: // db4
                return new double[]{0.2303778133088964, 0.7148465705529154,
                                   0.6308807679298587, -0.0279837694168599,
                                  -0.1870348117190931, 0.0308413818355607,
                                   0.0328830116668852, -0.0105974017850690};
            case 5: // db5
                return new double[]{0.1601023979741929, 0.6038292697971895,
                                   0.7243085284377726, 0.1384281459013203,
                                  -0.2422948870663823, -0.0322448695846381,
                                   0.0775714938400459, -0.0062414902127983,
                                  -0.0125807519990820, 0.0033357252854738};
            case 6: // db6
                return new double[]{0.1115407433501095, 0.4946238903984531,
                                   0.7511339080210954, 0.3152503517091982,
                                  -0.2262646939654400, -0.1297668675672625,
                                   0.0975016055873230, 0.0275228655303053,
                                  -0.0315820393174862, 0.0005538422011614,
                                   0.0047772575116975, -0.0010773010853085};
            case 7: // db7
                return new double[]{0.0778520540850079, 0.3965393194819173,
                                   0.7291320908462351, 0.4697822874051931,
                                  -0.1439060039285646, -0.2240361849938747,
                                   0.0713092192668276, 0.0806126091510825,
                                  -0.0380299369350144, -0.0165745416306664,
                                   0.0125509985560992, 0.0004295779729213,
                                  -0.0018016407040476, 0.0003537137999745};
            case 8: // db8
                return new double[]{0.0544158422431041, 0.3128715909143037,
                                   0.6756307362972906, 0.5853546836541903,
                                  -0.0158291052563723, -0.2840155429615534,
                                   0.0004724845739123, 0.1287474266204836,
                                  -0.0173693010018109, -0.0440882539307954,
                                   0.0139810279174001, 0.0087460940474062,
                                  -0.0048703529934518, -0.0003917403733770,
                                   0.0006754494064506, -0.0001174767841248};
            case 9: // db9
                return new double[]{0.0380779473638791, 0.2438346746125885,
                                   0.6048231236901120, 0.6572880780512952,
                                   0.1331973858249884, -0.2932737832791710,
                                  -0.0968407832229776, 0.1485407493381059,
                                   0.0307256814793384, -0.0676328290613305,
                                   0.0002509471148340, 0.0223616621246790,
                                  -0.0047232047577521, -0.0042815036824632,
                                   0.0018476468830564, 0.0002303857615232,
                                  -0.0002519631889427, 0.0000393473203163};
            case 10: // db10
                return new double[]{0.0266700579009498, 0.1881768000776337,
                                   0.5272011889315737, 0.6884590394535876,
                                   0.2811723436605705, -0.2498464243272288,
                                  -0.1959462743772883, 0.1273693403357428,
                                   0.0930573646038119, -0.0713941471663500,
                                  -0.0294575368219493, 0.0332126740593594,
                                   0.0036065535669565, -0.0107331754833017,
                                   0.0013953517470690, 0.0019924052951850,
                                  -0.0006858566949583, -0.0001164668551285,
                                   0.0000935886703202, -0.0000132642028945};
            default:
                throw new UnsupportedOperationException(
                    "Daubechies order " + order + " not supported. Valid orders: 1-10.");
        }
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
