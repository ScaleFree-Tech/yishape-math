package com.yishape.lab.math.plot.echarts.j3d;

import com.yishape.lab.math.plot.ColorPalette;

/**
 * 为 3D 多系列拆分调色板条目（二维 {@link ColorPalette} 逻辑的薄封装）。
 */
final class Echarts3dPalette {

    private Echarts3dPalette() {
    }

    private static final String[] FALLBACK = {
            "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
            "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc"
    };

    static String[] colors(String paletteKey, int n) {
        if (paletteKey != null && ColorPalette.hasPalette(paletteKey)) {
            String[] p = ColorPalette.getPalette(paletteKey);
            if (p != null && p.length > 0) {
                String[] result = new String[n];
                for (int i = 0; i < n; i++) result[i] = p[i % p.length];
                return result;
            }
        }
        String[] result = new String[n];
        for (int i = 0; i < n; i++) result[i] = FALLBACK[i % FALLBACK.length];
        return result;
    }
}
