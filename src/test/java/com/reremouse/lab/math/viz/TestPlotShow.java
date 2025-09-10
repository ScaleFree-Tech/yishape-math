package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.IVector;

/**
 *
 * @author lteb2
 */
public class TestPlotShow {

    public static void main(String args[]) {
        // 基础线图 / Basic line chart
        IPlot plot = Plots.of(800, 600);
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25});
        plot.line(x, y)
//                .title("销售趋势图", "2024年各月销售数据")
                .xlabel("月份")
                .ylabel("销售额（万元）")
                .show();

    }

}
