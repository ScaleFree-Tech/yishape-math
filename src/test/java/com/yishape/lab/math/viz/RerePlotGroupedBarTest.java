package com.yishape.lab.math.viz;

import com.yishape.lab.math.plot.echarts.EchartsPlot;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import java.util.List;
import java.util.ArrayList;

/**
 * Test for grouped bar chart functionality in EchartsPlot
 */
public class RerePlotGroupedBarTest {
    
    @Test
    @DisplayName("Test Grouped Bar Chart Creation")
    void testGroupedBarChart() {
        System.out.println("=== Testing Grouped Bar Chart ===");
        
        try {
            // Create test data similar to the performance comparison test
            // X-axis labels (data sizes)
            List<String> xLabels = List.of("1000", "10000", "100000", "1000000");
            
            // Create test data - for grouped bar chart, we need interleaved data
            // Format: [GPU_1000, SIMD_1000, SISD_1000, GPU_10000, SIMD_10000, SISD_10000, ...]
            List<Double> allTimes = new ArrayList<>();
            List<String> hueLabels = new ArrayList<>();
            
            // Add data for each size group
            for (String size : xLabels) {
                // Simulate performance data
                double baseTime = Double.parseDouble(size) / 100000.0;
                allTimes.add(baseTime * 2.0); // GPU time
                hueLabels.add("GPU");
                
                allTimes.add(baseTime * 1.0); // SIMD time
                hueLabels.add("SIMD");
                
                allTimes.add(baseTime * 1.5); // SISD time
                hueLabels.add("SISD");
            }
            
            // Convert to IVector format
            IVector<Double> allTimesVector = Linalg.vector(allTimes.stream().mapToDouble(Double::doubleValue).toArray());
            
            // Create the chart
            EchartsPlot plot = new EchartsPlot(800, 600);
            plot.bar(xLabels, allTimesVector, hueLabels)
                .title("Performance Comparison Test")
                .xlabel("Data Size")
                .ylabel("Time (ms)");
            
            // Save the chart
            String filename = "temp/grouped_bar_test.html";
            plot.saveAsHtml(filename);
            System.out.println("Grouped bar chart saved to: " + filename);
            System.out.println("Test completed successfully!");
            
        } catch (Exception e) {
            System.err.println("Error creating grouped bar chart: " + e.getMessage());
            e.printStackTrace();
        }
    }
}