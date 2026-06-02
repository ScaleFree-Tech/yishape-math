package com.yishape.lab.math.viz;

import com.yishape.lab.math.plot.echarts.EchartsPlot;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import java.util.List;
import java.util.ArrayList;

/**
 * Test that replicates the exact chart generation approach used in ComputerPerformanceComparisonTest
 */
public class PerformanceChartTest {
    
    @Test
    @DisplayName("Test Performance Comparison Chart Generation")
    void testPerformanceChartGeneration() {
        System.out.println("=== Testing Performance Comparison Chart Generation ===");
        
        try {
            // Replicate the exact data structure from ComputerPerformanceComparisonTest
            List<Integer> sizes = List.of(1000, 10000, 100000, 1000000);
            List<Double> gpuTimes = List.of(1.5, 2.0, 5.0, 20.0);
            List<Double> simdTimes = List.of(1.0, 1.5, 3.0, 15.0);
            List<Double> sisdTimes = List.of(2.0, 3.0, 8.0, 25.0);
            String operation = "Vector Addition";
            
            // Create x-axis labels (one for each data size)
            List<String> xLabels = new ArrayList<>();
            for (int i = 0; i < sizes.size(); i++) {
                xLabels.add(sizes.get(i).toString());
            }
            
            // Create the chart using the proper API for grouped bar charts
            EchartsPlot plot = new EchartsPlot(1000, 600);
            
            // For grouped bar charts, we need to provide all the data in a specific format
            // Let's try creating a single vector with all the data and corresponding hue labels
            List<Double> allTimes = new ArrayList<>();
            List<String> hueLabels = new ArrayList<>();
            
            // Add data in the correct order for grouped bars
            for (int i = 0; i < sizes.size(); i++) {
                allTimes.add(gpuTimes.get(i));
                hueLabels.add("GPU");
                
                allTimes.add(simdTimes.get(i));
                hueLabels.add("SIMD");
                
                allTimes.add(sisdTimes.get(i));
                hueLabels.add("SISD");
            }
            
            // Convert to IVector format
            IVector<Double> allTimesVector = Linalg.vector(allTimes.stream().mapToDouble(Double::doubleValue).toArray());
            
            // Create the chart using the proper API for grouped bar charts
            plot.bar(xLabels, allTimesVector, hueLabels)
                .title(operation + " Performance Comparison")
                .xlabel("Data Size")
                .ylabel("Time (ms)");
            
            // Save the chart
            String filename = "temp/test_performance_comparison.html";
            plot.saveAsHtml(filename);
            System.out.println("Performance comparison chart saved to: " + filename);
            System.out.println("Test completed successfully!");
            
        } catch (Exception e) {
            System.err.println("Error generating performance comparison chart: " + e.getMessage());
            e.printStackTrace();
        }
    }
}