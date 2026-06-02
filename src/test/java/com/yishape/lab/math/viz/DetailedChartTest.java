package com.yishape.lab.math.viz;

import com.yishape.lab.math.plot.echarts.EchartsPlot;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import java.util.List;
import java.util.ArrayList;

/**
 * Detailed test to understand the grouped bar chart implementation
 */
public class DetailedChartTest {
    
    @Test
    @DisplayName("Test Detailed Grouped Bar Chart")
    void testDetailedGroupedBarChart() {
        System.out.println("=== Testing Detailed Grouped Bar Chart ===");
        
        try {
            // Test case 1: Simple grouped bar chart
            System.out.println("\n--- Test Case 1: Simple grouped bar chart ---");
            List<String> xLabels1 = List.of("A", "B", "C");
            List<Double> data1 = List.of(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
            List<String> hue1 = List.of("Group1", "Group2", "Group1", "Group2", "Group1", "Group2");
            
            IVector<Double> vector1 = Linalg.vector(data1.stream().mapToDouble(Double::doubleValue).toArray());
            
            EchartsPlot plot1 = new EchartsPlot(800, 600);
            plot1.bar(xLabels1, vector1, hue1)
                .title("Test 1: Simple Grouped Bar Chart")
                .xlabel("Categories")
                .ylabel("Values");
            
            String filename1 = "temp/test1_grouped_bar.html";
            plot1.saveAsHtml(filename1);
            System.out.println("Test 1 chart saved to: " + filename1);
            
            // Test case 2: Performance comparison style
            System.out.println("\n--- Test Case 2: Performance comparison style ---");
            List<String> xLabels2 = List.of("1000", "10000", "100000");
            List<Double> gpuTimes = List.of(1.5, 2.0, 5.0);
            List<Double> simdTimes = List.of(1.0, 1.5, 3.0);
            List<Double> sisdTimes = List.of(2.0, 3.0, 8.0);
            
            // Interleaved data format as used in performance comparison
            List<Double> allTimes = new ArrayList<>();
            List<String> hueLabels = new ArrayList<>();
            
            for (int i = 0; i < xLabels2.size(); i++) {
                allTimes.add(gpuTimes.get(i));
                hueLabels.add("GPU");
                
                allTimes.add(simdTimes.get(i));
                hueLabels.add("SIMD");
                
                allTimes.add(sisdTimes.get(i));
                hueLabels.add("SISD");
            }
            
            IVector<Double> allTimesVector = Linalg.vector(allTimes.stream().mapToDouble(Double::doubleValue).toArray());
            
            EchartsPlot plot2 = new EchartsPlot(1000, 600);
            plot2.bar(xLabels2, allTimesVector, hueLabels)
                .title("Test 2: Performance Comparison Style")
                .xlabel("Data Size")
                .ylabel("Time (ms)");
            
            String filename2 = "temp/test2_performance_comparison.html";
            plot2.saveAsHtml(filename2);
            System.out.println("Test 2 chart saved to: " + filename2);
            
            System.out.println("Detailed test completed successfully!");
            
        } catch (Exception e) {
            System.err.println("Error in detailed test: " + e.getMessage());
            e.printStackTrace();
        }
    }
}