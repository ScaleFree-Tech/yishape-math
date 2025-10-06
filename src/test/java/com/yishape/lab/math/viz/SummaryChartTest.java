package com.yishape.lab.math.viz;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import java.util.List;

/**
 * Test that replicates the summary chart generation from ComputerPerformanceComparisonTest
 */
public class SummaryChartTest {
    
    @Test
    @DisplayName("Test Summary Chart Generation")
    void testSummaryChartGeneration() {
        System.out.println("=== Testing Summary Chart Generation ===");
        
        try {
            // Replicate the summary chart data structure
            double gpuAvg = 5.25;
            double simdAvg = 3.75;
            double sisdAvg = 8.75;
            
            // Create summary chart
            IVector<Double> avgTimes = Linalg.vector(new double[]{gpuAvg, simdAvg, sisdAvg});
            List<String> labels = List.of("GPU", "SIMD", "SISD");
            List<String> xLabels = List.of("GPU", "SIMD", "SISD");
            
            RerePlot plot = new RerePlot(800, 500);
            plot.bar(xLabels, avgTimes, labels)
                .title("Average Performance Comparison Across All Operations")
                .xlabel("Computer Type")
                .ylabel("Average Time (ms)");
            
            String filename = "temp/test_average_performance_comparison.html";
            plot.saveAsHtml(filename);
            System.out.println("Summary chart saved to: " + filename);
            System.out.println("Test completed successfully!");
            
        } catch (Exception e) {
            System.err.println("Error creating summary chart: " + e.getMessage());
            e.printStackTrace();
        }
    }
}