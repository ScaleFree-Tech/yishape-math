package com.yishape.lab.math.testframework;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Lightweight test result recorder for comprehensive validation testing.
 * Outputs structured results that can be compared against Python reference implementations.
 */
public class TestResult {
    public String module;
    public String testName;
    public String subTest;
    public boolean passed;
    public double error;
    public double relError;
    public double javaValue;
    public double refValue;
    public String javaResult;
    public String message;
    public long timeMs;

    public TestResult(String module, String testName, String subTest) {
        this.module = module;
        this.testName = testName;
        this.subTest = subTest;
        this.passed = true;
        this.error = 0;
        this.relError = 0;
        this.timeMs = 0;
    }

    public void fail(String message) {
        this.passed = false;
        this.message = message;
    }

    public void fail(String message, double javaValue, double refValue) {
        this.passed = false;
        this.message = message;
        this.javaValue = javaValue;
        this.refValue = refValue;
        this.error = Math.abs(javaValue - refValue);
        this.relError = refValue != 0 ? Math.abs(javaValue - refValue) / Math.abs(refValue) : (javaValue != 0 ? Double.POSITIVE_INFINITY : 0);
    }

    public void pass(double javaValue, double refValue) {
        this.passed = true;
        this.javaValue = javaValue;
        this.refValue = refValue;
        this.error = Math.abs(javaValue - refValue);
        this.relError = refValue != 0 ? Math.abs(javaValue - refValue) / Math.abs(refValue) : 0;
    }

    public void pass(String message) {
        this.passed = true;
        this.message = message;
    }

    public String toCsv() {
        return String.format("%s|%s|%s|%s|%.6e|%.6e|%s|%s|%d",
            module, testName, subTest,
            passed ? "PASS" : "FAIL",
            error, relError,
            javaResult != null ? javaResult.replace(",", ";") : "",
            message != null ? message.replace(",", ";") : "",
            timeMs);
    }

    public static String csvHeader() {
        return "module|testName|subTest|status|error|relError|javaResult|message|timeMs";
    }

    public static class Recorder {
        private final List<TestResult> results = new ArrayList<>();
        private final String module;
        private final String outputDir;

        public Recorder(String module, String outputDir) {
            this.module = module;
            this.outputDir = outputDir;
            new File(outputDir).mkdirs();
        }

        public TestResult record(String testName, String subTest) {
            TestResult r = new TestResult(module, testName, subTest);
            results.add(r);
            return r;
        }

        public void writeToFile() {
            String filename = outputDir + "/" + module + "_results_" +
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".csv";
            try (PrintWriter pw = new PrintWriter(new FileWriter(filename))) {
                pw.println(TestResult.csvHeader());
                for (TestResult r : results) {
                    pw.println(r.toCsv());
                }
            } catch (IOException e) {
                System.err.println("Failed to write results: " + e.getMessage());
            }
            // Also write a summary
            writeSummary();
        }

        public void writeSummary() {
            int total = results.size();
            int passed = (int) results.stream().filter(r -> r.passed).count();
            int failed = total - passed;

            String filename = outputDir + "/" + module + "_summary.txt";
            try (PrintWriter pw = new PrintWriter(new FileWriter(filename))) {
                pw.println("Module: " + module);
                pw.println("Total tests: " + total);
                pw.println("Passed: " + passed);
                pw.println("Failed: " + failed);
                pw.println("Pass rate: " + String.format("%.2f%%", 100.0 * passed / total));
                pw.println();
                pw.println("FAILED TESTS:");
                for (TestResult r : results) {
                    if (!r.passed) {
                        pw.println("  " + r.testName + " / " + r.subTest + ": " + r.message);
                    }
                }
            } catch (IOException e) {
                System.err.println("Failed to write summary: " + e.getMessage());
            }
        }

        public List<TestResult> getResults() { return results; }
        public int getPassed() { return (int) results.stream().filter(r -> r.passed).count(); }
        public int getFailed() { return (int) results.stream().filter(r -> !r.passed).count(); }
    }
}
