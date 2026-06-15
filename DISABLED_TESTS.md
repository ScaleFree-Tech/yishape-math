# Disabled Tests Tracking

Auto-generated 2026-06-16. Update when enabling or adding disabled tests.

## ⚠ Bug-Blocked (re-enable when fix lands)

| Test | Location | Blocked By | Details |
|------|----------|------------|---------|
| `testSubBroadcastDim1_vector` | `ProtocolContractTest.java:255` | GPU sub-broadcast backward | GPU returns NaN — scalar encoding or backward dispatch issue |
| `testPow_BcastDim1` | `ProtocolContractTest.java:289` | GPU f32 precision | f32 precision: pow^3 on GPU produces ~4e-3 diff on large values |
| `testMish` | `ProtocolContractTest.java:325` | GPU mish backward | GPU mish backward produces wrong gradient (8x error) |

## ⏸ Slow / Performance Benchmark (not bugs)

These are intentionally disabled to keep dev cycles fast. Run individually when needed.

| Test | Location | Category |
|------|----------|----------|
| `SIMDOptimizationTest` | `SIMDOptimizationTest.java:17` | perf benchmark |
| `SIMDSpecificOptimizationTest` | `SIMDSpecificOptimizationTest.java:18` | perf benchmark |
| `DmlComprehensiveTest` | `DmlComprehensiveTest.java:63` | validation suite |
| `RereLogisticRegressionDatasetsIntegrationTest` | `...:25` | integration |
| `DlasdPackedVsQrBidiagonalTest` | `...:46` | ref-lapack blocked |
| `HpcLargeScaleStatsTest` | `...:10` | perf benchmark |
| `HpcLargeScaleMLTest` | `...:10` | perf benchmark |
| `HpcLargeScaleLinalgTest` | `...:10` | perf benchmark |
| `PureJavaLargeScaleStatsTest` | `...:10` | perf benchmark |
| `PureJavaLargeScaleMLTest` | `...:10` | perf benchmark |
| `PureJavaLargeScaleLinalgTest` | `...:10` | perf benchmark |
| `OptimizerHpcBenchmarkTest` | `...:19` | perf benchmark |
| `RustSvdSpeedTest` | `...:18` | perf benchmark |
| `LargeScaleStatsTest` | `...:24` | perf benchmark |
| `PerformanceBugRegressionTest` | `...:30` | perf benchmark |
| `LargeScaleMLTest` | `...:33` | perf benchmark |
| `PerformanceBenchmarkTest` | `...:41` | perf benchmark |
| `RereSVDEigenTrueJavaPerformanceTest` | `...:23` | perf benchmark |
| `LargeScaleLinalgTest` | `...:22` | perf benchmark |
| `RereSVDEigenPureJavaPerformanceTest` | `...:23` | perf benchmark |
| `RereSVDEigenPerformanceTest` | `...:21` | perf benchmark |
| `LinProgIntegerProgLargeScaleSystematicTest` | `...:31` | perf benchmark |

## 🔧 yishape-dl Disabled Tests

| Test | Location | Blocked By |
|------|----------|------------|
| `ComprehensiveGpuTest` | `...:25` | Requires GPU hardware (commented out, not @Disabled) |
| `testDataLoader_10000Samples_stress` | `ComprehensiveDataLoaderTest.java:450` | Slow by design |
| `testComprehensiveTraining_100Epochs` | `ComprehensiveTrainingTest.java:171` | Slow by design |
| `testTraining_50Epochs` | `ComprehensiveTrainingTest.java:331` | Slow by design |
| `testTrainer_100EpochsConvergence` | `ComprehensiveTrainerTest.java:474` | Slow by design |
| `TranslatorBenchmarkTest` | `TranslatorBenchmarkTest.java:34` | Slow by design |
| `TrainMnistMain` | `TrainMnistMain.java:31` | Manual execution only |
| `TrainTextModelsMain` | `TrainTextModelsMain.java:37` | Manual execution only |
| `TrainVisionModelsMain` | `TrainVisionModelsMain.java:25` | Manual execution only |

---

**Convention:** When adding `@Disabled`, always include a reason string. Bug-blocked tests must reference a tracking issue or memory file.
