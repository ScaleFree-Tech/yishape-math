---
name: rust-native-deploy
description: Build and deploy Rust native DLLs for yishape-math-gpu and yishape-math-hpc into Maven JARs. Covers the full pipeline from cargo build to Maven install, common pitfalls (stale DLLs, resources directory pollution), and verification steps.
source: auto-skill
extracted_at: '2026-06-04T10:39:59.260Z'
---

# Rust Native DLL Deployment for yishape-math

## Architecture

Two Rust crates produce native DLLs loaded via Java FFM:

| Crate | Source | DLL | Maven project | JAR classifier |
|-------|--------|-----|---------------|----------------|
| `yishape_math_gpu` | `E:\rust_work\yishape_math_gpu` | `yishape_math_gpu.dll` | `E:\work\yishape-math-gpu` | `windows-x86_64` |
| `yishape_math_rust` | `E:\rust_work\yishape_math_rust` | `yishape_math_rust.dll` | `E:\work\yishape-math-hpc` | `windows-x86_64` |

## Deployment Steps

### 1. Compile Rust crate (always `--release`)

```bash
cd E:\rust_work\yishape_math_gpu  # or yishape_math_rust
cargo build --release
```

Output: `target/release/yishape_math_gpu.dll` (or `yishape_math_rust.dll`)

### 2. Copy DLL to `libs/` directory

```bash
copy E:\rust_work\yishape_math_gpu\target\release\yishape_math_gpu.dll ^
     E:\work\yishape-math-gpu\libs\windows-x86_64\yishape_math_gpu.dll /Y
```

The `libs/<classifier>/` directory holds prebuilt native libs for each platform.

### 3. Rebuild Maven JAR with `-DskipNativeBuild=true`

```bash
cd E:\work\yishape-math-gpu
mvn install -DskipTests -DskipNativeBuild=true
```

The `maven-antrun-plugin` copies from `libs/` → `target/classes/META-INF/native-libs/<classifier>/`, then packages platform-specific JARs.

`-DskipNativeBuild=true` tells the build to skip its own `cargo build` and use the prebuilt DLLs you just copied.

### 4. Verify

```bash
# Extract DLL from installed JAR and check size
cd %TEMP%
jar xf E:\maven_data\repository\com\yishape\lab\yishape-math-gpu\0.5.0\yishape-math-gpu-0.5.0-windows-x86_64.jar META-INF/native-libs/windows-x86_64/yishape_math_gpu.dll
dir META-INF\native-libs\windows-x86_64\yishape_math_gpu.dll
```

Compare file size and timestamp against your freshly compiled DLL.

## Critical Pitfalls

### 1. Stale DLL in `src/main/resources/`

**NEVER** place DLL files in `src/main/resources/META-INF/native-libs/`. The `maven-resources-plugin` may copy them into the JAR, overriding the fresh DLL from `libs/`. This was a real bug: an old DLL committed to git in `src/main/resources/` silently replaced the newly compiled one.

The correct `libs/` → `target/classes/` flow is managed entirely by `maven-antrun-plugin`. The `pom.xml` excludes `META-INF/native-libs/**` from the resources plugin:

```xml
<excludes>
    <exclude>META-INF/native-libs/**</exclude>
</excludes>
```

If you find a DLL in `src/main/resources/`, delete it — it's contamination.

### 2. `target/` not cleaned between builds

If `target/` still has old class files, the antrun `gpu-copy-prebuilt-libs` step copies from `libs/` correctly, but the `maven-resources-plugin` copies from `src/main/resources/` (which may have stale content). Always do a full clean if unsure:

```bash
rmdir /s /q target
mvn install -DskipTests -DskipNativeBuild=true
```

### 3. Runtime DLL loading priority

`GpuNativeLoader` loads DLLs in this order:
1. System property: `-Dyishape.gpu.library.path=<absolute-path>` (fastest for dev debugging)
2. Classpath extraction: from `META-INF/native-libs/<classifier>/` in the JAR
3. System library path: `System.loadLibrary("yishape_math_gpu")`

For quick testing without repackaging:
```bash
-Dyishape.gpu.library.path=E:\rust_work\yishape_math_gpu\target\release\yishape_math_gpu.dll
```

### 4. yishape-dl depends on both

`yishape-dl` at `E:\work\yishape-dl` depends on both `yishape-math-gpu` and `yishape-math-hpc`. After rebuilding either, yishape-dl picks up the new JAR automatically (same Maven coordinates). But if you modify Java code in yishape-dl too, remember to recompile:

```bash
cd E:\work\yishape-dl && mvn compile test-compile
```

## Full Rebuild (both projects)

```bash
# 1. Compile both Rust crates
cd E:\rust_work\yishape_math_gpu && cargo build --release
cd E:\rust_work\yishape_math_rust && cargo build --release

# 2. Copy DLLs
copy E:\rust_work\yishape_math_gpu\target\release\yishape_math_gpu.dll E:\work\yishape-math-gpu\libs\windows-x86_64\ /Y
copy E:\rust_work\yishape_math_rust\target\release\yishape_math_rust.dll E:\work\yishape-math-hpc\libs\windows-x86_64\ /Y

# 3. Rebuild Maven JARs
cd E:\work\yishape-math-gpu && mvn install -DskipTests -DskipNativeBuild=true -q
cd E:\work\yishape-math-hpc && mvn install -DskipTests -DskipNativeBuild=true -q

# 4. Rebuild yishape-dl (if Java code changed)
cd E:\work\yishape-dl && mvn compile test-compile -q
```
