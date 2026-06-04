---
name: bit-packed-double-debug
description: Debug pattern for bit-packed double parameters in yishape-math graph execution — Java packs integers into doubles via Double.longBitsToDouble(), Rust must unpack with f64::to_bits(), NOT 'as u64'. Covers the conv2d/maxpool2d scalarParam encoding and common decode bugs.
source: auto-skill
extracted_at: '2026-06-04T10:39:59.260Z'
---

# Bit-Packed Double Parameter Encoding

## The Pattern

yishape-math uses `Double.longBitsToDouble()` to pack multiple integer parameters (kernel size, stride, padding, channels) into a single `double` field for graph serialization. The JSON exporter writes this as a regular double value; the Rust executor must recover the original bits.

## Java Side (Encoding)

In `Conv2d.java` and `MaxPool2d.java`:

```java
// scalarParam: packs (kH, kW, stride) into bit fields
rv.scalarParam = Double.longBitsToDouble(
    ((long)kernelH << 16) | ((long)kernelW << 8) | stride);

// scalarParam2: packs (padding, outChannels) into bit fields
rv.scalarParam2 = Double.longBitsToDouble(
    ((long)padding << 16) | outChannels);
```

### Bit Layout

| Field | `scalarParam` bits | `scalarParam2` bits |
|-------|--------------------|---------------------|
| kernelH / padding | bits 23-16 | bits 31-16 |
| kernelW / outChannels | bits 15-8 | bits 15-0 |
| stride | bits 7-0 | — |

For `Linear`, `scalarParam` is a plain double (`outFeatures`), NOT bit-packed.
For `CrossEntropyLoss`, `scalarParam` is `B` (batch size), a plain double.

### The double values are subnormal

With typical parameters (kH=3, kW=3, stride=1), the packed long is `0x00030301 = 197377`. Converting:

```
Double.longBitsToDouble(197377) ≈ 9.75e-319
```

These are **extremely small denormalized numbers** — they look like zero in any normal float comparison but carry meaningful bits.

## Rust Side (Decoding)

### CORRECT: Use `.to_bits()` (bit reinterpretation)

```rust
let s_bits = scalar.to_bits();           // f64 → u64 bit pattern
let kh = ((s_bits >> 16) & 0xFF) as usize;
let kw = ((s_bits >> 8) & 0xFF) as usize;
let stride = (s_bits & 0xFF) as usize;
```

### WRONG: Using `as u64` (numeric truncation) — BUG!

```rust
let s_bits = scalar as u64;  // 9.75e-319 as u64 = 0 !!!
// All fields become 0 → stride=0 → divide-by-zero panic
```

`f64 as u64` is a **numeric cast** that truncates the float value to an integer. Since `9.75e-319` rounds to `0`, all decoded parameters become zero.

**This was a real production bug in `yishape_math_rust` (HPC) that caused every conv2d/maxpool2d graph execution to panic with `attempt to divide by zero`.**

## GraphExporter Serialization

`GraphExporter.java` serializes these as plain doubles:

```java
sb.append(",\"scalar\":").append(scalar);
```

For subnormal values, `Double.toString()` produces scientific notation like `9.75E-319`. Rust's JSON parser (using Ryu algorithm) handles subnormal round-trips with bit-exact precision, so `f64::to_bits()` recovers the original integer.

**Guard**: The exporter must also filter `Infinity`:

```java
if (!Double.isNaN(scalar) && !Double.isInfinite(scalar)) {
    sb.append(",\"scalar\":").append(scalar);
}
```

Without this, `Double.longBitsToDouble()` of certain bit patterns (e.g., dropout seed hitting `0x7FF0000000000000L`) produces `Infinity`, which is not valid JSON.

## exportShape Convention

`exportShape` on `RereDiffVector` must represent the **output** shape of the operation, not the input shape. The Rust graph executor uses `shape` to allocate buffers for both `value` and `grad`.

| Layer | exportShape (correct) | Notes |
|-------|----------------------|-------|
| Conv2d | `[B, outChannels, outH, outW]` | Output spatial dims |
| MaxPool2d | `[B, C, outH, outW]` | Channels unchanged |
| Linear | (not set) | 1D shape auto-derived from buffer size |
| CrossEntropyLoss | `[B, C]` on input logits; output is scalar | Scalar loss, shape=[1] |

After Java sets output shape, the Rust backward must derive input dimensions from the output shape:

```rust
// conv2d backward: shape = [B, out_ch, out_h, out_w]
let c_in = weight_data.len() / (out_ch * kh * kw);
let h_in = (out_h - 1) * stride + kh - 2 * pad;
let w_in = (out_w - 1) * stride + kw - 2 * pad;
```

## Debugging Checklist

When conv2d/maxpool2d graph execution crashes:

1. **`attempt to divide by zero`** → Check `scalar.to_bits()` vs `scalar as u64`
2. **`index out of bounds`** → Check `exportShape` is output shape, not input shape
3. **NaN/Inf loss values** → Check `Double.isInfinite()` guard in `GraphExporter`
4. **Wrong output values** → Verify bit field layout matches between Java encoder and Rust decoder
5. **`Assertion failed: dst_ncols == rhs_ncols`** → Check matrix dimension derivation in backward pass (linear backward had `(in_f, out_f)` swapped to `(out_f, in_f)`)
