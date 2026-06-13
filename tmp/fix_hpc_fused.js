const fs = require('fs');
const path = 'E:/rust_work/yishape_math_rust/src/graph.rs';
let content = fs.readFileSync(path, 'utf8');

// 1. Fix Java HPC_PATTERN (remove phantom tags)
const javaPath = 'E:/work/yishape-math/src/main/java/com/yishape/lab/math/autodiff/graph/GraphOpSchema.java';
let javaContent = fs.readFileSync(javaPath, 'utf8');
javaContent = javaContent.replace(
  `        /** {unary}{Reduce} tags with HPC faer implementations. */
        public static final Set<String> HPC_PATTERN = new HashSet<>(GPU_PATTERN); // currently identical, but independent copy`,
  `        /** {unary}{Reduce} tags with HPC faer implementations. */
        public static final Set<String> HPC_PATTERN = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "squareSum", "squareMean", "reluSum", "reluMean", "expSum", "expMean",
            "absSum", "absMean", "logSum", "logMean", "sigmoidSum", "sigmoidMean",
            "tanhSum", "tanhMean", "siluSum", "siluMean", "mishSum", "mishMean",
            "mulSum", "mulMean", "powSum", "powMean"
        )));`
);
fs.writeFileSync(javaPath, javaContent, 'utf8');
console.log('Fixed Java HPC_PATTERN');

// 2. Add 16 missing fused ops to HPC Rust forward dispatch
// Insert after "mulMean" forward handler, before "dot"
const mulMeanEnd = content.indexOf('        "dot" => {');
const forwardInsert = `
        "geluSum" => {
            let a = inputs.first().ok_or("geluSum: missing input")?;
            Ok(vec![a.iter().map(|v| 0.5 * v * (1.0 + (v * 0.7071067811865476).tanh())).sum()])
        }
        "geluMean" => {
            let a = inputs.first().ok_or("geluMean: missing input")?;
            let n = a.len() as f64;
            Ok(vec![a.iter().map(|v| 0.5 * v * (1.0 + (v * 0.7071067811865476).tanh())).sum::<f64>() / n])
        }
        "sinSum" => {
            let a = inputs.first().ok_or("sinSum: missing input")?;
            Ok(vec![a.iter().map(|v| v.sin()).sum()])
        }
        "sinMean" => {
            let a = inputs.first().ok_or("sinMean: missing input")?;
            let n = a.len() as f64;
            Ok(vec![a.iter().map(|v| v.sin()).sum::<f64>() / n])
        }
        "cosSum" => {
            let a = inputs.first().ok_or("cosSum: missing input")?;
            Ok(vec![a.iter().map(|v| v.cos()).sum()])
        }
        "cosMean" => {
            let a = inputs.first().ok_or("cosMean: missing input")?;
            let n = a.len() as f64;
            Ok(vec![a.iter().map(|v| v.cos()).sum::<f64>() / n])
        }
        "leakyReluSum" => {
            let a = inputs.first().ok_or("leakyReluSum: missing input")?;
            if !has_scalar { return Err("leakyReluSum: missing alpha scalar".into()); }
            let alpha = scalar;
            Ok(vec![a.iter().map(|v| if *v > 0.0 { *v } else { alpha * v }).sum()])
        }
        "leakyReluMean" => {
            let a = inputs.first().ok_or("leakyReluMean: missing input")?;
            let n = a.len() as f64;
            if !has_scalar { return Err("leakyReluMean: missing alpha scalar".into()); }
            let alpha = scalar;
            Ok(vec![a.iter().map(|v| if *v > 0.0 { *v } else { alpha * v }).sum::<f64>() / n])
        }
        "eluSum" => {
            let a = inputs.first().ok_or("eluSum: missing input")?;
            if !has_scalar { return Err("eluSum: missing alpha scalar".into()); }
            let alpha = scalar;
            Ok(vec![a.iter().map(|v| if *v > 0.0 { *v } else { alpha * (v.exp() - 1.0) }).sum()])
        }
        "eluMean" => {
            let a = inputs.first().ok_or("eluMean: missing input")?;
            let n = a.len() as f64;
            if !has_scalar { return Err("eluMean: missing alpha scalar".into()); }
            let alpha = scalar;
            Ok(vec![a.iter().map(|v| if *v > 0.0 { *v } else { alpha * (v.exp() - 1.0) }).sum::<f64>() / n])
        }
        "seluSum" => {
            let a = inputs.first().ok_or("seluSum: missing input")?;
            let scale = 1.05070098;
            let alpha = 1.67326324;
            Ok(vec![a.iter().map(|v| scale * if *v > 0.0 { *v } else { alpha * (v.exp() - 1.0) }).sum()])
        }
        "seluMean" => {
            let a = inputs.first().ok_or("seluMean: missing input")?;
            let n = a.len() as f64;
            let scale = 1.05070098;
            let alpha = 1.67326324;
            Ok(vec![a.iter().map(|v| scale * if *v > 0.0 { *v } else { alpha * (v.exp() - 1.0) }).sum::<f64>() / n])
        }
        "softplusSum" => {
            let a = inputs.first().ok_or("softplusSum: missing input")?;
            if !has_scalar { return Err("softplusSum: missing beta scalar".into()); }
            let beta = scalar;
            Ok(vec![a.iter().map(|v| (1.0 + (beta * v).exp()).ln() / beta).sum()])
        }
        "softplusMean" => {
            let a = inputs.first().ok_or("softplusMean: missing input")?;
            let n = a.len() as f64;
            if !has_scalar { return Err("softplusMean: missing beta scalar".into()); }
            let beta = scalar;
            Ok(vec![a.iter().map(|v| (1.0 + (beta * v).exp()).ln() / beta).sum::<f64>() / n])
        }
        "hardtanhSum" => {
            let a = inputs.first().ok_or("hardtanhSum: missing input")?;
            if !has_scalar { return Err("hardtanhSum: missing min/max scalar".into()); }
            let max_val = scalar;
            let min_val = param2;
            Ok(vec![a.iter().map(|v| v.max(min_val).min(max_val)).sum()])
        }
        "hardtanhMean" => {
            let a = inputs.first().ok_or("hardtanhMean: missing input")?;
            let n = a.len() as f64;
            if !has_scalar { return Err("hardtanhMean: missing min/max scalar".into()); }
            let max_val = scalar;
            let min_val = param2;
            Ok(vec![a.iter().map(|v| v.max(min_val).min(max_val)).sum::<f64>() / n])
        }
`;
content = content.slice(0, mulMeanEnd) + forwardInsert + content.slice(mulMeanEnd);
fs.writeFileSync(path, content, 'utf8');
console.log('Added 16 forward fused ops to HPC Rust');

// 3. Add 16 missing fused ops to HPC Rust backward dispatch
// Insert after "mulMean" backward handler, before "dot"
const bwMulMeanEnd = content.indexOf('        "dot" => {\n            let a = inputs.first().ok_or("missing dot input a")?;');
const bwInsert = `
        "geluSum" | "geluMean" => {
            let a = inputs.first().ok_or("geluSum/geluMean: missing input")?;
            let g = grad[0];
            let scale = 0.5;
            let c = 0.7071067811865476;
            Ok(vec![a.iter().map(|v| { let t = (c * v).tanh(); g * scale * v * (1.0 - t * t) * c + scale * (1.0 + t) }).collect()])
        }
        "sinSum" | "sinMean" => {
            let a = inputs.first().ok_or("sinSum: missing input")?;
            let g = grad[0];
            Ok(vec![a.iter().map(|v| g * v.cos()).collect()])
        }
        "cosSum" | "cosMean" => {
            let a = inputs.first().ok_or("cosSum: missing input")?;
            let g = grad[0];
            Ok(vec![a.iter().map(|v| -g * v.sin()).collect()])
        }
        "leakyReluSum" | "leakyReluMean" => {
            let a = inputs.first().ok_or("leakyReluSum: missing input")?;
            let g = grad[0];
            if !has_scalar { return Err("leakyReluSum: missing alpha scalar".into()); }
            let alpha = scalar;
            Ok(vec![a.iter().map(|v| if *v > 0.0 { g } else { g * alpha }).collect()])
        }
        "eluSum" | "eluMean" => {
            let a = inputs.first().ok_or("eluSum: missing input")?;
            let g = grad[0];
            if !has_scalar { return Err("eluSum: missing alpha scalar".into()); }
            let alpha = scalar;
            Ok(vec![a.iter().map(|v| if *v > 0.0 { g } else { g * alpha * v.exp() }).collect()])
        }
        "seluSum" | "seluMean" => {
            let a = inputs.first().ok_or("seluSum: missing input")?;
            let g = grad[0];
            let scale = 1.05070098;
            let alpha = 1.67326324;
            Ok(vec![a.iter().map(|v| if *v > 0.0 { g * scale } else { g * scale * alpha * v.exp() }).collect()])
        }
        "softplusSum" | "softplusMean" => {
            let a = inputs.first().ok_or("softplusSum: missing input")?;
            let g = grad[0];
            if !has_scalar { return Err("softplusSum: missing beta scalar".into()); }
            let beta = scalar;
            Ok(vec![a.iter().map(|v| { let s = (beta * v).exp(); g * s / (1.0 + s) }).collect()])
        }
        "hardtanhSum" | "hardtanhMean" => {
            let a = inputs.first().ok_or("hardtanhSum: missing input")?;
            let g = grad[0];
            if !has_scalar { return Err("hardtanhSum: missing min/max scalar".into()); }
            let max_val = scalar;
            let min_val = param2;
            Ok(vec![a.iter().map(|v| if *v >= max_val { 0.0 } else if *v <= min_val { 0.0 } else { g }).collect()])
        }
`;
content = content.slice(0, bwMulMeanEnd) + bwInsert + content.slice(bwMulMeanEnd);
fs.writeFileSync(path, content, 'utf8');
console.log('Added 16 backward fused ops to HPC Rust');
