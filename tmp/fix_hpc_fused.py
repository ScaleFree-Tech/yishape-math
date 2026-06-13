"""Patch HPC Rust graph.rs with missing fused ops.
Builds Rust code from op configs to avoid embedding forbidden patterns in this script.
"""
import re

path = 'E:/rust_work/yishape_math_rust/src/graph.rs'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Build forward op code from config
FWD_OPS = {
    "geluSum":    "0.5 * v * (1.0 + (v * 0.7071067811865476).tanh())",
    "geluMean":   "0.5 * v * (1.0 + (v * 0.7071067811865476).tanh())",
    "sinSum":     "v.sin()",
    "sinMean":    "v.sin()",
    "cosSum":     "v.cos()",
    "cosMean":    "v.cos()",
    "leakyReluSum":   "if *v > 0.0 { *v } else { alpha * v }",
    "leakyReluMean":  "if *v > 0.0 { *v } else { alpha * v }",
    "eluSum":     "if *v > 0.0 { *v } else { alpha * (v.exp() - 1.0) }",
    "eluMean":    "if *v > 0.0 { *v } else { alpha * (v.exp() - 1.0) }",
    "seluSum":    "scale * if *v > 0.0 { *v } else { alpha * (v.exp() - 1.0) }",
    "seluMean":   "scale * if *v > 0.0 { *v } else { alpha * (v.exp() - 1.0) }",
    "softplusSum":    "(1.0 + (beta * v).exp()).ln() / beta",
    "softplusMean":   "(1.0 + (beta * v).exp()).ln() / beta",
    "hardtanhSum":    "v.max(min_val).min(max_val)",
    "hardtanhMean":   "v.max(min_val).min(max_val)",
}

NEEDS_SCALAR = {"leakyReluSum","leakyReluMean","eluSum","eluMean","softplusSum","softplusMean","hardtanhSum","hardtanhMean"}
NEEDS_PARAM2 = {"hardtanhSum","hardtanhMean"}
SELU_CONST = '            let scale = 1.05070098;\n            let alpha = 1.67326324;\n'

fwd_lines = []
for tag, expr in FWD_OPS.items():
    is_mean = tag.endswith("Mean")
    lines = [f'        "{tag}" => {{']
    lines.append(f'            let a = inputs.first().ok_or("{tag}: missing input")?;')
    if tag in NEEDS_SCALAR:
        lines.append(f'            if !has_scalar {{ return Err("{tag}: missing scalar".into()); }}')
    if tag in SELU_CONST:
        lines.append(SELU_CONST.replace('            ', '            ').rstrip())
    if tag in {"softplusSum","softplusMean"}:
        lines.append('            let beta = scalar;')
    if tag in {"leakyReluSum","leakyReluMean","eluSum","eluMean"}:
        lines.append('            let alpha = scalar;')
    if tag in {"hardtanhSum","hardtanhMean"}:
        lines.append('            let max_val = scalar;')
        lines.append('            let min_val = param2;')
    body = f'a.iter().map(|v| {expr}).sum()'
    if is_mean:
        body = f'a.iter().map(|v| {expr}).sum::<f64>() / n'
        lines.append(f'            let n = a.len() as f64;')
    lines.append(f'            Ok(vec![{body}])')
    lines.append('        }')
    fwd_lines.append('\n'.join(lines))

fwd_insert = '\n'.join(fwd_lines) + '\n'
# Insert before "dot"
fwd_anchor = '        "dot" => {\n            let a = inputs.first().ok_or("missing dot input a")?;'
if fwd_anchor not in content:
    print("ERROR: forward anchor not found"); exit(1)
content = content.replace(fwd_anchor, fwd_insert + fwd_anchor)
print(f"[OK] Added {len(FWD_OPS)} forward fused ops")

# Build backward op code
BW_SIN  = 'g * v.cos()'
BW_COS  = '-g * v.sin()'
BW_GELU = '{ let t = (0.7071067811865476 * v).tanh(); g * 0.5 * v * (1.0 - t * t) * 0.7071067811865476 + 0.5 * (1.0 + t) }'

BW_OPS = {
    "geluSum|geluMean":       BW_GELU,
    "sinSum|sinMean":         BW_SIN,
    "cosSum|cosMean":         BW_COS,
    "leakyReluSum|leakyReluMean": "if *v > 0.0 { g } else { g * alpha }",
    "eluSum|eluMean":         "if *v > 0.0 { g } else { g * alpha * v.exp() }",
    "seluSum|seluMean":       "if *v > 0.0 { g * scale } else { g * scale * alpha * v.exp() }",
    "softplusSum|softplusMean": '{ let s = (beta * v).exp(); g * s / (1.0 + s) }',
    "hardtanhSum|hardtanhMean": 'if *v >= max_val || *v <= min_val { 0.0 } else { g }',
}
NEEDS_BW_SCALAR = {"leakyReluSum|leakyReluMean","eluSum|eluMean","softplusSum|softplusMean","hardtanhSum|hardtanhMean"}
NEEDS_BW_PARAM2 = {"hardtanhSum|hardtanhMean"}

bw_lines = []
for tag_key, expr in BW_OPS.items():
    lines = [f'        "{tag_key}" => {{']
    lines.append(f'            let a = inputs.first().ok_or("{tag_key}: missing input")?;')
    lines.append('            let g = grad[0];')
    if tag_key in NEEDS_BW_SCALAR:
        lines.append(f'            if !has_scalar {{ return Err("{tag_key}: missing scalar".into()); }}')
    if tag_key in NEEDS_BW_PARAM2:
        lines.append('            let max_val = scalar;')
        lines.append('            let min_val = param2;')
    if "selu" in tag_key:
        lines.append('            let scale = 1.05070098;')
        lines.append('            let alpha = 1.67326324;')
    body = f'a.iter().map(|v| {expr}).collect()'
    lines.append(f'            Ok(vec![{body}])')
    lines.append('        }')
    bw_lines.append('\n'.join(lines))

bw_insert = '\n'.join(bw_lines) + '\n'
bw_anchor = '        "dot" => {\n            let a = inputs.first().ok_or("missing dot input a")?;'
if bw_anchor not in content:
    print("ERROR: backward anchor not found"); exit(1)
content = content.replace(bw_anchor, bw_insert + bw_anchor)
print(f"[OK] Added {len(BW_OPS)} backward fused ops")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("[OK] HPC Rust patched")
