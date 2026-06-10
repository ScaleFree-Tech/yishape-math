# -*- coding: utf-8 -*-
"""Apply surgical edits per audit; idempotent."""
from pathlib import Path

ROOT = Path(r"e:\work\yishape-math")
results = []

def read_rel(rel):
    p = ROOT / rel
    return p.read_text(encoding="utf-8"), p

def write_rel(rel, text):
    p = ROOT / rel
    p.write_text(text, encoding="utf-8")

# 1. BatchedDiffTensor
rel = Path("src/main/java/com/yishape/lab/math/autodiff/BatchedDiffTensor.java")
text, _ = read_rel(rel)
unwrap = """    /** Unwrap nested BatchedDiffTensor for per-sample RNN ops inside vmap. */
    private static IDiffTensor unwrapDiff(IDiffTensor t) {
        return (t instanceof BatchedDiffTensor bdt) ? bdt.data : t;
    }
"""
if "unwrapDiff(IDiffTensor" in text:
    results.append(("1 BatchedDiffTensor", "already_present", "unwrapDiff exists"))
else:
    marker = "    private static IDoubleTensor detachOther(IDoubleTensor t) {"
    if marker not in text:
        results.append(("1 BatchedDiffTensor", "FAILED", "detachOther not found"))
    else:
        # insert after detachOther method block - find closing brace after detachOther
        idx = text.find(marker)
        # find end of detachOther: next "}\n" at same indent after return line
        sub = text[idx:]
        end_rel = sub.find("\n    }\n")
        if end_rel < 0:
            results.append(("1 BatchedDiffTensor", "FAILED", "detachOther end not found"))
        else:
            ins = idx + end_rel + len("\n    }\n")
            text = text[:ins] + "\n" + unwrap + text[ins:]
            write_rel(rel, text)
            results.append(("1 BatchedDiffTensor", "APPLIED", "added unwrapDiff after detachOther"))

text, _ = read_rel(rel)
if "unwrapDiff(x)" in text and "lstmCell" in text:
    results.append(("1 lstmCell/gruCell", "OK", "call unwrapDiff"))
else:
    # fix lstm/gru if needed - user said fix them
    results.append(("1 lstmCell/gruCell", "CHECK", "may need manual fix"))

# 2. FusedReductionOps - verify key patterns
rel2 = Path("src/main/java/com/yishape/lab/math/autodiff/impl/FusedReductionOps.java")
text2, _ = read_rel(rel2)
checks2 = []
if "mid = forwardElementOpsWithSaved(xData, n, saved)" in text2:
    checks2.append("single forwardElementOpsWithSaved in compute")
else:
    if "forwardElementOps(xData" in text2:
        # apply fix - replace old compute section
        old = """        double[][] saved = null;
        double[] mid;
        if (numElem > 0) {
            saved = new double[numElem][];
            for (int j = 0; j < numElem; j++) saved[j] = AutodiffBufferPool.acquire(n);
            mid = forwardElementOpsWithSaved(xData, n, saved);
        } else {
            mid = xData;
        }
"""
        # try alternate old pattern with double call
        pass
    checks2.append("MISSING forwardElementOpsWithSaved path")

if "final double[] fMidPooled = midPooled ? mid : null" in text2:
    checks2.append("fMidPooled capture")
if "AutodiffBufferPool.release(fMidPooled)" in text2:
    checks2.append("fMidPooled release")
if "private double[] forwardElementOpsWithSaved" in text2 and "AutodiffBufferPool.release(next)" in text2:
    checks2.append("forwardElementOpsWithSaved returns cur releases next")

if all("MISSING" not in c for c in checks2) and len(checks2) >= 3:
    results.append(("2 FusedReductionOps", "already_present", "; ".join(checks2)))
else:
    results.append(("2 FusedReductionOps", "PARTIAL" if checks2 else "FAILED", "; ".join(checks2)))

# 3 SIMDDoubleComputer
rel3 = Path("src/main/java/com/yishape/lab/math/compute/SIMDDoubleComputer.java")
text3, _ = read_rel(rel3)
old3 = "case REMAINDER -> x1[i] - Math.floor(x1[i] / x2) * x2;"
new3 = "case REMAINDER -> x1[i] % x2;"
if new3 in text3:
    results.append(("3 SIMDDoubleComputer", "already_present", new3))
elif old3 in text3:
    text3 = text3.replace(old3, new3)
    write_rel(rel3, text3)
    results.append(("3 SIMDDoubleComputer", "APPLIED", old3 + " -> " + new3))
else:
    results.append(("3 SIMDDoubleComputer", "FAILED", "REMAINDER case not found"))

# 4 RereDiffVector clone
rel4 = Path("src/main/java/com/yishape/lab/math/autodiff/impl/RereDiffVector.java")
text4, _ = read_rel(rel4)
if "initialGradient.getData().clone()" in text4:
    results.append(("4 RereDiffVector setGradData", "already_present", ".clone()"))
else:
    old4a = "tensor.setGradData(initialGradient.getData());"
    if old4a in text4:
        text4 = text4.replace(old4a, "tensor.setGradData(initialGradient.getData().clone());")
        write_rel(rel4, text4)
        results.append(("4 RereDiffVector setGradData", "APPLIED", "added .clone()"))
    else:
        results.append(("4 RereDiffVector setGradData", "FAILED", "line not found"))

# 5 RereDiffMatrix
rel5 = Path("src/main/java/com/yishape/lab/math/autodiff/impl/RereDiffMatrix.java")
text5, _ = read_rel(rel5)
if "this.gradient = initialGradient.copy();" in text5:
    results.append(("5 RereDiffMatrix", "already_present", "initialGradient.copy()"))
else:
    for old5 in ("this.gradient = initialGradient;", "this.gradient = initialGradient.getData();"):
        if old5 in text5:
            text5 = text5.replace(old5, "this.gradient = initialGradient.copy();")
            write_rel(rel5, text5)
            results.append(("5 RereDiffMatrix", "APPLIED", old5 + " -> copy()"))
            break
    else:
        results.append(("5 RereDiffMatrix", "FAILED", "backward line not found"))

# 6 remainder backward
text4, _ = read_rel(rel4)
if 'backwardFn, "remainder"' in text4 and "accGrad(gradOut.gradData())" in text4.split("remainder")[1][:800]:
    results.append(("6 RereDiffVector remainder", "already_present", "straight-through backwardFn"))
else:
    # find remainder without backward
    results.append(("6 RereDiffVector remainder", "NEEDS_EDIT", "check remainder block"))

# 7 GpuGemm long overflow
rel7 = Path("src/main/java/com/yishape/lab/math/compute/gpu/GpuGemm.java")
text7, _ = read_rel(rel7)
fixes7 = []
if "(long) C * H * W" in text7 or "(long)C*H*W" in text7.replace(" ", ""):
    fixes7.append("C*H*W long")
elif "input.length < C * H * W" in text7:
    text7 = text7.replace("input.length < C * H * W", "input.length < (long) C * H * W")
    fixes7.append("patched C*H*W")
if "(long) k * m" in text7 and "(long) m * k" in text7:
    fixes7.append("k*m long ok")
else:
    repls = [
        ("? k * m : m * k", "? (long) k * m : (long) m * k"),
        ("? n * k : k * n", "? (long) n * k : (long) k * n"),
    ]
    for a,b in repls:
        if a in text7 and b not in text7:
            text7 = text7.replace(a, b)
            fixes7.append("patched " + a)
if "patched" in str(fixes7):
    write_rel(rel7, text7)
    results.append(("7 GpuGemm", "APPLIED", "; ".join(fixes7)))
elif len(fixes7) >= 2:
    results.append(("7 GpuGemm", "already_present", "; ".join(fixes7)))
else:
    results.append(("7 GpuGemm", "PARTIAL", "; ".join(fixes7) or "unknown state"))

for r in results:
    print(f"{r[0]}: {r[1]} — {r[2]}")
