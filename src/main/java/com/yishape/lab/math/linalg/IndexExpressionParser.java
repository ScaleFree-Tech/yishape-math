package com.yishape.lab.math.linalg;

import java.util.regex.Pattern;

/**
 * 统一索引表达式解析器
 * Unified Index Expression Parser
 * <p>
 * 统一处理切片表达式、花式索引、布尔索引，支持负数索引，类似numpy的索引语义。
 * Unified handling of slice expressions, fancy indexing and boolean indexing
 * with negative indexing support, similar to numpy semantics.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 */
public class IndexExpressionParser {

    // ========== 正则表达式 ==========

    private static final Pattern SLICE_PATTERN =
        Pattern.compile("^(-?\\d*):(-?\\d*)(?::(-?\\d+))?$");

    /**
     * Sentinel value for negative-step slices where end was not specified
     * by the user (equivalent to Python None → "before the beginning").
     * Must be a value no user expression can produce.
     */
    private static final int NEG_STEP_END_SENTINEL = Integer.MIN_VALUE;

    // ========== 切片结果类 ==========

    /**
     * 切片结果类，包含原始表达式参数和解析后的实际索引范围。
     * Slice result class containing original expression parameters and resolved actual indices.
     *
     * <p>重要：{@code actualEnd} 的语义由 {@code inclusiveEnd} 标志决定：</p>
     * <ul>
     *   <li>如果 {@code inclusiveEnd == false}（默认），则 {@code actualEnd} 是<strong>不包含</strong>的边界（与Python/Numpy一致），迭代终止条件为 {@code i &lt; actualEnd}</li>
     *   <li>如果 {@code inclusiveEnd == true}，则 {@code actualEnd} 是<strong>包含</strong>的边界，迭代终止条件为 {@code i &lt;= actualEnd}</li>
     * </ul>
     * <p>Important: the meaning of {@code actualEnd} is controlled by {@code inclusiveEnd}:</p>
     * <ul>
     *   <li>If {@code inclusiveEnd == false} (default), {@code actualEnd} is <strong>exclusive</strong> (Python/Numpy semantics), loop condition is {@code i &lt; actualEnd}</li>
     *   <li>If {@code inclusiveEnd == true}, {@code actualEnd} is <strong>inclusive</strong>, loop condition is {@code i &lt;= actualEnd}</li>
     * </ul>
     */
    public static class SliceResult {
        /** 原始表达式中的start（可能为负数或空字符串表示None） */
        public final int start;
        /** 原始表达式中的end（可能为负数或空字符串表示None） */
        public final int end;
        /** 步长（永不为零） */
        public final int step;
        /** 解析后的实际起始索引（已处理负数索引和越界截断） */
        public final int actualStart;
        /** 解析后的实际结束索引（语义由 inclusiveEnd 决定） */
        public final int actualEnd;
        /**
         * actualEnd 是否为包含边界。
         * 当为 true 时，迭代终止条件为 i &lt;= actualEnd。
         * 当为 false 时，迭代终止条件为 i &lt; actualEnd。
         */
        public final boolean inclusiveEnd;

        public SliceResult(int start, int end, int step, int actualStart, int actualEnd) {
            this(start, end, step, actualStart, actualEnd, false);
        }

        public SliceResult(int start, int end, int step, int actualStart, int actualEnd, boolean inclusiveEnd) {
            this.start = start;
            this.end = end;
            this.step = step;
            this.actualStart = actualStart;
            this.actualEnd = actualEnd;
            this.inclusiveEnd = inclusiveEnd;
        }

        /**
         * 返回实际结束索引的包含版本（如果 inclusiveEnd 为 true 则与 actualEnd 相同，
         * 否则返回 actualEnd - 1）。
         * 用于需要明确包含边界值的场景。
         */
        public int getInclusiveActualEnd() {
            return inclusiveEnd ? actualEnd : actualEnd - Integer.signum(step);
        }

        @Override
        public String toString() {
            return String.format("SliceResult{start=%d, end=%d, step=%d, actualStart=%d, actualEnd=%d, inclusiveEnd=%s}",
                start, end, step, actualStart, actualEnd, inclusiveEnd);
        }
    }

    // ========== 花式索引结果类 ==========

    /**
     * 花式索引结果类，包含解析后的实际索引数组。
     */
    public static class FancyIndexResult {
        public final int[] indices;

        public FancyIndexResult(int[] indices) {
            this.indices = indices;
        }
    }

    // ========== 布尔索引结果类 ==========

    /**
     * 布尔索引结果类，包含提取出的元素值索引数组和提取后的元素数量。
     */
    public static class BooleanIndexResult {
        /** true位置对应的索引数组 */
        public final int[] trueIndices;
        /** true位置的数量 */
        public final int count;

        public BooleanIndexResult(int[] trueIndices, int count) {
            this.trueIndices = trueIndices;
            this.count = count;
        }
    }

    // ========== 切片解析 ==========

    /**
     * 解析切片表达式。
     * 支持格式：start:end:step 或 start:end 或 :end 或 start: 或 : 等
     * 支持负数索引，语义与 NumPy 一致。
     *
     * @param sliceExpression 切片表达式，如 "1:3", ":-1", "::2", "1:-1:2" 等
     * @param maxSize 最大尺寸（用于处理负数索引和越界截断）
     * @return 解析结果
     * @throws IllegalArgumentException 如果表达式格式不正确或 step 为 0
     */
    public static SliceResult parse(String sliceExpression, int maxSize) {
        if (sliceExpression == null || sliceExpression.trim().isEmpty()) {
            throw new IllegalArgumentException("切片表达式不能为空 / Slice expression cannot be empty");
        }

        if (maxSize < 0) {
            throw new IllegalArgumentException("最大尺寸不能为负数 / Max size cannot be negative: " + maxSize);
        }

        String trimmed = sliceExpression.trim();
        var matcher = SLICE_PATTERN.matcher(trimmed);

        if (!matcher.matches()) {
            throw new IllegalArgumentException("无效的切片表达式: " + sliceExpression + " / Invalid slice expression: " + sliceExpression);
        }

        // 解析 start
        String startStr = matcher.group(1);
        int origStart = startStr.isEmpty() ? 0 : Integer.parseInt(startStr);

        // 解析 end
        String endStr = matcher.group(2);
        int origEnd = endStr.isEmpty() ? maxSize : Integer.parseInt(endStr);

        // 解析 step
        String stepStr = matcher.group(3);
        int step = stepStr == null ? 1 : Integer.parseInt(stepStr);

        // 验证 step
        if (step == 0) {
            throw new IllegalArgumentException("步长不能为0 / Step cannot be zero");
        }

        // Apply step-direction-aware defaults for resolution.
        // origStart/origEnd preserve the original expression values (stored in SliceResult).
        int resStart, resEnd;
        if (step > 0) {
            resStart = origStart;
            resEnd = origEnd;
        } else {
            // Negative step: default start = last element, default end = "before beginning" sentinel
            resStart = startStr.isEmpty() ? maxSize - 1 : origStart;
            resEnd = endStr.isEmpty() ? NEG_STEP_END_SENTINEL : origEnd;
        }

        SliceResult resolved = resolveSlice(resStart, resEnd, step, maxSize);
        // Return result with original expression values preserved
        return new SliceResult(origStart, origEnd, step, resolved.actualStart, resolved.actualEnd, resolved.inclusiveEnd);
    }

    /**
     * 将原始的 (start, end, step, maxSize) 解析为带边界的 SliceResult。
     * 这是核心解析逻辑，供 parse(String, int) 和内部使用。
     */
    static SliceResult resolveSlice(int start, int end, int step, int maxSize) {
        int actualStart = start;
        int actualEnd = end;

        // Detect sentinel: end was not specified by user for a negative step
        boolean endIsSentinel = (step < 0 && end == NEG_STEP_END_SENTINEL);

        // Resolve negative indices (but NOT the sentinel)
        if (actualStart < 0) {
            actualStart = maxSize + actualStart;
        }
        if (actualEnd < 0 && !endIsSentinel) {
            actualEnd = maxSize + actualEnd;
        }

        if (step > 0) {
            // Positive step: end is exclusive (standard Python/NumPy semantics)
            if (actualStart < 0) {
                actualStart = 0;
            }
            if (actualEnd > maxSize) {
                actualEnd = maxSize;
            }
            // If start >= end, produce empty slice
            if (actualStart > actualEnd) {
                actualStart = actualEnd;
            }
        } else {
            // Negative step

            // Resolve sentinel: "before the beginning" → go to index 0 inclusive
            if (endIsSentinel) {
                actualEnd = -1; // sentinel meaning "to index 0, inclusive"
            }

            // Clamp start to valid index range
            if (actualStart < 0) {
                actualStart = -1;
            }
            if (actualStart >= maxSize) {
                actualStart = maxSize - 1;
            }

            // Clamp end: values past maxSize mean "to the end" → sentinel -1
            if (actualEnd > maxSize) {
                actualEnd = -1;
            }
            if (actualEnd < -1) {
                actualEnd = -1;
            }

            // Empty slice: start <= end and end is not the "to beginning" sentinel
            if (actualStart <= actualEnd && actualEnd != -1) {
                actualStart = actualEnd;
            }
        }

        return new SliceResult(start, end, step, actualStart, actualEnd);
    }

    /**
     * 解析切片表达式，支持指定 actualEnd 的包含性。
     * 当 end 为 -1 时使用包含语义（inclusiveEnd=true），表示"到开头，包含"。
     */
    public static SliceResult parse(String sliceExpression, int maxSize, boolean forceInclusiveEnd) {
        SliceResult result = parse(sliceExpression, maxSize);
        if (forceInclusiveEnd && result.step < 0 && result.actualEnd == -1) {
            // 对于 ::-1 这类表达式，设置包含语义
            // actualEnd = -1, inclusiveEnd = true 表示"到索引0，包含"
            return new SliceResult(result.start, result.end, result.step,
                result.actualStart, result.actualEnd, true);
        }
        return result;
    }

    // ========== 花式索引解析 ==========

    /**
     * 解析花式索引（位置数组），支持负数索引，返回解析后的实际索引数组。
     * 不修改原始数组，返回新数组。
     *
     * @param positions 原始位置索引数组（支持负数索引）
     * @param maxSize 最大尺寸
     * @return 包含解析后索引数组的结果对象
     * @throws IndexOutOfBoundsException 如果任何索引超出范围
     * @throws IllegalArgumentException 如果 positions 为 null
     */
    public static FancyIndexResult resolveFancyIndex(int[] positions, int maxSize) {
        if (positions == null) {
            throw new IllegalArgumentException("索引数组不能为null / Index array cannot be null");
        }
        int[] resolved = new int[positions.length];
        for (int i = 0; i < positions.length; i++) {
            int idx = positions[i];
            if (idx < 0) {
                idx = maxSize + idx;
            }
            if (idx < 0 || idx >= maxSize) {
                throw new IndexOutOfBoundsException("花式索引超出范围: " + positions[i] + " (max=" + maxSize + ") / Fancy index out of bounds: " + positions[i] + " (max=" + maxSize + ")");
            }
            resolved[i] = idx;
        }
        return new FancyIndexResult(resolved);
    }

    // ========== 布尔索引解析 ==========

    /**
     * 解析布尔索引，返回 true 位置对应的索引数组。
     *
     * @param booleanIndex 布尔数组
     * @return 包含 true 位置索引和计数的结果对象
     * @throws IllegalArgumentException 如果 booleanIndex 为 null
     */
    public static BooleanIndexResult resolveBooleanIndex(boolean[] booleanIndex) {
        if (booleanIndex == null) {
            throw new IllegalArgumentException("布尔索引数组不能为null / Boolean index array cannot be null");
        }
        // 第一次遍历：计数
        int count = 0;
        for (boolean b : booleanIndex) {
            if (b) count++;
        }
        // 第二次遍历：收集索引
        int[] indices = new int[count];
        int idx = 0;
        for (int i = 0; i < booleanIndex.length; i++) {
            if (booleanIndex[i]) {
                indices[idx++] = i;
            }
        }
        return new BooleanIndexResult(indices, count);
    }

    // ========== 切片大小计算（统一方法，消除重复） ==========

    /**
     * 计算切片结果的元素个数。
     * 这是切片操作的核心计算方法，与具体数据类型无关。
     *
     * @param sliceResult 切片结果
     * @return 切片中的元素个数（始终 >= 0）
     */
    public static int calculateSliceSize(SliceResult sliceResult) {
        return calculateSliceSize(
            sliceResult.actualStart,
            sliceResult.actualEnd,
            sliceResult.step,
            sliceResult.inclusiveEnd
        );
    }

    /**
     * 根据原始参数计算切片结果的元素个数。
     * 这是最底层的计算方法，支持 inclusiveEnd 语义。
     *
     * @param actualStart 实际起始索引
     * @param actualEnd 实际结束索引（语义由 inclusiveEnd 决定）
     * @param step 步长
     * @param inclusiveEnd actualEnd 是否为包含边界
     * @return 元素个数（始终 >= 0）
     */
    public static int calculateSliceSize(int actualStart, int actualEnd, int step, boolean inclusiveEnd) {
        if (step > 0) {
            if (actualStart >= actualEnd) {
                return 0;
            }
            return Math.max(0, (actualEnd - actualStart + step - 1) / step);
        } else {
            // Negative step
            int absStep = Math.abs(step);
            if (inclusiveEnd) {
                // actualEnd is inclusive
                if (actualEnd == -1) {
                    // Sentinel: go to index 0 inclusive
                    if (actualStart < 0) {
                        return 0;
                    }
                    return Math.max(0, actualStart / absStep + 1);
                }
                if (actualStart <= actualEnd) {
                    return 0;
                }
                return Math.max(0, (actualStart - actualEnd) / absStep + 1);
            } else {
                // actualEnd is exclusive (standard Python semantics)
                if (actualStart <= actualEnd) {
                    return 0;
                }
                if (actualEnd == -1) {
                    // Sentinel: go to index 0 inclusive (-1 sentinel always means "to 0 inclusive")
                    if (actualStart < 0) {
                        return 0;
                    }
                    return Math.max(0, actualStart / absStep + 1);
                }
                // Exclusive end with negative step: last included index = actualEnd + 1
                return Math.max(0, (actualStart - actualEnd - 1) / absStep + 1);
            }
        }
    }

    /**
     * 简化版：根据 actualStart、actualEnd（不包含）、step 计算切片大小。
     * 兼容旧的 -1 哨兵语义。
     */
    public static int calculateSliceSizeLegacy(int actualStart, int actualEnd, int step) {
        if (step > 0) {
            if (actualStart >= actualEnd) {
                return 0;
            }
            return Math.max(0, (actualEnd - actualStart + step - 1) / step);
        } else {
            // 负数步长，使用 -1 哨兵语义
            int absStep = Math.abs(step);
            if (actualStart <= actualEnd) {
                return 0;
            }
            if (actualEnd == -1) {
                // 到开头，包含
                return Math.max(0, actualStart / absStep + 1);
            } else {
                // actualEnd 是 exclusive 的，包含边界是 actualEnd - 1
                return Math.max(0, (actualStart - actualEnd - 1) / absStep + 1);
            }
        }
    }

    // ========== 索引数组生成 ==========

    /**
     * 根据 SliceResult 生成索引数组。
     * 支持正数和负数步长。
     *
     * @param sliceResult 切片结果
     * @return 索引数组
     */
    public static int[] generateIndices(SliceResult sliceResult) {
        int size = calculateSliceSize(sliceResult);
        if (size == 0) {
            return new int[0];
        }
        int[] indices = new int[size];
        int idx = 0;
        if (sliceResult.step > 0) {
            for (int i = sliceResult.actualStart; i < sliceResult.actualEnd; i += sliceResult.step) {
                indices[idx++] = i;
            }
        } else {
            if (sliceResult.inclusiveEnd) {
                if (sliceResult.actualEnd == -1) {
                    // ::-1 特殊处理：actualStart 到 0
                    for (int i = sliceResult.actualStart; i >= 0; i += sliceResult.step) {
                        indices[idx++] = i;
                    }
                } else {
                    for (int i = sliceResult.actualStart; i >= sliceResult.actualEnd; i += sliceResult.step) {
                        indices[idx++] = i;
                    }
                }
            } else {
                if (sliceResult.actualEnd == -1) {
                    // actualEnd = -1 作为哨兵，表示到开头（包含）
                    for (int i = sliceResult.actualStart; i >= 0; i += sliceResult.step) {
                        indices[idx++] = i;
                    }
                } else {
                    for (int i = sliceResult.actualStart; i > sliceResult.actualEnd; i += sliceResult.step) {
                        indices[idx++] = i;
                    }
                }
            }
        }
        return indices;
    }

    /**
     * 生成花式索引对应的元素值数组（从数据源中提取）。
     * 返回的数组索引对应 positions 数组的顺序。
     *
     * @param data 数据数组
     * @param positions 原始位置索引（支持负数）
     * @return 从数据中提取的元素值数组
     */
    public static double[] fancyGetData(double[] data, int[] positions) {
        FancyIndexResult resolved = resolveFancyIndex(positions, data.length);
        double[] result = new double[positions.length];
        for (int i = 0; i < resolved.indices.length; i++) {
            result[i] = data[resolved.indices[i]];
        }
        return result;
    }

    /**
     * 生成花式索引对应的元素值数组（从数据源中提取，float版本）。
     */
    public static float[] fancyGetData(float[] data, int[] positions) {
        FancyIndexResult resolved = resolveFancyIndex(positions, data.length);
        float[] result = new float[positions.length];
        for (int i = 0; i < resolved.indices.length; i++) {
            result[i] = data[resolved.indices[i]];
        }
        return result;
    }

    /**
     * 根据布尔索引从数据源中提取元素。
     */
    public static double[] booleanGetData(double[] data, boolean[] booleanIndex) {
        BooleanIndexResult resolved = resolveBooleanIndex(booleanIndex);
        if (booleanIndex.length != data.length) {
            throw new IllegalArgumentException("布尔索引长度(" + booleanIndex.length + ")与数据长度(" + data.length + ")不匹配 / Boolean index length mismatch");
        }
        double[] result = new double[resolved.count];
        for (int i = 0; i < resolved.trueIndices.length; i++) {
            result[i] = data[resolved.trueIndices[i]];
        }
        return result;
    }

    /**
     * 根据布尔索引从数据源中提取元素（float版本）。
     */
    public static float[] booleanGetData(float[] data, boolean[] booleanIndex) {
        BooleanIndexResult resolved = resolveBooleanIndex(booleanIndex);
        if (booleanIndex.length != data.length) {
            throw new IllegalArgumentException("布尔索引长度(" + booleanIndex.length + ")与数据长度(" + data.length + ")不匹配 / Boolean index length mismatch");
        }
        float[] result = new float[resolved.count];
        for (int i = 0; i < resolved.trueIndices.length; i++) {
            result[i] = data[resolved.trueIndices[i]];
        }
        return result;
    }

    // ========== 辅助方法 ==========

    /**
     * 检查表达式是否为切片表达式
     */
    public static boolean isSliceExpression(String expression) {
        if (expression == null || expression.trim().isEmpty()) {
            return false;
        }
        return SLICE_PATTERN.matcher(expression.trim()).matches();
    }

    /**
     * 检查表达式是否为简单整数索引表达式
     */
    public static boolean isIndexExpression(String expression) {
        if (expression == null || expression.trim().isEmpty()) {
            return false;
        }
        try {
            Integer.parseInt(expression.trim());
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    /**
     * 解析单个整数索引（支持负数）
     */
    public static int parseIndex(String indexExpression, int maxSize) {
        if (indexExpression == null || indexExpression.trim().isEmpty()) {
            throw new IllegalArgumentException("索引表达式不能为空 / Index expression cannot be empty");
        }
        if (maxSize <= 0) {
            throw new IllegalArgumentException("最大尺寸必须大于0 / Max size must be greater than 0");
        }
        try {
            int index = Integer.parseInt(indexExpression.trim());
            int actualIndex = index < 0 ? maxSize + index : index;
            if (actualIndex < 0 || actualIndex >= maxSize) {
                throw new IndexOutOfBoundsException("索引超出范围: " + index + " / Index out of bounds: " + index);
            }
            return actualIndex;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("无效的索引表达式: " + indexExpression + " / Invalid index expression: " + indexExpression);
        }
    }

    /**
     * 花式索引赋值（批量设值）。
     * @param data 数据数组
     * @param positions 原始位置索引（支持负数）
     * @param values 要设置的值数组
     */
    public static void fancySetData(double[] data, int[] positions, double[] values) {
        if (positions.length != values.length) {
            throw new IllegalArgumentException("索引数组长度(" + positions.length + ")与值数组长度(" + values.length + ")不匹配 / Index array length mismatch");
        }
        FancyIndexResult resolved = resolveFancyIndex(positions, data.length);
        for (int i = 0; i < resolved.indices.length; i++) {
            data[resolved.indices[i]] = values[i];
        }
    }

    /**
     * 花式索引赋值（float版本）
     */
    public static void fancySetData(float[] data, int[] positions, float[] values) {
        if (positions.length != values.length) {
            throw new IllegalArgumentException("索引数组长度(" + positions.length + ")与值数组长度(" + values.length + ")不匹配 / Index array length mismatch");
        }
        FancyIndexResult resolved = resolveFancyIndex(positions, data.length);
        for (int i = 0; i < resolved.indices.length; i++) {
            data[resolved.indices[i]] = values[i];
        }
    }
}
