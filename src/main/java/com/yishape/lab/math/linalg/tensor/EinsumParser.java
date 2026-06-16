package com.yishape.lab.math.linalg.tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Einstein summation convention parser.
 * Parses subscript strings into classified axes for composing einsum from basic ops.
 *
 * <p>Supports up to 2 inputs with explicit output (arrow notation).
 * Each axis label must be a single lowercase letter (a-z).
 * Classifies axes into batch (appear in all inputs + output), contract (summed over,
 * appear in inputs but not output), and kept (appear in one input + output, not batch).</p>
 */
public final class EinsumParser {

    private EinsumParser() {}

    /**
     * Parse an einsum subscript and validate against input shapes.
     *
     * @param subscript   e.g. "bij,bjk->bik"
     * @param inputShapes shape arrays for each input tensor
     * @return classified axis specification
     * @throws IllegalArgumentException if subscript is malformed or shapes mismatch
     * @throws UnsupportedOperationException if >2 inputs
     */
    public static EinsumSpec parse(String subscript, int[]... inputShapes) {
        String[] parts = subscript.split("->");
        if (parts.length != 2) {
            throw new IllegalArgumentException("einsum requires explicit output arrow (->): " + subscript);
        }
        String[] inputStrs = parts[0].split(",");
        if (inputStrs.length != inputShapes.length) {
            throw new IllegalArgumentException(
                "einsum input count mismatch: " + inputStrs.length + " subscripts, " + inputShapes.length + " tensors");
        }
        if (inputStrs.length > 2) {
            throw new UnsupportedOperationException(
                "einsum with >2 inputs not yet supported: " + subscript);
        }

        String outputStr = parts[1];

        validateLabels(inputStrs, outputStr);
        validateShapes(inputStrs, inputShapes);

        Map<Character, Integer> axisSizes = buildAxisSizes(inputStrs, inputShapes);

        List<Set<Character>> perInputAxes = new ArrayList<>();
        Set<Character> allInputAxes = new LinkedHashSet<>();
        for (String s : inputStrs) {
            Set<Character> axes = charsetOf(s);
            perInputAxes.add(axes);
            allInputAxes.addAll(axes);
        }

        Set<Character> outputAxes = charsetOf(outputStr);

        // batch axes: appear in ALL inputs AND in output
        Set<Character> batchAxes = new LinkedHashSet<>(allInputAxes);
        for (Set<Character> s : perInputAxes) batchAxes.retainAll(s);
        batchAxes.retainAll(outputAxes);

        // contract axes: appear in inputs but NOT in output
        Set<Character> contractAxes = new LinkedHashSet<>(allInputAxes);
        contractAxes.removeAll(outputAxes);

        return new EinsumSpec(inputStrs, outputStr, batchAxes, contractAxes,
                              axisSizes, perInputAxes, outputAxes);
    }

    private static void validateLabels(String[] inputStrs, String outputStr) {
        for (String s : inputStrs) {
            for (char c : s.toCharArray()) {
                if (c < 'a' || c > 'z') {
                    throw new IllegalArgumentException(
                        "einsum axis labels must be lowercase a-z, got: '" + c + "' in \"" + s + "\"");
                }
            }
        }
        for (char c : outputStr.toCharArray()) {
            if (c < 'a' || c > 'z') {
                throw new IllegalArgumentException(
                    "einsum axis labels must be lowercase a-z, got: '" + c + "' in output");
            }
        }
    }

    private static void validateShapes(String[] inputStrs, int[][] inputShapes) {
        for (int i = 0; i < inputStrs.length; i++) {
            if (inputStrs[i].length() != inputShapes[i].length) {
                throw new IllegalArgumentException(
                    "einsum: input " + i + " label \"" + inputStrs[i] + "\" has "
                    + inputStrs[i].length() + " axes but shape has "
                    + inputShapes[i].length + " dims");
            }
        }
    }

    private static Map<Character, Integer> buildAxisSizes(String[] inputStrs, int[][] inputShapes) {
        Map<Character, Integer> axisSizes = new LinkedHashMap<>();
        for (int i = 0; i < inputStrs.length; i++) {
            for (int j = 0; j < inputStrs[i].length(); j++) {
                char c = inputStrs[i].charAt(j);
                int size = inputShapes[i][j];
                Integer existing = axisSizes.get(c);
                if (existing != null && existing != size) {
                    throw new IllegalArgumentException(
                        "einsum: axis '" + c + "' size conflict: " + existing + " vs " + size);
                }
                axisSizes.put(c, size);
            }
        }
        return axisSizes;
    }

    private static Set<Character> charsetOf(String s) {
        Set<Character> set = new LinkedHashSet<>();
        for (char c : s.toCharArray()) set.add(c);
        return set;
    }

    // ==================== Spec ====================

    public static class EinsumSpec {
        public final String[] inputLabels;
        public final String outputLabels;
        public final Set<Character> batchAxes;
        public final Set<Character> contractAxes;
        public final Map<Character, Integer> axisSizes;
        public final List<Set<Character>> perInputAxes;
        public final Set<Character> outputAxes;

        EinsumSpec(String[] inputLabels, String outputLabels,
                   Set<Character> batchAxes, Set<Character> contractAxes,
                   Map<Character, Integer> axisSizes,
                   List<Set<Character>> perInputAxes, Set<Character> outputAxes) {
            this.inputLabels = inputLabels;
            this.outputLabels = outputLabels;
            this.batchAxes = batchAxes;
            this.contractAxes = contractAxes;
            this.axisSizes = axisSizes;
            this.perInputAxes = perInputAxes;
            this.outputAxes = outputAxes;
        }

        public boolean twoInputs() { return inputLabels.length == 2; }
        public boolean singleInput() { return inputLabels.length == 1; }

        /**
         * Permute order to align input A for bmm: [batch... | kept... | contract...].
         */
        public int[] permuteA() {
            return permuteForInput(0, true);
        }

        /**
         * Permute order to align input B for bmm: [batch... | contract... | kept...].
         */
        public int[] permuteB() {
            return permuteForInput(1, false);
        }

        private int[] permuteForInput(int idx, boolean keptBeforeContract) {
            String labels = inputLabels[idx];
            Set<Character> inputAxes = perInputAxes.get(idx);

            Set<Character> keptAxes = new LinkedHashSet<>(inputAxes);
            keptAxes.retainAll(outputAxes);
            keptAxes.removeAll(batchAxes);

            List<Integer> order = new ArrayList<>();
            for (int i = 0; i < labels.length(); i++) {
                if (batchAxes.contains(labels.charAt(i))) order.add(i);
            }
            if (keptBeforeContract) {
                for (int i = 0; i < labels.length(); i++) {
                    if (keptAxes.contains(labels.charAt(i))) order.add(i);
                }
                for (int i = 0; i < labels.length(); i++) {
                    if (contractAxes.contains(labels.charAt(i))) order.add(i);
                }
            } else {
                for (int i = 0; i < labels.length(); i++) {
                    if (contractAxes.contains(labels.charAt(i))) order.add(i);
                }
                for (int i = 0; i < labels.length(); i++) {
                    if (keptAxes.contains(labels.charAt(i))) order.add(i);
                }
            }
            return order.stream().mapToInt(Integer::intValue).toArray();
        }

        /**
         * Compute 3D reshape for bmm: [batchProd, keptProd, contractProd].
         */
        public int[] reshapeTo3D(int inputIdx, int[] shape) {
            int batchProd = 1, keptProd = 1, contractProd = 1;
            String labels = inputLabels[inputIdx];
            Set<Character> inputAxes = perInputAxes.get(inputIdx);
            Set<Character> keptAxes = new LinkedHashSet<>(inputAxes);
            keptAxes.retainAll(outputAxes);
            keptAxes.removeAll(batchAxes);

            for (int i = 0; i < labels.length(); i++) {
                char c = labels.charAt(i);
                int sz = shape[i];
                if (batchAxes.contains(c)) batchProd *= sz;
                else if (keptAxes.contains(c)) keptProd *= sz;
                else if (contractAxes.contains(c)) contractProd *= sz;
            }
            // Return [batch, contract, kept] — the order bmm expects: [B, K, N]
            return new int[]{batchProd, contractProd, keptProd};
        }

        /**
         * Output shape after bmm: [batch_dims..., kept_a_dims..., kept_b_dims...].
         */
        public int[] outputShape(int[] shapeA, int[] shapeB) {
            List<Integer> outShape = new ArrayList<>();
            for (char c : outputLabels.toCharArray()) {
                Integer sz = axisSizes.get(c);
                if (sz != null) {
                    outShape.add(sz);
                } else {
                    int sza = -1, szb = -1;
                    int ia = inputLabels[0].indexOf(c);
                    int ib = inputLabels[1].indexOf(c);
                    if (ia >= 0) sza = shapeA[ia];
                    if (ib >= 0) szb = shapeB[ib];
                    if (sza > 0) outShape.add(sza);
                    else if (szb > 0) outShape.add(szb);
                }
            }
            return outShape.stream().mapToInt(Integer::intValue).toArray();
        }
    }
}
