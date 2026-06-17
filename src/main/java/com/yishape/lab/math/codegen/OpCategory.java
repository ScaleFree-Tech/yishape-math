package com.yishape.lab.math.codegen;

/**
 * Categories for compute operations in the unified op schema.
 */
public enum OpCategory {
    /** Element-wise binary ops: add, sub, mul, div */
    BINARY,
    /** Element-wise binary with scalar: addScalar, mulScalar, etc. */
    BINARY_SCALAR,
    /** Element-wise unary ops: sqrt, exp, log, neg, etc. */
    UNARY,
    /** Activation functions: relu, sigmoid, tanh, gelu, etc. */
    ACTIVATION,
    /** Reduction ops: sum, mean, max, etc. */
    REDUCE,
    /** Linear algebra: matmul, mmul, dot, bmm, cross */
    LINALG,
    /** Data movement: transpose, reshape, flatten, cat, gather, slice, etc. */
    VIEW,
    /** Random / regularization: dropout */
    RANDOM,
    /** Normalization: layerNorm, batchNorm, rmsNorm, groupNorm, instanceNorm */
    NORMALIZATION,
    /** Deep learning: conv2d, convTranspose2d, maxpool2d, etc. */
    DL,
    /** Attention: mha, scaledDotProductAttention */
    ATTENTION,
    /** State-space models: selectiveScan, selectiveScan2, trapezoidalScan */
    SSM,
    /** Embedding lookup */
    EMBEDDING,
    /** Loss functions: bceLoss, focalLoss, diceLoss, softmaxCrossEntropy */
    LOSS,
    /** Graph structure: leaf, constant */
    GRAPH,
    /** Fused ops: {unary}{Reduce}, compound specials */
    FUSED,
    /** Other / miscellaneous */
    OTHER
}
