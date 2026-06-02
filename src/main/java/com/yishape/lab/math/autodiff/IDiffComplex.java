package com.yishape.lab.math.autodiff;

import java.io.Serializable;

import com.yishape.lab.math.core.Complex;
import com.yishape.lab.math.linalg.complex.IComplexMatrix.IComplexVector;

/**
 * Differentiable complex vector using Wirtinger calculus.
 * 基于 Wirtinger 微积分的可微复向量。
 *
 * <p>Gradients are stored with respect to the complex variable (holomorphic ops use conjugate rules).
 * 梯度相对于复变量存储（全纯运算遵循共轭 Wirtinger 规则）。</p>
 */
public interface IDiffComplex extends Serializable {

    long serialVersionUID = 1L;

    IComplexVector getValue();

    IComplexVector getGradient();

    boolean isLeaf();

    void backward();

    void backward(IComplexVector initialGradient);

    void zeroGradient();

    IDiffComplex add(IDiffComplex other);

    IDiffComplex sub(IDiffComplex other);

    IDiffComplex mul(IDiffComplex other);

    IDiffComplex div(IDiffComplex other);

    IDiffComplex scale(Complex scalar);

    IDiffComplex conjugate();

    IDiffComplex exp();

    IDiffComplex log();

    IDiffComplex sin();

    IDiffComplex cos();

    IDiffComplex tan();

    IDiffComplex tanh();

    IDiffComplex sigmoid();

    IDiffComplex relu();

    IDiffComplex abs();

    IDiffComplex sqrt();

    IDiffComplex square();

    IDiffComplex neg();

    IDiffComplex pow(double n);

    IDiffComplex sum();

    IDiffComplex innerProduct(IDiffComplex other);

    IDiffComplex grad();
}
