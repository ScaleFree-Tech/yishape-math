package com.reremouse.lab.math.stats.bayes.models.kernels;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.bayes.models.GaussianProcess;

/**
 * Matérn核函数
 * Matérn Kernel Function
 * 
 * <p>Matérn核是一类灵活的核函数，通过参数ν控制函数的平滑性。
 * 当ν→∞时，Matérn核收敛到RBF核。</p>
 * <p>Matérn kernel is a flexible class of kernel functions that controls 
 * function smoothness through parameter ν. As ν→∞, Matérn kernel converges to RBF kernel.</p>
 * 
 * <p>核函数形式：k(x, x') = σ² * (2^(1-ν) / Γ(ν)) * (√(2ν) * r / l)^ν * K_ν(√(2ν) * r / l)</p>
 * <p>其中 r = ||x - x'||，K_ν是修正贝塞尔函数</p>
 * <p>超参数：[σ² (signal variance), l (length scale), ν (smoothness)]</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MaternKernel implements GaussianProcess.KernelFunction {
    
    @Override
    public double evaluate(IVector x1, IVector x2, IVector hyperparameters) {
        if (hyperparameters.size() != 3) {
            throw new IllegalArgumentException("Matérn kernel requires 3 hyperparameters: [signal_variance, length_scale, nu]");
        }
        
        double signalVariance = hyperparameters.get(0).doubleValue();
        double lengthScale = hyperparameters.get(1).doubleValue();
        double nu = hyperparameters.get(2).doubleValue();
        
        if (signalVariance <= 0 || lengthScale <= 0 || nu <= 0) {
            throw new IllegalArgumentException("Hyperparameters must be positive");
        }
        
        // 计算欧几里得距离
        double distance = 0.0;
        for (int i = 0; i < x1.size(); i++) {
            double diff = x1.get(i).doubleValue() - x2.get(i).doubleValue();
            distance += diff * diff;
        }
        distance = Math.sqrt(distance);
        
        // 如果距离为0，返回信号方差
        if (distance < 1e-12) {
            return signalVariance;
        }
        
        // 计算标准化距离
        double scaledDistance = Math.sqrt(2 * nu) * distance / lengthScale;
        
        // 计算Matérn核值
        return signalVariance * maternFunction(scaledDistance, nu);
    }
    
    @Override
    public IVector gradient(IVector x1, IVector x2, IVector hyperparameters) {
        if (hyperparameters.size() != 3) {
            throw new IllegalArgumentException("Matérn kernel requires 3 hyperparameters: [signal_variance, length_scale, nu]");
        }
        
        double signalVariance = hyperparameters.get(0).doubleValue();
        double lengthScale = hyperparameters.get(1).doubleValue();
        double nu = hyperparameters.get(2).doubleValue();
        
        // 计算欧几里得距离
        double distance = 0.0;
        for (int i = 0; i < x1.size(); i++) {
            double diff = x1.get(i).doubleValue() - x2.get(i).doubleValue();
            distance += diff * diff;
        }
        distance = Math.sqrt(distance);
        
        IVector gradient = Linalg.vector(3);
        
        if (distance < 1e-12) {
            // 距离为0时的梯度
            gradient.set(0, 1.0);  // 关于信号方差
            gradient.set(1, 0.0);  // 关于长度尺度
            gradient.set(2, 0.0);  // 关于nu
            return gradient;
        }
        
        double scaledDistance = Math.sqrt(2 * nu) * distance / lengthScale;
        double kernelValue = signalVariance * maternFunction(scaledDistance, nu);
        
        // 关于信号方差的梯度
        gradient.set(0, kernelValue / signalVariance);
        
        // 关于长度尺度的梯度
        double dMaternDScaled = maternFunctionDerivative(scaledDistance, nu);
        double dScaledDLength = -scaledDistance / lengthScale;
        gradient.set(1, signalVariance * dMaternDScaled * dScaledDLength);
        
        // 关于nu的梯度（简化计算）
        gradient.set(2, 0.0);  // 复杂的解析梯度，这里简化为0
        
        return gradient;
    }
    
    @Override
    public int getNumHyperparameters() {
        return 3;
    }
    
    @Override
    public String getName() {
        return "Matérn";
    }
    
    /**
     * 计算Matérn函数值
     * Calculate Matérn function value
     */
    private double maternFunction(double r, double nu) {
        if (r < 1e-12) {
            return 1.0;
        }
        
        // 对于常见的nu值，使用简化公式
        if (Math.abs(nu - 0.5) < 1e-6) {
            // nu = 1/2: exponential kernel
            return Math.exp(-r);
        } else if (Math.abs(nu - 1.5) < 1e-6) {
            // nu = 3/2
            return (1.0 + r) * Math.exp(-r);
        } else if (Math.abs(nu - 2.5) < 1e-6) {
            // nu = 5/2
            return (1.0 + r + r * r / 3.0) * Math.exp(-r);
        } else {
            // 一般情况：使用近似公式
            double factor = Math.pow(2.0, 1.0 - nu) / gamma(nu);
            double bessel = modifiedBesselSecondKind(nu, r);
            return factor * Math.pow(r, nu) * bessel;
        }
    }
    
    /**
     * 计算Matérn函数的导数
     * Calculate derivative of Matérn function
     */
    private double maternFunctionDerivative(double r, double nu) {
        if (r < 1e-12) {
            return 0.0;
        }
        
        // 对于常见的nu值，使用简化公式的导数
        if (Math.abs(nu - 0.5) < 1e-6) {
            // nu = 1/2
            return -Math.exp(-r);
        } else if (Math.abs(nu - 1.5) < 1e-6) {
            // nu = 3/2
            return -r * Math.exp(-r);
        } else if (Math.abs(nu - 2.5) < 1e-6) {
            // nu = 5/2
            return -(r + r * r / 3.0) * Math.exp(-r);
        } else {
            // 一般情况：数值导数
            double h = 1e-8;
            return (maternFunction(r + h, nu) - maternFunction(r - h, nu)) / (2 * h);
        }
    }
    
    /**
     * 伽马函数近似
     * Gamma function approximation
     */
    private double gamma(double x) {
        // Stirling近似
        if (x > 12) {
            return Math.sqrt(2 * Math.PI / x) * Math.pow(x / Math.E, x);
        }
        
        // 对于小值，使用查表法
        if (Math.abs(x - 0.5) < 1e-6) {
            return Math.sqrt(Math.PI);
        } else if (Math.abs(x - 1.0) < 1e-6) {
            return 1.0;
        } else if (Math.abs(x - 1.5) < 1e-6) {
            return 0.5 * Math.sqrt(Math.PI);
        } else if (Math.abs(x - 2.0) < 1e-6) {
            return 1.0;
        } else if (Math.abs(x - 2.5) < 1e-6) {
            return 1.5 * 0.5 * Math.sqrt(Math.PI);
        } else {
            // 递归关系：Γ(x+1) = x * Γ(x)
            if (x > 1) {
                return (x - 1) * gamma(x - 1);
            } else {
                return gamma(x + 1) / x;
            }
        }
    }
    
    /**
     * 修正贝塞尔函数第二类近似
     * Modified Bessel function of the second kind approximation
     */
    private double modifiedBesselSecondKind(double nu, double x) {
        if (x < 1e-12) {
            return Double.POSITIVE_INFINITY;
        }
        
        // 对于大的x值，使用渐近展开
        if (x > 10) {
            return Math.sqrt(Math.PI / (2 * x)) * Math.exp(-x);
        }
        
        // 对于小的x值，使用级数展开的前几项
        if (Math.abs(nu - 0.5) < 1e-6) {
            return Math.sqrt(Math.PI / (2 * x)) * Math.exp(-x);
        } else if (Math.abs(nu - 1.5) < 1e-6) {
            return Math.sqrt(Math.PI / (2 * x)) * Math.exp(-x) * (1 + 1 / x);
        } else {
            // 简化近似
            return Math.sqrt(Math.PI / (2 * x)) * Math.exp(-x);
        }
    }
}