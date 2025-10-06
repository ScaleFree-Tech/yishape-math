package com.yishape.lab.math.optimize;

import com.yishape.lab.math.optimize.newton.RereOnlineAdam;
import com.yishape.lab.math.optimize.newton.RereOnlineSGD;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 优化器使用示例
 * Optimizer Usage Examples
 * 
 * 展示如何使用批量优化器(SGD, Adam)和在线优化器(OnlineSGD, OnlineAdam)
 * 
 * @author lteb2
 */
public class OptimizerExample {
    
    /**
     * 简单的二次函数目标函数: f(x) = (x-2)² + (y-3)²
     * 最优解应该是 x=2, y=3，最小值为0
     */
    static class QuadraticObjective implements IObjectiveFunction {
        @Override
        public double computeObjective(IVector x) {
            double x1 = x.get(0).doubleValue();
            double x2 = x.get(1).doubleValue();
            return Math.pow(x1 - 2.0, 2) + Math.pow(x2 - 3.0, 2);
        }
    }
    
    /**
     * 二次函数的梯度: ∇f(x) = [2(x-2), 2(y-3)]
     */
    static class QuadraticGradient implements IGradientFunction {
        @Override
        public IVector computeGradient(IVector x) {
            double x1 = x.get(0).doubleValue();
            double x2 = x.get(1).doubleValue();
            
            double[] grad = {
                2.0 * (x1 - 2.0),
                2.0 * (x2 - 3.0)
            };
            
            return Linalg.vector(grad);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("=== 优化器使用示例 ===\n");
        
        // 创建目标函数和梯度函数
        QuadraticObjective objFun = new QuadraticObjective();
        QuadraticGradient grdFun = new QuadraticGradient();
        
        // 初始点
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        
        System.out.println("目标函数: f(x,y) = (x-2)² + (y-3)²");
        System.out.println("理论最优解: x=2, y=3, f_min=0");
        System.out.println("初始点: " + initX);
        System.out.println("初始损失: " + objFun.computeObjective(initX));
        System.out.println();
        
        
        // 测试批量LBFGS
        testBatchLBFGS(initX, objFun, grdFun);
        
        // 测试在线SGD
        testOnlineSGD(initX, objFun, grdFun);
        
        // 测试在线Adam
        testOnlineAdam(initX, objFun, grdFun);
    }
    
    private static void testBatchLBFGS(IVector initX, QuadraticObjective objFun, QuadraticGradient grdFun) {
        System.out.println("=== 批量LBFGS优化 ===");
        
        RereLBFGS sgd = new RereLBFGS();
        
        OptResult result = sgd.optimize(initX, objFun, grdFun);
        
        System.out.println("最终损失: " + result.getOptimalValue());
        System.out.println("最优解: " + result.getOptimalPoint());
        System.out.println();
    }
    

    
    private static void testOnlineSGD(IVector initX, QuadraticObjective objFun, QuadraticGradient grdFun) {
        System.out.println("=== 在线SGD优化 ===");
        
        RereOnlineSGD onlineSgd = new RereOnlineSGD(0.1, 0.9)  // 学习率0.1，动量0.9
                .setVerbose(false);
        
        // 初始化优化器
        onlineSgd.initialize(initX);
        
        // 模拟在线学习过程
        IVector currentParams = onlineSgd.getCurrentParams();
        for (int i = 0; i < 100; i++) {
            IVector gradient = grdFun.computeGradient(currentParams);
            double loss = objFun.computeObjective(currentParams);
            
            currentParams = onlineSgd.step(gradient, loss);
            
            // 检查收敛
            if (loss < 1e-6) {
                System.out.println("在线SGD在第" + (i+1) + "步收敛");
                break;
            }
        }
        
        double finalLoss = objFun.computeObjective(currentParams);
        System.out.println("最终损失: " + finalLoss);
        System.out.println("最优解: " + currentParams);
        System.out.println("总步数: " + onlineSgd.getCurrentStep());
        System.out.println();
    }
    
    private static void testOnlineAdam(IVector initX, QuadraticObjective objFun, QuadraticGradient grdFun) {
        System.out.println("=== 在线Adam优化 ===");
        
        RereOnlineAdam onlineAdam = new RereOnlineAdam(0.1)  // 学习率0.1
                .setVerbose(false);
        
        // 初始化优化器
        onlineAdam.initialize(initX);
        
        // 模拟在线学习过程
        IVector currentParams = onlineAdam.getCurrentParams();
        for (int i = 0; i < 100; i++) {
            IVector gradient = grdFun.computeGradient(currentParams);
            double loss = objFun.computeObjective(currentParams);
            
            currentParams = onlineAdam.step(gradient, loss);
            
            // 检查收敛
            if (loss < 1e-6) {
                System.out.println("在线Adam在第" + (i+1) + "步收敛");
                break;
            }
        }
        
        double finalLoss = objFun.computeObjective(currentParams);
        System.out.println("最终损失: " + finalLoss);
        System.out.println("最优解: " + currentParams);
        System.out.println("总步数: " + onlineAdam.getCurrentStep());
        System.out.println();
    }
}