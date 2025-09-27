package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import org.junit.jupiter.api.Test;

public class ManualSimplexTest {
    
    @Test
    public void testManualSimplex() {
        System.out.println("=== 手动单纯形法验证 ===");
        
        // 问题: min 2*x1 + x2
        // 约束: x1 + x2 = 2.5
        // 期望最优解: x1=0, x2=2.5, 目标值=2.5
        
        System.out.println("原问题: min 2*x1 + x2");
        System.out.println("约束: x1 + x2 = 2.5");
        System.out.println("期望最优解: x1=0, x2=2.5, 目标值=2.5");
        System.out.println();
        
        // 第一阶段：添加人工变量s1
        // min s1
        // 约束: x1 + x2 + s1 = 2.5
        // 初始基解: x1=0, x2=0, s1=2.5
        
        System.out.println("第一阶段：添加人工变量s1");
        System.out.println("min s1");
        System.out.println("约束: x1 + x2 + s1 = 2.5");
        System.out.println("初始基解: x1=0, x2=0, s1=2.5 (目标值=2.5)");
        System.out.println();
        
        // 初始tableau
        System.out.println("初始tableau:");
        System.out.println("约束行: [1, 1, 1, 2.5]");
        System.out.println("目标行: [0, 0, 1, 0]");
        System.out.println("消除人工变量系数后:");
        System.out.println("目标行: [0, 0, 1, 0] - [1, 1, 1, 2.5] = [-1, -1, 0, -2.5]");
        System.out.println();
        
        // 第一次迭代
        System.out.println("第一次迭代:");
        System.out.println("入基变量选择: x1和x2的系数都是-1");
        System.out.println("根据原始目标函数系数选择: x2系数(1) < x1系数(2)，选择x2");
        System.out.println("出基变量选择: 比值测试 2.5/1 = 2.5，s1出基");
        System.out.println("pivot(0, 1): 以(0,1)位置的1.0为枢轴");
        System.out.println();
        
        // 手动计算pivot结果
        System.out.println("pivot计算:");
        System.out.println("枢轴行(行0): [1, 1, 1, 2.5] / 1.0 = [1, 1, 1, 2.5]");
        System.out.println("目标行(行1): [-1, -1, 0, -2.5] - (-1) * [1, 1, 1, 2.5]");
        System.out.println("            = [-1, -1, 0, -2.5] + [1, 1, 1, 2.5]");
        System.out.println("            = [0, 0, 1, 0]");
        System.out.println();
        
        System.out.println("第一阶段结束后的tableau:");
        System.out.println("约束行: [1, 1, 1, 2.5]");
        System.out.println("目标行: [0, 0, 1, 0]");
        System.out.println("基变量: x2=2.5-x1-s1, 当x1=0,s1=0时，x2=2.5");
        System.out.println("第一阶段目标值: 0 (人工变量s1=0)");
        System.out.println();
        
        // 第二阶段转换
        System.out.println("第二阶段转换:");
        System.out.println("移除人工变量列，设置原始目标函数");
        System.out.println("新tableau:");
        System.out.println("约束行: [1, 1, 2.5]");
        System.out.println("目标行: [2, 1, 0]");
        System.out.println();
        
        System.out.println("消除基变量x2在目标函数中的系数:");
        System.out.println("x2是基变量，需要消除目标行中x2的系数1");
        System.out.println("目标行: [2, 1, 0] - 1 * [1, 1, 2.5] = [1, 0, -2.5]");
        System.out.println();
        
        System.out.println("第二阶段最终tableau:");
        System.out.println("约束行: [1, 1, 2.5]");
        System.out.println("目标行: [1, 0, -2.5]");
        System.out.println();
        
        System.out.println("第二阶段分析:");
        System.out.println("目标行中x1的系数是1 > 0，已经最优");
        System.out.println("基变量: x2 = 2.5 - x1");
        System.out.println("当x1=0时，x2=2.5");
        System.out.println("目标值: 2*0 + 1*2.5 = 2.5");
        System.out.println();
        
        System.out.println("结论: 正确的最优解应该是 x1=0, x2=2.5, 目标值=2.5");
    }
}