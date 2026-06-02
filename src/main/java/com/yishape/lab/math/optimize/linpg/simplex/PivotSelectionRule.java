package com.yishape.lab.math.optimize.linpg.simplex;

/**
 * 枢轴选择规则枚举，用于控制单纯形法中选择入基变量的策略
 * PivotSelectionRule实现
 * 
 * @author lteb2
 */
public enum PivotSelectionRule {
    
    /**
     * Dantzig规则：选择目标函数行中系数最负的变量作为入基变量
     * 这是标准的单纯形法规则，通常收敛最快
     */
    DANTZIG,
    
    /**
     * Bland规则：选择目标函数行中第一个负系数的变量作为入基变量
     * 这可以防止单纯形法陷入循环，保证收敛
     */
    BLAND,
    
    /**
     * Steep-Edge规则：基于梯度信息选择入基变量
     * 考虑目标函数改进的方向和幅度，通常比Dantzig规则更高效
     * 特别适用于大规模线性规划问题
     */
    STEEP_EDGE
}