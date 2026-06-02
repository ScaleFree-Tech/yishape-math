package com.yishape.lab.math.optimize.mclp;

/**
 * 多目标线性规划求解器类型枚举
 * Multi-Criteria Linear Programming Solver Type Enum
 *
 * @author lteb2
 */
public enum MclpSolverType {

    /**
     * 加权求和法 / Weighted Sum Method
     * <p>将多个目标函数通过权重进行线性组合，转化为单目标优化问题。
     * Works well for convex Pareto fronts but cannot find non-convex solutions.</p>
     */
    WeightedSum("加权求和法", "Weighted Sum Method"),

    /**
     * 字典序法 / Lexicographic Method
     * <p>按优先级逐个优化目标函数，前一个目标达到最优后才考虑下一个。
     * Suitable when objectives have clear priority order.</p>
     */
    Lexicographic("字典序法", "Lexicographic Method"),

    /**
     * 目标规划法 / Goal Programming
     * <p>设定各目标的目标值，最小化与目标值的偏差。
     * Minimizes deviations from specified goal values.</p>
     */
    GoalProgramming("目标规划法", "Goal Programming"),

    /**
     * Pareto最优解法 / Pareto Optimal Method
     * <p>生成完整的Pareto前沿，帮助决策者了解各目标间的权衡关系。
     * Generates complete Pareto frontier for decision maker analysis.</p>
     */
    Pareto("Pareto最优解法", "Pareto Optimal Method"),

    /**
     * 层次分析法 / Analytic Hierarchy Process
     * <p>基于成对比较矩阵计算目标权重，适用于复杂决策问题。
     * Calculates weights based on pairwise comparison matrix.</p>
     */
    Ahp("层次分析法", "Analytic Hierarchy Process"),

    /**
     * TOPSIS方法 / Technique for Order Preference by Similarity to Ideal Solution
     * <p>基于理想解和负理想解的距离进行排序。
     * Ranks alternatives based on distance to ideal and anti-ideal solutions.</p>
     */
    Topsis("TOPSIS法", "TOPSIS Method"),

    /**
     * 交互式STEM方法 / STEM Interactive Method
     * <p>决策者逐步交互，通过反馈调整偏好找到满意解。
     * Interactive method where decision maker provides preferences iteratively.</p>
     */
    Interactive("交互式STEM法", "STEM Interactive Method");

    private final String chineseName;
    private final String englishName;

    MclpSolverType(String chineseName, String englishName) {
        this.chineseName = chineseName;
        this.englishName = englishName;
    }

    /**
     * 获取中文名称
     * @return 中文名称
     */
    public String getChineseName() {
        return chineseName;
    }

    /**
     * 获取英文名称
     * @return 英文名称
     */
    public String getEnglishName() {
        return englishName;
    }

    /**
     * 获取求解器描述
     * @return 描述字符串
     */
    public String getDescription() {
        return chineseName + " / " + englishName;
    }

    @Override
    public String toString() {
        return name() + ": " + getDescription();
    }
}
