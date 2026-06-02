package com.yishape.lab.math.optimize.mcqp;

/**
 * 多目标二次规划求解器类型枚举
 * Multi-Criteria Quadratic Programming Solver Type Enum
 *
 * <p>定义支持的MCQP求解方法，包括：
 * - WeightedSum: 加权求和法
 * - Lexicographic: 字典序法
 * - GoalProgramming: 目标规划法
 * - Pareto: Pareto最优解法
 * - Ahp: 层次分析法
 * - Topsis: TOPSIS法
 * - Interactive: 交互式STEM法</p>
 *
 * @author lteb2
 */
public enum McqpSolverType {
    /** 加权求和法 / Weighted Sum Method */
    WeightedSum("加权求和法", "Weighted Sum Method"),

    /** 字典序法 / Lexicographic Method */
    Lexicographic("字典序法", "Lexicographic Method"),

    /** 目标规划法 / Goal Programming Method */
    GoalProgramming("目标规划法", "Goal Programming Method"),

    /** Pareto最优解法 / Pareto Optimal Method */
    Pareto("Pareto最优解法", "Pareto Optimal Method"),

    /** 层次分析法 / Analytic Hierarchy Process */
    Ahp("层次分析法(AHP)", "Analytic Hierarchy Process (AHP)"),

    /** TOPSIS方法 / TOPSIS Method */
    Topsis("TOPSIS法", "TOPSIS Method"),

    /** 交互式STEM方法 / Interactive STEM Method */
    Interactive("交互式STEM法", "Interactive STEM Method");

    private final String chineseName;
    private final String englishName;

    McqpSolverType(String chineseName, String englishName) {
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
     * 获取描述信息
     * @return 格式为 "中文名 / English Name"
     */
    public String getDescription() {
        return chineseName + " / " + englishName;
    }
}
