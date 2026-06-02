package com.yishape.lab.math.optimize.linpg;

/**
 * 整数规划接口，定义整数规划和混合整数规划求解器的通用方法
 * Integer Programming Interface that defines common methods for integer and mixed-integer programming solvers
 *
 * <p>该接口扩展了线性规划求解器接口，添加了整数变量和二进制变量的支持。
 * 实现该接口的类可以使用分支定界法、分支切割法等算法求解整数规划问题。
 * This interface extends the linear programming solver interface, adding support for integer and binary variables.
 * Classes implementing this interface can use branch-and-bound, branch-and-cut algorithms to solve integer programming problems.</p>
 *
 * <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * IIntegerProg solver = new RereIntegerProg();
 * solver.setBinaryVariables(0, 1, 2, 3, 4); // 设置5个0-1变量
 * OptResult result = solver.solve(c, A_ub, b_ub, A_eq, b_eq, initX);
 * </pre>
 *
 * @author lteb2
 * @see ILinProgSolver
 * @see RereIntegerProg
 */
public interface IIntegerProg extends ILinProgSolver {

    /**
     * 设置指定索引的变量为整数变量
     * Set a variable at the specified index as an integer variable
     *
     * @param variableIndex 变量索引，从0开始计数 / Variable index, 0-based
     * @throws IllegalArgumentException 如果变量索引为负数 / If variable index is negative
     */
    public void setIntegerVariable(int variableIndex);

    /**
     * 批量添加整数变量
     * Add multiple integer variables at once
     *
     * @param variableIndices 变量索引数组，可以包含多个索引 / Array of variable indices, can contain multiple indices
     * @throws IllegalArgumentException 如果任何变量索引为负数 / If any variable index is negative
     */
    public void addIntegerVariables(int... variableIndices);


    /**
     * 设置所有变量为整数变量，优化时能够自动识别所有整数变量
     * Set all variables as integer variables for automatic identification during optimization
     *
     * <p>调用此方法后，优化器将把所有变量视为整数变量。
     * After calling this method, the optimizer will treat all variables as integer variables.</p>
     */
    public void setAllVariablesInteger();

    /**
     * 设置0-1变量（二进制变量），其取值只能为0或1
     * Set a variable as a binary (0-1) variable with value restricted to 0 or 1
     *
     * @param variableIndex 变量索引，从0开始计数 / Variable index, 0-based
     * @throws IllegalArgumentException 如果变量索引为负数 / If variable index is negative
     */
    public void setBinaryVariable(int variableIndex);

    /**
     * 批量添加0-1变量（二进制变量）
     * Add multiple binary (0-1) variables at once
     *
     * @param variableIndices 变量索引数组 / Array of variable indices
     * @throws IllegalArgumentException 如果任何变量索引为负数 / If any variable index is negative
     */
    public void addBinaryVariables(int... variableIndices);


    /**
     * 设置所有变量为0-1变量（二进制变量），优化时自动识别所有0-1变量
     * Set all variables as binary (0-1) variables for automatic identification during optimization
     *
     * <p>调用此方法后，优化器将把所有变量视为0-1变量。
     * After calling this method, the optimizer will treat all variables as binary variables.</p>
     */
    public void setAllVariablesBinary();

    /**
     * 设置详细输出模式，用于调试和监控优化过程
     * Set verbose mode for debugging and monitoring the optimization process
     *
     * @param verbose 是否输出详细信息 / Whether to output detailed information
     */
    public void setVerbose(boolean verbose);

    /**
     * 检查当前是否处于详细输出模式
     * Check if verbose mode is currently enabled
     *
     * @return 如果启用详细输出返回true，否则返回false / Returns true if verbose output is enabled, false otherwise
     */
    public boolean isVerbose();

}