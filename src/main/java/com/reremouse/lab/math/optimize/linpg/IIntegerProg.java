package com.reremouse.lab.math.optimize.linpg;

/**
 * 整数规划接口
 *
 * @author lteb2
 */
public interface IIntegerProg extends ILinProgSolver {

    /**
     * 设置整数变量
     *
     * @param variableIndex 变量索引
     */
    public void setIntegerVariable(int variableIndex);

    /**
     * 添加整数变量
     *
     * @param variableIndices 变量索引数组
     */
    public void addIntegerVariables(int... variableIndices);


    /**
     * 设置所有变量为整数变量，优化时能够自动识别所有整数变量
     */
    public void setAllVariablesInteger();

    /**
     * 设置0-1变量（二进制变量）
     *
     * @param variableIndex 变量索引
     */
    public void setBinaryVariable(int variableIndex);

    /**
     * 添加0-1变量（二进制变量）
     *
     * @param variableIndices 变量索引数组
     */
    public void addBinaryVariables(int... variableIndices);


    /**
     * 设置所有变量为0-1变量（二进制变量），优化时自动能够识别所有0-1变量
     */
    public void setAllVariablesBinary();

}
