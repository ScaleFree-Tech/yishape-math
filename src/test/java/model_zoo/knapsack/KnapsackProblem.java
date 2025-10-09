package model_zoo.knapsack;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;

/**
 * 🎒 探险家的背包问题：一场智慧与选择的博弈
 * 
 * 想象你是一位即将踏上冒险之旅的探险家，面前摆放着各种珍贵的物品。
 * 你的背包容量有限，需要做出明智的选择，让背包里的物品总价值最大化。
 * 
 * 这个例子展示了如何使用整数规划求解经典的0-1背包问题。
 * 在0-1背包问题中，每个物品要么完整地放入背包(1)，要么不放入(0)。
 * 
 * 📖 相关文档：请参阅 knapsack_introduction.md 了解详细的理论介绍
 */
public class KnapsackProblem {
    public static void main(String[] args) {
        System.out.println("🎒=== 探险家的背包问题 ===🎒");
        System.out.println("一场智慧与选择的博弈即将开始...");
        System.out.println();
        
        // 🎯 探险家的珍贵物品清单
        // 每件物品都有其独特的价值和重量
        String[] itemNames = {
            "珠宝💎", 
            "古籍📚", 
            "相机📷", 
            "手表⌚", 
            "笔记本💻",
            "帐篷⛺",
            "食物🍎"
        };
        
        // 💰 每件物品的价值（探险家的评估）
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        
        // ⚖️ 每件物品的重量（公斤）
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        
        // 🎒 背包的最大承重能力
        double capacity = 100.0;
        
        // 📋 展示探险家面临的选择
        System.out.println("🎯 探险家的物品清单：");
        System.out.println("物品名称\t\t\t价值\t重量");
        System.out.println("========================================");
        for (int i = 0; i < itemNames.length; i++) {
            System.out.printf("%-20s\t%.1f\t%.1f\n", itemNames[i], values[i], weights[i]);
        }
        System.out.println("========================================");
        System.out.println("🎒 背包最大承重: " + capacity + " 公斤");
        System.out.println();
        
        // 🧮 将探险家的选择转化为数学问题
        System.out.println("📐 数学建模（将现实问题转化为数学语言）：");
        System.out.println();
        System.out.println("🎯 目标函数（最大化总价值）：");
        System.out.println("  最大化: 60*x1 + 100*x2 + 120*x3 + 80*x4 + 150*x5 + 200*x6 + 50*x7");
        System.out.println("  其中 xi = 1 表示选择第i个物品，xi = 0 表示不选择");
        System.out.println();
        System.out.println("⚖️ 约束条件：");
        System.out.println("  重量约束（不能超过背包容量）：");
        System.out.println("    10*x1 + 20*x2 + 30*x3 + 40*x4 + 50*x5 + 60*x6 + 10*x7 ≤ 100");
        System.out.println();
        System.out.println("  0-1变量约束（每个物品要么选要么不选）：");
        System.out.println("    x1, x2, x3, x4, x5, x6, x7 ∈ {0, 1}");
        System.out.println();
        
        // 🔄 转换为求解器可处理的形式（求解器执行最小化）
        // 最小化: -sum(values[i] * x[i]) 等价于最大化 sum(values[i] * x[i])
        var c = Linalg.vector(values).multiplyScalar(-1.0);
        
        // 📊 约束矩阵（重量约束）
        // 创建1×7的矩阵，表示一个约束条件：weights[0]*x[0] + weights[1]*x[1] + ... + weights[6]*x[6] <= capacity
        var A_ub = Linalg.matrix(new double[][]{weights});
        
        // 📏 约束向量（背包容量限制）
        var b_ub = Linalg.vector(new double[]{capacity});
//        ILinProgSolver base = new ComMath4LinProgSolver();
        ILinProgSolver base = new RereSimplexLinProgSolver();
//        ILinProgSolver base = new NaiveSimplexLinProgSolver();
        // 🤖 创建整数规划求解器
        RereIntegerProg solver = new RereIntegerProg(base);
        
        // 🔢 设置所有变量为二进制变量（0-1变量）
        solver.setAllVariablesBinary();
        
        // ⚙️ 设置求解器参数以确保找到真正的整数解
        solver.setMaxDepth(100);           // 增加搜索深度
        solver.setGapTolerance(1e-10);     // 设置更严格的间隙容忍度
        solver.setTolerance(1e-10);        // 设置更严格的容忍度
        solver.setMaxIterations(10000);    // 增加最大迭代次数
        solver.setVerbose(false);          // 关闭详细输出以减少干扰
        
        
        System.out.println("🔍 正在求解0-1整数规划问题...");
        System.out.println("💭 探险家正在思考最优的选择策略...");
        System.out.println();
        
        // 🚀 求解0-1整数规划问题
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        // ✅ 检查是否找到可行解
        if (result == null) {
            System.out.println("❌ 没有找到可行解！探险家陷入了困境...");
            return;
        }
        
        // 📊 提取最优解
        IVector solution = result.getOptimalPoint();
        double rawOptimalValue = result.getOptimalValue(); // 求解器返回的原始值
        double optimalValue = -rawOptimalValue; // 转换回最大化问题的结果
        
        // 🔍 调试信息
        System.out.println("🔍 调试信息:");
        System.out.println("   求解器返回的原始目标函数值: " + rawOptimalValue);
        System.out.println("   转换后的最大化目标函数值: " + optimalValue);
        
        // 🎉 输出最优解
        System.out.println("🏆=== 探险家的最优选择 ===🏆");
        System.out.println("📋 决策向量: " + solution);
        System.out.println("💰 最大总价值: " + optimalValue);
        System.out.println();
        
        // 🔍 详细的解决方案分析
        System.out.println("📈=== 选择分析 ===📈");
        double totalWeight = 0;
        double totalValue = 0;
        
        System.out.println("🎒 探险家最终选择的物品：");
        System.out.println("物品名称\t\t\t选择\t价值\t重量");
        System.out.println("================================================");
        for (int i = 0; i < solution.size(); i++) {
            // 四舍五入处理数值精度问题
            int selected = (int) Math.round(solution.get(i).doubleValue());
            if (selected == 1) {
                System.out.printf("%-20s\t%s\t%.1f\t%.1f\n", itemNames[i], "✅", values[i], weights[i]);
                totalWeight += weights[i];
                totalValue += values[i];
            } else {
                System.out.printf("%-20s\t%s\t%.1f\t%.1f\n", itemNames[i], "❌", values[i], weights[i]);
            }
        }
        System.out.println("================================================");
        System.out.println("📦 总重量: " + totalWeight + " ≤ " + capacity + " 公斤");
        System.out.println("💎 总价值: " + totalValue);
        System.out.println();
        
        // 🔍 验证0-1约束
        System.out.println("🔍=== 0-1约束验证 ===🔍");
        boolean allBinary = true;
        for (int i = 0; i < solution.size(); i++) {
            double value = solution.get(i).doubleValue();
            // 检查数值是否为0或1（考虑数值误差）
            boolean isBinary = Math.abs(value) < 1e-6 || Math.abs(value - 1.0) < 1e-6;
            allBinary &= isBinary;
            System.out.printf("x%d = %.6f (是否为0-1: %s)\n", i+1, value, isBinary ? "✅是" : "❌否");
        }
        System.out.println("所有变量都是0-1: " + (allBinary ? "✅是" : "❌否"));
        System.out.println();
        
        // 📝 智慧总结
        System.out.println("🎓=== 探险家的智慧总结 ===🎓");
        System.out.println("这是一个经典的0-1整数规划问题（0-1背包问题）。");
        System.out.println();
        System.out.println("🔑 关键特征：");
        System.out.println("1. 🔢 每个变量只能是0或1（要么选择，要么不选择）");
        System.out.println("2. 🎯 目标是最大化总价值");
        System.out.println("3. ⚖️ 受到重量约束的限制");
        System.out.println("4. 🌳 使用分支定界法求解");
        System.out.println();
        System.out.println("💡 探险家学到的道理：");
        System.out.println("   在有限的资源下，智慧的选择比盲目的贪婪更有价值！");
        System.out.println();
        System.out.println("📖 想了解更多？请查看 knapsack_introduction.md 文档！");
    }
}