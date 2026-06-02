package model_zoo.yishape_linalg;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.Plots;

/**
 * YiShape-Math 向量与矩阵运算示例：创建、算术、乘法、范数、点积与 SVD 分解。
 */
public class YishapeLinalgMatrixExample {

    public static void main(String[] args) {
        System.out.println("======== YiShape-Math：向量与矩阵 =========");

        var v1 = Linalg.vector(new double[] {1, 2, 3, 4});
        var v2 = Linalg.vector(new double[] {0.5, 1.5, 2.5, 3.5});
        System.out.println("v1 = " + v1);
        System.out.println("v2 = " + v2);
        System.out.println("v1 + v2 = " + v1.add(v2));
        System.out.println("v1 · v2 (dot) = " + v1.dot(v2));
        System.out.println("||v1||_2 = " + v1.norm2());
        System.out.println("mean(v1) = " + v1.mean() + ", std(v1, ddof=1) = " + v1.std(1));

        var a = Linalg.matrix(new double[][] {
            {1, 2, 3},
            {4, 5, 6}
        });
        var b = Linalg.matrix(new double[][] {
            {1, 0},
            {0, 1},
            {1, 1}
        });
        System.out.println("A (2x3) = \n" + a);
        System.out.println("B (3x2) = \n" + b);
        var ab = a.mmul(b);
        System.out.println("A @ B (2x2) = \n" + ab);

        var eye = Linalg.eye(3);
        System.out.println("I_3 + rand(3,3)*0.01 ≈ \n" + eye.add(Linalg.rand(3, 3).multiplyScalar(0.01)));

        var small = Linalg.matrix(new double[][] {
            {3, 2, 11},
            {2, 6, 21},
            {11, 21, 8},
        });
        var eig = small.eigen();
        System.out.println("Eigen 特征分解： " +eig);
        Plots.bar(eig._1).title("特征值分布").show();
        var usv = small.svd();
        System.out.println("SVD 奇异值 σ = " + usv._2);
        Plots.bar(usv._2).title("奇异值分布").show();
        var det = small.det();
        System.out.println("行列式 = " + det);
        System.out.println("======== 示例结束（向量与矩阵） =========");
    }
}
