package com.yishape.lab.audio.embedding;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioIO;
import com.yishape.lab.audio.core.AudioUtil;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.util.List;

/**
 *
 * @author lteb2
 */
public class RealIVectorTest {

    public static void main(String args[]) {

        String path = "F:\\music\\test\\";
        String f1 = path + "20.杨友友-野花做了场玫瑰花的梦.mp3";

        try {
            AudioData ad = AudioIO.readAudio(f1);

            IMatrix<Double> mfcc = AudioUtil.calculateMFCCMatrix(ad);

            // 创建i-vector模型
            int ivectorDim = 100;      // i-vector维度
            int numComponents = 64;    // UBM高斯分量数（减少以加快示例运行）
            int featureDim = 13;       // MFCC特征维度

            IVectorEmbedding model = new IVectorEmbedding(ivectorDim, numComponents, featureDim);

            long startTime = System.currentTimeMillis();
            model.train(List.of(mfcc));
            long endTime = System.currentTimeMillis();

            System.out.println("模型训练完成!");
            System.out.println("训练时间: " + (endTime - startTime) + " ms");
            System.out.println("模型状态: " + (model.isTrained() ? "已训练" : "未训练"));
            System.out.println();

            // 测试i-vector提取
            System.out.println("=== 测试i-vector提取 ===");

            IVector<Double> ivector = model.embed(mfcc);

            System.out.println("提取的i-vector:");
            System.out.println("- 维度: " + ivector.length());
            System.out.println("- L2范数: " + String.format("%.6f", ivector.norm2()));
            System.out.println("- 前10个元素: " + formatVector(ivector, 10));
            System.out.println();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**
     * 格式化向量输出
     */
    private static String formatVector(IVector<Double> vector, int maxElements) {
        StringBuilder sb = new StringBuilder("[");
        int limit = Math.min(maxElements, vector.length());

        for (int i = 0; i < limit; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(String.format("%.4f", vector.get(i)));
        }

        if (vector.length() > maxElements) {
            sb.append(", ...");
        }

        sb.append("]");
        return sb.toString();
    }

}
