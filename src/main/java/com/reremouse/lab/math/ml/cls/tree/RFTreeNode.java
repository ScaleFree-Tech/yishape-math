package com.reremouse.lab.math.ml.cls.tree;

import com.reremouse.lab.math.linalg.IVector;
import java.util.Map;

/**
 * 分类决策树节点类
 * <p>
 * 表示分类决策树中的一个节点，可以是内部节点（包含分裂条件）或叶子节点（包含预测标签）。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RFTreeNode {
    
    /** 特征索引（内部节点） */
    private int featureIndex;
    
    /** 分裂阈值（内部节点） */
    private double threshold;
    
    /** 左子节点 */
    private RFTreeNode leftChild;
    
    /** 右子节点 */
    private RFTreeNode rightChild;
    
    /** 预测标签（叶子节点） */
    private String leafLabel;
    
    /** 是否为叶子节点 */
    private boolean isLeaf;
    
    /** 节点深度 */
    private int depth;
    
    /** 样本数量 */
    private int sampleCount;
    
    /** 分裂增益 */
    private double gain;
    
    /** 类别分布 */
    private Map<Integer, Integer> classCounts;
    
    /**
     * 叶子节点构造函数
     * @param leafLabel 叶子节点的预测标签
     * @param depth 节点深度
     * @param sampleCount 样本数量
     * @param classCounts 类别分布
     */
    public RFTreeNode(String leafLabel, int depth, int sampleCount, 
                                Map<Integer, Integer> classCounts) {
        this.leafLabel = leafLabel;
        this.isLeaf = true;
        this.depth = depth;
        this.sampleCount = sampleCount;
        this.classCounts = classCounts;
        this.gain = 0.0;
    }
    
    /**
     * 内部节点构造函数
     * @param featureIndex 分裂特征索引
     * @param threshold 分裂阈值
     * @param leftChild 左子节点
     * @param rightChild 右子节点
     * @param depth 节点深度
     * @param sampleCount 样本数量
     * @param gain 分裂增益
     * @param classCounts 类别分布
     */
    public RFTreeNode(int featureIndex, double threshold, 
                                RFTreeNode leftChild, RFTreeNode rightChild,
                                int depth, int sampleCount, double gain, 
                                Map<Integer, Integer> classCounts) {
        this.featureIndex = featureIndex;
        this.threshold = threshold;
        this.leftChild = leftChild;
        this.rightChild = rightChild;
        this.isLeaf = false;
        this.depth = depth;
        this.sampleCount = sampleCount;
        this.gain = gain;
        this.classCounts = classCounts;
    }
    
    /**
     * 预测单个样本
     * @param features 特征向量
     * @return 预测标签
     */
    public String predict(IVector features) {
        if (isLeaf) {
            return leafLabel;
        }
        
        // 根据分裂条件选择子节点
        if (features.get(featureIndex).doubleValue() <= threshold) {
            return leftChild.predict(features);
        } else {
            return rightChild.predict(features);
        }
    }
    
    /**
     * 获取树的深度
     * @return 树的深度
     */
    public int getTreeDepth() {
        if (isLeaf) {
            return depth;
        }
        
        int leftDepth = leftChild != null ? leftChild.getTreeDepth() : depth;
        int rightDepth = rightChild != null ? rightChild.getTreeDepth() : depth;
        
        return Math.max(leftDepth, rightDepth);
    }
    
    /**
     * 获取叶子节点数量
     * @return 叶子节点数量
     */
    public int getLeafCount() {
        if (isLeaf) {
            return 1;
        }
        
        int leftLeaves = leftChild != null ? leftChild.getLeafCount() : 0;
        int rightLeaves = rightChild != null ? rightChild.getLeafCount() : 0;
        
        return leftLeaves + rightLeaves;
    }
    
    /**
     * 获取节点信息字符串
     * @return 节点信息
     */
    public String getNodeInfo() {
        if (isLeaf) {
            return String.format("Leaf[label=%s, samples=%d, depth=%d]", 
                    leafLabel, sampleCount, depth);
        } else {
            return String.format("Node[feature=%d, threshold=%.3f, samples=%d, depth=%d, gain=%.3f]",
                    featureIndex, threshold, sampleCount, depth, gain);
        }
    }
    
    /**
     * 打印树结构（递归）
     * @param prefix 前缀字符串
     * @return 树结构字符串
     */
    public String printTree(String prefix) {
        StringBuilder sb = new StringBuilder();
        sb.append(prefix).append(getNodeInfo()).append("\n");
        
        if (!isLeaf) {
            if (leftChild != null) {
                sb.append(leftChild.printTree(prefix + "  L: "));
            }
            if (rightChild != null) {
                sb.append(rightChild.printTree(prefix + "  R: "));
            }
        }
        
        return sb.toString();
    }
    
    // ==================== Getters and Setters ====================
    
    public int getFeatureIndex() {
        return featureIndex;
    }
    
    public double getThreshold() {
        return threshold;
    }
    
    public RFTreeNode getLeftChild() {
        return leftChild;
    }
    
    public RFTreeNode getRightChild() {
        return rightChild;
    }
    
    public String getLeafLabel() {
        return leafLabel;
    }
    
    public boolean isLeaf() {
        return isLeaf;
    }
    
    public int getDepth() {
        return depth;
    }
    
    public int getSampleCount() {
        return sampleCount;
    }
    
    public double getGain() {
        return gain;
    }
    
    public Map<Integer, Integer> getClassCounts() {
        return classCounts;
    }
    
    public void setFeatureIndex(int featureIndex) {
        this.featureIndex = featureIndex;
    }
    
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
    
    public void setLeftChild(RFTreeNode leftChild) {
        this.leftChild = leftChild;
    }
    
    public void setRightChild(RFTreeNode rightChild) {
        this.rightChild = rightChild;
    }
    
    public void setLeafLabel(String leafLabel) {
        this.leafLabel = leafLabel;
    }
    
    public void setLeaf(boolean leaf) {
        isLeaf = leaf;
    }
    
    public void setDepth(int depth) {
        this.depth = depth;
    }
    
    public void setSampleCount(int sampleCount) {
        this.sampleCount = sampleCount;
    }
    
    public void setGain(double gain) {
        this.gain = gain;
    }
    
    public void setClassCounts(Map<Integer, Integer> classCounts) {
        this.classCounts = classCounts;
    }
}