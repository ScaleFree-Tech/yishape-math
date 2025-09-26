package com.reremouse.lab.math.ml.cls.tree;

/**
 * 决策树节点类
 * <p>
 * 表示决策树中的一个节点，可以是内部节点（包含分裂条件）或叶子节点（包含预测值）。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class XGTreeNode {
    
    /** 分裂特征索引 */
    private int featureIndex;
    
    /** 分裂阈值 */
    private double threshold;
    
    /** 左子节点 */
    private XGTreeNode leftChild;
    
    /** 右子节点 */
    private XGTreeNode rightChild;
    
    /** 叶子节点的预测值 */
    private double leafValue;
    
    /** 是否为叶子节点 */
    private boolean isLeaf;
    
    /** 节点深度 */
    private int depth;
    
    /** 节点包含的样本数量 */
    private int sampleCount;
    
    /** 节点的增益值 */
    private double gain;
    
    /**
     * 创建叶子节点
     * @param leafValue 叶子节点的预测值
     * @param depth 节点深度
     * @param sampleCount 样本数量
     */
    public XGTreeNode(double leafValue, int depth, int sampleCount) {
        this.leafValue = leafValue;
        this.isLeaf = true;
        this.depth = depth;
        this.sampleCount = sampleCount;
    }
    
    /**
     * 创建内部节点
     * @param featureIndex 分裂特征索引
     * @param threshold 分裂阈值
     * @param leftChild 左子节点
     * @param rightChild 右子节点
     * @param depth 节点深度
     * @param sampleCount 样本数量
     * @param gain 增益值
     */
    public XGTreeNode(int featureIndex, double threshold, XGTreeNode leftChild, 
                   XGTreeNode rightChild, int depth, int sampleCount, double gain) {
        this.featureIndex = featureIndex;
        this.threshold = threshold;
        this.leftChild = leftChild;
        this.rightChild = rightChild;
        this.isLeaf = false;
        this.depth = depth;
        this.sampleCount = sampleCount;
        this.gain = gain;
    }
    
    /**
     * 预测单个样本
     * @param features 特征值数组
     * @return 预测值
     */
    public double predict(double[] features) {
        if (isLeaf) {
            return leafValue;
        }
        
        if (features[featureIndex] <= threshold) {
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
     * 计算特征重要性
     * @param featureImportance 特征重要性数组
     */
    public void computeFeatureImportance(double[] featureImportance) {
        if (!isLeaf && featureIndex >= 0 && featureIndex < featureImportance.length) {
            featureImportance[featureIndex] += gain * sampleCount;
            
            if (leftChild != null) {
                leftChild.computeFeatureImportance(featureImportance);
            }
            if (rightChild != null) {
                rightChild.computeFeatureImportance(featureImportance);
            }
        }
    }
    
    // ==================== Getters and Setters ====================
    
    public int getFeatureIndex() {
        return featureIndex;
    }
    
    public void setFeatureIndex(int featureIndex) {
        this.featureIndex = featureIndex;
    }
    
    public double getThreshold() {
        return threshold;
    }
    
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
    
    public XGTreeNode getLeftChild() {
        return leftChild;
    }
    
    public void setLeftChild(XGTreeNode leftChild) {
        this.leftChild = leftChild;
    }
    
    public XGTreeNode getRightChild() {
        return rightChild;
    }
    
    public void setRightChild(XGTreeNode rightChild) {
        this.rightChild = rightChild;
    }
    
    public double getLeafValue() {
        return leafValue;
    }
    
    public void setLeafValue(double leafValue) {
        this.leafValue = leafValue;
    }
    
    public boolean isLeaf() {
        return isLeaf;
    }
    
    public void setLeaf(boolean isLeaf) {
        this.isLeaf = isLeaf;
    }
    
    public int getDepth() {
        return depth;
    }
    
    public void setDepth(int depth) {
        this.depth = depth;
    }
    
    public int getSampleCount() {
        return sampleCount;
    }
    
    public void setSampleCount(int sampleCount) {
        this.sampleCount = sampleCount;
    }
    
    public double getGain() {
        return gain;
    }
    
    public void setGain(double gain) {
        this.gain = gain;
    }
}