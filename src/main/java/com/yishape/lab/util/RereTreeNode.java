package com.yishape.lab.util;

import java.util.List;

/**
 * 树节点 / Tree Node
 * @author lteb2
 * @param <T> 数据类型 / Data type
 * @version 1.0
 * @since 1.0
 */
public class RereTreeNode<T> {

    /** 节点ID / Node ID */
    private String id;
    /** 父节点 / Parent node */
    private RereTreeNode father;
    /** 子节点列表 / List of child nodes */
    private List<RereTreeNode> children;

    /** 节点数据 / Node data */
    private T data;

    /**
     * 创建树节点 / Create tree node
     *
     * @param id 节点ID / Node ID
     */
    public RereTreeNode(String id) {
        this.id = id;
    }

    /**
     * 获取节点ID / Get node ID
     * @return 节点ID / Node ID
     */
    public String getId() {
        return id;
    }

    /**
     * 设置节点ID / Set node ID
     * @param id 节点ID / Node ID
     */
    public void setId(String id) {
        this.id = id;
    }

    /**
     * 获取父节点 / Get parent node
     * @return 父节点 / Parent node
     */
    public RereTreeNode getFather() {
        return father;
    }

    /**
     * 设置父节点 / Set parent node
     * @param father 父节点 / Parent node
     */
    public void setFather(RereTreeNode father) {
        this.father = father;
    }

    /**
     * 获取子节点列表 / Get children
     * @return 子节点列表 / List of child nodes
     */
    public List<RereTreeNode> getChildren() {
        return children;
    }

    /**
     * 设置子节点列表 / Set children
     * @param children 子节点列表 / List of child nodes
     */
    public void setChildren(List<RereTreeNode> children) {
        this.children = children;
    }

    /**
     * 获取节点数据 / Get node data
     * @return 节点数据 / Node data
     */
    public T getData() {
        return data;
    }

    /**
     * 设置节点数据 / Set node data
     * @param data 节点数据 / Node data
     */
    public void setData(T data) {
        this.data = data;
    }
}