package com.reremouse.lab.util;

import java.util.List;

/**
 * 树节点
 * @author RereMouse
 */
public class RereTreeNode<T> {

    private String id;
    private RereTreeNode father;//父节点
    private List<RereTreeNode> children;//子节点

    private T data;

    public RereTreeNode(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public RereTreeNode getFather() {
        return father;
    }

    public void setFather(RereTreeNode father) {
        this.father = father;
    }

    public List<RereTreeNode> getChildren() {
        return children;
    }

    public void setChildren(List<RereTreeNode> children) {
        this.children = children;
    }

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }

}
