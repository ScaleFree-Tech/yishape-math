package com.yishape.lab.util;

import java.util.UUID;

/**
 * 树结构 / Tree Structure
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereTree {

    private String id = UUID.randomUUID().toString();
    private RereTreeNode root;//根节点

    /**
     * 获取树ID / Get tree ID
     * @return 树ID / Tree ID
     */
    public String getId() {
        return id;
    }

    /**
     * 设置树ID / Set tree ID
     * @param id 树ID / Tree ID
     */
    public void setId(String id) {
        this.id = id;
    }

    /**
     * 获取根节点 / Get root node
     * @return 根节点 / Root node
     */
    public RereTreeNode getRoot() {
        return root;
    }

    /**
     * 设置根节点 / Set root node
     * @param root 根节点 / Root node
     */
    public void setRoot(RereTreeNode root) {
        this.root = root;
    }
}