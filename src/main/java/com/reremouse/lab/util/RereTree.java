package com.reremouse.lab.util;

import java.util.UUID;

/**
 *
 * @author RereMouse
 */
public class RereTree {

    private String id = UUID.randomUUID().toString();
    private RereTreeNode root;//根节点

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public RereTreeNode getRoot() {
        return root;
    }

    public void setRoot(RereTreeNode root) {
        this.root = root;
    }

}
