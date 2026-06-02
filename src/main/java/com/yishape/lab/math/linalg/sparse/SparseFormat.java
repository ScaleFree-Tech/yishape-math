package com.yishape.lab.math.linalg.sparse;

public enum SparseFormat {
    CSR,
    CSC,
    COO;

    public String toString() {
        return name();
    }
}