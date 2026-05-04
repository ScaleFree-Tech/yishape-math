package com.yishape.lab.math.linalg.decomposition.impl.support.lasd;

/**
 * LAPACK DLAMRG：将两个已排序数组合并为一个升序序列的索引（1-based 下标）。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class Dlamrg {

    private Dlamrg() {
    }

    /**
     * @param n1     第一段长度（升序 A[1..n1]）
     * @param n2     第二段长度（升序 A[n1+1..n1+n2]）
     * @param a      1-based，a[1..n1+n2] 为两段拼接
     * @param index  输出 1-based：index[k] = 取 a 的位置（下标 1..n1+n2）
     * @param indexOffset 起始写入下标（常为 1，对应 Fortran INDEX(1)）
     */
    public static void dlamrg(int n1, int n2, double[] a, int[] index, int indexOffset) {
        int ind1 = 1;
        int ind2 = n1 + 1;
        int n1sv = n1;
        int n2sv = n2;
        int p = indexOffset;
        while (n1sv > 0 && n2sv > 0) {
            if (a[ind1] <= a[ind2]) {
                index[p++] = ind1;
                ind1++;
                n1sv--;
            } else {
                index[p++] = ind2;
                ind2++;
                n2sv--;
            }
        }
        while (n2sv > 0) {
            index[p++] = ind2;
            ind2++;
            n2sv--;
        }
        while (n1sv > 0) {
            index[p++] = ind1;
            ind1++;
            n1sv--;
        }
    }
}
