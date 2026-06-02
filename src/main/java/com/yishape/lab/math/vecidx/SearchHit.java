package com.yishape.lab.math.vecidx;

import java.io.Serializable;

/**
 * 向量近邻查询结果。
 *
 * <p>{@code id} 为与索引条目关联的全局标识（通常为 UUID 字符串或矩阵行号的字符串形式）；
 * {@code distance} 与构建索引时所选度量一致。</p>
 *
 * @param id       结果条目的全局标识（String）
 * @param distance 与查询向量的距离或相似度值
 */
public record SearchHit(String id, double distance) implements Serializable {

    private static final long serialVersionUID = 1L;
}
