package com.yishape.lab.math.compute.hpc;

/**
 * <p><strong>HPC 运行时总开关</strong>：一键启用/禁用所有原生加速尝试，无需重启 JVM。</p>
 *
 * <p>与 {@link HpcConfig} 的分工：</p>
 * <ul>
 *   <li>{@code HpcSwitch} — 控制<strong>是否尝试</strong>调用 HPC（运行时热切换）</li>
 *   <li>{@code HpcConfig} — 控制<strong>满足什么规模阈值</strong>后才尝试（系统属性配置）</li>
 * </ul>
 *
 * <p>只有当 {@code HpcSwitch} <strong>且</strong> {@code HpcConfig} 同时允许时，各调用点
 * （{@link HpcGemm}、{@link HpcLapackDecomps}、{@link HpcOptionalRuntime} 等）
 * 才会真正发起 HPC 调用；任一关闭即回退纯 Java。</p>
 *
 * <p><strong>默认状态</strong>：{@code true}（启用尝试），与 {@code HpcConfig} 的默认行为一致。<br>
 * <strong>线程安全</strong>：{@code volatile} 保证可见性，无锁。</p>
 *
 * <h3>使用示例</h3>
 * <pre>{@code
 * // 1. 一键禁用 HPC（例如调试或对比测试）
 * HpcSwitch.disable();
 *
 * // 2. 查询当前状态
 * if (HpcSwitch.isEnabled()) { ... }
 *
 * // 3. 临时禁用，执行完自动恢复
 * HpcSwitch.runWith(false, () -> {
 *     IMatrix<Double> C = A.mmul(B); // 强制走纯 Java
 * });
 *
 * // 4. 切换状态
 * boolean nowOn = HpcSwitch.toggle();
 * }</pre>
 *
 * @since 0.5.0
 * @see HpcConfig
 */
public final class HpcSwitch {

    private static volatile boolean enabled = true;

    private HpcSwitch() {
    }

    /**
     * 启用 HPC 尝试。此后各调用点在满足 {@link HpcConfig} 规模阈值时会尝试走 HPC 路径。
     */
    public static void enable() {
        enabled = true;
    }

    /**
     * 禁用 HPC 尝试。此后所有调用点直接回退纯 Java，不走任何 HPC 路径。
     */
    public static void disable() {
        enabled = false;
    }

    /**
     * 切换 HPC 尝试状态：若当前启用则禁用，反之启用。
     *
     * @return 切换后的状态
     */
    public static boolean toggle() {
        enabled = !enabled;
        return enabled;
    }

    /**
     * 查询当前是否允许 HPC 尝试。
     *
     * @return {@code true} 表示允许尝试 HPC；{@code false} 表示强制回退纯 Java
     */
    public static boolean isEnabled() {
        return enabled;
    }

    /**
     * 在指定的 HPC 状态下执行代码块，执行完毕后<strong>自动恢复</strong>之前的状态。
     *
     * <p>适用于测试或对比场景：需要临时切换 HPC 状态执行一段代码，无需手动管理恢复逻辑。</p>
     *
     * <pre>{@code
     * HpcSwitch.runWith(false, () -> {
     *     // 这段代码内 HPC 被禁用
     *     double[][] c = computer.mmul(a, b);
     * });
     * // 离开 runWith 后，HPC 自动恢复到之前的状态
     * }</pre>
     *
     * @param state 执行期间的临时状态：{@code true}=启用，{@code false}=禁用
     * @param task  要执行的代码块
     */
    public static void runWith(boolean state, Runnable task) {
        synchronized (HpcSwitch.class) {
            boolean previous = enabled;
            try {
                enabled = state;
                task.run();
            } finally {
                enabled = previous;
            }
        }
    }
}
