package com.yishape.lab.math.plot;

/**
 * 可选宿主集成：当 classpath 上存在 MathStudio 的 {@code PlotBridge} 且已注册 sink 时，
 * 将 {@link IPlot} 转发到 IDE 内嵌图表面板；否则静默返回 {@code false}，由调用方走默认 {@code show()} 行为。
 *
 * <p>使用反射避免 yishape-math 对 MathStudio 的编译依赖；单独使用 yishape-math（无 PlotBridge）时
 * 类加载失败或方法不可用，均落入 {@code false}，库行为与原先一致。</p>
 */
public final class PlotHostBridge {

    private PlotHostBridge() {}

    /**
     * 若当前运行在与 MathStudio 集成的环境中，将 {@code plot} 交给宿主嵌入面板并返回 {@code true}；
     * 否则返回 {@code false}。
     */
    public static boolean trySendToIde(IPlot plot) {
        if (plot == null) {
            return false;
        }
        ClassLoader cl = Thread.currentThread().getContextClassLoader();
        if (cl == null) {
            cl = PlotHostBridge.class.getClassLoader();
        }
        try {
            Class<?> bridge = Class.forName("com.reremouse.mathstudio.execution.PlotBridge", false, cl);
            Object has = bridge.getMethod("hasSink").invoke(null);
            if (!Boolean.TRUE.equals(has)) {
                return false;
            }
            bridge.getMethod("send", IPlot.class).invoke(null, plot);
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        } catch (Throwable e) {
            return false;
        }
    }
}
