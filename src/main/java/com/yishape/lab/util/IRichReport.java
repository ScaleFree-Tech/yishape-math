package com.yishape.lab.util;

/**
 * 丰富报告接口 / Rich Report Interface
 * <p>
 * 为模型结果、评估指标等提供统一的结构化报表输出能力。
 * Provides a unified structured report output capability for model results, evaluation metrics, etc.
 * </p>
 *
 * @author RereMouse
 * @since 1.0
 */
public interface IRichReport {

    /**
     * 获取完整格式化报告 / Get full formatted report
     *
     * @return 完整报告字符串 / Full report string
     */
    String toReport();

    /**
     * 获取简要报告（默认返回 toReport() 的第一行）/ Get brief report
     *
     * @return 简要报告字符串 / Brief report string
     */
    default String toBriefReport() {
        String full = toReport();
        if (full == null || full.isEmpty()) {
            return "";
        }
        int nl = full.indexOf('\n');
        return nl > 0 ? full.substring(0, nl) : full;
    }
}
