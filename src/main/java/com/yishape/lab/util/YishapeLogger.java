package com.yishape.lab.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Unified logging facade wrapping SLF4J with global level filtering from
 * {@link YishapeConfig}.
 *
 * <p>The effective level is the stricter of YishapeConfig's level and the
 * SLF4J binding's level. This means the library can only further restrict
 * output, never amplify it beyond what the application's SLF4J binding allows.</p>
 *
 * <p>Uses SLF4J {@code {}} placeholder syntax. For i18n messages, callers
 * resolve the string via {@link Messages} before passing it to log methods:</p>
 * <pre>
 * log.info(Messages.get("cv.fold.completed", k));
 * </pre>
 */
public final class YishapeLogger {

    private final Logger delegate;

    private YishapeLogger(Class<?> clazz) {
        this.delegate = LoggerFactory.getLogger(clazz);
    }

    public static YishapeLogger getLogger(Class<?> clazz) {
        return new YishapeLogger(clazz);
    }

    // ── Level guard methods (avoids useless string building) ──

    public boolean isDebugEnabled() {
        return YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.DEBUG) && delegate.isDebugEnabled();
    }

    public boolean isTraceEnabled() {
        return YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.TRACE) && delegate.isTraceEnabled();
    }

    public boolean isInfoEnabled() {
        return YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.INFO) && delegate.isInfoEnabled();
    }

    public boolean isWarnEnabled() {
        return YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.WARN) && delegate.isWarnEnabled();
    }

    // ── Standard log methods ──

    public void error(String msg, Object... args) {
        if (YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.ERROR)) {
            delegate.error(msg, args);
        }
    }

    public void error(String msg, Throwable t) {
        if (YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.ERROR)) {
            delegate.error(msg, t);
        }
    }

    public void warn(String msg, Object... args) {
        if (YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.WARN)) {
            delegate.warn(msg, args);
        }
    }

    public void warn(String msg, Throwable t) {
        if (YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.WARN)) {
            delegate.warn(msg, t);
        }
    }

    public void info(String msg, Object... args) {
        if (YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.INFO)) {
            delegate.info(msg, args);
        }
    }

    public void debug(String msg, Object... args) {
        if (YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.DEBUG)) {
            delegate.debug(msg, args);
        }
    }

    public void trace(String msg, Object... args) {
        if (YishapeConfig.isLevelEnabled(YishapeConfig.LogLevel.TRACE)) {
            delegate.trace(msg, args);
        }
    }

    // ── Access underlying SLF4J logger ──

    public Logger unwrap() {
        return delegate;
    }
}
