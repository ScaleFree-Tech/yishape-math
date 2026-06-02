package com.yishape.lab.util;

import java.util.Locale;

/**
 * Global configuration singleton for yishape-math logging and i18n.
 *
 * <p>Controls log level (upper bound over SLF4J binding level) and locale for
 * {@link Messages} lookups. Supports both system properties and programmatic
 * API, with programmatic calls taking precedence.</p>
 *
 * <h3>System properties</h3>
 * <pre>
 * -Dyishape.log.profile=DEV|PROD|SILENT|NORMAL|TRACE
 * -Dyishape.log.level=DEBUG|INFO|WARN|ERROR|OFF
 * -Dyishape.locale=zh|en
 * </pre>
 */
public final class YishapeConfig {

    // ── Profile ──────────────────────────────────────────

    public enum Profile {
        /** All logging suppressed. */
        SILENT(LogLevel.OFF, Locale.ENGLISH),
        /** Production: WARN + English. */
        PROD(LogLevel.WARN, Locale.ENGLISH),
        /** Daily use: INFO + English (default). */
        NORMAL(LogLevel.INFO, Locale.ENGLISH),
        /** Development: DEBUG + Chinese. */
        DEV(LogLevel.DEBUG, Locale.SIMPLIFIED_CHINESE),
        /** Performance tuning: TRACE + English. */
        TRACE(LogLevel.TRACE, Locale.ENGLISH);

        final LogLevel level;
        final Locale locale;

        Profile(LogLevel level, Locale locale) {
            this.level = level;
            this.locale = locale;
        }
    }

    public enum LogLevel {
        OFF(0), ERROR(1), WARN(2), INFO(3), DEBUG(4), TRACE(5);

        final int severity;

        LogLevel(int severity) {
            this.severity = severity;
        }

        boolean isEnabled(LogLevel candidate) {
            return candidate.severity > 0 && candidate.severity <= this.severity;
        }
    }

    // ── State (volatile for multi-threaded visibility) ──

    private static volatile LogLevel logLevel;
    private static volatile Locale locale;
    private static volatile Profile profile;

    static {
        // 1. Try yishape.log.profile system property
        String profileStr = System.getProperty("yishape.log.profile");
        if (profileStr != null && !profileStr.isBlank()) {
            try {
                profile = Profile.valueOf(profileStr.trim().toUpperCase());
                logLevel = profile.level;
                locale = profile.locale;
            } catch (IllegalArgumentException ignored) {
                // Fall through to manual property parsing
            }
        }

        // 2. If no profile or invalid, read individual properties
        if (logLevel == null) {
            String levelStr = System.getProperty("yishape.log.level");
            if (levelStr != null && !levelStr.isBlank()) {
                try {
                    logLevel = LogLevel.valueOf(levelStr.trim().toUpperCase());
                } catch (IllegalArgumentException ignored) {
                }
            }
        }
        if (locale == null) {
            String locStr = System.getProperty("yishape.locale");
            if ("zh".equals(locStr) || "zh_CN".equals(locStr) || "zh-CN".equals(locStr)) {
                locale = Locale.SIMPLIFIED_CHINESE;
            } else if (locStr != null && !locStr.isBlank()) {
                locale = Locale.forLanguageTag(locStr.replace('_', '-'));
            }
        }

        // 3. Defaults
        if (logLevel == null) {
            logLevel = Profile.NORMAL.level;
        }
        if (locale == null) {
            locale = Profile.NORMAL.locale;
        }
        if (profile == null) {
            profile = Profile.NORMAL;
        }
    }

    private YishapeConfig() {
    }

    // ── Profile ──────────────────────────────────────────

    public static void setProfile(Profile p) {
        profile = p;
        logLevel = p.level;
        locale = p.locale;
        Messages.clearCache();
    }

    public static Profile getProfile() {
        return profile;
    }

    // ── Log level ────────────────────────────────────────

    public static void setLogLevel(LogLevel level) {
        logLevel = level;
    }

    public static LogLevel getLogLevel() {
        return logLevel;
    }

    static boolean isLevelEnabled(LogLevel candidate) {
        return logLevel.isEnabled(candidate);
    }

    // ── Locale ───────────────────────────────────────────

    public static void setLocale(Locale loc) {
        locale = loc;
        Messages.clearCache();
    }

    public static Locale getLocale() {
        return locale;
    }

    // ── Feature flags (bridge existing simd.* properties) ──

    private static final boolean SIMD_PERF_MON = Boolean.parseBoolean(
            System.getProperty("simd.performance.monitoring", "false"));
    private static final boolean SIMD_DETAIL_LOG = Boolean.parseBoolean(
            System.getProperty("simd.detailed.logging", "false"));

    /**
     * Whether SIMD performance monitoring is active. Enabled by system property
     * {@code simd.performance.monitoring} or by DEV/TRACE profiles.
     */
    public static boolean isSimdPerformanceMonitoring() {
        return SIMD_PERF_MON || profile == Profile.DEV || profile == Profile.TRACE;
    }

    /**
     * Whether SIMD detailed logging is active. Enabled by system property
     * {@code simd.detailed.logging} or by TRACE profile.
     */
    public static boolean isSimdDetailedLogging() {
        return SIMD_DETAIL_LOG || profile == Profile.TRACE;
    }
}
