package com.yishape.lab.util;

import java.text.MessageFormat;
import java.util.Locale;
import java.util.ResourceBundle;

/**
 * i18n message catalog backed by {@link ResourceBundle}.
 *
 * <p>Message keys follow the convention {@code module.component.action.status}.
 * When a key is not found in the bundle, the key itself is returned as a
 * fallback (never null).</p>
 *
 * <p>The active locale is controlled by {@link YishapeConfig#getLocale()}.
 * ResourceBundle caches are cleared on locale switch.</p>
 */
public final class Messages {

    private static final String BASE_NAME = "i18n.Messages";
    private static volatile ResourceBundle cachedBundle;
    private static volatile Locale cachedLocale;

    private Messages() {
    }

    /**
     * Returns the localized message for {@code key}, formatted with
     * {@link MessageFormat} using the given arguments.
     *
     * @param key  message key; returned as-is if not found in bundle
     * @param args MessageFormat parameters ({0}, {1}, ...)
     * @return formatted localized string, never null
     */
    public static String get(String key, Object... args) {
        ResourceBundle bundle = bundle();
        String pattern;
        try {
            pattern = bundle.getString(key);
        } catch (java.util.MissingResourceException e) {
            pattern = key;
        }
        if (args == null || args.length == 0) {
            return pattern;
        }
        try {
            return MessageFormat.format(pattern, args);
        } catch (IllegalArgumentException e) {
            return pattern;
        }
    }

    /**
     * Returns the localized message for a specific locale without changing
     * the global locale setting.
     */
    public static String getFor(Locale loc, String key, Object... args) {
        ResourceBundle bundle = ResourceBundle.getBundle(BASE_NAME, loc);
        String pattern;
        try {
            pattern = bundle.getString(key);
        } catch (java.util.MissingResourceException e) {
            pattern = key;
        }
        if (args == null || args.length == 0) {
            return pattern;
        }
        try {
            return MessageFormat.format(pattern, args);
        } catch (IllegalArgumentException e) {
            return pattern;
        }
    }

    /** Returns the current locale used for message lookups. */
    public static Locale currentLocale() {
        return YishapeConfig.getLocale();
    }

    static void clearCache() {
        cachedBundle = null;
        cachedLocale = null;
        ResourceBundle.clearCache();
    }

    private static ResourceBundle bundle() {
        Locale loc = YishapeConfig.getLocale();
        ResourceBundle b = cachedBundle;
        if (b != null && loc.equals(cachedLocale)) {
            return b;
        }
        b = ResourceBundle.getBundle(BASE_NAME, loc);
        cachedBundle = b;
        cachedLocale = loc;
        return b;
    }
}
