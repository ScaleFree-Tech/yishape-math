package com.yishape.lab.math.ml;

import com.yishape.lab.util.JsonUtil;
import com.yishape.lab.util.YishapeLogger;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * JSON-based model persistence interface.
 * <p>
 * Each model exports its inference-essential parameters via {@link #toParams()} and restores from
 * them via {@link #fromParams(Map)}. The default {@link #save(String)} / {@link #load(String)} use
 * JSON with a {@code @type} envelope for reflection-based instantiation.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public interface ISerializableModel {

    /**
     * Export inference-essential parameters as a map.
     * Values must be JSON-serializable: String, Number, Boolean, arrays, nested maps/lists.
     */
    Map<String, Object> toParams();

    /**
     * Restore model state from a parameter map produced by {@link #toParams()}.
     */
    void fromParams(Map<String, Object> params);

    /**
     * Save model to a JSON file.
     */
    default void save(String path) {
        Map<String, Object> envelope = new LinkedHashMap<>();
        envelope.put("@type", getClass().getName());
        envelope.put("params", toParams());
        String json = JsonUtil.toJson(envelope);
        try {
            Files.writeString(Path.of(path), json, StandardCharsets.UTF_8);
        } catch (Exception e) {
            YishapeLogger.getLogger(ISerializableModel.class).error("Failed to save model to {}", path, e);
        }
    }

    /**
     * Load model from a JSON file.
     *
     * @param path file path
     * @return deserialized model, or null on failure
     */
    static ISerializableModel load(String path) {
        try {
            String json = Files.readString(Path.of(path), StandardCharsets.UTF_8);
            Map<String, Object> envelope = JsonUtil.fromJson(json);
            String typeName = (String) envelope.get("@type");
            if (typeName == null) {
                YishapeLogger.getLogger(ISerializableModel.class).warn("JSON missing @type field: {}", path);
                return null;
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> params = (Map<String, Object>) envelope.get("params");
            if (params == null) {
                YishapeLogger.getLogger(ISerializableModel.class).warn("JSON missing params field: {}", path);
                return null;
            }
            Class<?> clazz = Class.forName(typeName);
            Object instance = clazz.getDeclaredConstructor().newInstance();
            if (instance instanceof ISerializableModel model) {
                model.fromParams(params);
                return model;
            }
            YishapeLogger.getLogger(ISerializableModel.class).warn("Loaded class is not ISerializableModel: {}", typeName);
            return null;
        } catch (Exception e) {
            YishapeLogger.getLogger(ISerializableModel.class).warn("Failed to load model from {}", path, e);
            return null;
        }
    }
}
