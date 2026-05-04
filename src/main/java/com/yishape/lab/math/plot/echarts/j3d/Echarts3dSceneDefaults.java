package com.yishape.lab.math.plot.echarts.j3d;

import java.util.HashMap;
import java.util.Map;

/**
 * GL 场景的默认视图控制 / 光照 / 后处理，与 JavaFX 侧 {@link com.yishape.lab.math.plot.javafx.j3d.JavaFx3dThemeConfig} 对应。
 *
 * @author lteb2
 */
final class Echarts3dSceneDefaults {

    private Echarts3dSceneDefaults() {
    }

    static void initialize(Map<String, Object> sceneConfig) {
        sceneConfig.clear();
        sceneConfig.put("viewControl", viewControl());
        sceneConfig.put("light", lighting());
        sceneConfig.put("postEffect", postEffect());
    }

    /** 反序列化后若缺键则补齐，不改变已有自定义项 */
    static void ensureDefaults(Map<String, Object> sceneConfig) {
        sceneConfig.putIfAbsent("viewControl", viewControl());
        sceneConfig.putIfAbsent("light", lighting());
        sceneConfig.putIfAbsent("postEffect", postEffect());
    }

    private static Map<String, Object> viewControl() {
        Map<String, Object> control = new HashMap<>();
        control.put("projection", "perspective");
        control.put("autoRotate", false);
        control.put("autoRotateSpeed", 10);
        control.put("distance", 200);
        control.put("alpha", 20);
        control.put("beta", 40);
        control.put("minDistance", 50);
        control.put("maxDistance", 400);
        control.put("panMouseButton", "left");
        control.put("rotateMouseButton", "left");
        control.put("zoomSensitivity", 1);
        return control;
    }

    private static Map<String, Object> lighting() {
        Map<String, Object> light = new HashMap<>();
        light.put("main", Map.of(
                "intensity", 1.2,
                "shadow", true,
                "shadowQuality", "high",
                "alpha", 30,
                "beta", 30
        ));
        light.put("ambient", Map.of("intensity", 0.4));
        light.put("ambientCubemap", Map.of(
                "diffuse", Map.of("intensity", 0.5),
                "specular", Map.of("intensity", 0.5)
        ));
        return light;
    }

    private static Map<String, Object> postEffect() {
        Map<String, Object> effect = new HashMap<>();
        effect.put("enable", true);
        effect.put("bloom", Map.of("enable", false, "intensity", 0.1));
        effect.put("SSAO", Map.of(
                "enable", true,
                "radius", 2,
                "intensity", 1.2,
                "quality", "medium"
        ));
        return effect;
    }
}
