package com.yishape.lab.math.plot.javafx.j3d;

import javafx.scene.paint.Color;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * JavaFX 3D主题配置类，支持材质和光照参数的可配置化。
 * <p>
 * 允许通过主题系统配置：
 * <ul>
 *   <li>材质属性：漫反射、镜面反射、自发光</li>
 *   <li>光照属性：环境光、点光源、方向光</li>
 *   <li>阴影属性：启用/禁用、质量等级</li>
 *   <li>后期效果：抗锯齿、辉光</li>
 * </ul>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class JavaFx3dThemeConfig implements Serializable, Cloneable {

    private static final long serialVersionUID = 1L;

    // 材质属性
    private Color diffuseColor = Color.WHITE;
    private Color specularColor = Color.color(1, 1, 1, 0.38);
    private double specularPower = 42.0;
    private double opacity = 1.0;

    // 光照属性
    private Color ambientLightColor = Color.color(0.42, 0.42, 0.45);
    private double ambientIntensity = 0.4;

    private Color pointLightColor = Color.WHITE;
    private double pointLightIntensity = 1.0;
    private double[] pointLightPosition = {400, -280, -340};

    private Color fillLightColor = Color.color(0.95, 0.96, 1.0);
    private double fillLightIntensity = 0.6;
    private double[] fillLightPosition = {-420, 160, 360};

    // 阴影属性
    private boolean shadowEnabled = true;
    private String shadowQuality = "medium"; // low, medium, high

    // 后期效果
    private boolean antialiasingEnabled = true;
    private boolean bloomEnabled = false;
    private double bloomIntensity = 0.1;

    // 相机属性
    private double cameraFOV = 45.0;
    private double cameraNearClip = 0.1;
    private double cameraFarClip = 8000.0;
    private double cameraInitialDistance = 690.0;

    // 场景属性
    private Color backgroundColor = Color.WHITE;
    private boolean gridEnabled = true;
    private Color gridColor = Color.color(0.8, 0.8, 0.8, 0.3);

    // 预设主题配置
    public static JavaFx3dThemeConfig getDefaultConfig() {
        return new JavaFx3dThemeConfig();
    }

    public static JavaFx3dThemeConfig getDarkConfig() {
        JavaFx3dThemeConfig config = new JavaFx3dThemeConfig();
        config.backgroundColor = Color.color(0.12, 0.12, 0.12);
        config.ambientLightColor = Color.color(0.55, 0.55, 0.58);
        config.ambientIntensity = 0.5;
        config.pointLightColor = Color.color(0.93, 0.95, 1.0);
        config.fillLightColor = Color.color(0.5, 0.55, 0.65);
        config.gridColor = Color.color(0.4, 0.4, 0.4, 0.3);
        return config;
    }

    public static JavaFx3dThemeConfig getAcademicConfig() {
        JavaFx3dThemeConfig config = new JavaFx3dThemeConfig();
        config.backgroundColor = Color.color(0.98, 0.98, 0.98);
        config.specularPower = 32.0;
        config.gridEnabled = true;
        config.gridColor = Color.color(0.85, 0.85, 0.85, 0.5);
        config.antialiasingEnabled = true;
        return config;
    }

    public static JavaFx3dThemeConfig getFuturisticConfig() {
        JavaFx3dThemeConfig config = new JavaFx3dThemeConfig();
        config.backgroundColor = Color.color(0.06, 0.06, 0.14);
        config.ambientLightColor = Color.color(0.2, 0.3, 0.5);
        config.pointLightColor = Color.color(0, 1, 0.65); // 霓虹绿
        config.specularColor = Color.color(0, 1, 0.65, 0.5);
        config.specularPower = 64.0;
        config.bloomEnabled = true;
        config.bloomIntensity = 0.3;
        return config;
    }

    // Getters and Setters

    public Color getDiffuseColor() { return diffuseColor; }
    public void setDiffuseColor(Color c) { this.diffuseColor = c; }

    public Color getSpecularColor() { return specularColor; }
    public void setSpecularColor(Color c) { this.specularColor = c; }

    public double getSpecularPower() { return specularPower; }
    public void setSpecularPower(double p) { this.specularPower = p; }

    public double getOpacity() { return opacity; }
    public void setOpacity(double o) { this.opacity = o; }

    public Color getAmbientLightColor() { return ambientLightColor; }
    public void setAmbientLightColor(Color c) { this.ambientLightColor = c; }

    public double getAmbientIntensity() { return ambientIntensity; }
    public void setAmbientIntensity(double i) { this.ambientIntensity = i; }

    public Color getPointLightColor() { return pointLightColor; }
    public void setPointLightColor(Color c) { this.pointLightColor = c; }

    public double getPointLightIntensity() { return pointLightIntensity; }
    public void setPointLightIntensity(double i) { this.pointLightIntensity = i; }

    public double[] getPointLightPosition() { return pointLightPosition.clone(); }
    public void setPointLightPosition(double[] pos) {
        this.pointLightPosition = pos != null ? pos.clone() : new double[]{400, -280, -340};
    }

    public Color getFillLightColor() { return fillLightColor; }
    public void setFillLightColor(Color c) { this.fillLightColor = c; }

    public double getFillLightIntensity() { return fillLightIntensity; }
    public void setFillLightIntensity(double i) { this.fillLightIntensity = i; }

    public double[] getFillLightPosition() { return fillLightPosition.clone(); }
    public void setFillLightPosition(double[] pos) {
        this.fillLightPosition = pos != null ? pos.clone() : new double[]{-420, 160, 360};
    }

    public boolean isShadowEnabled() { return shadowEnabled; }
    public void setShadowEnabled(boolean e) { this.shadowEnabled = e; }

    public String getShadowQuality() { return shadowQuality; }
    public void setShadowQuality(String q) { this.shadowQuality = q; }

    public boolean isAntialiasingEnabled() { return antialiasingEnabled; }
    public void setAntialiasingEnabled(boolean e) { this.antialiasingEnabled = e; }

    public boolean isBloomEnabled() { return bloomEnabled; }
    public void setBloomEnabled(boolean e) { this.bloomEnabled = e; }

    public double getBloomIntensity() { return bloomIntensity; }
    public void setBloomIntensity(double i) { this.bloomIntensity = i; }

    public double getCameraFOV() { return cameraFOV; }
    public void setCameraFOV(double fov) { this.cameraFOV = fov; }

    public double getCameraNearClip() { return cameraNearClip; }
    public void setCameraNearClip(double c) { this.cameraNearClip = c; }

    public double getCameraFarClip() { return cameraFarClip; }
    public void setCameraFarClip(double c) { this.cameraFarClip = c; }

    public double getCameraInitialDistance() { return cameraInitialDistance; }
    public void setCameraInitialDistance(double d) { this.cameraInitialDistance = d; }

    public Color getBackgroundColor() { return backgroundColor; }
    public void setBackgroundColor(Color c) { this.backgroundColor = c; }

    public boolean isGridEnabled() { return gridEnabled; }
    public void setGridEnabled(boolean e) { this.gridEnabled = e; }

    public Color getGridColor() { return gridColor; }
    public void setGridColor(Color c) { this.gridColor = c; }

    /**
     * 从Map加载配置
     */
    public void fromMap(Map<String, Object> map) {
        if (map == null) return;

        if (map.containsKey("diffuseColor")) {
            diffuseColor = parseColor(map.get("diffuseColor"));
        }
        if (map.containsKey("specularColor")) {
            specularColor = parseColor(map.get("specularColor"));
        }
        if (map.containsKey("specularPower")) {
            specularPower = ((Number) map.get("specularPower")).doubleValue();
        }
        if (map.containsKey("ambientIntensity")) {
            ambientIntensity = ((Number) map.get("ambientIntensity")).doubleValue();
        }
        if (map.containsKey("shadowEnabled")) {
            shadowEnabled = (Boolean) map.get("shadowEnabled");
        }
        if (map.containsKey("antialiasingEnabled")) {
            antialiasingEnabled = (Boolean) map.get("antialiasingEnabled");
        }
        if (map.containsKey("backgroundColor")) {
            backgroundColor = parseColor(map.get("backgroundColor"));
        }
    }

    /**
     * 转换为Map
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("diffuseColor", colorToString(diffuseColor));
        map.put("specularColor", colorToString(specularColor));
        map.put("specularPower", specularPower);
        map.put("ambientIntensity", ambientIntensity);
        map.put("shadowEnabled", shadowEnabled);
        map.put("antialiasingEnabled", antialiasingEnabled);
        map.put("backgroundColor", colorToString(backgroundColor));
        return map;
    }

    private Color parseColor(Object obj) {
        if (obj instanceof Color) return (Color) obj;
        if (obj instanceof String) return javafx.scene.paint.Color.valueOf((String) obj);
        return Color.WHITE;
    }

    private String colorToString(Color c) {
        return String.format("#%02X%02X%02X",
                (int) (c.getRed() * 255),
                (int) (c.getGreen() * 255),
                (int) (c.getBlue() * 255));
    }

    @Override
    public JavaFx3dThemeConfig clone() {
        try {
            JavaFx3dThemeConfig cloned = (JavaFx3dThemeConfig) super.clone();
            cloned.pointLightPosition = this.pointLightPosition.clone();
            cloned.fillLightPosition = this.fillLightPosition.clone();
            return cloned;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }
}
