package com.yishape.lab.math.ml;

import org.slf4j.LoggerFactory;

import java.io.*;

/**
 * 可序列化模型接口 / Serializable Model Interface
 * <p>
 * 定义可序列化模型的接口，支持将模型保存到本地文件系统以及从本地加载。
 * Defines the interface for serializable models that can be saved to local filesystem and loaded when needed.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface ISerializableModel extends Serializable {

    /**
     * 将模型保存在本地 / Save Model to Local Storage
     *
     * @param path 保存路径 / Save path
     */
    public void save(String path);

    /**
     * 从本地加载模型 / Load Model from Local Storage
     *
     * @param path 加载路径 / Load path
     * @return 加载的模型，如果加载失败则返回null / Loaded model, or null if loading fails
     */
    public static ISerializableModel load(String path) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            Object obj = ois.readObject();
            
            // 验证对象类型与期望的模型类型是否匹配
            if (obj instanceof ISerializableModel) {
                ISerializableModel model = (ISerializableModel) obj;
                
                // 可以添加类型检查逻辑，确保加载的模型与期望的modelType匹配
                // 这里我们只是返回加载的对象，但在实际应用中可能需要更严格的类型验证
                
                return model;
            } else {
                LoggerFactory.getLogger(ISerializableModel.class).warn("加载的对象不是有效的ISerializableModel实例");
                return null;
            }
        } catch (IOException | ClassNotFoundException e) {
//            log.error("exception", e);
            return null;
        }
    }
    
}