package com.reremouse.lab.math.ml;

import java.io.*;

/**
 * 可以保存在本地的模型，适合需要时直接加载进行推理
 * @author lteb2
 */
public interface ISerializableModel extends Serializable {
    
    /**
     * 将模型保存在本地
     * @param path 保存路径
     */
    public void save(String path);
    
    /**
     * 从本地加载模型
     * @param path 加载路径
     * @return 
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
                System.err.println("加载的对象不是有效的ISerializableModel实例");
                return null;
            }
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }
    
}