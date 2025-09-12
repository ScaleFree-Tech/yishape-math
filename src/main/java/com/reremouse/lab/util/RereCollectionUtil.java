package com.reremouse.lab.util;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 集合工具类 - 改进版本
 * 
 * 改进说明：
 * 1. 添加线程安全支持
 * 2. 优化反射性能，添加方法缓存
 * 3. 改进异常处理和日志记录
 * 4. 添加参数验证
 * 5. 保持原有API兼容性
 *
 * @author RereMouse
 */
public class RereCollectionUtil {

    private static final Logger logger = LoggerFactory.getLogger(RereCollectionUtil.class);
    
    // 方法缓存，避免重复反射
    private static final ConcurrentHashMap<Class<?>, Method> methodCache = new ConcurrentHashMap<>();
    
    // 读写锁保护方法缓存
    private static final ReadWriteLock cacheLock = new ReentrantReadWriteLock();

    /**
     * 从列表中删除某元素（按getId()匹配）
     * 线程安全版本
     *
     * @param <T> 列表元素类型
     * @param ls 列表
     * @param id 元素ID
     * @return 修改后的列表
     */
    public static <T> List<T> deleteElementFromList(List<T> ls, String id) {
        return (List<T>) deleteElementFromCollection(ls, id);
    }

    /**
     * 从集合中删除某元素（按getId()匹配）
     * 线程安全版本，支持 CopyOnWriteArraySet
     *
     * @param <T> 集合元素类型
     * @param ls 集合
     * @param id 元素ID
     * @return 修改后的集合
     */
    public static <T> Collection<T> deleteElementFromCollection(Collection<T> ls, String id) {
        if (ls == null) {
            logger.warn("集合参数为空，无法执行删除操作");
            return ls;
        }
        
        if (StringUtils.isBlank(id)) {
            logger.warn("ID参数为空，无法执行删除操作");
            return ls;
        }
        
        try {
            // 检查集合类型，CopyOnWriteArraySet 需要特殊处理
            if (ls.getClass().getName().contains("CopyOnWriteArraySet")) {
                // 对于 CopyOnWriteArraySet，直接使用 remove() 方法
                T elementToRemove = null;
                for (T x : ls) {
                    if (x == null) {
                        continue;
                    }
                    
                    try {
                        String xid = getIdFromObject(x);
                        if (id.equals(xid)) {
                            elementToRemove = x;
                            break;
                        }
                    } catch (Exception e) {
                        logger.warn("获取对象ID时发生错误，跳过该元素: {}", x.getClass().getSimpleName(), e);
                    }
                }
                
                if (elementToRemove != null) {
                    boolean removed = ls.remove(elementToRemove);
                    if (removed) {
                        logger.debug("成功删除元素，ID: {}", id);
                    } else {
                        logger.warn("删除元素失败，ID: {}", id);
                    }
                } else {
                    logger.debug("未找到要删除的元素，ID: {}", id);
                }
            } else {
                // 对于其他集合类型，使用迭代器删除
                synchronized (ls) {
                    Iterator<T> it = ls.iterator();
                    while (it.hasNext()) {
                        T x = it.next();
                        if (x == null) {
                            continue;
                        }
                        
                        try {
                            String xid = getIdFromObject(x);
                            if (id.equals(xid)) {
                                it.remove();
                                logger.debug("成功删除元素，ID: {}", id);
                                break; // 找到并删除后立即退出
                            }
                        } catch (Exception e) {
                            logger.warn("获取对象ID时发生错误，跳过该元素: {}", x.getClass().getSimpleName(), e);
                        }
                    }
                }
            }
        } catch (Exception e) {
            logger.error("删除集合元素时发生错误，ID: {}", id, e);
        }
        
        return ls;
    }

    /**
     * 从对象获取ID，使用缓存优化反射性能
     *
     * @param obj 对象
     * @return ID字符串
     * @throws Exception 如果获取失败
     */
    private static String getIdFromObject(Object obj) throws Exception {
        if (obj == null) {
            throw new IllegalArgumentException("对象不能为空");
        }
        
        Class<?> clazz = obj.getClass();
        
        // 先从缓存中获取方法
        Method method = getCachedMethod(clazz);
        if (method == null) {
            // 缓存中没有，通过反射获取
            method = clazz.getMethod("getId");
            // 缓存方法
            cacheMethod(clazz, method);
        }
        
        Object result = method.invoke(obj);
        return result != null ? result.toString() : "";
    }

    /**
     * 从缓存中获取方法
     *
     * @param clazz 类
     * @return 方法，如果不存在则返回null
     */
    private static Method getCachedMethod(Class<?> clazz) {
        cacheLock.readLock().lock();
        try {
            return methodCache.get(clazz);
        } finally {
            cacheLock.readLock().unlock();
        }
    }

    /**
     * 缓存方法
     *
     * @param clazz 类
     * @param method 方法
     */
    private static void cacheMethod(Class<?> clazz, Method method) {
        cacheLock.writeLock().lock();
        try {
            // 双重检查，避免重复缓存
            if (!methodCache.containsKey(clazz)) {
                methodCache.put(clazz, method);
                logger.debug("缓存方法 getId 用于类: {}", clazz.getSimpleName());
            }
        } finally {
            cacheLock.writeLock().unlock();
        }
    }

    /**
     * 批量删除元素（按ID列表匹配）
     * 新增功能：支持批量删除，支持 CopyOnWriteArraySet
     *
     * @param <T> 集合元素类型
     * @param ls 集合
     * @param ids ID列表
     * @return 修改后的集合
     */
    public static <T> Collection<T> deleteElementsFromCollection(Collection<T> ls, List<String> ids) {
        if (ls == null || ids == null || ids.isEmpty()) {
            logger.warn("集合或ID列表为空，无法执行批量删除操作");
            return ls;
        }
        
        int deletedCount = 0;
        try {
            // 检查集合类型，CopyOnWriteArraySet 需要特殊处理
            if (ls.getClass().getName().contains("CopyOnWriteArraySet")) {
                // 对于 CopyOnWriteArraySet，收集要删除的元素然后批量删除
                List<T> elementsToRemove = new ArrayList<>();
                for (T x : ls) {
                    if (x == null) {
                        continue;
                    }
                    
                    try {
                        String xid = getIdFromObject(x);
                        if (ids.contains(xid)) {
                            elementsToRemove.add(x);
                        }
                    } catch (Exception e) {
                        logger.warn("获取对象ID时发生错误，跳过该元素: {}", x.getClass().getSimpleName(), e);
                    }
                }
                
                // 批量删除
                for (T element : elementsToRemove) {
                    if (ls.remove(element)) {
                        deletedCount++;
                    }
                }
            } else {
                // 对于其他集合类型，使用迭代器删除
                synchronized (ls) {
                    Iterator<T> it = ls.iterator();
                    while (it.hasNext()) {
                        T x = it.next();
                        if (x == null) {
                            continue;
                        }
                        
                        try {
                            String xid = getIdFromObject(x);
                            if (ids.contains(xid)) {
                                it.remove();
                                deletedCount++;
                            }
                        } catch (Exception e) {
                            logger.warn("获取对象ID时发生错误，跳过该元素: {}", x.getClass().getSimpleName(), e);
                        }
                    }
                }
            }
            
            logger.info("批量删除完成，删除了 {} 个元素", deletedCount);
        } catch (Exception e) {
            logger.error("批量删除集合元素时发生错误", e);
        }
        
        return ls;
    }

    /**
     * 查找集合中的元素（按ID匹配）
     * 新增功能：支持查找元素
     *
     * @param <T> 集合元素类型
     * @param ls 集合
     * @param id 元素ID
     * @return 找到的元素，如果不存在则返回null
     */
    public static <T> T findElementFromCollection(Collection<T> ls, String id) {
        if (ls == null || StringUtils.isBlank(id)) {
            logger.warn("集合或ID参数为空，无法执行查找操作");
            return null;
        }
        
        try {
            synchronized (ls) {
                for (T x : ls) {
                    if (x == null) {
                        continue;
                    }
                    
                    try {
                        String xid = getIdFromObject(x);
                        if (id.equals(xid)) {
                            logger.debug("找到元素，ID: {}", id);
                            return x;
                        }
                    } catch (Exception e) {
                        logger.warn("获取对象ID时发生错误，跳过该元素: {}", x.getClass().getSimpleName(), e);
                    }
                }
            }
        } catch (Exception e) {
            logger.error("查找集合元素时发生错误，ID: {}", id, e);
        }
        
        logger.debug("未找到元素，ID: {}", id);
        return null;
    }

    /**
     * 检查集合中是否包含指定ID的元素
     * 新增功能：支持存在性检查
     *
     * @param <T> 集合元素类型
     * @param ls 集合
     * @param id 元素ID
     * @return 如果包含则返回true，否则返回false
     */
    public static <T> boolean containsElement(Collection<T> ls, String id) {
        return findElementFromCollection(ls, id) != null;
    }

    /**
     * 清空方法缓存
     * 新增功能：支持缓存管理
     */
    public static void clearMethodCache() {
        cacheLock.writeLock().lock();
        try {
            int size = methodCache.size();
            methodCache.clear();
            logger.info("清空方法缓存，清除了 {} 个缓存项", size);
        } finally {
            cacheLock.writeLock().unlock();
        }
    }

    /**
     * 获取缓存统计信息
     * 新增功能：支持缓存监控
     *
     * @return 缓存大小
     */
    public static int getCacheSize() {
        cacheLock.readLock().lock();
        try {
            return methodCache.size();
        } finally {
            cacheLock.readLock().unlock();
        }
    }

    /**
     * 将集合转换为字符串表示
     * 改进原有方法
     *
     * @param <T> 集合元素类型
     * @param collection 集合
     * @return 字符串表示
     */
    public static <T> String collectionAsString(Collection<T> collection) {
        if (collection == null) {
            return "null";
        }
        
        if (collection.isEmpty()) {
            return "[]";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        
        boolean first = true;
        for (T item : collection) {
            if (!first) {
                sb.append(", ");
            }
            
            if (item == null) {
                sb.append("null");
            } else {
                try {
                    String id = getIdFromObject(item);
                    sb.append(item.getClass().getSimpleName()).append("(id=").append(id).append(")");
                } catch (Exception e) {
                    sb.append(item.toString());
                }
            }
            
            first = false;
        }
        
        sb.append("]");
        return sb.toString();
    }

    public static void main(String args[]) {
        // 测试代码
        logger.info("RereCollectionUtil 测试开始");
        
        // 测试缓存功能
        logger.info("当前缓存大小: {}", getCacheSize());
        
        // 测试清空缓存
        clearMethodCache();
        logger.info("清空后缓存大小: {}", getCacheSize());
    }
}
