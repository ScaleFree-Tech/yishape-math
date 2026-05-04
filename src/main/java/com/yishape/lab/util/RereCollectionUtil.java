package com.yishape.lab.util;

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
 * 集合工具类 - 改进版本 / Collection Utility Class - Improved Version
 * 
 * 改进说明 / Improvements:
 * 1. 添加线程安全支持 / Added thread safety support
 * 2. 优化反射性能，添加方法缓存 / Optimized reflection performance with method caching
 * 3. 改进异常处理和日志记录 / Improved exception handling and logging
 * 4. 添加参数验证 / Added parameter validation
 * 5. 保持原有API兼容性 / Maintained original API compatibility
 *
 * @author lteb2
 */
public class RereCollectionUtil {
    
    private static final Logger logger = LoggerFactory.getLogger(RereCollectionUtil.class);
    
    // 方法缓存，避免重复反射 / Method cache to avoid repeated reflection
    private static final ConcurrentHashMap<Class<?>, Method> methodCache = new ConcurrentHashMap<>();
    
    // 读写锁保护方法缓存 / Read-write lock to protect method cache
    private static final ReadWriteLock cacheLock = new ReentrantReadWriteLock();
    
    /**
     * 从列表中删除某元素（按getId()匹配）/ Remove element from list (match by getId())
     * 线程安全版本 / Thread-safe version
     *
     * @param <T> 列表元素类型 / List element type
     * @param ls 列表 / List
     * @param id 元素ID / Element ID
     * @return 修改后的列表 / Modified list
     */
    public static <T> List<T> deleteElementFromList(List<T> ls, String id) {
        return (List<T>) deleteElementFromCollection(ls, id);
    }
    
    /**
     * 从集合中删除某元素（按getId()匹配）/ Remove element from collection (match by getId())
     * 线程安全版本，支持CopyOnWriteArraySet / Thread-safe version, supports CopyOnWriteArraySet
     *
     * @param <T> 集合元素类型 / Collection element type
     * @param ls 集合 / Collection
     * @param id 元素ID / Element ID
     * @return 修改后的集合 / Modified collection
     */
    public static <T> Collection<T> deleteElementFromCollection(Collection<T> ls, String id) {
        if (ls == null) {
            logger.warn("集合参数为空，无法执行删除操作 / Collection parameter is null, cannot perform delete operation");
            return ls;
        }
        
        if (StringUtils.isBlank(id)) {
            logger.warn("ID参数为空，无法执行删除操作 / ID parameter is blank, cannot perform delete operation");
            return ls;
        }
        
        try {
            // 检查集合类型，CopyOnWriteArraySet 需要特殊处理 / Check collection type, CopyOnWriteArraySet needs special handling
            if (ls.getClass().getName().contains("CopyOnWriteArraySet")) {
                // 对于 CopyOnWriteArraySet，直接使用remove() 方法 / For CopyOnWriteArraySet, use remove() method directly
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
                        logger.warn("获取对象ID时发生错误，跳过该元素 / Error getting object ID, skipping element: {}", x.getClass().getSimpleName(), e);
                    }
                }
                
                if (elementToRemove != null) {
                    boolean removed = ls.remove(elementToRemove);
                    if (removed) {
                        logger.debug("成功删除元素，ID: {} / Successfully removed element, ID: {}", id, id);
                    } else {
                        logger.warn("删除元素失败，ID: {} / Failed to remove element, ID: {}", id, id);
                    }
                } else {
                    logger.debug("未找到要删除的元素，ID: {} / Element to remove not found, ID: {}", id, id);
                }
            } else {
                // 对于其他集合类型，使用迭代器删除 / For other collection types, use iterator to remove
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
                                logger.debug("成功删除元素，ID: {} / Successfully removed element, ID: {}", id, id);
                                break; // 找到并删除后立即退出 / Exit immediately after finding and removing
                            }
                        } catch (Exception e) {
                            logger.warn("获取对象ID时发生错误，跳过该元素 / Error getting object ID, skipping element: {}", x.getClass().getSimpleName(), e);
                        }
                    }
                }
            }
        } catch (Exception e) {
            logger.error("删除集合元素时发生错误，ID: {} / Error removing collection element, ID: {}", id, id, e);
        }
        
        return ls;
    }
    
    /**
     * 从对象获取ID，使用缓存优化反射性能 / Get ID from object, using cache to optimize reflection performance
     *
     * @param obj 对象 / Object
     * @return ID字符串 / ID string
     * @throws Exception 如果获取失败 / If retrieval fails
     */
    private static String getIdFromObject(Object obj) throws Exception {
        if (obj == null) {
            throw new IllegalArgumentException("对象不能为空 / Object cannot be null");
        }
        
        Class<?> clazz = obj.getClass();
        
        // 先从缓存中获取方法 / Get method from cache first
        Method method = getCachedMethod(clazz);
        if (method == null) {
            // 缓存中没有，通过反射获取 / Not in cache, get through reflection
            method = clazz.getMethod("getId");
            // 缓存方法 / Cache method
            cacheMethod(clazz, method);
        }
        
        Object result = method.invoke(obj);
        return result != null ? result.toString() : "";
    }
    
    /**
     * 从缓存中获取方法 / Get method from cache
     *
     * @param clazz 类 / Class
     * @return 方法，如果不存在则返回null / Method, returns null if not exists
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
     * 缓存方法 / Cache method
     *
     * @param clazz 类 / Class
     * @param method 方法 / Method
     */
    private static void cacheMethod(Class<?> clazz, Method method) {
        cacheLock.writeLock().lock();
        try {
            // 双重检查，避免重复缓存 / Double check to avoid duplicate caching
            if (!methodCache.containsKey(clazz)) {
                methodCache.put(clazz, method);
                logger.debug("缓存方法 getId 用于类 {} / Cached getId method for class {}", clazz.getSimpleName(), clazz.getSimpleName());
            }
        } finally {
            cacheLock.writeLock().unlock();
        }
    }
    
    /**
     * 批量删除元素（按ID列表匹配）/ Batch delete elements (match by ID list)
     * 新增功能：支持批量删除，支持 CopyOnWriteArraySet / New feature: supports batch deletion and CopyOnWriteArraySet
     *
     * @param <T> 集合元素类型 / Collection element type
     * @param ls 集合 / Collection
     * @param ids ID列表 / ID list
     * @return 修改后的集合 / Modified collection
     */
    public static <T> Collection<T> deleteElementsFromCollection(Collection<T> ls, List<String> ids) {
        if (ls == null || ids == null || ids.isEmpty()) {
            logger.warn("集合或ID列表为空，无法执行批量删除操作 / Collection or ID list is null/empty, cannot perform batch delete operation");
            return ls;
        }
        
        int deletedCount = 0;
        try {
            // 检查集合类型，CopyOnWriteArraySet 需要特殊处理 / Check collection type, CopyOnWriteArraySet needs special handling
            if (ls.getClass().getName().contains("CopyOnWriteArraySet")) {
                // 对于 CopyOnWriteArraySet，收集要删除的元素然后批量删除 / For CopyOnWriteArraySet, collect elements to remove then batch delete
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
                        logger.warn("获取对象ID时发生错误，跳过该元素 / Error getting object ID, skipping element: {}", x.getClass().getSimpleName(), e);
                    }
                }
                
                // 批量删除 / Batch delete
                for (T element : elementsToRemove) {
                    if (ls.remove(element)) {
                        deletedCount++;
                    }
                }
            } else {
                // 对于其他集合类型，使用迭代器删除 / For other collection types, use iterator to remove
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
                            logger.warn("获取对象ID时发生错误，跳过该元素 / Error getting object ID, skipping element: {}", x.getClass().getSimpleName(), e);
                        }
                    }
                }
            }
            
            logger.info("批量删除完成，删除了 {} 个元素 / Batch deletion completed, removed {} elements", deletedCount, deletedCount);
        } catch (Exception e) {
            logger.error("批量删除集合元素时发生错误 / Error during batch deletion of collection elements", e);
        }
        
        return ls;
    }
    
    /**
     * 查找集合中的元素（按ID匹配）/ Find element in collection (match by ID)
     * 新增功能：支持查找元素 / New feature: supports finding elements
     *
     * @param <T> 集合元素类型 / Collection element type
     * @param ls 集合 / Collection
     * @param id 元素ID / Element ID
     * @return 找到的元素，如果不存在则返回null / Found element, returns null if not exists
     */
    public static <T> T findElementFromCollection(Collection<T> ls, String id) {
        if (ls == null || StringUtils.isBlank(id)) {
            logger.warn("集合或ID参数为空，无法执行查找操作 / Collection or ID parameter is null/blank, cannot perform find operation");
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
                            logger.debug("找到元素，ID: {} / Found element, ID: {}", id, id);
                            return x;
                        }
                    } catch (Exception e) {
                        logger.warn("获取对象ID时发生错误，跳过该元素 / Error getting object ID, skipping element: {}", x.getClass().getSimpleName(), e);
                    }
                }
            }
        } catch (Exception e) {
            logger.error("查找集合元素时发生错误，ID: {} / Error finding collection element, ID: {}", id, id, e);
        }
        
        logger.debug("未找到元素，ID: {} / Element not found, ID: {}", id, id);
        return null;
    }
    
    /**
     * 检查集合中是否包含指定ID的元素 / Check if collection contains element with specified ID
     * 新增功能：支持存在性检查 / New feature: supports existence check
     *
     * @param <T> 集合元素类型 / Collection element type
     * @param ls 集合 / Collection
     * @param id 元素ID / Element ID
     * @return 如果包含则返回true，否则返回false / Returns true if contains, false otherwise
     */
    public static <T> boolean containsElement(Collection<T> ls, String id) {
        return findElementFromCollection(ls, id) != null;
    }
    
    /**
     * 清空方法缓存 / Clear method cache
     * 新增功能：支持缓存管理 / New feature: supports cache management
     */
    public static void clearMethodCache() {
        cacheLock.writeLock().lock();
        try {
            int size = methodCache.size();
            methodCache.clear();
            logger.info("清空方法缓存，清除了 {} 个缓存项 / Cleared method cache, removed {} cache entries", size, size);
        } finally {
            cacheLock.writeLock().unlock();
        }
    }
    
    /**
     * 获取缓存统计信息 / Get cache statistics
     * 新增功能：支持缓存监控 / New feature: supports cache monitoring
     *
     * @return 缓存大小 / Cache size
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
     * 将集合转换为字符串表示 / Convert collection to string representation
     * 改进原有方法 / Improved original method
     *
     * @param <T> 集合元素类型 / Collection element type
     * @param collection 集合 / Collection
     * @return 字符串表示 / String representation
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
    
    /**
     * 主方法，用于测试 / Main method, used for testing
     *
     * @param args 命令行参数 / Command line arguments
     */
    public static void main(String args[]) {
        // 测试代码 / Test code
        logger.info("RereCollectionUtil 测试开始 / RereCollectionUtil test started");
        
        // 测试缓存功能 / Test cache functionality
        logger.info("当前缓存大小: {} / Current cache size: {}", getCacheSize(), getCacheSize());
        
        // 测试清空缓存 / Test cache clearing
        clearMethodCache();
        logger.info("清空后缓存大小: {} / Cache size after clearing: {}", getCacheSize(), getCacheSize());
    }
}