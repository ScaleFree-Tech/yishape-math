package com.yishape.lab.util;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 本类对多线程操作进行统一控制，以避免在批量操作中过度混乱创建线程
 * This class provides unified control for multi-threading operations to avoid excessive chaotic thread creation in batch operations
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereExecutor {

    /**
     * 线程池，根据CPU数量创建 / Thread pool, created based on CPU count
     */
    private static ExecutorService exec = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    
    /**
     * 执行某线程 / Execute a thread
     * @param run 要执行的任务 / Task to execute
     */
    public static void execute(Runnable run) {
        exec.execute(run);
    }
}