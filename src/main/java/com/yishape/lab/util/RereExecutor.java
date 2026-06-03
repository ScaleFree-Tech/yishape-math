package com.yishape.lab.util;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 本类对多线程操作进行统一控制，以避免在批量操作中过度混乱创建线程
 * This class provides unified control for multi-threading operations to avoid excessive chaotic thread creation in batch operations
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereExecutor {

    /**
     * 线程池，根据CPU数量创建，使用 daemon 线程避免阻止 JVM 退出 /
     * Thread pool, created based on CPU count, using daemon threads to avoid blocking JVM shutdown
     */
    private static ExecutorService exec = Executors.newFixedThreadPool(
        Runtime.getRuntime().availableProcessors(),
        new ThreadFactory() {
            private final AtomicInteger count = new AtomicInteger(0);
            @Override
            public Thread newThread(Runnable r) {
                Thread t = new Thread(r, "yishape-executor-" + count.incrementAndGet());
                t.setDaemon(true);
                return t;
            }
        }
    );

    /**
     * 执行某线程 / Execute a thread
     * @param run 要执行的任务 / Task to execute
     */
    public static void execute(Runnable run) {
        exec.execute(run);
    }
}