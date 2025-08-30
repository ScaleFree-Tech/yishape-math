package com.reremouse.lab.util;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 本类对多线程操作进行统一控制，以避免在批量操作中过多混乱创建线程
 * @author RereMouse
 */
public class RereExecutor {

    /**
     * 根据CPU数量创建线程池
     */
    private static ExecutorService exec = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());    
    
    /**
     * 执行某线程
     * @param run 
     */
    public static void execute(Runnable run) {
        exec.execute(run);
    }
    
}
