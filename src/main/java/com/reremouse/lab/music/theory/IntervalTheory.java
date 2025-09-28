package com.reremouse.lab.music.theory;

/**
 * 音程理论 / Interval Theory Class
 * <p>
 * 提供音程相关的音乐理论功能
 * Provides music theory functionality related to intervals.
 * </p>
 */
public class IntervalTheory {

    /**
     * 计算音程 / Calculate interval
     * <p>
     * 计算两个音符之间的音程
     * Calculate interval between two notes.
     * </p>
     *
     * @param note1 第一个音符(0-11) / First note (0-11)
     * @param note2 第二个音符(0-11) / Second note (0-11)
     * @return 音程 (半音) / Interval (semitones)
     */
    public static int calculateInterval(int note1, int note2) {
        return (note2 - note1 + 12) % 12;
    }

    /**
     * 获取音程名称 / Get interval name
     *
     * @param semitones 半音数 / Semitones
     * @return 音程名称 / Interval name
     */
    public static String getIntervalName(int semitones) {
        String[] intervalNames = {
            "纯一度", "小二度", "大二度", "小三度", "大三度", "纯四度",
            "增四度", "纯五度", "小六度", "大六度", "小七度", "大七度"
        };
        return intervalNames[semitones % 12];
    }
}