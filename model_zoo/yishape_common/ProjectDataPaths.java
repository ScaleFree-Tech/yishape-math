package model_zoo.yishape_common;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 解析 {@code data} 目录下 CSV 的绝对路径，兼容从项目根目录或 {@code build/classes} 等目录启动时的当前工作目录差异。
 */
public final class ProjectDataPaths {

    private ProjectDataPaths() {
    }

    /**
     * @param csvFileName 例如 {@code "iris.csv"}
     * @return 存在的 {@code data/...} 绝对路径字符串
     */
    public static String resolveDataCsv(String csvFileName) {
        Path name = Paths.get(csvFileName);
        Path[] candidates = new Path[] {
            Paths.get("data").resolve(name),
            Paths.get("..", "data").resolve(name),
            Paths.get("..", "..", "data").resolve(name),
            Paths.get(System.getProperty("user.dir", ".")).resolve("data").resolve(name),
            Paths.get(System.getProperty("user.dir", ".")).resolve("..").resolve("data").resolve(name)
        };
        for (Path p : candidates) {
            Path abs = p.toAbsolutePath().normalize();
            if (Files.isRegularFile(abs)) {
                return abs.toString();
            }
        }
        throw new IllegalStateException(
                "找不到数据文件 data/" + csvFileName + "；请从 project_course 根目录运行，或确认 data 目录存在。");
    }
}
