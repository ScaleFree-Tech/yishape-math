package com.yishape.lab.math.util;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 读写 {@code .npy} 单文件格式（版本 1.0），支持 C 序、{@code &lt;f8} / {@code &lt;f4} 与 1D/2D。
 * <p>矩阵/向量优先使用 {@link IMatrix}{@code <Double>} / {@link IVector}{@code <Double>} 重载，与 {@link Linalg} 一致。</p>
 */
public final class NpyArrayIO {

    private static final byte[] MAGIC = new byte[]{(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};
    private static final Pattern HEADER_SHAPE = Pattern.compile("'shape'\\s*:\\s*\\(([^)]*)\\)");
    private static final Pattern HEADER_DESCR = Pattern.compile("'descr'\\s*:\\s*'([^']+)'");

    private NpyArrayIO() {
    }

    /**
     * 将双精度矩阵写入 {@code .npy}（与 {@link #writeDouble2D(OutputStream, double[][])} 数据布局一致）。
     */
    public static void writeDouble2D(OutputStream out, IMatrix<Double> data) throws IOException {
        Objects.requireNonNull(data, "data");
        writeDouble2D(out, data.toDoubleArray());
    }

    /**
     * 将双精度向量写入一维 {@code .npy}。
     */
    public static void writeDouble1D(OutputStream out, IVector<Double> data) throws IOException {
        Objects.requireNonNull(data, "data");
        writeDouble1D(out, data.toDoubleArray());
    }

    /**
     * 从流读取为 {@link IMatrix}{@code <Double>}（一维文件为 {@code 1×N} 矩阵）。
     */
    public static IMatrix<Double> readMatrix(InputStream in) throws IOException {
        return Linalg.matrix(readDouble2D(in));
    }

    /**
     * 从字节数组解析为矩阵（与 {@link #readMatrix(InputStream)} 一致）。
     */
    public static IMatrix<Double> fromByteArray(byte[] bytes) throws IOException {
        return readMatrix(new java.io.ByteArrayInputStream(bytes));
    }

    /**
     * 将矩阵序列化为字节数组。
     */
    public static byte[] toByteArray(IMatrix<Double> data) throws IOException {
        Objects.requireNonNull(data, "data");
        return toByteArrayDouble2D(data.toDoubleArray());
    }

    /**
     * 将二维 {@code float64} 数组写入 {@code .npy}（little-endian）。
     */
    public static void writeDouble2D(OutputStream out, double[][] data) throws IOException {
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(data, "data");
        int rows = data.length;
        int cols = data[0].length;
        String header = buildHeader("'descr': '<f8', 'fortran_order': False, 'shape': ("
                + rows + ", " + cols + "), ");
        byte[] headerBytes = padHeaderV1(header);
        out.write(MAGIC);
        out.write(1);
        out.write(0);
        int len = headerBytes.length;
        out.write(len & 0xff);
        out.write((len >> 8) & 0xff);
        out.write(headerBytes);
        ByteBuffer buf = ByteBuffer.allocate(rows * cols * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < rows; i++) {
            if (data[i].length != cols) {
                throw new IllegalArgumentException("行长度不一致 / ragged rows");
            }
            for (int j = 0; j < cols; j++) {
                buf.putDouble(data[i][j]);
            }
        }
        out.write(buf.array());
    }

    /**
     * 将一维 {@code float64} 写入 {@code .npy}。
     */
    public static void writeDouble1D(OutputStream out, double[] data) throws IOException {
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(data, "data");
        String header = buildHeader("'descr': '<f8', 'fortran_order': False, 'shape': ("
                + data.length + ", ), ");
        byte[] headerBytes = padHeaderV1(header);
        out.write(MAGIC);
        out.write(1);
        out.write(0);
        int len = headerBytes.length;
        out.write(len & 0xff);
        out.write((len >> 8) & 0xff);
        out.write(headerBytes);
        ByteBuffer buf = ByteBuffer.allocate(data.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (double v : data) {
            buf.putDouble(v);
        }
        out.write(buf.array());
    }

    private static String buildHeader(String inner) {
        return "{" + inner + "}";
    }

    /** 格式 1.0：使 10 + header 字节长度为 64 的倍数 */
    private static byte[] padHeaderV1(String dict) {
        String header = dict + "\n";
        int pad = (64 - ((10 + header.length()) % 64)) % 64;
        header = header + " ".repeat(pad);
        return header.getBytes(StandardCharsets.US_ASCII);
    }

    /**
     * 从 {@code .npy} 流读取为二维 double（若为一维则返回 1×N）。
     */
    public static double[][] readDouble2D(InputStream in) throws IOException {
        Objects.requireNonNull(in, "in");
        byte[] header = readFullHeader(in);
        String headerStr = new String(header, StandardCharsets.US_ASCII);
        Matcher mShape = HEADER_SHAPE.matcher(headerStr);
        Matcher mDescr = HEADER_DESCR.matcher(headerStr);
        if (!mDescr.find()) {
            throw new IOException("无法解析 descr / cannot parse descr");
        }
        String descr = mDescr.group(1);
        boolean f8 = "<f8".equals(descr) || "|f8".equals(descr);
        boolean f4 = "<f4".equals(descr) || "|f4".equals(descr);
        if (!f8 && !f4) {
            throw new IOException("仅支持 <f8/<f4 / only <f8 and <f4 supported");
        }
        int[] shape = parseShape(mShape);
        if (shape.length == 0) {
            throw new IOException("无法解析 shape / cannot parse shape");
        }
        int total = 1;
        for (int s : shape) {
            total *= s;
        }
        byte[] rest = in.readAllBytes();
        int expected = f8 ? total * 8 : total * 4;
        if (rest.length < expected) {
            throw new IOException("数据长度不足 / data truncated");
        }
        ByteBuffer bb = ByteBuffer.wrap(rest, 0, expected).order(ByteOrder.LITTLE_ENDIAN);
        double[] flat = new double[total];
        for (int i = 0; i < total; i++) {
            flat[i] = f8 ? bb.getDouble() : bb.getFloat();
        }
        if (shape.length == 1) {
            double[][] row = new double[1][shape[0]];
            row[0] = flat;
            return row;
        }
        if (shape.length == 2) {
            int r = shape[0];
            int c = shape[1];
            double[][] m = new double[r][c];
            int k = 0;
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    m[i][j] = flat[k++];
                }
            }
            return m;
        }
        throw new IOException("仅支持 1D/2D shape / only 1D and 2D supported");
    }

    private static int[] parseShape(Matcher mShape) {
        if (!mShape.find()) {
            return new int[0];
        }
        String inner = mShape.group(1).trim();
        if (inner.isEmpty()) {
            return new int[0];
        }
        String[] parts = inner.split(",");
        int[] tmp = new int[parts.length];
        int n = 0;
        for (String p : parts) {
            p = p.trim();
            if (!p.isEmpty()) {
                tmp[n++] = Integer.parseInt(p);
            }
        }
        return Arrays.copyOf(tmp, n);
    }

    private static byte[] readFullHeader(InputStream in) throws IOException {
        byte[] magic = in.readNBytes(6);
        if (magic.length < 6 || !Arrays.equals(Arrays.copyOf(magic, 6), MAGIC)) {
            throw new IOException("非 .npy 文件 / not a .npy file");
        }
        int major = in.read();
        in.read(); /* minor */
        if (major != 1) {
            throw new IOException("仅支持 .npy 格式版本 1.0 / only .npy format version 1.0");
        }
        int hlen = in.read() | (in.read() << 8);
        return in.readNBytes(hlen);
    }

    /**
     * 将二维数组写入字节数组（便于测试或序列化）。
     */
    public static byte[] toByteArrayDouble2D(double[][] data) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        writeDouble2D(bos, data);
        return bos.toByteArray();
    }
}
