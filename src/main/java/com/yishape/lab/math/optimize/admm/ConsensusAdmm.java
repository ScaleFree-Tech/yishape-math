package com.yishape.lab.math.optimize.admm;

import com.yishape.lab.math.linalg.IVector;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.function.UnaryOperator;

/**
 * 多块<strong>共识 ADMM</strong>的通用骨架：各块局部变量 {@code w_k}、对偶 {@code y_k}、共识 {@code z}，
 * 单轮为「各块 W 步 → {@code w̄, ȳ} → 全局 Z 步 → {@code y_k ← y_k + w_k - z}」。
 * <p>
 * 与 Boyd 等共识 / 交换方向标准形式一致；具体 W/Z 子问题由调用方注入
 * {@link LocalWStep}、{@link GlobalZStep}。对角 DDML 的 LP+光滑正则实现见
 * {@link com.yishape.lab.math.ml.dml.ddml.RereDiagDmlAdmmSolver}。
 * </p>
 *
 * @see <a href="https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf">Boyd et al., ADMM</a>
 */
public final class ConsensusAdmm {

    private ConsensusAdmm() {
    }

    /**
     * 外层循环配置：罚参数 {@code ρ} 从 {@link #rhoStart()} 起步，每轮Successful后乘
     * {@link #rhoMultiplier()} 直至 {@link #rhoCeiling()}。
     */
    public record Config(
            double rhoStart,
            double rhoMultiplier,
            double rhoCeiling,
            int maxOuterIterations,
            double errorTolerance
    ) {
        public Config {
            if (maxOuterIterations < 1) {
                throw new IllegalArgumentException("maxOuterIterations 须 >= 1");
            }
            if (errorTolerance <= 0) {
                throw new IllegalArgumentException("errorTolerance 须 > 0");
            }
            if (rhoStart <= 0) {
                throw new IllegalArgumentException("rhoStart 须 > 0");
            }
            if (rhoMultiplier <= 1.0) {
                throw new IllegalArgumentException("rhoMultiplier 须 > 1");
            }
            if (rhoCeiling < rhoStart) {
                throw new IllegalArgumentException("rhoCeiling 须 >= rhoStart");
            }
        }
    }

    /** 共识 {@code z}（及可选的后处理前）与每轮标量残差序列（如 ‖w̄-z+ȳ‖²）。 */
    public record Outcome(IVector<Double> z, List<Double> errors) {
    }

    /** 单轮 ADMM 输出。 */
    public record IterationResult(IVector<Double> zNew, double error) {
    }

    /**
     * 块局部 W 步：在约束与 ADMM 罚下更新 {@code w_k}。
     *
     * @param <K> 块键类型
     * @param <B> 块数据（邻接矩阵、三元组列表、子问题参数等）
     */
    @FunctionalInterface
    public interface LocalWStep<K, B> {
        IVector<Double> apply(K blockKey, B blockData, IVector<Double> wInit,
                IVector<Double> z, IVector<Double> y, double rho);
    }

    /**
     * 全局 Z 步：给定 {@code w̄}, {@code ȳ}, {@code ρ} 与当前参与块数，更新共识 {@code z}。
     */
    @FunctionalInterface
    public interface GlobalZStep {
        IVector<Double> apply(IVector<Double> zInit, IVector<Double> wBar,
                IVector<Double> yBar, double rho, int numBlocks);
    }

    /**
     * 每轮开始时的分块：例如打乱样本后切段；返回的 {@link Map} 应用 {@link LinkedHashMap} 等保证迭代顺序稳定。
     *
     * @param <K> 块键
     * @param <B> 块负载
     */
    @FunctionalInterface
    public interface BlockPartitioner<K, B> {
        Map<K, B> partition(int outerIterationIndex, Random random);
    }

    /**
     * 首轮根据出现的块键初始化 {@code wMap}、{@code yMap}（只调用一次，在第一轮分块之后）。
     *
     * @param <K> 块键
     */
    @FunctionalInterface
    public interface InitialDualPrimalMaps<K> {
        void init(Map<K, ?> firstRoundBlocks, Map<K, IVector<Double>> wMap, Map<K, IVector<Double>> yMap, int zDim);
    }

    /**
     * 共识惩罚项 {@code ‖w - z + y‖²}（与 Julia DDML {@code census_punish_l2} 一致），作停机标量。
     */
    public static double censusPunishL2Squared(IVector<Double> w, IVector<Double> z, IVector<Double> y) {
        Objects.requireNonNull(w, "w");
        Objects.requireNonNull(z, "z");
        Objects.requireNonNull(y, "y");
        IVector<Double> d = w.sub(z).add(y);
        return d.innerProductValue(d);
    }

    /**
     * 向量算术平均。
     */
    public static IVector<Double> mean(List<IVector<Double>> rows) {
        if (rows == null || rows.isEmpty()) {
            throw new IllegalArgumentException("rows 不能为空");
        }
        IVector<Double> acc = rows.get(0).copy();
        for (int i = 1; i < rows.size(); i++) {
            acc = acc.add(rows.get(i));
        }
        double inv = 1.0 / rows.size();
        return acc.multiplyByScalar(inv);
    }

    /**
     * 执行<strong>一轮</strong>共识 ADMM：更新所有块的 {@code w_k}、共识 {@code z}、对偶 {@code y_k}。
     *
     * @param blocks   当前轮分块；键须与 {@code wMap}/{@code yMap} 中已有条目对应（首轮先 {@link InitialDualPrimalMaps#init}）
     * @param wMap     各块 primal，将被原地更新
     * @param yMap     各块对偶，将被原地更新
     * @param z        当前共识
     * @param rho      当前罚参数
     * @param localW   局部 W 步
     * @param globalZ  全局 Z 步
     * @return 新 {@code z} 与标量 {@link censusPunishL2Squared}{@code (w̄, z_new, ȳ)}
     */
    public static <K, B> IterationResult singleIteration(
            Map<K, B> blocks,
            Map<K, IVector<Double>> wMap,
            Map<K, IVector<Double>> yMap,
            IVector<Double> z,
            double rho,
            LocalWStep<K, B> localW,
            GlobalZStep globalZ) {
        Objects.requireNonNull(blocks, "blocks");
        Objects.requireNonNull(wMap, "wMap");
        Objects.requireNonNull(yMap, "yMap");
        Objects.requireNonNull(z, "z");
        Objects.requireNonNull(localW, "localW");
        Objects.requireNonNull(globalZ, "globalZ");

        Map<K, IVector<Double>> ws = new LinkedHashMap<>();
        List<IVector<Double>> ys = new ArrayList<>();
        List<IVector<Double>> wForMean = new ArrayList<>();

        for (Map.Entry<K, B> en : blocks.entrySet()) {
            K key = en.getKey();
            IVector<Double> y0 = yMap.get(key);
            IVector<Double> w0 = wMap.get(key);
            if (y0 == null || w0 == null) {
                continue;
            }
            ys.add(y0);
            IVector<Double> wt = localW.apply(key, en.getValue(), w0, z, y0, rho);
            ws.put(key, wt);
            wMap.put(key, wt);
            wForMean.add(wt);
        }

        if (wForMean.isEmpty()) {
            throw new IllegalStateException("无有效块参与本轮迭代");
        }

        IVector<Double> wBar = mean(wForMean);
        IVector<Double> yBar = mean(ys);
        int nBloc = ws.size();
        IVector<Double> zNew = globalZ.apply(z.copy(), wBar, yBar, rho, nBloc);

        for (K k : new ArrayList<>(yMap.keySet())) {
            IVector<Double> yPrev = yMap.get(k);
            IVector<Double> wp = ws.get(k);
            if (yPrev != null && wp != null) {
                yMap.put(k, yPrev.add(wp).sub(zNew));
            }
        }

        double err = censusPunishL2Squared(wBar, zNew, yBar);
        return new IterationResult(zNew, err);
    }

    /**
     * 完整外层循环：每轮由 {@link BlockPartitioner} 生成分块，调用 {@link #singleIteration}，
     * 按 {@link Config} 更新 {@code ρ}，直至残差小于阈值或达到最大轮数；最后对 {@code z} 作 {@code finalizeZ}。
     *
     * @param zInitial      初始共识
     * @param config        罚参数与停机
     * @param partitioner   每轮分块（可内含打乱）
     * @param mapInitializer 首轮写入 {@code wMap}/{@code yMap}
     * @param localW        局部 W 步
     * @param globalZ       全局 Z 步
     * @param finalizeZ     例如非负截断；若无需处理可传 {@link UnaryOperator#identity()}
     * @param random        传入 {@code partitioner}；若 {@code null} 则用每轮迭代的默认种子式 {@link Random}
     */
    public static <K, B> Outcome run(
            IVector<Double> zInitial,
            Config config,
            BlockPartitioner<K, B> partitioner,
            InitialDualPrimalMaps<K> mapInitializer,
            LocalWStep<K, B> localW,
            GlobalZStep globalZ,
            UnaryOperator<IVector<Double>> finalizeZ,
            Random random) {
        Objects.requireNonNull(zInitial, "zInitial");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(partitioner, "partitioner");
        Objects.requireNonNull(mapInitializer, "mapInitializer");
        Objects.requireNonNull(localW, "localW");
        Objects.requireNonNull(globalZ, "globalZ");
        Objects.requireNonNull(finalizeZ, "finalizeZ");

        Map<K, IVector<Double>> wMap = new LinkedHashMap<>();
        Map<K, IVector<Double>> yMap = new LinkedHashMap<>();
        IVector<Double> z = zInitial;
        double rho = config.rhoStart();
        List<Double> errors = new ArrayList<>();
        IVector<Double> zOut = z;

        int itr = 1;
        while (itr <= config.maxOuterIterations()) {
            Random rng = random != null ? random : new Random(itr);
            Map<K, B> blocks = partitioner.partition(itr, rng);
            if (itr == 1) {
                mapInitializer.init(blocks, wMap, yMap, z.length());
            }
            IterationResult up = singleIteration(blocks, wMap, yMap, z, rho, localW, globalZ);
            z = up.zNew();
            zOut = z;
            errors.add(up.error());
            if (up.error() < config.errorTolerance()) {
                break;
            }
            if (rho < config.rhoCeiling()) {
                rho *= config.rhoMultiplier();
            }
            itr++;
        }

        return new Outcome(finalizeZ.apply(zOut.copy()), errors);
    }
}
