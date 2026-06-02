package com.yishape.lab.math.ml;

import com.yishape.lab.math.ml.dml.ISupervisedDml;
import com.yishape.lab.math.ml.dml.IUnsupervisedDml;
import com.yishape.lab.math.ml.dml.KernelDmlUtils.KernelType;
import com.yishape.lab.math.ml.dml.anmm.AnmmDml;
import com.yishape.lab.math.ml.dml.anmm.KanmmDml;
import com.yishape.lab.math.ml.dml.cmoml.CmomlDml;
import com.yishape.lab.math.ml.dml.ddml.RereDiagDml;
import com.yishape.lab.math.ml.dml.dmleig.DmleigDml;
import com.yishape.lab.math.ml.dml.dmlmj.DmlmjDml;
import com.yishape.lab.math.ml.dml.dmlmj.KDmlmjDml;
import com.yishape.lab.math.ml.dml.fisher.FisherDml;
import com.yishape.lab.math.ml.dml.gmml.GmmlDml;
import com.yishape.lab.math.ml.dml.itml.ItmlDml;
import com.yishape.lab.math.ml.dml.kda.KdaDml;
import com.yishape.lab.math.ml.dml.knn.CondensedNearestNeighbors;
import com.yishape.lab.math.ml.dml.knn.ReducedNearestNeighbors;
import com.yishape.lab.math.ml.dml.ldml.LdmlPairwiseDml;
import com.yishape.lab.math.ml.dml.llda.KLldaDml;
import com.yishape.lab.math.ml.dml.llda.LldaDml;
import com.yishape.lab.math.ml.dml.lmnn.KlmmnDml;
import com.yishape.lab.math.ml.dml.lmnn.LmnnDml;
import com.yishape.lab.math.ml.dml.lsi.LsiDml;
import com.yishape.lab.math.ml.dml.lsi.LsiMmcDml;
import com.yishape.lab.math.ml.dml.mcml.McmlDml;
import com.yishape.lab.math.ml.dml.multidml.MultiDmlKnn;
import com.yishape.lab.math.ml.dml.nca.NcaDml;
import com.yishape.lab.math.ml.dml.ncmc.NcmcDml;
import com.yishape.lab.math.ml.dml.ncmml.NcmmlDml;
import com.yishape.lab.math.ml.dml.odml.KodmlDml;
import com.yishape.lab.math.ml.dml.odml.OdmlDml;
import com.yishape.lab.math.ml.dml.rca.RcaDml;
import com.yishape.lab.math.ml.dml.withinclass.WithinClassDml;

/**
 * 距离度量学习（DML）工厂：与 {@link ClfWrapper} 并列，作为 {@link ML#dml} 的<strong>唯一推荐入口</strong>。
 *
 * <h2>设计约定（与分类 / 降维 Wrapper 一致）</h2>
 * <ol>
 *   <li><strong>对外类型</strong>：所有工厂方法仅声明返回 {@link ISupervisedDml}，不暴露具体实现类名。</li>
 *   <li><strong>默认与重载</strong>：无参版本 = 库内审慎默认；<strong>对结果影响大、且在教程里常被调的内容</strong>通过方法重载显式传入
 *       （与 {@code logisticRegression(l1,l2)}、{@code kNN(k)} 同级）。命名上，L2 / Ridge 统一为 {@code l2Weight}（与逻辑回归第二个惩罚系数同口径）。</li>
 *   <li><strong>何时只用本类</strong>：业务代码、流水线、需以多态持有「某一种度量学习器」时——一律
 *       {@code ML.dml.xxx(...)}。</li>
 *   <li><strong>何时 {@code new} 实现类</strong>：仅当需要<strong>尚未在本 Wrapper 中设重载或文档未承诺覆盖</strong>的调参项
 *       （如 ITML 约束对条数、DDML 的 ADMM 块大小、随机种子等）时，在 {@code com.yishape.lab.math.ml.dml.*} 子包中
 *       {@code new FooMetricLearning().setRareKnob(...).fit(...)}；实现类仍实现 {@link ISupervisedDml}，
 *       可与门面产出类型互换。</li>
 *   <li><strong>静态 {@code Algorithm.fit(...)}</strong>：各算法类保留的静态 {@code fit} 仅为<strong>脚本一行调用</strong>便利，
 *       新代码首选本 Wrapper 或实例 {@code new + set + fit}。</li>
 * </ol>
 *
 * @see ML#dml
 * @see com.yishape.lab.math.ml.dml
 * @see ISupervisedDml
 */
public final class DmlWrapper {

    /**
     * Fisher / LDA 式类内白化度量，默认 L2 岭 {@code l2Weight = 1e-3}。
     *
     * @return 监督度量学习器
     * @see FisherDml
     */
    public ISupervisedDml fisherWhitening() {
        return new FisherDml();
    }

    /**
     * Fisher / LDA 式类内白化，指定加在 {@code S_w} 对角上的 L2 岭强度 {@code l2Weight}（须为正）；
     * 与 {@link ClfWrapper#logisticRegression(double, double)} 中第二个参数 {@code l2Weight} 同属 Ridge 口径。
     *
     * @param l2Weight L2 正则（岭）强度
     * @return 监督度量学习器
     */
    public ISupervisedDml fisherWhitening(double l2Weight) {
        return new FisherDml().setL2Weight(l2Weight);
    }

    /**
     * RCA chunklet 白化，默认 L2 岭 {@code l2Weight = 1e-2}。
     *
     * @return 监督度量学习器
     * @see RcaDml
     */
    public ISupervisedDml rca() {
        return new RcaDml();
    }

    /**
     * RCA，指定 chunk 内协方差对角上的 L2 岭 {@code l2Weight}（须为正）。
     *
     * @param l2Weight L2 岭强度
     * @return 监督度量学习器
     */
    public ISupervisedDml rca(double l2Weight) {
        return new RcaDml().setL2Weight(l2Weight);
    }

    /**
     * 类内方差对角缩放，默认 L2 岭 {@code l2Weight = 1.0}。
     *
     * @return 监督度量学习器
     */
    public ISupervisedDml withinClass() {
        return new WithinClassDml();
    }

    /**
     * 类内方差对角缩放，指定逐维方差上的 L2 岭 {@code l2Weight}（须为正）。
     *
     * @param l2Weight L2 岭强度
     * @return 监督度量学习器
     */
    public ISupervisedDml withinClass(double l2Weight) {
        return new WithinClassDml().setL2Weight(l2Weight);
    }

    /**
     * ITML（Bregman–LogDet），采用库内默认 {@code γ}、迭代与先验。
     *
     * @return 监督度量学习器
     * @see ItmlDml
     */
    public ISupervisedDml itml() {
        return new ItmlDml();
    }

    /**
     * ITML，指定松弛 {@code γ} 与外层迭代上限（与收敛判据配合，是最常用的两项）。
     *
     * @param gamma   约束松弛（对应实现中投影步）
     * @param maxIter 最大迭代轮数
     * @return 监督度量学习器
     */
    public ISupervisedDml itml(double gamma, int maxIter) {
        return new ItmlDml().setGamma(gamma).setMaxIter(maxIter);
    }

    /**
     * ITML，指定松弛、迭代上限与先验类型（恒等或逆协方差）。
     *
     * @param gamma     约束松弛
     * @param maxIter   最大迭代轮数
     * @param priorKind {@link ItmlDml.PriorKind}
     * @return 监督度量学习器
     */
    public ISupervisedDml itml(double gamma, int maxIter, ItmlDml.PriorKind priorKind) {
        return new ItmlDml().setGamma(gamma).setMaxIter(maxIter).setPriorKind(priorKind);
    }

    /**
     * ITML，附带指定逆协方差先验在对角上的 L2 岭 {@code priorL2Weight}（仅当 {@code priorKind} 为
     * {@link ItmlDml.PriorKind#INVERSE_COVARIANCE} 时进入求逆；与全库 {@code l2Weight} 同 Ridge 口径）。
     *
     * @param gamma         约束松弛
     * @param maxIter       最大迭代轮数
     * @param priorKind     先验种类
     * @param priorL2Weight 样本协方差对角扰动（须非负）
     * @return 监督度量学习器
     */
    public ISupervisedDml itml(double gamma, int maxIter, ItmlDml.PriorKind priorKind,
            double priorL2Weight) {
        return new ItmlDml()
                .setGamma(gamma)
                .setMaxIter(maxIter)
                .setPriorKind(priorKind)
                .setPriorL2Weight(priorL2Weight);
    }

    /**
     * NCA 低秩嵌入，默认秩、迭代与学习率。
     *
     * @return 监督度量学习器
     * @see NcaDml
     */
    public ISupervisedDml nca() {
        return new NcaDml();
    }

    /**
     * NCA，指定嵌入秩、梯度迭代轮数与学习率（直接决定优化行为）。
     *
     * @param rank         低秩 {@code L} 的行数
     * @param maxIter      迭代轮数
     * @param learningRate 步长
     * @return 监督度量学习器
     */
    public ISupervisedDml nca(int rank, int maxIter, double learningRate) {
        return new NcaDml().setRank(rank).setMaxIter(maxIter).setLearningRate(learningRate);
    }

    /**
     * MCML 低秩嵌入，默认超参。
     *
     * @return 监督度量学习器
     * @see McmlDml
     */
    public ISupervisedDml mcml() {
        return new McmlDml();
    }

    /**
     * MCML，指定嵌入秩、迭代与学习率。
     *
     * @param rank         低秩维
     * @param maxIter      迭代轮数
     * @param learningRate 步长
     * @return 监督度量学习器
     */
    public ISupervisedDml mcml(int rank, int maxIter, double learningRate) {
        return new McmlDml().setRank(rank).setMaxIter(maxIter).setLearningRate(learningRate);
    }

    /**
     * 成对 logistic（LDML 风格）低秩，默认秩与步数。
     *
     * @return 监督度量学习器
     * @see LdmlPairwiseDml
     */
    public ISupervisedDml ldmlPairwise() {
        return new LdmlPairwiseDml();
    }

    /**
     * 成对 logistic，指定嵌入秩与随机 SGD 步数。
     *
     * @param rank      低秩维
     * @param maxSteps  配对抽样更新步数
     * @return 监督度量学习器
     */
    public ISupervisedDml ldmlPairwise(int rank, int maxSteps) {
        return new LdmlPairwiseDml().setRank(rank).setMaxSteps(maxSteps);
    }

    /**
     * 成对 logistic，指定秩、步数与学习率。
     *
     * @param rank         低秩维
     * @param maxSteps     SGD 步数
     * @param learningRate 步长
     * @return 监督度量学习器
     */
    public ISupervisedDml ldmlPairwise(int rank, int maxSteps, double learningRate) {
        return new LdmlPairwiseDml().setRank(rank).setMaxSteps(maxSteps).setLearningRate(learningRate);
    }

    /**
     * LMNN 风格三元组大间隔（低秩 SGD，非全秩 SDP），默认 margin 与步数。
     *
     * @return 监督度量学习器
     * @see LmnnDml
     */
    public ISupervisedDml lmnn() {
        return new LmnnDml();
    }

    /**
     * 三元组大间隔，指定嵌入秩、欧氏空间目标近邻数与 hinge 裕量。
     *
     * @param rank            低秩 {@code L} 行数
     * @param targetNeighbors 每锚点同类近邻数
     * @param margin          hinge 间隔
     * @return 监督度量学习器
     */
    public ISupervisedDml lmnn(int rank, int targetNeighbors, double margin) {
        return new LmnnDml()
                .setRank(rank)
                .setTargetNeighbors(targetNeighbors)
                .setMargin(margin);
    }

    /**
     * 三元组大间隔，额外指定 SGD 步数与学习率。
     *
     * @param rank            低秩维
     * @param targetNeighbors 目标近邻数
     * @param margin          hinge 间隔
     * @param maxSteps        SGD 更新上限
     * @param learningRate    步长
     * @return 监督度量学习器
     */
    public ISupervisedDml lmnn(int rank, int targetNeighbors, double margin,
            int maxSteps, double learningRate) {
        return new LmnnDml()
                .setRank(rank)
                .setTargetNeighbors(targetNeighbors)
                .setMargin(margin)
                .setMaxSteps(maxSteps)
                .setLearningRate(learningRate);
    }

    /**
     * 对角 DDML，无正则时走 LP；否则由 {@code l1Weight}/{@code l2Weight} 推断 L1、L2 或弹性网路径（与
     * {@link com.yishape.lab.math.ml.reg.RereLinearRegression} 对 {@code lambda1}/{@code lambda2} 的规则一致）。
     *
     * @return 监督度量学习器
     * @see RereDiagDml
     */
    public ISupervisedDml diagDml() {
        return new RereDiagDml();
    }

    /**
     * 对角 DDML：单参数表示<strong>等量弹性网</strong>总强度 {@code W}（等价于 {@code l1Weight=l2Weight=W/2}），
     * 与原先 {@link RereDiagDml} 仅设 {@code regWeight=W} 且默认混合系数 {@code 0.5} 时的行为一致；{@code W=0} 为无正则。
     *
     * @param elasticTotalWeight 非负；{@code 0} 表示无正则 LP 路径
     * @return 
     */
    public ISupervisedDml diagDml(double elasticTotalWeight) {
        if (elasticTotalWeight < 0) {
            throw new IllegalArgumentException("elasticTotalWeight 须 >= 0");
        }
        double half = 0.5 * elasticTotalWeight;
        return new RereDiagDml().setL1Weight(half).setL2Weight(half);
    }

    /**
     * 对角 DDML，指定 L1 与 L2 权重（均为 0 则无正则；仅其一非 0 为纯 L1/L2；均正为弹性网）。
     *
     * @param l1Weight L1 系数
     * @param l2Weight L2 系数
     * @return 
     */
    public ISupervisedDml diagDml(double l1Weight, double l2Weight) {
        return new RereDiagDml().setRegularization(l1Weight, l2Weight);
    }

    /**
     * 对角 DDML，在上述基础上限制三元组条数上限。
     *
     * @param l1Weight    L1 系数
     * @param l2Weight    L2 系数
     * @param maxTriplets 参与建模的三元组数量上限
     * @return
     */
    public ISupervisedDml diagDml(double l1Weight, double l2Weight, int maxTriplets) {
        return new RereDiagDml()
                .setRegularization(l1Weight, l2Weight)
                .setMaxTriplets(maxTriplets);
    }

    // ==================== DML-eig ====================

    /**
     * DML-eig：通过广义特征值优化学习马氏度量，最小化异类样本间的软间隔。
     *
     * @return 监督度量学习器
     * @see DmleigDml
     */
    public ISupervisedDml dmleig() {
        return new DmleigDml();
    }

    /**
     * DML-eig，指定正则化 {@code mu} 与最大迭代次数。
     *
     * @param mu      正则化强度
     * @param maxIter 最大外层迭代次数
     * @return 监督度量学习器
     */
    public ISupervisedDml dmleig(double mu, int maxIter) {
        return new DmleigDml().setMu(mu).setMaxIter(maxIter);
    }

    // ==================== GMML ====================

    /**
     * Geometric Mean Metric Learning（GMML），默认参数。
     *
     * @return 监督度量学习器
     * @see GmmlDml
     */
    public ISupervisedDml gmml() {
        return new GmmlDml();
    }

    /**
     * GMML，指定测地线步长与正则化强度。
     *
     * @param geodesicStep 测地线插值步长（须在 (0,1) 区间）
     * @param reg         正则化参数
     * @return 监督度量学习器
     */
    public ISupervisedDml gmml(double geodesicStep, double reg) {
        return new GmmlDml().setGeodesicStep(geodesicStep).setReg(reg);
    }

    // ==================== DML-MJ ====================

    /**
     * DML-MJ（Jeffrey 散度最大化），默认参数。
     *
     * @return 监督度量学习器
     * @see DmlmjDml
     */
    public ISupervisedDml dmlmj() {
        return new DmlmjDml();
    }

    /**
     * DML-MJ，指定输出维数与近邻参数。
     *
     * @param numDims   输出维数（null 表示自动）
     * @param nNeighbors 近邻数量
     * @return 监督度量学习器
     */
    public ISupervisedDml dmlmj(Integer numDims, int nNeighbors) {
        return new DmlmjDml().setNumDims(numDims).setNNeighbors(nNeighbors);
    }

    // ==================== Kernel DML-MJ ====================

    /**
     * 核化 DML-MJ，默认 RBF 核。
     *
     * @return 监督度量学习器
     * @see KDmlmjDml
     */
    public ISupervisedDml kdmlmj() {
        return new KDmlmjDml();
    }

    /**
     * 核化 DML-MJ，指定核类型与 gamma。
     *
     * @param kernelType 核类型
     * @param gamma      RBF 核参数
     * @return 监督度量学习器
     */
    public ISupervisedDml kdmlmj(KernelType kernelType, double gamma) {
        return new KDmlmjDml().setKernelType(kernelType).setGamma(gamma);
    }

    // ==================== CMOML ====================

    /**
     * Class Mean Metric Learning（CMOML），默认参数。
     *
     * @return 监督度量学习器
     * @see CmomlDml
     */
    public ISupervisedDml cmoml() {
        return new CmomlDml();
    }

    /**
     * CMOML，指定输出维数与正则化。
     *
     * @param numDims 输出维数（null 表示自动）
     * @param reg     正则化参数
     * @return 监督度量学习器
     */
    public ISupervisedDml cmoml(Integer numDims, double reg) {
        return new CmomlDml().setNumDims(numDims).setReg(reg);
    }

    // ==================== ANMM ====================

    /**
     * Average Neighborhood Margin Maximization（ANMM），默认参数。
     *
     * @return 监督度量学习器
     * @see AnmmDml
     */
    public ISupervisedDml anmm() {
        return new AnmmDml();
    }

    /**
     * ANMM，指定输出维数与近邻配置。
     *
     * @param numDims   输出维数（null 表示自动）
     * @param nFriends  同类近邻数量
     * @param nEnemies  异类近邻数量
     * @return 监督度量学习器
     */
    public ISupervisedDml anmm(Integer numDims, int nFriends, int nEnemies) {
        return new AnmmDml().setNumDims(numDims).setNFriends(nFriends).setNEnemies(nEnemies);
    }

    // ==================== Kernel ANMM ====================

    /**
     * 核化 ANMM，默认参数。
     *
     * @return 监督度量学习器
     * @see KanmmDml
     */
    public ISupervisedDml kanmm() {
        return new KanmmDml();
    }

    /**
     * 核化 ANMM，指定核类型、gamma 与近邻数。
     *
     * @param kernelType 核类型
     * @param gamma      RBF 核参数
     * @param k          近邻数量
     * @return 监督度量学习器
     */
    public ISupervisedDml kanmm(KernelType kernelType, double gamma, int k) {
        return new KanmmDml().setKernelType(kernelType).setGamma(gamma).setK(k);
    }

    // ==================== LLDA ====================

    /**
     * Local Fisher Discriminant Analysis（LLDA），默认参数。
     *
     * @return 监督度量学习器
     * @see LldaDml
     */
    public ISupervisedDml llda() {
        return new LldaDml();
    }

    /**
     * LLDA，指定输出维数与求解器。
     *
     * @param nComponents 输出维数
     * @param solver     求解器类型（SUGIYAMA 或 CLASSIC）
     * @return 监督度量学习器
     */
    public ISupervisedDml llda(int nComponents, LldaDml.SolverType solver) {
        return new LldaDml().setNComponents(nComponents).setSolver(solver);
    }

    // ==================== Kernel LLDA ====================

    /**
     * 核化 LLDA，默认 RBF 核。
     *
     * @return 监督度量学习器
     * @see KLldaDml
     */
    public ISupervisedDml kllda() {
        return new KLldaDml();
    }

    /**
     * 核化 LLDA，指定核参数与输出维数。
     *
     * @param kernelType   核类型
     * @param gamma        RBF 核参数
     * @param nComponents  输出维数
     * @return 监督度量学习器
     */
    public ISupervisedDml kllda(KernelType kernelType, double gamma, int nComponents) {
        return new KLldaDml().setKernelType(kernelType).setGamma(gamma).setNComponents(nComponents);
    }

    // ==================== KDA ====================

    /**
     * Kernel Discriminant Analysis（KDA），默认 RBF 核。
     *
     * @return 监督度量学习器
     * @see KdaDml
     */
    public ISupervisedDml kda() {
        return new KdaDml();
    }

    /**
     * KDA，指定核类型、gamma 与输出维数。
     *
     * @param kernelType   核类型
     * @param gamma        RBF 核参数
     * @param nComponents  输出维数（不超过类别数-1）
     * @return 监督度量学习器
     */
    public ISupervisedDml kda(KernelType kernelType, double gamma, int nComponents) {
        return new KdaDml().setKernelType(kernelType).setGamma(gamma).setNComponents(nComponents);
    }

    // ==================== Kernel LMNN ====================

    /**
     * 核化大间隔最近邻（KLMNN），默认参数。
     *
     * @return 监督度量学习器
     * @see KlmmnDml
     */
    public ISupervisedDml klmnn() {
        return new KlmmnDml();
    }

    /**
     * KLMNN，指定核类型、gamma、秩与间隔。
     *
     * @param kernelType 核类型
     * @param gamma      RBF 核参数
     * @param rank       低秩维数
     * @param margin     hinge 间隔
     * @return 监督度量学习器
     */
    public ISupervisedDml klmnn(KernelType kernelType, double gamma, int rank, double margin) {
        return new KlmmnDml().setKernelType(kernelType).setGamma(gamma)
                .setRank(rank).setMargin(margin);
    }

    // ==================== Online DML ====================

    /**
     * Online Distance Metric Learning（ODML），默认参数。
     *
     * @return 监督度量学习器
     * @see OdmlDml
     */
    public ISupervisedDml odml() {
        return new OdmlDml();
    }

    /**
     * ODML，指定学习率与攻击参数。
     *
     * @param learningRate 学习率
     * @param aggression  攻击参数
     * @return 监督度量学习器
     */
    public ISupervisedDml odml(double learningRate, double aggression) {
        return new OdmlDml().setLearningRate(learningRate).setAggression(aggression);
    }

    // ==================== Kernel Online DML ====================

    /**
     * 核化在线 DML（KODML），默认参数。
     *
     * @return 监督度量学习器
     * @see KodmlDml
     */
    public ISupervisedDml kodml() {
        return new KodmlDml();
    }

    /**
     * KODML，指定核类型与学习率。
     *
     * @param kernelType  核类型
     * @param gamma       RBF 核参数
     * @param learningRate 学习率
     * @return 监督度量学习器
     */
    public ISupervisedDml kodml(KernelType kernelType, double gamma, double learningRate) {
        return new KodmlDml().setKernelType(kernelType).setGamma(gamma)
                .setLearningRate(learningRate);
    }

    // ==================== NCMC ====================

    /**
     * Nearest Class with Multiple Centroids（NCMC），默认参数。
     *
     * @return 监督度量学习器
     * @see NcmcDml
     */
    public ISupervisedDml ncmc() {
        return new NcmcDml();
    }

    /**
     * NCMC，指定每类中心数与最大迭代次数。
     *
     * @param centroidsNum 每类中心数量
     * @param maxIter      最大迭代次数
     * @return 监督度量学习器
     */
    public ISupervisedDml ncmc(int centroidsNum, int maxIter) {
        return new NcmcDml().setCentroidsNum(centroidsNum).setMaxIter(maxIter);
    }

    // ==================== NCMML ====================

    /**
     * Nearest Class Mean Metric Learning（NCMML），默认参数。
     *
     * @return 监督度量学习器
     * @see NcmmlDml
     */
    public ISupervisedDml ncmml() {
        return new NcmmlDml();
    }

    /**
     * NCMML，指定输出维数与最大迭代次数。
     *
     * @param numDims  输出维数（null 表示自动）
     * @param maxIter  最大迭代次数
     * @return 监督度量学习器
     */
    public ISupervisedDml ncmml(Integer numDims, int maxIter) {
        return new NcmmlDml().setNumDims(numDims).setMaxIter(maxIter);
    }

    // ==================== LSI ====================

    /**
     * Locality Sensitive Indexing（LSI），局部保持投影，无需标签即可训练。
     *
     * @return 监督度量学习器（标签仅用于验证，不参与训练）
     * @see LsiDml
     */
    public IUnsupervisedDml lsi() {
        return new LsiDml();
    }

    /**
     * LSI，指定输出维数与近邻数量。
     *
     * @param nComponents 输出维数
     * @param nNeighbors 近邻数量
     * @return 非监督度量学习器
     */
    public IUnsupervisedDml lsi(int nComponents, int nNeighbors) {
        return new LsiDml().setNComponents(nComponents).setNNeighbors(nNeighbors);
    }

    /**
     * Mahalanobis Metric for Clustering (MMC)：监督型马氏度量学习。
     * 最小化同类样本对距离，最大化异类样本对距离（对数势垒约束）。
     * 学习一个正半定（PSD）马氏度量矩阵 A。
     *
     * @return 监督度量学习器
     * @see LsiMmcDml
     */
    public ISupervisedDml lsiMmc() {
        return new LsiMmcDml();
    }

    /**
     * MMC，指定最大迭代次数与收敛容忍度。
     *
     * @param maxIter    最大迭代次数
     * @param tolerance  收敛容忍度
     * @return 监督度量学习器
     */
    public ISupervisedDml lsiMmc(int maxIter, double tolerance) {
        return new LsiMmcDml().setMaxIter(maxIter).setTolerance(tolerance);
    }

    // ==================== CNN / RNN ====================

    /**
     * Condensed Nearest Neighbors（CNN）：对已标注样本进行压缩，返回的度量矩阵为全 1 常矩阵，
     * 实际距离计算完全依赖最近邻查表。
     *
     * @return 监督度量学习器（度量恒为 1）
     * @see CondensedNearestNeighbors
     */
    public ISupervisedDml cnn() {
        return new CondensedNearestNeighbors();
    }

    /**
     * Reduced Nearest Neighbors（RNN）：在 CNN 基础上进一步移除冗余样本。
     *
     * @return 监督度量学习器（度量恒为 1）
     * @see ReducedNearestNeighbors
     */
    public ISupervisedDml rnn() {
        return new ReducedNearestNeighbors();
    }

    // ==================== Multi-DML KNN ====================

    /**
     * 多 DML 集成 KNN：同时训练多个 DML 算法，通过投票进行预测。
     *
     * @param dmls 要集成的 DML 算法数组
     * @return 监督度量学习器（集成）
     * @see MultiDmlKnn
     */
    public ISupervisedDml multiDmlKnn(ISupervisedDml... dmls) {
        MultiDmlKnn multi = new MultiDmlKnn();
        for (ISupervisedDml dml : dmls) {
            multi.addDml(dml);
        }
        return multi;
    }
}
