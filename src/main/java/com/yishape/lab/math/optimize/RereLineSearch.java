package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IVector;
import java.io.Serializable;

/**
 * 线搜索：强 Wolfe 条件（Armijo + 曲率），含步长扩张与 zoom 阶段。
 * Strong Wolfe line search with bracket expansion and zoom (Nocedal &amp; Wright; SciPy scalar_search_wolfe2).
 *
 * <p>旧实现仅按固定因子缩小 {@code alpha}，无法在曲率不满足时增大步长，在 Rosenbrock 谷底等处易停滞。</p>
 *
 * @author lteb2
 */
public class RereLineSearch implements Serializable {

    private static final long serialVersionUID = 1L;

    private double c1 = 1e-4;
    private double c2 = 0.9;
    private double initialStepSize = 1.0;
    private int maxLineSearchIterations = 40;
    private int maxZoomIterations = 30;

    /** 为 true 时使用强 Wolfe + zoom（推荐于 L-BFGS）；非线性共轭梯度可与旧回溯行为更合拍 */
    private boolean strongWolfeEnabled = true;

    private static final double ZOOM_DELTA1 = 0.2;
    private static final double ZOOM_DELTA2 = 0.1;

    public RereLineSearch() {
    }

    public RereLineSearch(double c1, double c2, double initialStepSize) {
        this.c1 = c1;
        this.c2 = c2;
        this.initialStepSize = initialStepSize;
    }

    public boolean isStrongWolfeEnabled() {
        return strongWolfeEnabled;
    }

    public void setStrongWolfeEnabled(boolean strongWolfeEnabled) {
        this.strongWolfeEnabled = strongWolfeEnabled;
    }

    /**
     * @param phi0       {@code f(x)}
     * @param oldPhi0    上一外层迭代的 {@code f}；首次迭代传 {@code null}
     * @param derphi0    {@code ∇f(x)·d}（须 &lt; 0）
     */
    public double searchWithCachedDerivative(IVector x, IVector direction, IObjectiveFunction objFun,
            IGradientFunction grdFun, IVector grad, double phi0, Double oldPhi0,
            double derphi0) {
        return strongWolfe(x, direction, objFun, grdFun, phi0, oldPhi0, derphi0);
    }

    public double searchWithCachedDerivative(IVector x, IVector direction, IObjectiveFunction objFun,
            IGradientFunction grdFun, IVector grad, double cachedDirectionalDerivative) {
        double phi0 = objFun.computeObjective(x);
        return strongWolfe(x, direction, objFun, grdFun, phi0, null, cachedDirectionalDerivative);
    }

    public double search(IVector x, IVector direction, IObjectiveFunction objFun,
            IGradientFunction grdFun, IVector grad) {
        double phi0 = objFun.computeObjective(x);
        double derphi0 = grad.innerProductValue(direction);
        if (!strongWolfeEnabled) {
            return legacyBacktracking(x, direction, objFun, grdFun, grad, phi0, derphi0);
        }
        return strongWolfe(x, direction, objFun, grdFun, phi0, null, derphi0);
    }

    /**
     * 仅缩小步长的回溯 + 曲率检验（历史实现）。部分算法（如非线性 CG）与此配合更稳定。
     */
    private double legacyBacktracking(IVector x, IVector direction, IObjectiveFunction objFun,
            IGradientFunction grdFun, IVector grad, double phi0, double directionalDerivative) {

        if (directionalDerivative >= 0) {
            double alpha = 1e-8;
            IVector newX = x.add(direction.multiplyByScalar(alpha));
            double newValue = objFun.computeObjective(newX);
            if (newValue < phi0) {
                return alpha;
            }
            return 1e-12;
        }

        double alpha = initialStepSize;
        for (int i = 0; i < maxLineSearchIterations; i++) {
            IVector newX = x.add(direction.multiplyByScalar(alpha));
            double newValue = objFun.computeObjective(newX);
            if (newValue <= phi0 + c1 * alpha * directionalDerivative) {
                IVector newGrad = grdFun.computeGradient(newX);
                double newDirectionalDerivative = newGrad.innerProductValue(direction);
                if (Math.abs(newDirectionalDerivative) <= c2 * Math.abs(directionalDerivative)) {
                    return alpha;
                }
            }
            alpha *= 0.5;
            if (alpha < 1e-20) {
                break;
            }
        }
        return Math.min(initialStepSize, Math.max(1e-8, alpha));
    }

    private double strongWolfe(IVector x, IVector direction, IObjectiveFunction objFun,
            IGradientFunction grdFun, double phi0, Double oldPhi0, double derphi0) {

        if (derphi0 >= 0) {
            double alpha = 1e-8;
            IVector newX = x.add(direction.multiplyByScalar(alpha));
            double newVal = objFun.computeObjective(newX);
            if (newVal < phi0) {
                return alpha;
            }
            return 1e-12;
        }

        double alpha0 = 0.0;
        double alpha1;
        if (oldPhi0 != null && derphi0 != 0) {
            alpha1 = Math.min(initialStepSize, 1.01 * 2.0 * (phi0 - oldPhi0) / derphi0);
        } else {
            alpha1 = initialStepSize;
        }
        if (alpha1 < 0) {
            alpha1 = initialStepSize;
        }
        if (alpha1 == 0) {
            return armijoFallback(x, direction, objFun, phi0, derphi0);
        }

        double phiA0 = phi0;
        double derphiA0 = derphi0;
        double phiA1 = evalPhi(x, direction, objFun, alpha1);

        for (int i = 0; i < maxLineSearchIterations; i++) {
            boolean notFirst = i > 0;
            if (phiA1 > phi0 + c1 * alpha1 * derphi0 || (phiA1 >= phiA0 && notFirst)) {
                double star = zoom(alpha0, alpha1, phiA0, phiA1, derphiA0, x, direction, objFun, grdFun,
                    phi0, derphi0);
                if (!Double.isNaN(star)) {
                    return star;
                }
                break;
            }

            double derphiA1 = evalDerphi(x, direction, grdFun, alpha1);
            if (Math.abs(derphiA1) <= -c2 * derphi0) {
                return alpha1;
            }
            if (derphiA1 >= 0.0) {
                double star = zoom(alpha1, alpha0, phiA1, phiA0, derphiA1, x, direction, objFun, grdFun,
                    phi0, derphi0);
                if (!Double.isNaN(star)) {
                    return star;
                }
                break;
            }

            alpha0 = alpha1;
            alpha1 = 2.0 * alpha0;
            phiA0 = phiA1;
            derphiA0 = derphiA1;
            phiA1 = evalPhi(x, direction, objFun, alpha1);
        }

        return armijoFallback(x, direction, objFun, phi0, derphi0);
    }

    /**
     * Algorithm 3.6 (Zoom), Nocedal & Wright；与 SciPy {@code _zoom} 等价（无 extra_condition）。
     */
    private double zoom(double aLo, double aHi, double phiLo, double phiHi, double derphiLo,
            IVector x, IVector direction, IObjectiveFunction objFun, IGradientFunction grdFun,
            double phi0, double derphi0) {

        double phiRec = phi0;
        double aRec = 0.0;

        int i = 0;
        while (true) {
            double dalpha = aHi - aLo;
            double intervalLo = Math.min(aLo, aHi);
            double intervalHi = Math.max(aLo, aHi);

            double aJ;
            if (i > 0) {
                double cchk = ZOOM_DELTA1 * dalpha;
                Double cubic = cubicMin(aLo, phiLo, derphiLo, aHi, phiHi, aRec, phiRec);
                if (cubic != null && cubic <= intervalHi - cchk && cubic >= intervalLo + cchk) {
                    aJ = cubic;
                } else {
                    double qchk = ZOOM_DELTA2 * dalpha;
                    Double quad = quadMin(aLo, phiLo, derphiLo, aHi, phiHi);
                    if (quad != null && quad <= intervalHi - qchk && quad >= intervalLo + qchk) {
                        aJ = quad;
                    } else {
                        aJ = aLo + 0.5 * dalpha;
                    }
                }
            } else {
                aJ = aLo + 0.5 * dalpha;
            }

            double phiAj = evalPhi(x, direction, objFun, aJ);
            if (phiAj > phi0 + c1 * aJ * derphi0 || phiAj >= phiLo) {
                phiRec = phiHi;
                aRec = aHi;
                aHi = aJ;
                phiHi = phiAj;
            } else {
                double derphiAj = evalDerphi(x, direction, grdFun, aJ);
                if (Math.abs(derphiAj) <= -c2 * derphi0) {
                    return aJ;
                }
                if (derphiAj * (aHi - aLo) >= 0.0) {
                    phiRec = phiHi;
                    aRec = aHi;
                    aHi = aLo;
                    phiHi = phiLo;
                } else {
                    phiRec = phiLo;
                    aRec = aLo;
                }
                aLo = aJ;
                phiLo = phiAj;
                derphiLo = derphiAj;
            }

            i++;
            if (i > maxZoomIterations) {
                return Double.NaN;
            }
        }
    }

    private static Double cubicMin(double a, double fa, double fpa, double b, double fb, double c, double fc) {
        try {
            double db = b - a;
            double dc = c - a;
            double denom = db * db * dc * dc * (db - dc);
            if (denom == 0.0) {
                return null;
            }
            double cb = fc - fa - fpa * dc;
            double bb = fb - fa - fpa * db;
            double d1 = dc * dc;
            double d2 = -db * db;
            double d3 = -dc * dc * dc;
            double d4 = db * db * db;
            double A = (d1 * bb + d2 * cb) / denom;
            double B = (d3 * bb + d4 * cb) / denom;
            double radical = B * B - 3.0 * A * fpa;
            if (radical < 0.0 || A == 0.0) {
                return null;
            }
            double xmin = a + (-B + Math.sqrt(radical)) / (3.0 * A);
            if (!Double.isFinite(xmin)) {
                return null;
            }
            return xmin;
        } catch (ArithmeticException ignored) {
            return null;
        }
    }

    private static Double quadMin(double a, double fa, double fpa, double b, double fb) {
        try {
            double db = b - a;
            if (db == 0.0) {
                return null;
            }
            double B = (fb - fa - fpa * db) / (db * db);
            if (B == 0.0) {
                return null;
            }
            double xmin = a - fpa / (2.0 * B);
            if (!Double.isFinite(xmin)) {
                return null;
            }
            return xmin;
        } catch (ArithmeticException ignored) {
            return null;
        }
    }

    private static double evalPhi(IVector x, IVector direction, IObjectiveFunction objFun, double alpha) {
        return objFun.computeObjective(x.add(direction.multiplyByScalar(alpha)));
    }

    private static double evalDerphi(IVector x, IVector direction, IGradientFunction grdFun, double alpha) {
        IVector g = grdFun.computeGradient(x.add(direction.multiplyByScalar(alpha)));
        return g.innerProductValue(direction);
    }

    private double armijoFallback(IVector x, IVector direction, IObjectiveFunction objFun,
            double phi0, double derphi0) {
        double alpha = Math.min(initialStepSize, 1.0);
        for (int i = 0; i < maxLineSearchIterations; i++) {
            double phi = evalPhi(x, direction, objFun, alpha);
            if (phi <= phi0 + c1 * alpha * derphi0) {
                return alpha;
            }
            alpha *= 0.5;
            if (alpha < 1e-20) {
                break;
            }
        }
        return Math.max(1e-12, Math.min(initialStepSize, 1e-8));
    }

    public double getC1() {
        return c1;
    }

    public void setC1(double c1) {
        this.c1 = c1;
    }

    public double getC2() {
        return c2;
    }

    public void setC2(double c2) {
        this.c2 = c2;
    }

    public double getInitialStepSize() {
        return initialStepSize;
    }

    public void setInitialStepSize(double initialStepSize) {
        this.initialStepSize = initialStepSize;
    }

    public int getMaxLineSearchIterations() {
        return maxLineSearchIterations;
    }

    public void setMaxLineSearchIterations(int maxLineSearchIterations) {
        this.maxLineSearchIterations = maxLineSearchIterations;
    }

    public int getMaxZoomIterations() {
        return maxZoomIterations;
    }

    public void setMaxZoomIterations(int maxZoomIterations) {
        this.maxZoomIterations = Math.max(5, maxZoomIterations);
    }
}
