package com.yishape.lab.math.linalg.complex;

import com.yishape.lab.math.core.Complex;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

public interface IComplexMatrix {

    static IComplexMatrix fromRealImag(double[][] real, double[][] imag) {
        return new RereComplexDoubleMatrix(real, imag);
    }

    static IComplexMatrix fromPolar(double[][] magnitude, double[][] phase) {
        double[][] real = new double[magnitude.length][magnitude[0].length];
        double[][] imag = new double[magnitude.length][magnitude[0].length];
        for (int i = 0; i < magnitude.length; i++) {
            for (int j = 0; j < magnitude[i].length; j++) {
                real[i][j] = magnitude[i][j] * Math.cos(phase[i][j]);
                imag[i][j] = magnitude[i][j] * Math.sin(phase[i][j]);
            }
        }
        return new RereComplexDoubleMatrix(real, imag);
    }

    static IComplexMatrix fromComplex(Complex[][] data) {
        double[][] real = new double[data.length][data[0].length];
        double[][] imag = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                real[i][j] = data[i][j].real;
                imag[i][j] = data[i][j].imag;
            }
        }
        return new RereComplexDoubleMatrix(real, imag);
    }

    int rows();

    int cols();

    int getRowNum();

    int getColNum();

    Complex get(int row, int col);

    void put(int row, int col, Complex value);

    IMatrix<Double> real();

    IMatrix<Double> imag();

    IComplexMatrix conjugate();

    IComplexMatrix transpose();

    IComplexMatrix conjugateTranspose();

    IComplexMatrix add(IComplexMatrix other);

    IComplexMatrix sub(IComplexMatrix other);

    IComplexMatrix scale(double scalar);

    IComplexMatrix scale(Complex scalar);

    IComplexMatrix multiply(IComplexMatrix other);

    IComplexVector mmul(IComplexVector vector);

    IComplexMatrix mmul(IComplexMatrix other);

    IComplexMatrix hadamard(IComplexMatrix other);

    double frobeniusNorm();

    Complex trace();

    IComplexMatrix inv();

    Tuple2<IComplexMatrix, IComplexMatrix> lu();

    Tuple2<IComplexMatrix, IComplexMatrix> qr();

    Tuple2<IComplexVector, IComplexMatrix> eigen();

    Tuple3<IComplexMatrix, IComplexVector, IComplexMatrix> svd();

    Complex det();

    int rank();

    double cond();

    Complex[] diag();

    boolean isSquare();

    boolean isHermitian();

    IComplexMatrix copy();

    Complex[][] toComplexArray();

    String toString();

    interface IComplexVector {

        static IComplexVector fromRealImag(double[] real, double[] imag) {
            return new RereComplexDoubleVector(real, imag);
        }

        static IComplexVector fromPolar(double[] magnitude, double[] phase) {
            double[] real = new double[magnitude.length];
            double[] imag = new double[magnitude.length];
            for (int i = 0; i < magnitude.length; i++) {
                real[i] = magnitude[i] * Math.cos(phase[i]);
                imag[i] = magnitude[i] * Math.sin(phase[i]);
            }
            return new RereComplexDoubleVector(real, imag);
        }

        static IComplexVector fromComplex(Complex[] data) {
            double[] real = new double[data.length];
            double[] imag = new double[data.length];
            for (int i = 0; i < data.length; i++) {
                real[i] = data[i].real;
                imag[i] = data[i].imag;
            }
            return new RereComplexDoubleVector(real, imag);
        }

        int length();

        Complex get(int index);

        void set(int index, Complex value);

        IVector<Double> real();

        IVector<Double> imag();

        IComplexVector conjugate();

        IComplexVector add(IComplexVector other);

        IComplexVector sub(IComplexVector other);

        IComplexVector scale(double scalar);

        IComplexVector scale(Complex scalar);

        Complex innerProduct(IComplexVector other);

        IComplexMatrix outerProduct(IComplexVector other);

        IComplexVector pointwiseMultiply(IComplexVector other);

        IComplexVector pointwiseDivide(IComplexVector other);

        Complex sum();

        double magnitude();

        IComplexVector normalize();

        IComplexVector copy();

        Complex[] toComplexArray();

        String toString();
    }

    class RereComplexDoubleMatrix implements IComplexMatrix {
        private final double[][] real;
        private final double[][] imag;
        private final int rows;
        private final int cols;

        public RereComplexDoubleMatrix(double[][] real, double[][] imag) {
            if (real.length != imag.length || real[0].length != imag[0].length) {
                throw new IllegalArgumentException("Real and imaginary arrays must have same dimensions");
            }
            this.real = real;
            this.imag = imag;
            this.rows = real.length;
            this.cols = real[0].length;
        }

        @Override
        public int rows() { return rows; }

        @Override
        public int cols() { return cols; }

        @Override
        public int getRowNum() { return rows; }

        @Override
        public int getColNum() { return cols; }

        @Override
        public Complex get(int row, int col) {
            return new Complex(real[row][col], imag[row][col]);
        }

        @Override
        public void put(int row, int col, Complex value) {
            real[row][col] = value.real;
            imag[row][col] = value.imag;
        }

        @Override
        public IMatrix<Double> real() {
            return com.yishape.lab.math.linalg.Linalg.matrix(real);
        }

        @Override
        public IMatrix<Double> imag() {
            return com.yishape.lab.math.linalg.Linalg.matrix(imag);
        }

        @Override
        public IComplexMatrix conjugate() {
            double[][] imagNeg = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    imagNeg[i][j] = -imag[i][j];
                }
            }
            return new RereComplexDoubleMatrix(real, imagNeg);
        }

        @Override
        public IComplexMatrix transpose() {
            double[][] realT = new double[cols][rows];
            double[][] imagT = new double[cols][rows];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    realT[j][i] = real[i][j];
                    imagT[j][i] = imag[i][j];
                }
            }
            return new RereComplexDoubleMatrix(realT, imagT);
        }

        @Override
        public IComplexMatrix conjugateTranspose() {
            return conjugate().transpose();
        }

        @Override
        public IComplexMatrix add(IComplexMatrix other) {
            double[][] realNew = new double[rows][cols];
            double[][] imagNew = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Complex otherVal = other.get(i, j);
                    realNew[i][j] = real[i][j] + otherVal.real;
                    imagNew[i][j] = imag[i][j] + otherVal.imag;
                }
            }
            return new RereComplexDoubleMatrix(realNew, imagNew);
        }

        @Override
        public IComplexMatrix sub(IComplexMatrix other) {
            double[][] realNew = new double[rows][cols];
            double[][] imagNew = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Complex otherVal = other.get(i, j);
                    realNew[i][j] = real[i][j] - otherVal.real;
                    imagNew[i][j] = imag[i][j] - otherVal.imag;
                }
            }
            return new RereComplexDoubleMatrix(realNew, imagNew);
        }

        @Override
        public IComplexMatrix scale(double scalar) {
            double[][] realNew = new double[rows][cols];
            double[][] imagNew = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    realNew[i][j] = real[i][j] * scalar;
                    imagNew[i][j] = imag[i][j] * scalar;
                }
            }
            return new RereComplexDoubleMatrix(realNew, imagNew);
        }

        @Override
        public IComplexMatrix scale(Complex scalar) {
            double[][] realNew = new double[rows][cols];
            double[][] imagNew = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Complex val = get(i, j);
                    Complex result = val.multiply(scalar);
                    realNew[i][j] = result.real;
                    imagNew[i][j] = result.imag;
                }
            }
            return new RereComplexDoubleMatrix(realNew, imagNew);
        }

        @Override
        public IComplexMatrix multiply(IComplexMatrix other) {
            return mmul(other);
        }

        @Override
        public IComplexVector mmul(IComplexVector vector) {
            double[] realResult = new double[rows];
            double[] imagResult = new double[rows];
            Complex[] vecData = ((RereComplexDoubleVector) vector).toComplexArray();
            for (int i = 0; i < rows; i++) {
                Complex sum = Complex.ZERO;
                for (int j = 0; j < cols; j++) {
                    sum = sum.add(get(i, j).multiply(vecData[j]));
                }
                realResult[i] = sum.real;
                imagResult[i] = sum.imag;
            }
            return new RereComplexDoubleVector(realResult, imagResult);
        }

        @Override
        public IComplexMatrix mmul(IComplexMatrix other) {
            if (cols != other.rows()) {
                throw new IllegalArgumentException("Matrix dimensions not compatible for multiplication");
            }
            double[][] realNew = new double[rows][other.cols()];
            double[][] imagNew = new double[rows][other.cols()];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < other.cols(); j++) {
                    Complex sum = Complex.ZERO;
                    for (int k = 0; k < cols; k++) {
                        sum = sum.add(get(i, k).multiply(other.get(k, j)));
                    }
                    realNew[i][j] = sum.real;
                    imagNew[i][j] = sum.imag;
                }
            }
            return new RereComplexDoubleMatrix(realNew, imagNew);
        }

        @Override
        public IComplexMatrix hadamard(IComplexMatrix other) {
            double[][] realNew = new double[rows][cols];
            double[][] imagNew = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Complex result = get(i, j).multiply(other.get(i, j));
                    realNew[i][j] = result.real;
                    imagNew[i][j] = result.imag;
                }
            }
            return new RereComplexDoubleMatrix(realNew, imagNew);
        }

        @Override
        public double frobeniusNorm() {
            double sum = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    sum += real[i][j] * real[i][j] + imag[i][j] * imag[i][j];
                }
            }
            return Math.sqrt(sum);
        }

        @Override
        public Complex trace() {
            if (rows != cols) {
                throw new IllegalStateException("Trace requires square matrix");
            }
            Complex sum = Complex.ZERO;
            for (int i = 0; i < rows; i++) {
                sum = sum.add(get(i, i));
            }
            return sum;
        }

        @Override
        public IComplexMatrix inv() {
            if (rows != cols) {
                throw new IllegalStateException("Inverse requires square matrix");
            }
            int n = rows;
            double[][] realAug = new double[n][2 * n];
            double[][] imagAug = new double[n][2 * n];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    realAug[i][j] = real[i][j];
                    imagAug[i][j] = imag[i][j];
                }
                realAug[i][n + i] = 1;
            }

            for (int i = 0; i < n; i++) {
                double pivotR = realAug[i][i], pivotI = imagAug[i][i];
                double pivotMag = Math.sqrt(pivotR * pivotR + pivotI * pivotI);
                if (pivotMag < 1e-12) {
                    for (int k = i + 1; k < n; k++) {
                        double skR = realAug[k][i], skI = imagAug[k][i];
                        if (Math.sqrt(skR * skR + skI * skI) > 1e-12) {
                            double[] tempReal = realAug[i];
                            double[] tempImag = imagAug[i];
                            realAug[i] = realAug[k];
                            imagAug[i] = imagAug[k];
                            realAug[k] = tempReal;
                            imagAug[k] = tempImag;
                            pivotR = realAug[i][i];
                            pivotI = imagAug[i][i];
                            break;
                        }
                    }
                }

                Complex pivotInv = new Complex(pivotR, pivotI).reciprocal();
                for (int j = 0; j < 2 * n; j++) {
                    double oldReal = realAug[i][j], oldImag = imagAug[i][j];
                    realAug[i][j] = oldReal * pivotInv.real - oldImag * pivotInv.imag;
                    imagAug[i][j] = oldReal * pivotInv.imag + oldImag * pivotInv.real;
                }

                for (int k = 0; k < n; k++) {
                    if (k != i) {
                        double factorR = realAug[k][i], factorI = imagAug[k][i];
                        for (int j = 0; j < 2 * n; j++) {
                            realAug[k][j] -= (realAug[i][j] * factorR - imagAug[i][j] * factorI);
                            imagAug[k][j] -= (realAug[i][j] * factorI + imagAug[i][j] * factorR);
                        }
                    }
                }
            }

            double[][] resultReal = new double[n][n];
            double[][] resultImag = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    resultReal[i][j] = realAug[i][n + j];
                    resultImag[i][j] = imagAug[i][n + j];
                }
            }
            return new RereComplexDoubleMatrix(resultReal, resultImag);
        }

        @Override
        public Tuple2<IComplexMatrix, IComplexMatrix> lu() {
            if (rows != cols) {
                throw new IllegalStateException("LU decomposition requires square matrix");
            }
            int n = rows;
            double[][] lr = new double[n][n];
            double[][] li = new double[n][n];
            double[][] ur = new double[n][n];
            double[][] ui = new double[n][n];
            double[][] ar = new double[n][n];
            double[][] ai = new double[n][n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(real[i], 0, ar[i], 0, n);
                System.arraycopy(imag[i], 0, ai[i], 0, n);
                lr[i][i] = 1.0;
            }

            for (int k = 0; k < n; k++) {
                double pR = ar[k][k], pI = ai[k][k];
                double pMag2 = pR * pR + pI * pI;
                if (pMag2 < 1e-14) {
                    throw new IllegalStateException("Zero pivot at column " + k);
                }
                ur[k][k] = pR;
                ui[k][k] = pI;

                for (int i = k + 1; i < n; i++) {
                    double aR = ar[i][k], aI = ai[i][k];
                    double mR = (aR * pR + aI * pI) / pMag2;
                    double mI = (aI * pR - aR * pI) / pMag2;
                    lr[i][k] = mR; li[i][k] = mI;
                    for (int j = k + 1; j < n; j++) {
                        ar[i][j] -= (mR * ar[k][j] - mI * ai[k][j]);
                        ai[i][j] -= (mR * ai[k][j] + mI * ar[k][j]);
                    }
                    ar[i][k] = 0; ai[i][k] = 0;
                }
                for (int j = k + 1; j < n; j++) {
                    ur[k][j] = ar[k][j]; ui[k][j] = ai[k][j];
                }
            }
            return new Tuple2<>(new RereComplexDoubleMatrix(lr, li), new RereComplexDoubleMatrix(ur, ui));
        }

        @Override
        public Tuple2<IComplexMatrix, IComplexMatrix> qr() {
            int m = rows, n = cols;
            int minDim = Math.min(m, n);
            double[][] qrR = new double[m][n];
            double[][] qiR = new double[m][n];
            double[][] qrQ = new double[m][m];
            double[][] qiQ = new double[m][m];
            for (int i = 0; i < m; i++) {
                System.arraycopy(real[i], 0, qrR[i], 0, n);
                System.arraycopy(imag[i], 0, qiR[i], 0, n);
                qrQ[i][i] = 1.0;
            }

            for (int k = 0; k < minDim; k++) {
                double normX = 0;
                for (int i = k; i < m; i++) {
                    normX += qrR[i][k] * qrR[i][k] + qiR[i][k] * qiR[i][k];
                }
                normX = Math.sqrt(normX);
                if (normX < 1e-14) continue;

                double alphaR = qrR[k][k], alphaI = qiR[k][k];
                double alphaMag = Math.sqrt(alphaR * alphaR + alphaI * alphaI);
                double phaseR, phaseI;
                if (alphaMag < 1e-14) {
                    phaseR = normX; phaseI = 0;
                } else {
                    phaseR = alphaR / alphaMag; phaseI = -alphaI / alphaMag;
                }
                double tauR = alphaR + phaseR * normX;
                double tauI = alphaI + phaseI * normX;
                double tauMag = Math.sqrt(tauR * tauR + tauI * tauI);
                if (tauMag < 1e-14) continue;

                double[] vR = new double[m - k];
                double[] vI = new double[m - k];
                vR[0] = 1.0;
                double tauDenom = tauR * tauR + tauI * tauI;
                if (tauDenom < 1e-30) continue;
                for (int i = k + 1; i < m; i++) {
                    double xr = qrR[i][k], xi = qiR[i][k];
                    vR[i - k] = (xr * tauR + xi * tauI) / tauDenom;
                    vI[i - k] = (xi * tauR - xr * tauI) / tauDenom;
                }
                double vNorm2 = 1.0;
                for (int i = 1; i < vR.length; i++) {
                    vNorm2 += vR[i] * vR[i] + vI[i] * vI[i];
                }
                double betaR = 2.0 / vNorm2;

                for (int j = k; j < n; j++) {
                    double dotR = 0, dotI = 0;
                    for (int i = 0; i < vR.length; i++) {
                        double vr = vR[i], vi = vI[i];
                        double xr = qrR[k + i][j], xi = qiR[k + i][j];
                        dotR += vr * xr + vi * xi;
                        dotI += vr * xi - vi * xr;
                    }
                    dotR *= betaR; dotI *= betaR;
                    for (int i = 0; i < vR.length; i++) {
                        double vr = vR[i], vi = vI[i];
                        qrR[k + i][j] -= (vr * dotR - vi * dotI);
                        qiR[k + i][j] -= (vr * dotI + vi * dotR);
                    }
                }

                double[][] hR = new double[m][m];
                double[][] hI = new double[m][m];
                for (int i = 0; i < m; i++) { hR[i][i] = 1.0; }
                for (int i = 0; i < vR.length; i++) {
                    for (int jj = 0; jj < vR.length; jj++) {
                        hR[k + i][k + jj] -= betaR * (vR[i] * vR[jj] + vI[i] * vI[jj]);
                        hI[k + i][k + jj] -= betaR * (vI[i] * vR[jj] - vR[i] * vI[jj]);
                    }
                }
                double[][] tR = new double[m][m];
                double[][] tI = new double[m][m];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < m; j++) {
                        for (int p = 0; p < m; p++) {
                            tR[i][j] += qrQ[i][p] * hR[p][j] - qiQ[i][p] * hI[p][j];
                            tI[i][j] += qrQ[i][p] * hI[p][j] + qiQ[i][p] * hR[p][j];
                        }
                    }
                }
                qrQ = tR; qiQ = tI;
            }

            IComplexMatrix Q = new RereComplexDoubleMatrix(qrQ, qiQ);
            IComplexMatrix R = new RereComplexDoubleMatrix(qrR, qiR);
            return new Tuple2<>(Q, R);
        }

        @Override
        public Tuple2<IComplexVector, IComplexMatrix> eigen() {
            if (rows != cols) {
                throw new IllegalStateException("Eigen decomposition requires square matrix");
            }
            int n = rows;
            int maxIter = 200 * n;

            double[][] Hr = new double[n][n];
            double[][] Hi = new double[n][n];
            double[][] Qr = new double[n][n];
            double[][] Qi = new double[n][n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(real[i], 0, Hr[i], 0, n);
                System.arraycopy(imag[i], 0, Hi[i], 0, n);
                Qr[i][i] = 1.0;
            }

            for (int iter = 0; iter < maxIter; iter++) {
                boolean converged = true;
                for (int i = 0; i < n - 1; i++) {
                    double subMag = Math.sqrt(Hr[i + 1][i] * Hr[i + 1][i] + Hi[i + 1][i] * Hi[i + 1][i]);
                    if (subMag > 1e-12) { converged = false; break; }
                }
                if (converged) break;

                double[][] stepQr = new double[n][n];
                double[][] stepQi = new double[n][n];
                double[][] Rr = new double[n][n];
                double[][] Ri = new double[n][n];
                for (int i = 0; i < n; i++) {
                    System.arraycopy(Hr[i], 0, Rr[i], 0, n);
                    System.arraycopy(Hi[i], 0, Ri[i], 0, n);
                    stepQr[i][i] = 1.0;
                }

                for (int k = 0; k < n - 1; k++) {
                    double xr = Rr[k][k], xi = Ri[k][k];
                    double yr = Rr[k + 1][k], yi = Ri[k + 1][k];
                    double xNorm = Math.sqrt(xr * xr + xi * xi);
                    double yNorm = Math.sqrt(yr * yr + yi * yi);
                    double norm = Math.sqrt(xNorm * xNorm + yNorm * yNorm);
                    if (norm < 1e-14) continue;

                    double cr, ci, sr, si;
                    if (yNorm < 1e-14) {
                        cr = 1; ci = 0; sr = 0; si = 0;
                    } else {
                        cr = xNorm / norm; ci = 0;
                        double yrNorm = yr / yNorm, yiNorm = yi / yNorm;
                        sr = yNorm / norm;
                        si = 0;
                        double phaseR = (xr * yr + xi * yi) / (xNorm * yNorm);
                        double phaseI = (xi * yr - xr * yi) / (xNorm * yNorm);
                        double phaseMag = Math.sqrt(phaseR * phaseR + phaseI * phaseI);
                        if (phaseMag > 1e-14) {
                            sr = (phaseR / phaseMag) * sr;
                            si = -(phaseI / phaseMag) * sr;
                        }
                    }

                    for (int j = k; j < n; j++) {
                        double r1r = Rr[k][j], r1i = Ri[k][j];
                        double r2r = Rr[k + 1][j], r2i = Ri[k + 1][j];
                        Rr[k][j] = cr * r1r + sr * r2r - si * r2i;
                        Ri[k][j] = cr * r1i + sr * r2i + si * r2r;
                        Rr[k + 1][j] = -sr * r1r - si * r1i + cr * r2r;
                        Ri[k + 1][j] = -sr * r1i + si * r1r + cr * r2i;
                    }
                    for (int j = 0; j < n; j++) {
                        double q1r = stepQr[j][k], q1i = stepQi[j][k];
                        double q2r = stepQr[j][k + 1], q2i = stepQi[j][k + 1];
                        stepQr[j][k] = cr * q1r + sr * q2r - si * q2i;
                        stepQi[j][k] = cr * q1i + sr * q2i + si * q2r;
                        stepQr[j][k + 1] = -sr * q1r - si * q1i + cr * q2r;
                        stepQi[j][k + 1] = -sr * q1i + si * q1r + cr * q2i;
                    }
                }

                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        Hr[i][j] = 0; Hi[i][j] = 0;
                        for (int p = 0; p < n; p++) {
                            Hr[i][j] += Rr[i][p] * stepQr[p][j] - Ri[i][p] * stepQi[p][j];
                            Hi[i][j] += Rr[i][p] * stepQi[p][j] + Ri[i][p] * stepQr[p][j];
                        }
                    }
                }
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double qr = 0, qi = 0;
                        for (int p = 0; p < n; p++) {
                            qr += Qr[i][p] * stepQr[p][j] - Qi[i][p] * stepQi[p][j];
                            qi += Qr[i][p] * stepQi[p][j] + Qi[i][p] * stepQr[p][j];
                        }
                        Qr[i][j] = qr; Qi[i][j] = qi;
                    }
                }
            }

            double[] eigR = new double[n];
            double[] eigI = new double[n];
            for (int i = 0; i < n; i++) {
                eigR[i] = Hr[i][i];
                eigI[i] = Hi[i][i];
            }

            IComplexVector eigenvalues = new RereComplexDoubleVector(eigR, eigI);
            IComplexMatrix eigenvectors = new RereComplexDoubleMatrix(Qr, Qi);
            return new Tuple2<>(eigenvalues, eigenvectors);
        }

        @Override
        public Tuple3<IComplexMatrix, IComplexVector, IComplexMatrix> svd() {
            int m = rows, n = cols;
            int bRows = 2 * m, bCols = 2 * n;
            double[][] B = new double[bRows][bCols];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    B[i][j] = real[i][j];
                    B[i][j + n] = -imag[i][j];
                    B[i + m][j] = imag[i][j];
                    B[i + m][j + n] = real[i][j];
                }
            }

            com.yishape.lab.math.linalg.IDoubleMatrix Bmat = com.yishape.lab.math.linalg.IDoubleMatrix.of(B);
            var svdResult = Bmat.svd();
            com.yishape.lab.math.linalg.IMatrix<Double> Ub = svdResult._1;
            com.yishape.lab.math.linalg.IVector<Double> Sb = svdResult._2;
            com.yishape.lab.math.linalg.IMatrix<Double> Vb = svdResult._3;

            int minDim = Math.min(m, n);
            double[] singVals = new double[minDim];
            int svCount = 0;
            for (int i = 0; i < Sb.length() && svCount < minDim; i++) {
                double sv = Sb.get(i);
                if (sv > 1e-12) {
                    singVals[svCount++] = sv;
                    if (i + 1 < Sb.length() && Math.abs(Sb.get(i + 1) - sv) < 1e-10) i++;
                }
            }
            while (svCount < minDim) singVals[svCount++] = 0;

            double[][] uReal = new double[m][minDim];
            double[][] uImag = new double[m][minDim];
            double[][] vReal = new double[n][minDim];
            double[][] vImag = new double[n][minDim];

            for (int k = 0; k < minDim; k++) {
                for (int i = 0; i < m; i++) {
                    uReal[i][k] = Ub.get(i, 2 * k);
                    uImag[i][k] = -Ub.get(i + m, 2 * k);
                }
                for (int j = 0; j < n; j++) {
                    vReal[j][k] = Vb.get(j, 2 * k);
                    vImag[j][k] = -Vb.get(j + n, 2 * k);
                }
            }

            double[] sigReal = new double[minDim];
            double[] sigImag = new double[minDim];
            for (int i = 0; i < minDim; i++) sigReal[i] = singVals[i];

            IComplexMatrix U = new RereComplexDoubleMatrix(uReal, uImag);
            IComplexVector S = new RereComplexDoubleVector(sigReal, sigImag);
            IComplexMatrix V = new RereComplexDoubleMatrix(vReal, vImag);
            return new Tuple3<>(U, S, V);
        }

        @Override
        public Complex det() {
            if (rows != cols) {
                throw new IllegalStateException("Determinant requires square matrix");
            }
            int n = rows;
            if (n == 1) {
                return get(0, 0);
            }
            if (n == 2) {
                Complex a = get(0, 0), b = get(0, 1);
                Complex c = get(1, 0), d = get(1, 1);
                return a.multiply(d).subtract(b.multiply(c));
            }

            Complex det = Complex.ZERO;
            for (int j = 0; j < n; j++) {
                double[][] subReal = new double[n - 1][n - 1];
                double[][] subImag = new double[n - 1][n - 1];
                for (int i = 1; i < n; i++) {
                    int colIdx = 0;
                    for (int k = 0; k < n; k++) {
                        if (k != j) {
                            subReal[i - 1][colIdx] = real[i][k];
                            subImag[i - 1][colIdx] = imag[i][k];
                            colIdx++;
                        }
                    }
                }
                Complex minor = new RereComplexDoubleMatrix(subReal, subImag).det();
                Complex cofactor = minor.scale((j % 2 == 0) ? 1 : -1).multiply(get(0, j));
                det = det.add(cofactor);
            }
            return det;
        }

        @Override
        public int rank() {
            Tuple3<IComplexMatrix, IComplexVector, IComplexMatrix> svdResult = svd();
            IComplexVector S = svdResult._2;
            int r = 0;
            double maxSv = 0;
            for (int i = 0; i < S.length(); i++) {
                double mag = S.get(i).real;
                if (mag > maxSv) maxSv = mag;
            }
            double tol = maxSv * Math.max(rows, cols) * 1e-14;
            for (int i = 0; i < S.length(); i++) {
                if (Math.abs(S.get(i).real) > tol) r++;
            }
            return r;
        }

        @Override
        public double cond() {
            Tuple3<IComplexMatrix, IComplexVector, IComplexMatrix> svdResult = svd();
            IComplexVector S = svdResult._2;
            double maxSv = 0, minSv = Double.POSITIVE_INFINITY;
            for (int i = 0; i < S.length(); i++) {
                double mag = Math.abs(S.get(i).real);
                if (mag > maxSv) maxSv = mag;
                if (mag > 1e-14 && mag < minSv) minSv = mag;
            }
            if (Double.isInfinite(minSv)) return Double.POSITIVE_INFINITY;
            return maxSv / minSv;
        }

        @Override
        public Complex[] diag() {
            int n = Math.min(rows, cols);
            Complex[] diagVals = new Complex[n];
            for (int i = 0; i < n; i++) {
                diagVals[i] = get(i, i);
            }
            return diagVals;
        }

        @Override
        public boolean isSquare() {
            return rows == cols;
        }

        @Override
        public boolean isHermitian() {
            if (!isSquare()) return false;
            for (int i = 0; i < rows; i++) {
                for (int j = i + 1; j < cols; j++) {
                    Complex a = get(i, j);
                    Complex bConj = get(j, i).conjugate();
                    if (Math.abs(a.real - bConj.real) > 1e-10 || Math.abs(a.imag - bConj.imag) > 1e-10) {
                        return false;
                    }
                }
            }
            return true;
        }

        @Override
        public IComplexMatrix copy() {
            double[][] realCopy = new double[rows][cols];
            double[][] imagCopy = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                System.arraycopy(real[i], 0, realCopy[i], 0, cols);
                System.arraycopy(imag[i], 0, imagCopy[i], 0, cols);
            }
            return new RereComplexDoubleMatrix(realCopy, imagCopy);
        }

        @Override
        public Complex[][] toComplexArray() {
            Complex[][] result = new Complex[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] = get(i, j);
                }
            }
            return result;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("ComplexMatrix[\n");
            for (int i = 0; i < rows; i++) {
                sb.append("  [");
                for (int j = 0; j < cols; j++) {
                    sb.append(String.format("(%+.3f,%+.3fi)", real[i][j], imag[i][j]));
                    if (j < cols - 1) sb.append(", ");
                }
                sb.append("]\n");
            }
            sb.append("]");
            return sb.toString();
        }
    }

    class RereComplexDoubleVector implements IComplexVector {
        private final double[] real;
        private final double[] imag;
        private final int length;

        public RereComplexDoubleVector(double[] real, double[] imag) {
            if (real.length != imag.length) {
                throw new IllegalArgumentException("Real and imaginary arrays must have same length");
            }
            this.real = real;
            this.imag = imag;
            this.length = real.length;
        }

        @Override
        public int length() { return length; }

        @Override
        public Complex get(int index) {
            return new Complex(real[index], imag[index]);
        }

        @Override
        public void set(int index, Complex value) {
            real[index] = value.real;
            imag[index] = value.imag;
        }

        @Override
        public IVector<Double> real() {
            return com.yishape.lab.math.linalg.Linalg.vector(real);
        }

        @Override
        public IVector<Double> imag() {
            return com.yishape.lab.math.linalg.Linalg.vector(imag);
        }

        @Override
        public IComplexVector conjugate() {
            double[] imagNeg = new double[length];
            for (int i = 0; i < length; i++) {
                imagNeg[i] = -imag[i];
            }
            return new RereComplexDoubleVector(real, imagNeg);
        }

        @Override
        public IComplexVector add(IComplexVector other) {
            double[] realNew = new double[length];
            double[] imagNew = new double[length];
            Complex[] otherData = ((RereComplexDoubleVector) other).toComplexArray();
            for (int i = 0; i < length; i++) {
                Complex sum = get(i).add(otherData[i]);
                realNew[i] = sum.real;
                imagNew[i] = sum.imag;
            }
            return new RereComplexDoubleVector(realNew, imagNew);
        }

        @Override
        public IComplexVector sub(IComplexVector other) {
            double[] realNew = new double[length];
            double[] imagNew = new double[length];
            Complex[] otherData = ((RereComplexDoubleVector) other).toComplexArray();
            for (int i = 0; i < length; i++) {
                Complex diff = get(i).subtract(otherData[i]);
                realNew[i] = diff.real;
                imagNew[i] = diff.imag;
            }
            return new RereComplexDoubleVector(realNew, imagNew);
        }

        @Override
        public IComplexVector scale(double scalar) {
            double[] realNew = new double[length];
            double[] imagNew = new double[length];
            for (int i = 0; i < length; i++) {
                realNew[i] = real[i] * scalar;
                imagNew[i] = imag[i] * scalar;
            }
            return new RereComplexDoubleVector(realNew, imagNew);
        }

        @Override
        public IComplexVector scale(Complex scalar) {
            double[] realNew = new double[length];
            double[] imagNew = new double[length];
            for (int i = 0; i < length; i++) {
                Complex val = get(i).multiply(scalar);
                realNew[i] = val.real;
                imagNew[i] = val.imag;
            }
            return new RereComplexDoubleVector(realNew, imagNew);
        }

        @Override
        public Complex innerProduct(IComplexVector other) {
            Complex sum = Complex.ZERO;
            Complex[] otherData = ((RereComplexDoubleVector) other).toComplexArray();
            for (int i = 0; i < length; i++) {
                sum = sum.add(get(i).multiply(otherData[i].conjugate()));
            }
            return sum;
        }

        @Override
        public IComplexMatrix outerProduct(IComplexVector other) {
            Complex[] otherData = ((RereComplexDoubleVector) other).toComplexArray();
            double[][] realResult = new double[length][other.length()];
            double[][] imagResult = new double[length][other.length()];
            for (int i = 0; i < length; i++) {
                for (int j = 0; j < other.length(); j++) {
                    Complex result = get(i).multiply(otherData[j].conjugate());
                    realResult[i][j] = result.real;
                    imagResult[i][j] = result.imag;
                }
            }
            return new RereComplexDoubleMatrix(realResult, imagResult);
        }

        @Override
        public IComplexVector pointwiseMultiply(IComplexVector other) {
            double[] realNew = new double[length];
            double[] imagNew = new double[length];
            Complex[] otherData = ((RereComplexDoubleVector) other).toComplexArray();
            for (int i = 0; i < length; i++) {
                Complex result = get(i).multiply(otherData[i]);
                realNew[i] = result.real;
                imagNew[i] = result.imag;
            }
            return new RereComplexDoubleVector(realNew, imagNew);
        }

        @Override
        public IComplexVector pointwiseDivide(IComplexVector other) {
            double[] realNew = new double[length];
            double[] imagNew = new double[length];
            Complex[] otherData = ((RereComplexDoubleVector) other).toComplexArray();
            for (int i = 0; i < length; i++) {
                Complex result = get(i).divide(otherData[i]);
                realNew[i] = result.real;
                imagNew[i] = result.imag;
            }
            return new RereComplexDoubleVector(realNew, imagNew);
        }

        @Override
        public Complex sum() {
            Complex sum = Complex.ZERO;
            for (int i = 0; i < length; i++) {
                sum = sum.add(get(i));
            }
            return sum;
        }

        @Override
        public double magnitude() {
            double sum = 0;
            for (int i = 0; i < length; i++) {
                sum += real[i] * real[i] + imag[i] * imag[i];
            }
            return Math.sqrt(sum);
        }

        @Override
        public IComplexVector normalize() {
            double mag = magnitude();
            if (mag < 1e-12) {
                throw new ArithmeticException("Cannot normalize zero vector");
            }
            return scale(1.0 / mag);
        }

        @Override
        public IComplexVector copy() {
            double[] realCopy = new double[length];
            double[] imagCopy = new double[length];
            System.arraycopy(real, 0, realCopy, 0, length);
            System.arraycopy(imag, 0, imagCopy, 0, length);
            return new RereComplexDoubleVector(realCopy, imagCopy);
        }

        @Override
        public Complex[] toComplexArray() {
            Complex[] result = new Complex[length];
            for (int i = 0; i < length; i++) {
                result[i] = get(i);
            }
            return result;
        }

        public double[] getRealArray() { return real; }
        public double[] getImagArray() { return imag; }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("ComplexVector[");
            for (int i = 0; i < length; i++) {
                sb.append(String.format("(%+.3f,%+.3fi)", real[i], imag[i]));
                if (i < length - 1) sb.append(", ");
            }
            sb.append("]");
            return sb.toString();
        }
    }
}