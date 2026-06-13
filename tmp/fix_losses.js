const fs = require('fs');
const path = 'E:/work/yishape-math/src/main/java/com/yishape/lab/math/autodiff/impl/TangentDiffTensor.java';
let content = fs.readFileSync(path, 'utf8');

// Fix loss functions - replace fill_(0) with lossJVP
const lossJVPHelper = `
    /**
     * Compute scalar JVP for loss functions: JVP = dot(grad_input, tangent_input) + dot(grad_target, tangent_target).
     * Backpropagates through the loss node p, then computes dot product of input gradients with their tangents.
     */
    private static double[] lossJVP(RereDiffTensor lossNode, TangentDiffTensor input, TangentDiffTensor target) {
        lossNode.backward();
        DoubleVectorComputer comp = new DoubleVectorComputer();
        double jvp = 0;
        IDoubleTensor gx = input.primal.grad();
        if (gx != null) {
            double[] gxd = gx.toDoubleArray();
            double[] tx = input.tangent.toDoubleArray();
            double[] prod = comp.binaryOperate(gxd, tx, BinaryOperation.MULTIPLY);
            jvp += comp.reduceOperate(prod, ReduceOperation.SUM);
        }
        IDoubleTensor gt = target.primal.grad();
        if (gt != null) {
            double[] gtd = gt.toDoubleArray();
            double[] ty = target.tangent.toDoubleArray();
            double[] prod = comp.binaryOperate(gtd, ty, BinaryOperation.MULTIPLY);
            jvp += comp.reduceOperate(prod, ReduceOperation.SUM);
        }
        return new double[]{jvp};
    }
`;

// Fix smoothL1Loss
content = content.replace(
  '    @Override public IDiffTensor smoothL1Loss(IDiffTensor target, double beta) {\n        TangentDiffTensor t = (TangentDiffTensor) target;\n        RereDiffTensor p = (RereDiffTensor) primal.smoothL1Loss(t.primal, beta);\n        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),\n            List.of(this, t), p);\n    }',
  '    @Override public IDiffTensor smoothL1Loss(IDiffTensor target, double beta) {\n        TangentDiffTensor t = (TangentDiffTensor) target;\n        RereDiffTensor p = (RereDiffTensor) primal.smoothL1Loss(t.primal, beta);\n        double[] jvpVal = lossJVP(p, this, t);\n        return new TangentDiffTensor(p, new RereDoubleTensor(jvpVal, 1), List.of(this, t), p);\n    }'
);

// Fix bceLoss
content = content.replace(
  '    @Override public IDiffTensor bceLoss(IDiffTensor target) {\n        TangentDiffTensor t = (TangentDiffTensor) target;\n        RereDiffTensor p = (RereDiffTensor) primal.bceLoss(t.primal);\n        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),\n            List.of(this, t), p);\n    }',
  '    @Override public IDiffTensor bceLoss(IDiffTensor target) {\n        TangentDiffTensor t = (TangentDiffTensor) target;\n        RereDiffTensor p = (RereDiffTensor) primal.bceLoss(t.primal);\n        double[] jvpVal = lossJVP(p, this, t);\n        return new TangentDiffTensor(p, new RereDoubleTensor(jvpVal, 1), List.of(this, t), p);\n    }'
);

// Fix focalLoss
content = content.replace(
  '    @Override public IDiffTensor focalLoss(IDiffTensor target, double alpha, double gamma) {\n        TangentDiffTensor t = (TangentDiffTensor) target;\n        RereDiffTensor p = (RereDiffTensor) primal.focalLoss(t.primal, alpha, gamma);\n        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),\n            List.of(this, t), p);\n    }',
  '    @Override public IDiffTensor focalLoss(IDiffTensor target, double alpha, double gamma) {\n        TangentDiffTensor t = (TangentDiffTensor) target;\n        RereDiffTensor p = (RereDiffTensor) primal.focalLoss(t.primal, alpha, gamma);\n        double[] jvpVal = lossJVP(p, this, t);\n        return new TangentDiffTensor(p, new RereDoubleTensor(jvpVal, 1), List.of(this, t), p);\n    }'
);

// Fix diceLoss
content = content.replace(
  '    @Override public IDiffTensor diceLoss(IDiffTensor target, double smooth) {\n        TangentDiffTensor t = (TangentDiffTensor) target;\n        RereDiffTensor p = (RereDiffTensor) primal.diceLoss(t.primal, smooth);\n        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),\n            List.of(this, t), p);\n    }',
  '    @Override public IDiffTensor diceLoss(IDiffTensor target, double smooth) {\n        TangentDiffTensor t = (TangentDiffTensor) target;\n        RereDiffTensor p = (RereDiffTensor) primal.diceLoss(t.primal, smooth);\n        double[] jvpVal = lossJVP(p, this, t);\n        return new TangentDiffTensor(p, new RereDoubleTensor(jvpVal, 1), List.of(this, t), p);\n    }'
);

// Fix nllLoss
content = content.replace(
  '    @Override public IDiffTensor nllLoss(IDiffTensor target, int classDim) {\n        TangentDiffTensor t = (TangentDiffTensor) target;\n        RereDiffTensor p = (RereDiffTensor) primal.nllLoss(t.primal, classDim);\n        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),\n            List.of(this, t), p);\n    }',
  '    @Override public IDiffTensor nllLoss(IDiffTensor target, int classDim) {\n        TangentDiffTensor t = (TangentDiffTensor) target;\n        RereDiffTensor p = (RereDiffTensor) primal.nllLoss(t.primal, classDim);\n        double[] jvpVal = lossJVP(p, this, t);\n        return new TangentDiffTensor(p, new RereDoubleTensor(jvpVal, 1), List.of(this, t), p);\n    }'
);

// Insert lossJVP helper before the first loss method
const lossMethodPos = content.indexOf('    @Override public IDiffTensor smoothL1Loss');
content = content.slice(0, lossMethodPos) + lossJVPHelper + content.slice(lossMethodPos);

fs.writeFileSync(path, content, 'utf8');
console.log('Loss JVP fixes applied!');
