package com.yishape.lab.math.autodiff.impl;

import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.CustomGradientRegistry;
import com.yishape.lab.math.autodiff.IDiffVector;

/**
 * User-defined op with registered custom backward from {@link CustomGradientRegistry}.
 * 用户自定义算子，反向逻辑来自 {@link CustomGradientRegistry} 注册。
 */
public class CustomDiffVector extends RereDiffVector {

    private static final long serialVersionUID = 1L;

    private final String gradName;

    CustomDiffVector(IDoubleVector value, String gradName, List<RereDiffVector> inputs,
            Function<IDoubleVector, IDoubleVector[]> gradFn) {
        super(value, inputs, (Consumer<IDoubleVector>) (gradOut) -> {
            IDoubleVector[] grads = gradFn.apply(gradOut);
            for (int i = 0; i < inputs.size() && i < grads.length; i++) {
                inputs.get(i).accGrad(grads[i]);
            }
        });
        this.gradName = gradName;
    }

    public static IDiffVector create(String name, Function<IDiffVector[], IDiffVector> forwardFn,
            IDiffVector... inputs) {
        Function<IDoubleVector, IDoubleVector[]> gradFn = CustomGradientRegistry.get(name);
        if (gradFn == null) {
            throw new IllegalArgumentException(
                    "No gradient registered for '" + name + "'. Call AD.registerGradient() first.");
        }
        RereDiffVector[] rereInputs = Arrays.stream(inputs)
                .map(v -> (RereDiffVector) v).toArray(RereDiffVector[]::new);
        IDiffVector result = forwardFn.apply(inputs);
        IDoubleVector value = ((RereDiffVector) result).getValue().copy();
        return new CustomDiffVector(value, name, List.of(rereInputs), gradFn);
    }

    public String getGradientName() {
        return gradName;
    }
}
