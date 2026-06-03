package com.yishape.lab.math.autodiff.vmap;

/**
 *
 * @author lteb2
 */
public class VMap implements IVMap{
    
    //真正的vmap实现者
    IVMap base;
    
    static{
    
    
    }
    
    /**
     * 根据GPU、HPC、SIMD可用性检测和配置，获得真正的VMap实现
     * @return 
     */
    private IVMap fetchVMap(){
    
        return null;
    }
    
}
