package com.yishape.lab.math.linalg.decomposition;

import org.junit.jupiter.api.Test;

public class DecompsTest {
    
    @Test
    public void testDecompsFactory() {
        // This test simply verifies that the Decomps factory can be imported and used
        var qr = Decomps.createQR();
        var lu = Decomps.createLU();
        var svd = Decomps.createSVD();
        var chol = Decomps.createCholesky();
        var qrcp = Decomps.createQrcp();
        var qrcpFast = Decomps.createQrcpDgeqp3();
        var qrcpBlock = Decomps.createQrcpDlaqps();
        
        // If we reach this point, the imports are working correctly
        assert qr != null;
        assert lu != null;
        assert svd != null;
        assert chol != null;
        assert qrcp != null;
        assert qrcpFast != null;
        assert qrcpBlock != null;
    }
}