package com.yishape.lab.audio.embedding;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class IVectorEmbeddingSerializationTest {
    
    @Test
    public void testIVectorEmbeddingSaveAndLoad() {
        // Create a simple IVectorEmbedding model
        IVectorEmbedding embedding = new IVectorEmbedding(10, 8, 13);
        
        // Test save method
        String filePath = "test-ivector-model.ser";
        try {
            embedding.save(filePath);
            
            // Test load method
            IAudioEmbedding loadedEmbedding = IAudioEmbedding.load(filePath);
            
            assertNotNull(loadedEmbedding, "Loaded embedding should not be null");
            assertTrue(loadedEmbedding instanceof IVectorEmbedding, 
                      "Loaded embedding should be an IVectorEmbedding instance");
            
            IVectorEmbedding loadedIVector = (IVectorEmbedding) loadedEmbedding;
            assertEquals(embedding.getLen(), loadedIVector.getLen(), "Length should match");
            assertEquals(embedding.getNumComponents(), loadedIVector.getNumComponents(), "Num components should match");
            assertEquals(embedding.getMfccDim(), loadedIVector.getMfccDim(), "Mfcc dim should match");
            
            // Clean up test file
            java.io.File file = new java.io.File(filePath);
            if (file.exists()) {
                file.delete();
            }
            
        } catch (Exception e) {
            fail("Exception during save/load test: " + e.getMessage());
        }
    }
    
    @Test
    public void testOnlineIVectorEmbeddingSaveAndLoad() {
        // Create a simple OnlineIVectorEmbedding model
        OnlineIVectorEmbedding embedding = new OnlineIVectorEmbedding(10, 8, 13);
        
        // Test save method
        String filePath = "test-online-ivector-model.ser";
        try {
            embedding.save(filePath);
            
            // Test load method
            IAudioEmbedding loadedEmbedding = IAudioEmbedding.load(filePath);
            
            assertNotNull(loadedEmbedding, "Loaded embedding should not be null");
            assertTrue(loadedEmbedding instanceof OnlineIVectorEmbedding, 
                      "Loaded embedding should be an OnlineIVectorEmbedding instance");
            
            OnlineIVectorEmbedding loadedIVector = (OnlineIVectorEmbedding) loadedEmbedding;
            assertEquals(embedding.getLen(), loadedIVector.getLen(), "Length should match");
            assertEquals(embedding.getNumComponents(), loadedIVector.getNumComponents(), "Num components should match");
            assertEquals(embedding.getMfccDim(), loadedIVector.getMfccDim(), "MFCC dim should match");
            
            // Clean up test file
            java.io.File file = new java.io.File(filePath);
            if (file.exists()) {
                file.delete();
            }
            
        } catch (Exception e) {
            fail("Exception during save/load test: " + e.getMessage());
        }
    }
}