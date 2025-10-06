package com.yishape.lab.audio;

import com.yishape.lab.audio.core.AudioIO;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.core.UnsupportedAudioFormatException;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.io.IOException;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

public class MP3SupportTest {

    @Test
    public void testMP3FormatEnum() {
        // Test that MP3 is a supported format in the enum
        AudioFormat mp3Format = AudioFormat.MP3;
        assertNotNull(mp3Format);
        assertEquals("mp3", mp3Format.getExtension());
        assertTrue(mp3Format.isLossy());
        assertFalse(mp3Format.isLossless());
    }

    @Test
    public void testMP3FormatFromExtension() {
        // Test that we can get MP3 format from extension
        AudioFormat format = AudioFormat.fromExtension("mp3");
        assertEquals(AudioFormat.MP3, format);
        
        format = AudioFormat.fromExtension(".mp3");
        assertEquals(AudioFormat.MP3, format);
        
        format = AudioFormat.fromExtension("MP3");
        assertEquals(AudioFormat.MP3, format);
    }

    @Test
    public void testMP3NotYetImplemented() {
        // Create a temporary file with .mp3 extension
        try {
            Path tempFile = Files.createTempFile("test", ".mp3");
            // Write some dummy data to the file
            Files.write(tempFile, new byte[100]);
            
            // This should throw an exception since we haven't implemented full MP3 support yet
            // In our current implementation, we throw an exception with specific guidance
            assertThrows(UnsupportedAudioFormatException.class, () -> {
                AudioData data = AudioIO.readAudio(tempFile.toString());
            });
            
            // Clean up
            Files.deleteIfExists(tempFile);
        } catch (Exception e) {
            fail("Unexpected exception: " + e.getMessage());
        }
    }
}