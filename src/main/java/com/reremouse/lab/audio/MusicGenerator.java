package com.reremouse.lab.audio;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.Signals;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 音乐生成器类 / Music Generator Class
 * <p>
 * 提供音乐生成功能，包括生成音阶、和弦、旋律、节奏等。
 * 使用项目现有的signal包和linalg包功能进行音乐生成。
 * </p>
 * <p>
 * Provides music generation functionality including generating scales, chords, 
 * melodies, rhythms, etc. Uses existing signal and linalg package functionality for music generation.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MusicGenerator {
    
    private static final Random random = new Random();
    
    /**
     * 音符类 / Note Class
     */
    public static class Note {
        private final int pitch; // 音高 (半音数) / Pitch (semitones)
        private final double duration; // 持续时间 (拍) / Duration (beats)
        private final double velocity; // 力度 (0-1) / Velocity (0-1)
        private final double startTime; // 开始时间 (拍) / Start time (beats)
        
        public Note(int pitch, double duration, double velocity, double startTime) {
            this.pitch = pitch;
            this.duration = duration;
            this.velocity = velocity;
            this.startTime = startTime;
        }
        
        public int getPitch() { return pitch; }
        public double getDuration() { return duration; }
        public double getVelocity() { return velocity; }
        public double getStartTime() { return startTime; }
        
        @Override
        public String toString() {
            return String.format("Note{pitch=%d, duration=%.2f, velocity=%.2f, startTime=%.2f}", 
                               pitch, duration, velocity, startTime);
        }
    }
    
    /**
     * 旋律类 / Melody Class
     */
    public static class Melody {
        private final List<Note> notes;
        private final double tempo; // 节拍 (BPM) / Tempo (BPM)
        private final int timeSignature; // 拍号 / Time signature
        
        public Melody(List<Note> notes, double tempo, int timeSignature) {
            this.notes = new ArrayList<>(notes);
            this.tempo = tempo;
            this.timeSignature = timeSignature;
        }
        
        public List<Note> getNotes() { return new ArrayList<>(notes); }
        public double getTempo() { return tempo; }
        public int getTimeSignature() { return timeSignature; }
        
        /**
         * 转换为音频数据 / Convert to audio data
         *
         * @param sampleRate 采样率 / Sample rate
         * @return 音频数据 / Audio data
         */
        public AudioData toAudioData(double sampleRate) {
            return generateMelodyAudio(this, sampleRate);
        }
        
        @Override
        public String toString() {
            return String.format("Melody{notes=%d, tempo=%.2f, timeSignature=%d}", 
                               notes.size(), tempo, timeSignature);
        }
    }
    
    /**
     * 生成随机旋律 / Generate random melody
     * <p>
     * 根据指定的音阶和参数生成随机旋律。
     * Generate random melody based on specified scale and parameters.
     * </p>
     *
     * @param rootNote 根音 (0-11) / Root note (0-11)
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param numNotes 音符数量 / Number of notes
     * @param tempo 节拍 (BPM) / Tempo (BPM)
     * @param timeSignature 拍号 / Time signature
     * @return 生成的旋律 / Generated melody
     */
    public static Melody generateRandomMelody(int rootNote, MusicTheory.ScaleType scaleType, 
                                            int octave, int numNotes, double tempo, int timeSignature) {
        int[] scale = MusicTheory.generateScale(rootNote, scaleType);
        List<Note> notes = new ArrayList<>();
        
        // double beatDuration = 60.0 / tempo; // 每拍的持续时间 (秒) / Duration of each beat (seconds)
        double currentTime = 0;
        
        for (int i = 0; i < numNotes; i++) {
            // 随机选择音阶中的音符 / Randomly select note from scale
            int scaleNote = scale[random.nextInt(scale.length)];
            int pitch = scaleNote + octave * 12;
            
            // 随机持续时间 (1/4拍 到 2拍) / Random duration (1/4 beat to 2 beats)
            double duration = (0.25 + random.nextDouble() * 1.75);
            
            // 随机力度 / Random velocity
            double velocity = 0.5 + random.nextDouble() * 0.5;
            
            notes.add(new Note(pitch, duration, velocity, currentTime));
            currentTime += duration;
        }
        
        return new Melody(notes, tempo, timeSignature);
    }
    
    /**
     * 生成和弦进行 / Generate chord progression
     * <p>
     * 根据指定的调性和和弦类型生成和弦进行。
     * Generate chord progression based on specified key and chord types.
     * </p>
     *
     * @param key 调性 / Key
     * @param chordTypes 和弦类型序列 / Chord type sequence
     * @param octave 八度 / Octave
     * @param chordDuration 每个和弦的持续时间 (拍) / Duration of each chord (beats)
     * @param tempo 节拍 (BPM) / Tempo (BPM)
     * @return 和弦进行音频数据 / Chord progression audio data
     */
    public static AudioData generateChordProgression(MusicTheory.Key key, 
                                                   MusicTheory.ChordType[] chordTypes,
                                                   int octave, double chordDuration, double tempo) {
        double beatDuration = 60.0 / tempo;
        double noteDuration = chordDuration * beatDuration;
        int sampleRate = 44100;
        
        IVector<Double> totalSamples = Linalg.zeros(0);
        
        for (MusicTheory.ChordType chordType : chordTypes) {
            // 生成和弦 / Generate chord
            MusicTheory.Chord chord = new MusicTheory.Chord(key.getRootNote(), chordType);
            AudioData chordAudio = MusicTheory.generateChordAudio(
                chord.getRootNote(), chord.getChordType(), octave, 
                noteDuration, sampleRate, 0.3);
            
            // 添加到总音频中 / Add to total audio
            if (totalSamples.length() == 0) {
                totalSamples = chordAudio.getSamples();
            } else {
                // 简单的音频拼接 / Simple audio concatenation
                IVector<Double> newSamples = Linalg.zeros(totalSamples.length() + chordAudio.getLength());
                for (int i = 0; i < totalSamples.length(); i++) {
                    newSamples.set(i, totalSamples.get(i));
                }
                for (int i = 0; i < chordAudio.getLength(); i++) {
                    newSamples.set(totalSamples.length() + i, chordAudio.getSamples().get(i));
                }
                totalSamples = newSamples;
            }
        }
        
        return new AudioData(totalSamples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成音阶练习 / Generate scale exercise
     * <p>
     * 生成指定音阶的上下行练习音频。
     * Generate ascending and descending scale exercise audio.
     * </p>
     *
     * @param rootNote 根音 (0-11) / Root note (0-11)
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param noteDuration 每个音符的持续时间 (秒) / Duration of each note (seconds)
     * @param sampleRate 采样率 / Sample rate
     * @return 音阶练习音频数据 / Scale exercise audio data
     */
    public static AudioData generateScaleExercise(int rootNote, MusicTheory.ScaleType scaleType,
                                                int octave, double noteDuration, double sampleRate) {
        int[] scale = MusicTheory.generateScale(rootNote, scaleType);
        
        // 生成上行音阶 / Generate ascending scale
        IVector<Double> ascendingSamples = Linalg.zeros(0);
        for (int note : scale) {
            double frequency = MusicTheory.noteToFrequency(MusicTheory.NOTE_NAMES[note], octave);
            IVector<Double> noteSamples = Signals.sineWave(
                (int)(noteDuration * sampleRate), frequency, sampleRate, 0.3, 0.0);
            ascendingSamples = concatenateAudio(ascendingSamples, noteSamples);
        }
        
        // 生成下行音阶 / Generate descending scale
        IVector<Double> descendingSamples = Linalg.zeros(0);
        for (int i = scale.length - 1; i >= 0; i--) {
            int note = scale[i];
            double frequency = MusicTheory.noteToFrequency(MusicTheory.NOTE_NAMES[note], octave);
            IVector<Double> noteSamples = Signals.sineWave(
                (int)(noteDuration * sampleRate), frequency, sampleRate, 0.3, 0.0);
            descendingSamples = concatenateAudio(descendingSamples, noteSamples);
        }
        
        // 合并上行和下行 / Merge ascending and descending
        IVector<Double> totalSamples = concatenateAudio(ascendingSamples, descendingSamples);
        
        return new AudioData(totalSamples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成节拍器 / Generate metronome
     * <p>
     * 生成指定节拍的节拍器音频。
     * Generate metronome audio with specified tempo.
     * </p>
     *
     * @param bpm 每分钟节拍数 / Beats per minute
     * @param duration 持续时间 (秒) / Duration (seconds)
     * @param sampleRate 采样率 / Sample rate
     * @param accentEvery 每几个节拍重音 / Accent every N beats
     * @return 节拍器音频数据 / Metronome audio data
     */
    public static AudioData generateMetronome(double bpm, double duration, double sampleRate, int accentEvery) {
        double beatInterval = 60.0 / bpm; // 每拍间隔 (秒) / Beat interval (seconds)
        int totalSamples = (int)(duration * sampleRate);
        int samplesPerBeat = (int)(beatInterval * sampleRate);
        
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        for (int beat = 0; beat * samplesPerBeat < totalSamples; beat++) {
            int startSample = beat * samplesPerBeat;
            int endSample = Math.min(startSample + samplesPerBeat, totalSamples);
            int beatSamples = endSample - startSample;
            
            // 生成节拍音 / Generate beat sound
            double frequency = beat % accentEvery == 0 ? 1000 : 800; // 重音频率更高 / Accent has higher frequency
            double amplitude = beat % accentEvery == 0 ? 0.5 : 0.3; // 重音幅度更大 / Accent has higher amplitude
            
            IVector<Double> beatAudio = Signals.sineWave(
                beatSamples, frequency, sampleRate, amplitude, 0.0);
            
            // 添加短促的节拍音 / Add short beat sound
            int clickDuration = (int)(sampleRate * 0.1); // 0.1秒的节拍音 / 0.1 second beat sound
            for (int i = 0; i < Math.min(clickDuration, beatSamples); i++) {
                samples.set(startSample + i, beatAudio.get(i));
            }
        }
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成琶音 / Generate arpeggio
     * <p>
     * 生成指定和弦的琶音音频。
     * Generate arpeggio audio for specified chord.
     * </p>
     *
     * @param chord 和弦 / Chord
     * @param octave 八度 / Octave
     * @param noteDuration 每个音符的持续时间 (秒) / Duration of each note (seconds)
     * @param sampleRate 采样率 / Sample rate
     * @return 琶音音频数据 / Arpeggio audio data
     */
    public static AudioData generateArpeggio(MusicTheory.Chord chord, int octave, 
                                           double noteDuration, double sampleRate) {
        int[] chordNotes = chord.getNotes();
        IVector<Double> totalSamples = Linalg.zeros(0);
        
        // 上行琶音 / Ascending arpeggio
        for (int note : chordNotes) {
            double frequency = MusicTheory.noteToFrequency(MusicTheory.NOTE_NAMES[note], octave);
            IVector<Double> noteSamples = Signals.sineWave(
                (int)(noteDuration * sampleRate), frequency, sampleRate, 0.3, 0.0);
            totalSamples = concatenateAudio(totalSamples, noteSamples);
        }
        
        // 下行琶音 / Descending arpeggio
        for (int i = chordNotes.length - 1; i >= 0; i--) {
            int note = chordNotes[i];
            double frequency = MusicTheory.noteToFrequency(MusicTheory.NOTE_NAMES[note], octave);
            IVector<Double> noteSamples = Signals.sineWave(
                (int)(noteDuration * sampleRate), frequency, sampleRate, 0.3, 0.0);
            totalSamples = concatenateAudio(totalSamples, noteSamples);
        }
        
        return new AudioData(totalSamples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成随机音乐片段 / Generate random music segment
     * <p>
     * 生成包含旋律、和弦和节奏的随机音乐片段。
     * Generate random music segment including melody, chords, and rhythm.
     * </p>
     *
     * @param duration 持续时间 (秒) / Duration (seconds)
     * @param tempo 节拍 (BPM) / Tempo (BPM)
     * @param sampleRate 采样率 / Sample rate
     * @return 音乐片段音频数据 / Music segment audio data
     */
    public static AudioData generateRandomMusicSegment(double duration, double tempo, double sampleRate) {
        // 随机选择调性和音阶 / Randomly select key and scale
        int rootNote = random.nextInt(12);
        MusicTheory.ScaleType[] scaleTypes = MusicTheory.ScaleType.values();
        MusicTheory.ScaleType scaleType = scaleTypes[random.nextInt(scaleTypes.length)];
        
        // 生成旋律 / Generate melody
        Melody melody = generateRandomMelody(rootNote, scaleType, 4, 
                                          (int)(duration * tempo / 60), tempo, 4);
        
        // 生成和弦进行 / Generate chord progression
        MusicTheory.Key key = new MusicTheory.Key(rootNote, scaleType);
        MusicTheory.ChordType[] chordTypes = {
            MusicTheory.ChordType.MAJOR,
            MusicTheory.ChordType.MINOR,
            MusicTheory.ChordType.MAJOR_7TH,
            MusicTheory.ChordType.MINOR_7TH
        };
        
        int numChords = (int)(duration * tempo / 60 / 4); // 每4拍一个和弦 / One chord per 4 beats
        MusicTheory.ChordType[] progression = new MusicTheory.ChordType[numChords];
        for (int i = 0; i < numChords; i++) {
            progression[i] = chordTypes[random.nextInt(chordTypes.length)];
        }
        
        AudioData chordAudio = generateChordProgression(key, progression, 3, 4, tempo);
        
        // 混合旋律和和弦 / Mix melody and chords
        return mixAudio(melody.toAudioData(sampleRate), chordAudio, 0.7, 0.3);
    }
    
    /**
     * 生成旋律音频 / Generate melody audio
     */
    private static AudioData generateMelodyAudio(Melody melody, double sampleRate) {
        double beatDuration = 60.0 / melody.getTempo();
        int totalSamples = 0;
        
        // 计算总长度 / Calculate total length
        for (Note note : melody.getNotes()) {
            int noteSamples = (int)(note.getDuration() * beatDuration * sampleRate);
            totalSamples = Math.max(totalSamples, (int)(note.getStartTime() * beatDuration * sampleRate) + noteSamples);
        }
        
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        // 生成每个音符 / Generate each note
        for (Note note : melody.getNotes()) {
            double frequency = MusicTheory.semitonesToFrequency(note.getPitch());
            int startSample = (int)(note.getStartTime() * beatDuration * sampleRate);
            int noteSamples = (int)(note.getDuration() * beatDuration * sampleRate);
            
            IVector<Double> noteAudio = Signals.sineWave(
                noteSamples, frequency, sampleRate, note.getVelocity() * 0.5, 0.0);
            
            // 添加音符到总音频中 / Add note to total audio
            for (int i = 0; i < noteAudio.length() && startSample + i < totalSamples; i++) {
                samples.set(startSample + i, samples.get(startSample + i) + noteAudio.get(i));
            }
        }
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 拼接音频 / Concatenate audio
     */
    private static IVector<Double> concatenateAudio(IVector<Double> audio1, IVector<Double> audio2) {
        if (audio1.length() == 0) {
            return audio2;
        }
        if (audio2.length() == 0) {
            return audio1;
        }
        
        IVector<Double> result = Linalg.zeros(audio1.length() + audio2.length());
        
        for (int i = 0; i < audio1.length(); i++) {
            result.set(i, audio1.get(i));
        }
        for (int i = 0; i < audio2.length(); i++) {
            result.set(audio1.length() + i, audio2.get(i));
        }
        
        return result;
    }
    
    /**
     * 混合音频 / Mix audio
     */
    private static AudioData mixAudio(AudioData audio1, AudioData audio2, double weight1, double weight2) {
        int maxLength = Math.max(audio1.getLength(), audio2.getLength());
        IVector<Double> mixedSamples = Linalg.zeros(maxLength);
        
        // 混合两个音频 / Mix two audio files
        for (int i = 0; i < maxLength; i++) {
            double sample1 = i < audio1.getLength() ? audio1.getSamples().get(i) : 0;
            double sample2 = i < audio2.getLength() ? audio2.getSamples().get(i) : 0;
            mixedSamples.set(i, sample1 * weight1 + sample2 * weight2);
        }
        
        return new AudioData(mixedSamples, audio1.getSampleRate(), 1, 16, AudioFormat.WAV);
    }
}
