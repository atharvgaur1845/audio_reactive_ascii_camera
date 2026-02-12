import cv2
import numpy as np
import pyaudio
import sys
import os
import time
import shutil
import threading
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# AESTHETICS
BG_DITHER_STRENGTH = 25.0   # Grittiness of background
BG_TRAIL_DECAY = 0.90       # 0.90 = Long psychedelic trails
FG_CONTRAST = 1.2           # Make person pop more
FG_BRIGHTNESS = 10          # Slight boost for visibility

# AUDIO BANDS
BAND_BASS = (20, 100)
BAND_MID = (300, 2000)
BAND_TREBLE = (3000, 8000)

# ASCII RAMPS
# 1. High Fidelity (For Body/Face) - Dense to show details
RAMP_REAL = list(" $B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
# 2. Abstract (For Background) - Sparse/Matrix style
RAMP_VOID = list(" @%#*+=-:. ")

# VOID PALETTE (Background Only)
PALETTE_VOID = np.array([
    [5, 0, 5],        # Void Black
    [40, 0, 60],      # Deep Purple
    [0, 100, 100],    # Dark Teal
    [180, 0, 255],    # Neon Purple
    [0, 255, 255]     # Cyber Cyan
], dtype=np.uint8)

# ═══════════════════════════════════════════════════════════════
#  AUDIO ENGINE (FFT)
# ═══════════════════════════════════════════════════════════════
class AudioReactor:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
        self.lock = threading.Lock()
        self.bass = 0.0
        self.mid = 0.0
        self.treble = 0.0
        self.beat = False
        
        self.bass_hist = deque(maxlen=20)
        self.start()

    def start(self):
        try:
            dev_idx = None
            # Auto-detect loopback/monitor if available
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                if "monitor" in info['name'].lower() or "stereo" in info['name'].lower():
                    dev_idx = i
                    break
            
            self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100,
                                      input=True, input_device_index=dev_idx, frames_per_buffer=1024)
            threading.Thread(target=self._run, daemon=True).start()
            print(f"✓ Audio: {'System Loopback' if dev_idx else 'Mic/Default'}")
        except Exception as e:
            print(f"⚠ Audio Error: {e}")

    def _run(self):
        while self.running:
            try:
                data = self.stream.read(1024, exception_on_overflow=False)
                y = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                fft = np.abs(np.fft.rfft(y * np.hanning(len(y))))
                freqs = np.fft.rfftfreq(len(y), 1.0/44100)

                def get_energy(low, high):
                    m = (freqs >= low) & (freqs < high)
                    return np.mean(fft[m]) * 5.0 if np.any(m) else 0

                b = get_energy(*BAND_BASS)
                m = get_energy(*BAND_MID)
                t = get_energy(*BAND_TREBLE)
                
                # Beat logic
                self.bass_hist.append(b)
                avg = sum(self.bass_hist)/len(self.bass_hist)
                is_beat = b > avg * 1.4 and b > 0.1

                with self.lock:
                    self.bass = np.clip(b, 0, 1)
                    self.mid = np.clip(m, 0, 1)
                    self.treble = np.clip(t, 0, 1)
                    self.beat = is_beat
            except: pass

    def get(self):
        with self.lock: return self.bass, self.mid, self.treble, self.beat
    
    def stop(self):
        self.running = False
        self.p.terminate()

# ═══════════════════════════════════════════════════════════════
#  RENDER ENGINE
# ═══════════════════════════════════════════════════════════════

def get_dither_matrix(h, w):
    # 4x4 Bayer Matrix
    bayer = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ]) / 16.0
    return np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]

def main():
    # 1. Setup MediaPipe
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "selfie_segmenter.tflite")
    
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        print("Downloading segmentation model...")
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite", MODEL_PATH)

    options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        output_category_mask=True)
    segmenter = vision.ImageSegmenter.create_from_options(options)

    # 2. Init
    audio = AudioReactor()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Background Feedback Buffer
    bg_buffer = None
    
    # Pre-compute ASCII lookup arrays for speed
    def create_lookup(ramp):
        return np.array(ramp)
    
    CHARS_REAL = create_lookup(RAMP_REAL)
    CHARS_VOID = create_lookup(RAMP_VOID)

    sys.stdout.write("\033[2J\033[?25l")

    try:
        while True:
            # A. Inputs
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            bass, mid, treble, beat = audio.get()
            
            # Sizing (Terminal Auto-Fit)
            cols, rows = shutil.get_terminal_size()
            t_h, t_w = rows - 1, cols
            
            # B. Segmentation
            # We process segmentation on a small image for speed, then scale up
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            seg_result = segmenter.segment(mp_img)
            mask_raw = seg_result.category_mask.numpy_view()
            
            # Resize everything to Terminal Size
            # 1. Frame (Real Color)
            frame_small = cv2.resize(frame, (t_w, t_h), interpolation=cv2.INTER_AREA)
            # 2. Mask (Boolean: True=Person)
            mask_small = cv2.resize(mask_raw, (t_w, t_h), interpolation=cv2.INTER_NEAREST)
            is_person = mask_small == 0 # MediaPipe: 0 is person
            
            # C. PIPELINE 1: THE REALISTIC PERSON
            # Enhance contrast/brightness so you pop against the void
            fg_frame = cv2.convertScaleAbs(frame_small, alpha=FG_CONTRAST, beta=FG_BRIGHTNESS)
            fg_gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)
            
            # Map to Dense ASCII
            # Normalize 0-255 to 0-(len-1)
            fg_indices = (fg_gray.astype(float) / 255 * (len(CHARS_REAL) - 1)).astype(int)
            fg_chars = CHARS_REAL[fg_indices]

            # D. PIPELINE 2: THE VOID BACKGROUND
            # 1. Feedback Loop (Trails)
            if bg_buffer is None: bg_buffer = frame_small.astype(float)
            
            # Decay trails based on Bass (More bass = longer trails)
            decay = BG_TRAIL_DECAY + (bass * 0.08)
            decay = min(0.98, decay)
            
            # Blend: New Frame + Old Buffer
            # CRITICAL: We only feed the BACKGROUND into the buffer to avoid smearing the person
            # Create a frame where person is blacked out
            bg_input = frame_small.copy()
            bg_input[is_person] = 0 
            
            bg_buffer = cv2.addWeighted(bg_input.astype(float), 1.0 - decay, bg_buffer, decay, 0)
            
            # 2. Reactivity (Zoom/Shift)
            bg_render = bg_buffer.astype(np.uint8)
            if beat: # Screen pump on kick
                M = cv2.getRotationMatrix2D((t_w//2, t_h//2), (np.random.rand()-0.5)*2, 1.0 + (bass * 0.05))
                bg_render = cv2.warpAffine(bg_render, M, (t_w, t_h))

            # 3. Dither & Quantize
            bg_gray = cv2.cvtColor(bg_render, cv2.COLOR_BGR2GRAY)
            # Add Bayer noise
            dither_mat = get_dither_matrix(t_h, t_w)
            bg_noisy = np.clip(bg_gray + (dither_mat * BG_DITHER_STRENGTH), 0, 255).astype(np.uint8)
            
            # Quantize color (Palette mapping)
            # Map brightness buckets to our 5 void colors
            bins = np.linspace(0, 255, len(PALETTE_VOID))
            p_indices = np.digitize(bg_noisy, bins) - 1
            p_indices = np.clip(p_indices, 0, len(PALETTE_VOID)-1)
            bg_colors = PALETTE_VOID[p_indices]
            
            # Map to Sparse ASCII
            bg_char_indices = (bg_noisy.astype(float) / 255 * (len(CHARS_VOID) - 1)).astype(int)
            bg_chars = CHARS_VOID[bg_char_indices]

            # E. EDGE DETECTION (AURA)
            # Find the edge of the mask
            mask_uint8 = (is_person * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edge_map = np.zeros_like(mask_uint8)
            cv2.drawContours(edge_map, contours, -1, 255, 1) # 1px thick edge
            is_edge = edge_map > 0

            # F. COMPOSITING (The Merge)
            # We build the output string loop manually for control
            
            out_lines = []
            
            # Vectorize the data for fast loop
            # fg_colors = fg_frame (BGR)
            # bg_colors = bg_colors (BGR from palette)
            
            for y in range(t_h):
                line_parts = []
                last_color = None
                
                # Pre-fetch row data
                r_mask = is_person[y]
                r_edge = is_edge[y]
                
                # Person Data
                r_fg_char = fg_chars[y]
                r_fg_col = fg_frame[y]
                
                # BG Data
                r_bg_char = bg_chars[y]
                r_bg_col = bg_colors[y]
                
                for x in range(t_w):
                    # DECISION LOGIC
                    if r_edge[x] and treble > 0.4:
                        # 1. The Aura (Edge) - Pulses White/Electric Blue on Treble
                        char = "░" # Sparkle char
                        b, g, r = 255, 255, 200 # Electric
                    
                    elif r_mask[x]:
                        # 2. The Person (Realistic)
                        char = r_fg_char[x]
                        b, g, r = r_fg_col[x]
                        # Slight glitch if heavy beat
                        if beat and np.random.rand() < 0.05:
                            b, g, r = 255, 255, 255 # Glitch white
                            
                    else:
                        # 3. The Void (Background)
                        char = r_bg_char[x]
                        b, g, r = r_bg_col[x]
                    
                    # Optimization: Only write color code if changed
                    curr_color = (r, g, b)
                    if curr_color != last_color:
                        line_parts.append(f"\033[38;2;{r};{g};{b}m")
                        last_color = curr_color
                    
                    line_parts.append(char)
                
                out_lines.append("".join(line_parts))

            # Draw
            sys.stdout.write("\033[H" + "\n".join(out_lines))
            sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[0m\n")
        audio.stop()
        cap.release()

if __name__ == "__main__":
    main()