import cv2
import numpy as np
import pyaudio
import sys
import os
import time
import shutil
import threading
from collections import deque

# ═══════════════════════════════════════════════════════════════
#  ADVANCED CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# AESTHETICS
FONT_RATIO = 0.5            # Char height/width ratio (0.5 is standard terminal)
DITHER_STRENGTH = 20.0      # How gritty the image looks (0 = clean, 40 = noisy)
FEEDBACK_DECAY = 0.85       # Trail length (0.0 = no trails, 0.95 = infinite infinite)
USE_OPTICAL_FLOW = True     # Turn off if CPU struggles

# AUDIO BANDS (Hz)
BAND_SUB = (20, 60)
BAND_KICK = (60, 150)
BAND_MID = (300, 2000)
BAND_HIGH = (2500, 10000)

# CYBERPUNK PALETTE (Strict Quantization)
# Format: BGR (OpenCV uses BGR)
PALETTE = np.array([
    [10, 5, 20],      # 0: VOID (Deep Purple/Black)
    [50, 20, 100],    # 1: SHADOW (Dark Violet)
    [180, 50, 180],   # 2: MID (Magenta)
    [0, 255, 255],    # 3: HIGHLIGHT (Cyan)
    [220, 255, 220]   # 4: PEAK (White-ish)
], dtype=np.uint8)

# ASCII MAPS (Density Sorted)
# Complex, technical char set
ASCII_CHARS = np.array(list(" .'`^,:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkha*#MW&8%B@$"))
ASCII_MAP_LEN = len(ASCII_CHARS)

# ═══════════════════════════════════════════════════════════════
#  HIGH-PERFORMANCE AUDIO ENGINE (FFT)
# ═══════════════════════════════════════════════════════════════

class AudioReactor:
    def __init__(self, rate=44100, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
        self.lock = threading.Lock()
        
        # Reactive State
        self.sub = 0.0
        self.kick = 0.0
        self.mid = 0.0
        self.high = 0.0
        self.on_beat = False
        
        # Smoothing
        self.kick_history = deque(maxlen=20)
        
        self.start()

    def start(self):
        try:
            # Try to find a loopback device, else default mic
            dev_idx = None
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                # Look for "monitor" or "stereo mix" for system audio
                if "monitor" in info['name'].lower() or "stereo" in info['name'].lower():
                    dev_idx = i
                    break
            
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                input_device_index=dev_idx,
                frames_per_buffer=self.chunk
            )
            t = threading.Thread(target=self._process, daemon=True)
            t.start()
            print(f"✓ Audio Engine: {'System Loopback' if dev_idx else 'Microphone'}")
        except Exception as e:
            print(f"⚠ Audio Failed: {e}")

    def _process(self):
        while self.running:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                # Normalize and Window
                y = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                y = y * np.hanning(len(y))
                
                # FFT
                fft = np.abs(np.fft.rfft(y))
                freqs = np.fft.rfftfreq(len(y), 1.0/self.rate)
                
                # Band Calculation
                def get_band(low, high):
                    mask = (freqs >= low) & (freqs < high)
                    if not np.any(mask): return 0.0
                    return np.mean(fft[mask]) * 10.0 # Boost gain

                s = get_band(*BAND_SUB)
                k = get_band(*BAND_KICK)
                m = get_band(*BAND_MID)
                h = get_band(*BAND_HIGH)
                
                # Beat Detection (Simple Dynamic Threshold)
                self.kick_history.append(k)
                avg_kick = sum(self.kick_history) / len(self.kick_history)
                is_beat = k > (avg_kick * 1.5) and k > 0.1

                with self.lock:
                    self.sub = np.clip(s, 0, 1)
                    self.kick = np.clip(k, 0, 1)
                    self.mid = np.clip(m, 0, 1)
                    self.high = np.clip(h, 0, 1)
                    self.on_beat = is_beat

            except Exception:
                pass

    def get_state(self):
        with self.lock:
            return self.sub, self.kick, self.mid, self.high, self.on_beat

    def stop(self):
        self.running = False
        self.p.terminate()

# ═══════════════════════════════════════════════════════════════
#  VIDEO PROCESSING (NUMPY VECTORIZATION)
# ═══════════════════════════════════════════════════════════════

# Pre-computed Bayer Matrix for Dithering
BAYER_4x4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
]) / 16.0

def apply_dither(gray, strength):
    h, w = gray.shape
    # Tile the Bayer matrix to cover the image
    tiled_bayer = np.tile(BAYER_4x4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    # Add noise based on matrix
    noisy = gray.astype(np.float32) + (tiled_bayer * strength - (strength / 2))
    return np.clip(noisy, 0, 255).astype(np.uint8)

def render_ascii_frame(frame, sub, kick, mid, high, on_beat):
    """
    The Core Rendering Pipeline:
    1. Pre-process (Resize, Contrast)
    2. Feedback Loop (Ghosting)
    3. Dithering (Texture)
    4. Quantization (Palette Mapping)
    5. ASCII Conversion
    """
    
    # 1. Resize to Terminal
    cols, rows = shutil.get_terminal_size()
    rows -= 1 # Prevent scrolling
    # Correct aspect ratio for font
    target_h = rows
    target_w = int(cols)
    
    # Resize with area interpolation for smoothness before FX
    small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # 2. Glitch Effects (Displacement)
    # On heavy bass/beat, shake the image RGB channels
    if kick > 0.4:
        shift = int(kick * 2)
        # BGR channels
        small[:, :, 0] = np.roll(small[:, :, 0], shift, axis=1) # Blue shift
        small[:, :, 2] = np.roll(small[:, :, 2], -shift, axis=1) # Red shift
    
    # 3. High Contrast & Dither
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Contrast stretch
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply Ordered Dithering
    dithered = apply_dither(gray, DITHER_STRENGTH)
    
    # 4. Quantization (The "Posterize" Look)
    # Map 0-255 to 0-4 (Indices for Palette)
    # We use np.digitize for fast binning
    bins = np.array([50, 100, 180, 230]) # Thresholds
    indices = np.digitize(dithered, bins)
    
    # 5. Color Mapping
    # Vectorized lookup: Create an image of RGB values based on indices
    colored_frame = PALETTE[indices]
    
    # Flash effect on beat (Invert colors)
    if on_beat and kick > 0.6:
        colored_frame = 255 - colored_frame
        
    # 6. ASCII Mapping
    # Map indices (0-4) to ASCII chars
    # We map brightness 0-255 to 0-(len-1)
    ascii_indices = (dithered.astype(np.float32) / 255.0 * (ASCII_MAP_LEN - 1)).astype(int)
    chars = ASCII_CHARS[ascii_indices]
    
    return chars, colored_frame

# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    audio = AudioReactor()
    cap = cv2.VideoCapture(0)
    
    # Optimize Camera
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Optical Flow State
    prev_gray = None
    feedback_buffer = None
    
    # Clear Screen
    sys.stdout.write("\033[2J\033[?25l")
    
    try:
        while True:
            # 1. Capture & Audio
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1) # Mirror
            
            sub, kick, mid, high, beat = audio.get_state()
            
            # 2. Optical Flow & Feedback (The "Smear" Effect)
            # Resize for flow calc (performance)
            h, w = frame.shape[:2]
            small_h, small_w = 200, int(200 * (w/h))
            curr_gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (small_w, small_h))
            
            if prev_gray is not None and USE_OPTICAL_FLOW:
                # Calculate flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Calculate magnitude
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Feedback Effect:
                # If there is movement, blend previous frame trails
                # The "decay" depends on the SUB-BASS. More bass = longer trails.
                current_decay = FEEDBACK_DECAY + (sub * 0.1)
                current_decay = min(0.98, current_decay)
                
                if feedback_buffer is None:
                    feedback_buffer = frame.astype(np.float32)
                
                # Blend: New Frame + (Old Frame * Decay)
                # But only where there is motion? No, global feedback looks better for "void" style
                feedback_buffer = cv2.addWeighted(frame.astype(np.float32), 1.0 - current_decay, feedback_buffer, current_decay, 0)
                
                # On heavy kick, Zoom the feedback buffer out (Screen Pump)
                if kick > 0.5:
                    center_x, center_y = w // 2, h // 2
                    zoom = 1 + (kick * 0.05)
                    M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom)
                    feedback_buffer = cv2.warpAffine(feedback_buffer, M, (w, h))

                frame_to_render = feedback_buffer.astype(np.uint8)
            else:
                frame_to_render = frame
                
            prev_gray = curr_gray_small

            # 3. Render ASCII
            chars, colors = render_ascii_frame(frame_to_render, sub, kick, mid, high, beat)
            
            # 4. Construct Output String (Fastest Python method)
            # Create a list of strings row by row
            # We use ANSI TrueColor escape codes: \033[38;2;R;G;Bm
            
            out_lines = []
            rows, cols, _ = colors.shape
            
            # Vectorized string construction is hard in Python, falling back to optimized loop
            # But we can minimize string ops
            for y in range(rows):
                line = []
                # Pre-calculate row data to avoid repeated indexing
                row_chars = chars[y]
                row_colors = colors[y]
                
                last_color = None
                
                for x in range(cols):
                    c = row_chars[x]
                    b, g, r = row_colors[x] # OpenCV is BGR
                    
                    # Optimization: Only write color code if it changes
                    # (Simple RLE for terminal output)
                    current_color = (r, g, b)
                    if current_color != last_color:
                        line.append(f"\033[38;2;{r};{g};{b}m")
                        last_color = current_color
                    
                    line.append(c)
                out_lines.append("".join(line))
            
            # Single Write
            sys.stdout.write("\033[H" + "\n".join(out_lines))
            sys.stdout.flush()
            
            # Cap FPS to prevent tearing
            # time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[0m\033[2J")
        sys.stdout.flush()
        audio.stop()
        cap.release()

if __name__ == "__main__":
    main()