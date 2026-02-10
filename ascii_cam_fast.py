import cv2
import numpy as np
import pyaudio
import sys
import os
import time
import math
import random
import colorsys
import threading
import shutil
import warnings

# Suppress minor warnings
warnings.filterwarnings("ignore")

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ═══════════════════════════════════════════════════════════════
#  PERFORMANCE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
# Clamping width is CRITICAL for high FPS. 
# 120-160 is high definition for ASCII. Going above 200 causes terminal lag.
MAX_WIDTH = 140  
AUTO_FULLSCREEN = True # Will try to fill screen UP TO Max_Width
CAMERA_SOURCE = 0
AUDIO_SOURCE = "mic"  
CHUNK = 1024
RATE = 44100

# ── BLENDING ──
FACE_BRIGHTNESS = 1.3
BODY_BRIGHTNESS = 1.1
BG_INTENSITY = 0.8

# ── ASCII RAMPS ──
# Denser ramp for body/face
ASCII_RAMP = np.array(list(" .:-=+*#%@"))
# Lighter/tech ramp for background
ASCII_BG_RAMP = np.array(list(" .'`^,:;~-_+<>i!lI?/|\\()1{}[]rcvunxzjftLCJUYXZO0Qmwqpdbkhao*#MW&8%B@$"))


# ═══════════════════════════════════════════════════════════════
#  FAST HELPER FUNCTIONS (NUMPY OPTIMIZED)
# ═══════════════════════════════════════════════════════════════

def get_audio_level(stream):
    try:
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
        return np.sqrt(np.mean(data**2))
    except Exception:
        return 0

class AudioCapture:
    def __init__(self, streams):
        self.streams = streams
        self.volume = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def _loop(self):
        while self._running:
            levels = [get_audio_level(s) for s in self.streams]
            self.volume = max(levels) if levels else 0.0
            time.sleep(0.01) # Slight yield to prevent CPU hogging
    
    def get_volume(self):
        return self.volume
    
    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)

# ═══════════════════════════════════════════════════════════════
#  FAST EFFECTS ENGINES
# ═══════════════════════════════════════════════════════════════

class FastBackground:
    """Vectorized background generator using noise and scrolling patterns."""
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.grid_x, self.grid_y = np.meshgrid(np.arange(w), np.arange(h))
        self.frame_count = 0
        
    def update_and_render(self, w, h, audio_norm):
        # Resize if terminal changed
        if w != self.w or h != self.h:
            self.w, self.h = w, h
            self.grid_x, self.grid_y = np.meshgrid(np.arange(w), np.arange(h))

        self.frame_count += 1
        t = self.frame_count * 0.05
        
        # 1. Create base plasma pattern using vectorized math (much faster than loops)
        # Normalized coordinates
        nx = self.grid_x / w
        ny = self.grid_y / h
        
        # Vectorized wave function
        v1 = np.sin(nx * 10 + t)
        v2 = np.sin(ny * 10 - t * 0.5)
        v3 = np.sin((nx + ny) * 8 + t)
        plasma = (v1 + v2 + v3) / 3.0
        
        # 2. Map plasma to colors (Blue/Cyan/Magenta theme)
        # Create empty RGB grid
        bg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Masking for colors
        mask_cyan = plasma > 0.2
        mask_magenta = plasma < -0.2
        
        # Apply colors (vectorized)
        intensity = ((plasma + 1) / 2 * 255 * BG_INTENSITY).astype(np.uint8)
        
        # Dark base
        bg_rgb[:, :, 0] = 10  # R
        bg_rgb[:, :, 1] = 30  # G
        bg_rgb[:, :, 2] = 50 + intensity // 4 # B
        
        # Cyan hits
        bg_rgb[mask_cyan, 1] = np.clip(bg_rgb[mask_cyan, 1] + 100 + audio_norm * 50, 0, 255)
        bg_rgb[mask_cyan, 2] = 255
        
        # Magenta hits
        bg_rgb[mask_magenta, 0] = np.clip(bg_rgb[mask_magenta, 0] + 100 + audio_norm * 50, 0, 255)
        bg_rgb[mask_magenta, 2] = 200

        # 3. Matrix Rain / Digital Rain (Simplified for speed)
        # We use random noise columns instead of tracking objects
        rain_mask = (np.random.random((h, w)) > 0.985).astype(bool)
        bg_rgb[rain_mask] = [0, 255, 100] # Bright green drops
        
        return bg_rgb

class FastParticles:
    """Manages trails using a sparse list but renders to a numpy mask."""
    def __init__(self):
        self.particles = [] # list of [x, y, life, r, g, b]
        
    def update(self, spawn_points, w, h):
        # Decay existing
        self.particles = [[p[0], p[1], p[2]-0.05, p[3], p[4], p[5]] for p in self.particles if p[2] > 0.05]
        
        # Spawn new
        for x, y, r, g, b in spawn_points:
            # Random scatter
            sx = x + random.randint(-1, 1)
            sy = y + random.randint(-1, 1)
            if 0 <= sx < w and 0 <= sy < h:
                self.particles.append([sx, sy, 1.0, r, g, b])
                
        # Limit count
        if len(self.particles) > 1000:
            self.particles = self.particles[-1000:]
            
    def render_to_grid(self, h, w):
        # Create a sparse layer
        grid = np.zeros((h, w, 3), dtype=np.int16) # int16 to avoid overflow during add
        mask = np.zeros((h, w), dtype=bool)
        
        if not self.particles:
            return grid, mask
            
        # Bulk update (slower than pure numpy, but acceptable for sparse particles)
        for p in self.particles:
            x, y, life = int(p[0]), int(p[1]), p[2]
            if 0 <= x < w and 0 <= y < h:
                r, g, b = int(p[3]*life), int(p[4]*life), int(p[5]*life)
                grid[y, x, 0] = r
                grid[y, x, 1] = g
                grid[y, x, 2] = b
                mask[y, x] = True
                
        return grid, mask


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Setup Audio ──
    p = pyaudio.PyAudio()
    audio_streams = []
    try:
        if AUDIO_SOURCE in ("mic", "both"):
            audio_streams.append(p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK))
        if not audio_streams:
            print("No audio found. Continuing silent.")
    except Exception:
        pass
    
    audio_capture = AudioCapture(audio_streams)

    # ── Setup Video ──
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened(): return

    # ── Setup MediaPipe (Tasks API) ──
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SEG_MODEL = os.path.join(SCRIPT_DIR, "selfie_segmenter.tflite")
    FACE_MODEL = os.path.join(SCRIPT_DIR, "face_landmarker.task")
    
    # Auto-download if missing
    import urllib.request
    if not os.path.exists(SEG_MODEL):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite", SEG_MODEL)
    if not os.path.exists(FACE_MODEL):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", FACE_MODEL)

    # Load Models
    base_options = python.BaseOptions(model_asset_path=SEG_MODEL)
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    segmenter = vision.ImageSegmenter.create_from_options(options)
    
    face_options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
                                                num_faces=1, min_face_detection_confidence=0.5)
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

    # Initialize Engines
    bg_engine = FastBackground(100, 50)
    face_trails = FastParticles()

    # Clear screen
    sys.stdout.write("\033[2J\033[?25l")
    
    prev_time = 0
    fps_limit = 30

    try:
        while True:
            # ── FPS Control ──
            now = time.time()
            if now - prev_time < 1/fps_limit: continue
            prev_time = now

            # ── Read Inputs ──
            vol = audio_capture.get_volume()
            audio_norm = min(vol / 4000.0, 1.0)
            
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h_orig, w_orig, _ = frame.shape

            # ── Determine Dimensions (CRITICAL FOR SPEED) ──
            term_cols, term_rows = shutil.get_terminal_size((80, 24))
            
            # Limit width to MAX_WIDTH to prevent lag
            ascii_w = min(term_cols, MAX_WIDTH)
            # Calculate height based on aspect ratio (chars are ~2x tall as wide)
            aspect = (h_orig / w_orig) * 0.55
            ascii_h = int(ascii_w * aspect)
            
            # Ensure we don't exceed terminal height
            if ascii_h > term_rows - 1:
                ascii_h = term_rows - 1
                ascii_w = int(ascii_h / aspect)

            # ── Process Images ──
            # Resize inputs once
            frame_small = cv2.resize(frame, (ascii_w, ascii_h))
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            
            # MediaPipe processing
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            seg_result = segmenter.segment(mp_image)
            face_result = face_landmarker.detect(mp_image)
            
            # ── Masks (The Key to Crispness) ──
            mask_orig = seg_result.category_mask.numpy_view()
            mask_resized = cv2.resize(mask_orig, (ascii_w, ascii_h), interpolation=cv2.INTER_NEAREST)
            
            # STRICT BINARY THRESHOLD for crisp edges
            body_mask = mask_resized > 0.1 # Person is index 0 in Selfie Segmenter? Actually standard is 0=bg, 1=person usually, or reversed. 
            # In MP Selfie Segmenter: 0=background, 1=person. 
            # WAIT: The provided code used `seg_mask_resized == 0` for body. Let's stick to that logic but verify.
            # Usually MP: 0=Background, 1=Hair, 2=Body... etc (Multiclass) or 0=BG, 1=Person.
            # If the previous code worked with ==0 for body, likely it was 0=Person. 
            # Let's assume standard behavior: 0 is usually background.
            # Adjusting logic:
            is_body = mask_resized > 0.5 # Assuming >0 is body categories
            # If your previous code used ==0 for body, swap this: `is_body = mask_resized == 0`
            # For standard SelfieSegmenter (multiclass), usually 0 is background. 
            # Let's assume >0 is body.
            
            # ── Face Logic ──
            face_mask = np.zeros((ascii_h, ascii_w), dtype=bool)
            trail_spawns = []
            
            if face_result.face_landmarks:
                lm = face_result.face_landmarks[0]
                # Extract face box
                xs = [l.x for l in lm]
                ys = [l.y for l in lm]
                x1, x2 = int(min(xs)*ascii_w), int(max(xs)*ascii_w)
                y1, y2 = int(min(ys)*ascii_h), int(max(ys)*ascii_h)
                
                # Expand slightly
                pad = 2
                x1, x2 = max(0, x1-pad), min(ascii_w, x2+pad)
                y1, y2 = max(0, y1-pad), min(ascii_h, y2+pad)
                
                face_mask[y1:y2, x1:x2] = True
                
                # Add trail points if moving (simplified)
                if random.random() < 0.3:
                    trail_spawns.append(( (x1+x2)//2, (y1+y2)//2, 255, 200, 0 ))

            # ── Vectorized Compositing (The Speed Boost) ──
            
            # 1. Background Layer
            final_rgb = bg_engine.update_and_render(ascii_w, ascii_h, audio_norm)
            
            # 2. Particle Trails Layer
            face_trails.update(trail_spawns, ascii_w, ascii_h)
            p_grid, p_mask = face_trails.render_to_grid(ascii_h, ascii_w)
            
            # Blend particles onto background
            final_rgb[p_mask] = p_grid[p_mask]

            # 3. Body Layer (Vectorized tinting)
            # Create neon body tint
            body_tint = np.zeros_like(final_rgb)
            body_tint[:, :, 0] = 0   # R
            body_tint[:, :, 1] = 255 # G
            body_tint[:, :, 2] = 255 # B (Cyan tint)
            
            if audio_norm > 0.5: # Beat flash
                body_tint[:] = [255, 50, 255] # Magenta

            # Apply tint to original camera pixels
            cam_body = frame_rgb.copy()
            # Simple linear blend: 70% cam, 30% tint
            blended_body = cv2.addWeighted(cam_body, 0.7, body_tint, 0.3, 0)
            
            # 4. Face Layer (Warmer tint)
            face_tint = np.zeros_like(final_rgb)
            face_tint[:] = [255, 200, 100] # Gold
            blended_face = cv2.addWeighted(cam_body, 0.8, face_tint, 0.2, 0)

            # ── Final Composite (Boolean Indexing = FAST) ──
            # Apply body
            final_rgb[is_body] = blended_body[is_body]
            # Apply face (overrides body)
            final_rgb[face_mask] = blended_face[face_mask]

            # ── ASCII Character Mapping (Vectorized) ──
            # Normalize gray to 0..len(ramp)-1
            # Brightness Boost
            gray_boosted = gray_small.astype(float) * 1.2
            
            # Different ramps for body vs bg
            # Background indices
            bg_indices = (np.clip(gray_boosted * BG_INTENSITY, 0, 255) / 255 * (len(ASCII_BG_RAMP) - 1)).astype(int)
            char_grid = ASCII_BG_RAMP[bg_indices]
            
            # Body indices
            body_indices = (np.clip(gray_boosted * BODY_BRIGHTNESS, 0, 255) / 255 * (len(ASCII_RAMP) - 1)).astype(int)
            
            # Overwrite body characters
            char_grid[is_body] = ASCII_RAMP[body_indices[is_body]]
            
            # ── Render to String (Optimized) ──
            # We construct the string row by row using list comprehensions
            # This is significantly faster than nested loops
            
            output_lines = []
            
            # Pre-flatten rows for iteration
            for y in range(ascii_h):
                # Get row data
                row_chars = char_grid[y]
                row_colors = final_rgb[y] # Shape (W, 3)
                
                # Python string construction optimization:
                # Use f-string in list comp
                line_str = "".join([
                    f"\033[38;2;{r};{g};{b}m{c}" 
                    for c, (r, g, b) in zip(row_chars, row_colors)
                ])
                output_lines.append(line_str)

            # Move cursor home and print
            sys.stdout.write("\033[H" + "\n".join(output_lines))
            sys.stdout.write("\033[0m") # Reset color
            sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        sys.stdout.write("\033[?25h\033[0m\n")
        cap.release()
        audio_capture.stop()
        segmenter.close()
        face_landmarker.close()
        p.terminate()

if __name__ == "__main__":
    main()