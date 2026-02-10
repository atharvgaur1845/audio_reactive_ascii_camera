import cv2
import numpy as np
import pyaudio
import sys
import os
import time
import math
import random
import colorsys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
WIDTH = 160           # Higher resolution for better detail
CAMERA_SOURCE = 0     # 0 = default webcam, 1 = external/phone, or "http://IP:PORT/video"
CHUNK = 1024          # Audio chunk size
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# ASCII gradients
# Standard high-density ASCII ramp (sorted by visual brightness)
ASCII_RAMP = list(" .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkha*#MW&8%B@$")
ASCII_BODY = ASCII_RAMP
ASCII_FACE = ASCII_RAMP # Use same detailed ramp for face
ASCII_BG = ASCII_RAMP
# Sci-fi glyphs
MATRIX_CHARS = list("01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン")
HEX_CHARS = list("0123456789ABCDEF:><[]{}|/-\\")
HUD_CHARS = list("╔╗╚╝═║╠╣")

# ═══════════════════════════════════════════════════════════════
#  NEON COLOR PALETTES
# ═══════════════════════════════════════════════════════════════

# Dynamic palettes that can be cycled
PALETTES = {
    "CYBERPUNK": [
        (0, 255, 255),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
    ],
    "MATRIX": [
        (0, 255, 0),      # Matrix Green
        (0, 128, 0),      # Dark Green
        (150, 255, 150),  # Pale Green
        (0, 50, 0),       # Very Dark Green
    ],
    "PLASMA": [
        (255, 0, 0),      # Red
        (255, 100, 0),    # Orange
        (255, 0, 100),    # Pink
        (100, 0, 255),    # Purple
    ]
}

# Face colors — cyberpunk gold/amber
FACE_COLORS = [
    (255, 215, 0),    # Gold
    (255, 165, 0),    # Orange
    (255, 69, 0),     # Red-orange
    (255, 255, 100),  # Bright yellow
]

# Background sci-fi palette
BG_BASE_COLOR = (10, 5, 20)       # Deep dark purple/black


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB tuples."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def sample_gradient(colors, t):
    """Sample a color from a list of gradient keyframes at position t (0-1)."""
    t = t % 1.0
    n = len(colors) - 1
    idx = t * n
    i = int(idx)
    frac = idx - i
    i = min(i, n - 1)
    return lerp_color(colors[i], colors[i + 1], frac)


def get_audio_level(stream):
    """Reads audio chunk and returns RMS volume."""
    try:
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
        rms = np.sqrt(np.mean(data**2))
        return rms
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════
#  SCI-FI BACKGROUND EFFECTS ENGINE
# ═══════════════════════════════════════════════════════════════

class SciFiBackground:
    """Manages animated sci-fi background effects: Matrix rain, ripples, pulses."""

    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.frame_count = 0
        
        # Matrix rain
        self.rain_cols = []
        for x in range(width):
            self.rain_cols.append({
                'y': random.randint(-height, height),
                'speed': random.randint(1, 3),
                'len': random.randint(5, 15),
                'active': random.random() < 0.2
            })
            
        # Holographic ripples (x, y, radius, max_radius)
        self.ripples = []

    def update(self, audio_level, audio_norm):
        self.frame_count += 1
        
        # Update Rain
        for col in self.rain_cols:
            if col['active']:
                col['y'] += col['speed']
                if col['y'] > self.h + col['len']:
                    col['y'] = -random.randint(5, 20)
                    col['active'] = random.random() < 0.3 # Randomly restart or stop
            else:
                if random.random() < 0.01:
                    col['active'] = True
                    col['y'] = -col['len']

        # Spawn ripples on loud audio
        if audio_norm > 0.6 and random.random() < 0.3:
            self.ripples.append({
                'x': random.randint(0, self.w),
                'y': random.randint(0, self.h),
                'r': 0,
                'max_r': random.randint(10, 30),
                'life': 1.0
            })
            
        # Update Ripples
        for r in self.ripples[:]:
            r['r'] += 1.5
            r['life'] -= 0.05
            if r['life'] <= 0 or r['r'] > r['max_r']:
                self.ripples.remove(r)

    def get_effect(self, x, y, brightness):
        # 1. Matrix Rain
        col = self.rain_cols[x]
        if col['active']:
            dist = y - col['y']
            if 0 <= dist < col['len']:
                norm_dist = dist / col['len'] # 0 (head) to 1 (tail)
                
                # Head is bright white
                if dist < 1:
                    return random.choice(MATRIX_CHARS), 200, 255, 200
                
                # Tail is green/fading
                g_val = int(255 * (1.0 - norm_dist))
                return random.choice(MATRIX_CHARS), 0, max(50, g_val), 0
        
        # 2. Ripples
        in_ripple = False
        ripple_strength = 0
        for r in self.ripples:
            d = math.sqrt((x - r['x'])**2 + (y - r['y'])**2)
            if abs(d - r['r']) < 2.0:
                 in_ripple = True
                 ripple_strength = max(ripple_strength, r['life'])

        if in_ripple:
             val = int(255 * ripple_strength)
             return random.choice(HEX_CHARS), val, int(val*0.5), 255 # Blue-ish ripple

        # 3. Default Background
        n = len(ASCII_BG)
        idx = min(int(brightness / 255.0 * (n - 1)), n - 1)
        char = ASCII_BG[idx]
        
        base = BG_BASE_COLOR
        b_factor = brightness / 255.0
        return char, int(base[0]*b_factor), int(base[1]*b_factor), int(base[2]*b_factor)


# ═══════════════════════════════════════════════════════════════
#  BODY EFFECTS ENGINE
# ═══════════════════════════════════════════════════════════════

class BodyEffects:
    """Manages body visualization: Digital aura, circuit patterns, dynamic gradients."""
    
    def __init__(self):
        self.frame_count = 0
        self.pattern_offset = 0.0
        self.current_palette_name = "CYBERPUNK"
        self.target_palette_name = "CYBERPUNK"
        
    def update(self, audio_norm):
        self.frame_count += 1
        self.pattern_offset += 0.1 + (audio_norm * 0.2)
        
        # Switch palettes based on intensity
        if audio_norm > 0.8:
            self.target_palette_name = "PLASMA"
        elif audio_norm > 0.4:
             self.target_palette_name = "CYBERPUNK"
        else:
             self.target_palette_name = "MATRIX"
             
        if self.current_palette_name != self.target_palette_name:
             self.current_palette_name = self.target_palette_name

    def get_color(self, x, y, brightness, w, h, frame_small, is_aura=False, blend_factor=0.5):
        
        if is_aura:
            # Aura is electric blue/white, sparse for effect
            if random.random() < 0.4:
                return ':', 100, 200, 255
            return ' ', 0, 0, 0
            
        # ─── REALISM: STRICTLY USE BRIGHTNESS FOR CHARACTER ───
        n = len(ASCII_BODY)
        # Gamma correction for better midtones
        norm_b = brightness / 255.0
        idx = min(int(norm_b * (n - 1)), n - 1)
        char = ASCII_BODY[idx]

        # ─── COLOR LOGIC ───
        # Get original color
        b_orig, g_orig, r_orig = frame_small[y, x] # OpenCV is BGR
        
        # Circuit / Data Pattern affects COLOR only
        nx = x / w
        ny = y / h
        
        wave = math.sin(nx * 10 + ny * 10 - self.pattern_offset) 
        wave2 = math.cos(nx * 20 - ny * 5 + self.pattern_offset * 0.5)
        pattern = (wave + wave2) / 2.0 # -1 to 1
        
        # Base Color from Palette
        palette = PALETTES[self.current_palette_name]
        t = (pattern + 1.0) / 2.0 
        r_neon, g_neon, b_neon = sample_gradient(palette, t)
        
        # Blend original with neon
        # We want the original structure but the neon vibe
        r = int(r_orig * blend_factor + r_neon * (1 - blend_factor))
        g = int(g_orig * blend_factor + g_neon * (1 - blend_factor))
        b = int(b_orig * blend_factor + b_neon * (1 - blend_factor))
        
        # Modulate by brightness (so dark areas stay dark)
        # But don't double-darken too much since original color already has brightness info
        # Let's just boost saturation/brightness slightly if needed
        
        # Highlight bright spots white
        if brightness > 230:
            r, g, b = 255, 255, 255
            
        return char, r, g, b


# ═══════════════════════════════════════════════════════════════
#  FACE EFFECTS ENGINE
# ═══════════════════════════════════════════════════════════════

class FaceEffects:
    """Animated face-region effects: HUD, Glitch, Highlights."""

    def __init__(self):
        self.scan_line_y = 0
        self.scan_speed = 2
        self.frame_count = 0

    def update(self, face_top, face_bottom):
        self.frame_count += 1
        h = face_bottom - face_top
        if h > 0:
            self.scan_line_y = face_top + (self.frame_count * self.scan_speed) % h

    def get_color(self, x, y, brightness, face_rect, audio_norm, frame_small):
        ft, fb, fl, fr = face_rect
        b_orig, g_orig, r_orig = frame_small[y, x]
        
        # HUD Corners
        if x == fl and y == ft: return '╔', 255, 255, 0
        if x == fr and y == ft: return '╗', 255, 255, 0
        if x == fl and y == fb: return '╚', 255, 255, 0
        if x == fr and y == fb: return '╝', 255, 255, 0
        
        # ─── REALISM: STRICTLY USE BRIGHTNESS FOR CHARACTER ───
        n = len(ASCII_FACE)
        norm_b = brightness / 255.0
        idx = min(int(norm_b * (n - 1)), n - 1)
        char = ASCII_FACE[idx]

        # Base face color — cycling gold/amber
        t = (time.time() * 0.5)
        base_color = sample_gradient(FACE_COLORS, t)
        
        # Blend: Mostly original natural color, tinted by the sci-fi base color
        blend = 0.7 # 70% original, 30% tint
        r = int(r_orig * blend + base_color[0] * (1 - blend))
        g = int(g_orig * blend + base_color[1] * (1 - blend))
        b = int(b_orig * blend + base_color[2] * (1 - blend))
        
        # Slight brightness boost for face to ensure it pops
        r = min(255, int(r * 1.2))
        g = min(255, int(g * 1.2))
        b = min(255, int(b * 1.2))
        
        # Glitch (Color only, keep char unless very strong)
        if audio_norm > 0.7 and random.random() < 0.1:
            r, g, b = 255, 255, 255 # Flash white
            if random.random() < 0.5:
                char = random.choice(MATRIX_CHARS)

        return char, r, g, b


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Setup Audio ──
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Error opening audio: {e}")
        return

    # ── Setup Video ──
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Could not open video source: {CAMERA_SOURCE}")
        return

    # ── Setup MediaPipe Tasks ──
    
    # 1. Selfie Segmentation
    seg_base = python.BaseOptions(model_asset_path="selfie_segmenter.tflite")
    seg_opts = vision.ImageSegmenterOptions(base_options=seg_base,
                                            output_category_mask=True)
    segmenter = vision.ImageSegmenter.create_from_options(seg_opts)

    # 2. Face Landmark Detection
    mesh_base = python.BaseOptions(model_asset_path="face_landmarker.task")
    mesh_opts = vision.FaceLandmarkerOptions(base_options=mesh_base,
                                             num_faces=1,
                                             min_face_detection_confidence=0.5,
                                             min_face_presence_confidence=0.5,
                                             min_tracking_confidence=0.5)
    face_landmarker = vision.FaceLandmarker.create_from_options(mesh_opts)


    # Clear screen and hide cursor
    sys.stdout.write("\033[2J\033[?25l")
    sys.stdout.flush()

    # Will be initialized on first frame
    scifi_bg = None
    body_fx = BodyEffects()
    face_fx = FaceEffects()
    
    # ─── Frame Rate Control ───
    fps_limit = 60
    prev_time = 0
    
    print("Starting Sci-Fi ASCII Camera (Tasks API)... Press Ctrl+C to stop.")
    time.sleep(0.5)
    sys.stdout.write("\033[2J")

    try:
        while True:
            time_elapsed = time.time() - prev_time
            if time_elapsed < 1.0 / fps_limit:
                 continue
            prev_time = time.time()

            # ── AUDIO ──
            vol = get_audio_level(stream)
            audio_norm = min(vol / 5000.0, 1.0)  # 0..1 normalized
            reactivity = int(vol / 50)

            # ── VIDEO ──
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontally
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create MP Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # ── Compute ASCII dimensions ──
            h_orig, w_orig = gray.shape
            aspect = w_orig / h_orig
            new_h = int(WIDTH / aspect / 0.55)

            # ── SEGMENTATION (Tasks API) ──
            seg_result = segmenter.segment(mp_image)
            category_mask = seg_result.category_mask.numpy_view() # uint8, 255 or 0 usually check docs
            
            # Resize mask to ASCII dimensions
            # category_mask is usually same size as input image
            seg_mask_resized = cv2.resize(category_mask, (WIDTH, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Masks - Check index (0=background, 1=person for selfie segmenter usually, or vice versa)
            # Standard selfie segmenter: index 1 is body. 
            body_mask = seg_mask_resized > 0 
            
            # Create strict boolean masks
            aura_body_mask = seg_mask_resized > 0
            
            # Simple erosion for "inner body" vs "aura"
            kernel = np.ones((3,3), np.uint8)
            eroded_body = cv2.erode(seg_mask_resized, kernel, iterations=1)
            
            is_body = eroded_body > 0
            is_aura = (aura_body_mask) & (~is_body)
            

            # ── FACE DETECTION (Tasks API) ──
            face_result = face_landmarker.detect(mp_image)
            face_top, face_bottom, face_left, face_right = 0, 0, 0, 0
            has_face = False

            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0] # List of NormalizedLandmark
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                
                face_left = int(min(xs) * WIDTH)
                face_right = int(max(xs) * WIDTH)
                face_top = int(min(ys) * new_h)
                face_bottom = int(max(ys) * new_h)
                
                # Padding
                pad_x = int((face_right - face_left) * 0.2)
                pad_y = int((face_bottom - face_top) * 0.2)
                face_left = max(0, face_left - pad_x)
                face_right = min(WIDTH - 1, face_right + pad_x)
                face_top = max(0, face_top - pad_y)
                face_bottom = min(new_h - 1, face_bottom + pad_y)
                has_face = True

            face_mask = np.zeros((new_h, WIDTH), dtype=bool)
            if has_face:
                face_mask[face_top:face_bottom + 1, face_left:face_right + 1] = True
                face_mask = face_mask & is_body

            # ── Resize grayscale ──
            gray_resized = cv2.resize(gray, (WIDTH, new_h))
            
            # ── Contrast Enhancement (CLAHE) ──
            # This brings out facial features significantly
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_enhanced = clahe.apply(gray_resized)
            
            # Audio boost
            gray_boosted = np.clip(gray_enhanced.astype(int) + reactivity * 2, 0, 255).astype(np.uint8)
            
            # Resize color frame for sampling
            frame_small = cv2.resize(frame_rgb, (WIDTH, new_h))

            # ── Initialize/Update Effects ──
            if scifi_bg is None or scifi_bg.w != WIDTH or scifi_bg.h != new_h:
                scifi_bg = SciFiBackground(WIDTH, new_h)

            scifi_bg.update(vol, audio_norm)
            body_fx.update(audio_norm)
            if has_face:
                face_fx.update(face_top, face_bottom)

            # ── RENDER ──
            output_lines = []
            face_rect = (face_top, face_bottom, face_left, face_right)

            for y in range(new_h):
                line_chars = []
                for x in range(WIDTH):
                    brightness = int(gray_boosted[y, x])

                    if face_mask[y, x]:
                        # FACE
                        char, r, g, b = face_fx.get_color(x, y, brightness, face_rect, audio_norm, frame_small)
                    
                    elif is_body[y, x]:
                        # BODY
                        char, r, g, b = body_fx.get_color(x, y, brightness, WIDTH, new_h, frame_small, is_aura=False)
                        
                    elif is_aura[y, x]:
                        # AURA
                        char, r, g, b = body_fx.get_color(x, y, brightness, WIDTH, new_h, frame_small, is_aura=True)
                        
                    else:
                        # BACKGROUND
                        char, r, g, b = scifi_bg.get_effect(x, y, brightness)

                    line_chars.append(f"\033[38;2;{r};{g};{b}m{char}")

                output_lines.append("".join(line_chars))

            sys.stdout.write("\033[H")
            sys.stdout.write("\n".join(output_lines))
            sys.stdout.write("\033[0m")
            sys.stdout.flush()

    except KeyboardInterrupt:
        pass

    # ── Cleanup ──
    sys.stdout.write("\033[?25h")
    sys.stdout.write("\033[0m\n")
    sys.stdout.flush()
    print("Exiting Sci-Fi ASCII Camera...")

    # Tasks API: Clean up usually happens automatically or via 'with' block, 
    # but explicit close() is good habit if available, or just letting them go out of scope.
    segmenter.close()
    face_landmarker.close()
    cap.release()
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()