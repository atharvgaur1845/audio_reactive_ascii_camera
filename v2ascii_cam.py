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
from collections import OrderedDict

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
WIDTH = 200           # Fallback width (used if AUTO_FULLSCREEN is False)
AUTO_FULLSCREEN = True  # True = auto-fill terminal, False = use WIDTH above
CAMERA_SOURCE = 0     # 0 = default webcam, 1 = external/phone, or "http://IP:PORT/video"
AUDIO_SOURCE = "mic" # "mic" = microphone, "system" = PC audio (music), "both" = mix of both
CHUNK = 1024          # Audio chunk size
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# ── EFFECT BLEND RATIOS (tune these!) ──
# Each pair: (effect_amount, camera_amount) — they should sum to 1.0
BG_EFFECT_BLEND = 1.0    # Background: 90% sci-fi effects, 10% real camera
BG_CAMERA_BLEND = 0.0
BODY_EFFECT_BLEND = 0.2   # Body: 30% neon effects, 70% real camera
BODY_CAMERA_BLEND = 0.8
FACE_EFFECT_BLEND = 0.350  # Face: 10% tint, 90% real camera (keep face natural)
FACE_CAMERA_BLEND = 0.75
FACE_BRIGHTNESS_BOOST = 1.5 # Brightness multiplier for face clarity
BODY_BRIGHTNESS_BOOST = 2.3  # Brightness multiplier for body clarity
AUDIO_REACT_BODY = 0.1       # How much music affects body (0.0 = none, 1.0 = full)
AUDIO_REACT_FACE = 0.1       # How much music affects face (0.0 = none, 1.0 = full)
BG_INTENSITY = 1.0           # Background effects brightness (0.0 = dark, 1.0 = full bright)
MUSIC_DISTORTION_STRENGTH = 1.0  # How strong the psychedelic music distortion is
MUSIC_BG_FLOW_STRENGTH = 2.0     # How much music drives flowing neon background patterns

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

# Background sci-fi palette — Cyberpunk Cyan/Blue
BG_BASE_COLOR = (0, 40, 60)  # Deep cyberpunk dark cyan/teal


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


class Envelope:
    def __init__(self, attack=0.15, decay=0.5):
        self.value = 0.0
        self.attack = attack
        self.decay = decay
    
    def update(self, target, dt):
        if target > self.value:
            k = 1.0 - math.exp(-dt / max(1e-6, self.attack))
        else:
            k = 1.0 - math.exp(-dt / max(1e-6, self.decay))
        self.value += (target - self.value) * k
        return self.value


def compute_audio_features(raw_data):
    """Compute RMS and Bass energy from raw audio bytes."""
    if not raw_data: 
        return 0.0, 0.0
        
    data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
    if len(data) == 0: 
        return 0.0, 0.0
        
    # RMS total
    rms = np.sqrt(np.mean(data**2))
    
    # FFT for bass
    # Use Hanning window to reduce spectral leakage
    windowed = data * np.hanning(len(data))
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(data), d=1.0/RATE)
    
    # Bass range: 20-150 Hz
    bass_idx = np.where((freqs >= 20) & (freqs <= 150))[0]
    bass_energy = spec[bass_idx].mean() if len(bass_idx) > 0 else 0.0
    
    return rms, bass_energy


def get_audio_level(stream):
    """Reads audio chunk and returns (rms, bass_energy)."""
    try:
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        return compute_audio_features(raw_data)
    except Exception:
        return 0.0, 0.0


class AudioCapture:
    """Threaded audio capture to prevent blocking the render loop."""
    def __init__(self, streams):
        self.streams = streams
        self.rms = 0.0
        self.bass = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def _loop(self):
        while self._running:
            rms_vals = []
            bass_vals = []
            for s in self.streams:
                r, b = get_audio_level(s)
                rms_vals.append(r)
                bass_vals.append(b)
            
            # Take max of connected streams
            self.rms = max(rms_vals) if rms_vals else 0.0
            self.bass = max(bass_vals) if bass_vals else 0.0
    
    def get_levels(self):
        return self.rms, self.bass
    
    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


def find_monitor_device(p):
    """Find PulseAudio/PipeWire monitor source for capturing system audio."""
    monitor_idx = None
    monitor_name = ""
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            name = info.get('name', '')
            # Monitor sources contain 'monitor' in their name
            if 'monitor' in name.lower() and info.get('maxInputChannels', 0) > 0:
                monitor_idx = i
                monitor_name = name
                break
        except Exception:
            continue
    return monitor_idx, monitor_name


# ═══════════════════════════════════════════════════════════════
#  TRIPPY BACKGROUND EFFECTS ENGINE
# ═══════════════════════════════════════════════════════════════

TRIPPY_CHARS = list('░▒▓█╬╠╣╦╩┼┤├┬┴│─┌┐└┘◆◇○●∞≈≡±×÷∆∇∂∫Σπφψω')
GLITCH_CHARS = list('!@#$%^&*<>{}[]|/\\~`?=+')
CYBER_WORDS = ['CYBER', 'HACK', 'NEURAL', 'SYNC', 'VOID', 'FLUX', 'GLITCH', 'NEON',
               'DATA', 'CODE', 'PULSE', 'WAVE', 'NODE', 'GRID', 'LINK', 'CORE',
               'ZERO', 'ROOT', 'SYS', 'NET', 'BIT', 'HEX', 'BOOT', 'INIT']

class SciFiBackground:
    """Chaotic cyber background: plasma, glitch, text trails, scan lines, sparks."""

    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.frame_count = 0
        self.audio_norm = 0
        
        # Matrix rain - aggressive
        self.rain_cols = []
        for x in range(width):
            self.rain_cols.append({
                'y': random.randint(-height, height),
                'speed': random.randint(1, 5),
                'len': random.randint(3, 12),
                'active': random.random() < 0.35,
                'hue': random.random()
            })
        
        # Glitch blocks
        self.glitch_blocks = []
        
        # Sparks
        self.sparks = {}
        
        # Plasma phase
        self.plasma_phase = random.random() * 6.28
        
        # Horizontal text trails (scrolling words)
        self.h_trails = []
        for _ in range(5):
            self.h_trails.append({
                'y': random.randint(0, height - 1),
                'x': random.randint(-30, width),
                'word': random.choice(CYBER_WORDS),
                'speed': random.randint(2, 6),
                'hue': random.random()
            })
        
        # Vertical text trails (falling words)
        self.v_trails = []
        for _ in range(5):
            self.v_trails.append({
                'x': random.randint(0, width - 1),
                'y': random.randint(-20, height),
                'word': random.choice(CYBER_WORDS),
                'speed': random.randint(1, 4),
                'hue': random.random()
            })
        
        # Scan lines
        self.scan_y = 0
        self.scan_speed = 2

    def update(self, audio_level, audio_norm):
        self.frame_count += 1
        self.audio_norm = audio_norm
        self.plasma_phase += 0.08 + audio_norm * 0.15
        
        # Update Rain
        for col in self.rain_cols:
            if col['active']:
                col['y'] += col['speed']
                if col['y'] > self.h + col['len']:
                    col['y'] = -random.randint(3, 15)
                    col['speed'] = random.randint(1, 5)
                    col['hue'] = random.random()
                    col['active'] = random.random() < 0.4
            else:
                if random.random() < 0.03 + audio_norm * 0.05:
                    col['active'] = True
                    col['y'] = -col['len']
        
        # Glitch blocks on audio
        if audio_norm > 0.3 and random.random() < 0.4:
            bw = random.randint(3, 20)
            bh = random.randint(2, 8)
            self.glitch_blocks.append({
                'x': random.randint(0, max(1, self.w - bw)),
                'y': random.randint(0, max(1, self.h - bh)),
                'w': bw, 'h': bh,
                'life': random.uniform(0.3, 1.0),
                'hue': random.random(),
                'char': random.choice(TRIPPY_CHARS)
            })
        
        for g in self.glitch_blocks[:]:
            g['life'] -= 0.08
            if g['life'] <= 0:
                self.glitch_blocks.remove(g)
        
        # Sparks
        num_sparks = int(3 + audio_norm * 15)
        for _ in range(num_sparks):
            sx = random.randint(0, self.w - 1)
            sy = random.randint(0, self.h - 1)
            self.sparks[(sx, sy)] = random.uniform(0.3, 1.0)
        
        dead = []
        for k, v in self.sparks.items():
            self.sparks[k] = v - 0.15
            if self.sparks[k] <= 0:
                dead.append(k)
        for k in dead:
            del self.sparks[k]
        
        # Update horizontal text trails
        for ht in self.h_trails:
            ht['x'] += ht['speed']
            if ht['x'] > self.w + 10:
                ht['x'] = -len(ht['word']) * 2
                ht['y'] = random.randint(0, self.h - 1)
                ht['word'] = random.choice(CYBER_WORDS)
                ht['hue'] = random.random()
        
        # Update vertical text trails
        for vt in self.v_trails:
            vt['y'] += vt['speed']
            if vt['y'] > self.h + 10:
                vt['y'] = -len(vt['word']) * 2
                vt['x'] = random.randint(0, self.w - 1)
                vt['word'] = random.choice(CYBER_WORDS)
                vt['hue'] = random.random()
        
        # Scan line
        self.scan_y = (self.scan_y + self.scan_speed) % self.h
        
        # Spawn more trails on loud audio
        if audio_norm > 0.5 and random.random() < 0.2:
            self.h_trails.append({
                'y': random.randint(0, self.h - 1),
                'x': -10,
                'word': random.choice(CYBER_WORDS),
                'speed': random.randint(3, 8),
                'hue': random.random()
            })
        # Cap trails
        if len(self.h_trails) > 12:
            self.h_trails = self.h_trails[-12:]
        if len(self.v_trails) > 10:
            self.v_trails = self.v_trails[-10:]

    def get_effect(self, x, y, brightness):
        inten = BG_INTENSITY
        
        # 1. Scan line - bright horizontal sweep
        if abs(y - self.scan_y) < 1:
            hue = (self.frame_count * 0.02) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.8 * inten)
            return '─', int(r*255), int(g*255), int(b*255)
        
        # 2. Horizontal text trails
        for ht in self.h_trails:
            if y == ht['y']:
                ci = x - ht['x']
                if 0 <= ci < len(ht['word']):
                    char = ht['word'][ci]
                    r, g, b = colorsys.hsv_to_rgb(ht['hue'], 0.9, 0.9 * inten)
                    return char, int(r*255), int(g*255), int(b*255)
        
        # 3. Vertical text trails
        for vt in self.v_trails:
            if x == vt['x']:
                ci = y - vt['y']
                if 0 <= ci < len(vt['word']):
                    char = vt['word'][ci]
                    r, g, b = colorsys.hsv_to_rgb(vt['hue'], 0.9, 0.85 * inten)
                    return char, int(r*255), int(g*255), int(b*255)
        
        # 4. Sparks
        if (x, y) in self.sparks:
            life = self.sparks[(x, y)]
            hue = (x * 0.01 + y * 0.01 + self.frame_count * 0.1) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, life * inten)
            return random.choice(GLITCH_CHARS), int(r*255), int(g*255), int(b*255)
        
        # 5. Glitch blocks
        for gb in self.glitch_blocks:
            if gb['x'] <= x < gb['x'] + gb['w'] and gb['y'] <= y < gb['y'] + gb['h']:
                r, g, b = colorsys.hsv_to_rgb(gb['hue'], 0.9, gb['life'] * 0.6 * inten)
                return gb['char'], int(r*255), int(g*255), int(b*255)
        
        # 6. Matrix Rain
        col = self.rain_cols[x]
        if col['active']:
            dist = y - col['y']
            if 0 <= dist < col['len']:
                norm_dist = dist / col['len']
                intensity = 1.0 - norm_dist
                hue = (col['hue'] + self.frame_count * 0.005) % 1.0
                
                if dist < 1:
                    r, g, b = colorsys.hsv_to_rgb(hue, 0.3, 0.9 * inten)
                    return random.choice(MATRIX_CHARS), int(r*255), int(g*255), int(b*255)
                
                r, g, b = colorsys.hsv_to_rgb(hue, 0.7, intensity * 0.5 * inten)
                return random.choice(MATRIX_CHARS), int(r*255), int(g*255), int(b*255)
        
        # 7. Plasma base - dark with subtle color
        t = self.plasma_phase
        v1 = math.sin(x * 0.04 + t)
        v2 = math.sin(y * 0.06 - t * 0.7)
        v3 = math.sin((x * 0.03 + y * 0.04) + t * 0.5)
        v4 = math.sin(math.sqrt(max(0.01, (x - self.w/2)**2 + (y - self.h/2)**2)) * 0.08 - t)
        plasma = (v1 + v2 + v3 + v4) / 4.0
        
        hue = ((plasma + 1.0) / 2.0 + self.frame_count * 0.008) % 1.0
        sat = 0.6 + self.audio_norm * 0.3
        val = (0.08 + abs(plasma) * 0.12 + self.audio_norm * 0.06) * inten
        
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        
        if random.random() < 0.02 + self.audio_norm * 0.03:
            char = random.choice(TRIPPY_CHARS)
        else:
            n = len(ASCII_BG)
            idx = min(int((abs(plasma) + 0.1) * (n - 1)), n - 1)
            char = ASCII_BG[idx]
        
        return char, int(r*255), int(g*255), int(b*255)


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
        body_audio = audio_norm * AUDIO_REACT_BODY
        self.pattern_offset += 0.1 + (body_audio * 0.2)
        
        # Switch palettes based on intensity
        if body_audio > 0.8:
            self.target_palette_name = "PLASMA"
        elif body_audio > 0.4:
             self.target_palette_name = "CYBERPUNK"
        else:
             self.target_palette_name = "MATRIX"
             
        if self.current_palette_name != self.target_palette_name:
             self.current_palette_name = self.target_palette_name

    def get_color(self, x, y, brightness, w, h, pixel_rgb, is_aura=False):
        
        if is_aura:
            # Aura is electric blue/white, sparse for effect
            # Use coherent noise instead of random flicker
            shimmer = math.sin(x * 0.3 + y * 0.3 + self.frame_count * 0.2)
            if shimmer > 0.6:
                return ':', 100, 200, 255
            return ' ', 0, 0, 0
            
        # ─── REALISM: STRICTLY USE BRIGHTNESS FOR CHARACTER ───
        n = len(ASCII_BODY)
        norm_b = brightness / 255.0
        idx = min(int(norm_b * (n - 1)), n - 1)
        char = ASCII_BODY[idx]

        # ─── COLOR LOGIC ───
        # pixel_rgb is (R, G, B) from the camera
        r_orig, g_orig, b_orig = int(pixel_rgb[0]), int(pixel_rgb[1]), int(pixel_rgb[2])
        
        # Circuit / Data Pattern for neon tint
        nx = x / w
        ny = y / h
        wave = math.sin(nx * 10 + ny * 10 - self.pattern_offset) 
        wave2 = math.cos(nx * 20 - ny * 5 + self.pattern_offset * 0.5)
        pattern = (wave + wave2) / 2.0
        
        palette = PALETTES[self.current_palette_name]
        t = (pattern + 1.0) / 2.0 
        r_neon, g_neon, b_neon = sample_gradient(palette, t)
        
        # Camera color + neon tint (uses config)
        blend = BODY_CAMERA_BLEND
        effect = BODY_EFFECT_BLEND
        r = int(r_orig * blend + r_neon * effect * norm_b)
        g = int(g_orig * blend + g_neon * effect * norm_b)
        b = int(b_orig * blend + b_neon * effect * norm_b)
        
        # Brightness boost for body clarity
        boost = BODY_BRIGHTNESS_BOOST
        r = min(255, int(r * boost))
        g = min(255, int(g * boost))
        b = min(255, int(b * boost))
        
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

    def get_color(self, x, y, brightness, face_rect, audio_norm, pixel_rgb):
        ft, fb, fl, fr = face_rect
        r_orig, g_orig, b_orig = int(pixel_rgb[0]), int(pixel_rgb[1]), int(pixel_rgb[2])
        
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

        # Face uses camera color + subtle tint (uses config)
        t = (time.time() * 0.5)
        base_color = sample_gradient(FACE_COLORS, t)
        
        blend = FACE_CAMERA_BLEND
        effect = FACE_EFFECT_BLEND
        r = int(r_orig * blend + base_color[0] * effect)
        g = int(g_orig * blend + base_color[1] * effect)
        b = int(b_orig * blend + base_color[2] * effect)
        
        # Brightness boost for face clarity (uses config)
        boost = FACE_BRIGHTNESS_BOOST
        r = min(255, int(r * boost))
        g = min(255, int(g * boost))
        b = min(255, int(b * boost))
        
        # Glitch on loud audio
        if audio_norm * AUDIO_REACT_FACE > 0.7 and random.random() < 0.1:
            r, g, b = 255, 255, 255
            if random.random() < 0.5:
                char = random.choice(MATRIX_CHARS)

        return char, r, g, b


# ═══════════════════════════════════════════════════════════════
#  FACE MOTION TRAIL
# ═══════════════════════════════════════════════════════════════

class FaceTrail:
    """Leaves bright, long-lasting trippy neon light trails when the face moves."""
    
    TRAIL_BRIGHT_CHARS = list('█▓▒░★✦⚡●◆♦♢✧')
    TRAIL_FADE_CHARS = list('▒░·•✧○◇')
    TRAIL_DIM_CHARS = list('·.,:;')
    
    def __init__(self):
        self.trail = OrderedDict()  # (x, y) -> {'life': float, 'hue': float, 'birth_hue': float}
        self.prev_center = None
        self.frame_count = 0
        self.trail_hue = 0.0
    
    def update(self, face_rect, has_face):
        self.frame_count += 1
        self.trail_hue = (self.trail_hue + 0.025) % 1.0
        
        if has_face:
            ft, fb, fl, fr = face_rect
            cx = (fl + fr) // 2
            cy = (ft + fb) // 2
            fw = max(fr - fl, 1)
            fh = max(fb - ft, 1)
            
            if self.prev_center is not None:
                px, py = self.prev_center
                dx = abs(cx - px)
                dy = abs(cy - py)
                
                # Only trail if face is moving enough
                if dx > 1 or dy > 1:
                    speed = math.sqrt(dx**2 + dy**2)
                    trail_life = min(1.0, 0.7 + speed * 0.05)  # Faster = brighter start
                    
                    # Deposit trail pixels along the face boundary — 2 pixels wide
                    for i in range(0, max(fw, fh), 1):  # Every pixel, not every 2
                        # Top and bottom edges (2 pixels thick)
                        tx = fl + (i % fw)
                        for offset in range(2):
                            self.trail[(tx, ft + offset)] = {'life': trail_life, 'hue': self.trail_hue, 'seed': random.random()}
                            self.trail[(tx, fb - offset)] = {'life': trail_life, 'hue': (self.trail_hue + 0.3) % 1.0, 'seed': random.random()}
                        # Left and right edges (2 pixels thick)
                        ty = ft + (i % fh)
                        for offset in range(2):
                            self.trail[(fl + offset, ty)] = {'life': trail_life, 'hue': (self.trail_hue + 0.5) % 1.0, 'seed': random.random()}
                            self.trail[(fr - offset, ty)] = {'life': trail_life, 'hue': (self.trail_hue + 0.7) % 1.0, 'seed': random.random()}
                    
                    # Interpolate between prev and current position for smooth trails
                    steps = max(int(speed), 2)
                    for s in range(steps):
                        t = s / steps
                        ix = int(px + (cx - px) * t)
                        iy = int(py + (cy - py) * t)
                        # Deposit a small glow around interpolated points
                        for ox in range(-2, 3):
                            for oy in range(-1, 2):
                                self.trail[(ix + ox, iy + oy)] = {
                                    'life': trail_life * 0.8,
                                    'hue': (self.trail_hue + t * 0.5) % 1.0,
                                    'seed': random.random()
                                }
                    
                    # Scatter bright sparkles behind the face — more particles
                    num_sparkles = 20 + int(speed * 3)
                    for _ in range(num_sparkles):
                        # Sparkles trail BEHIND the direction of motion
                        sx = px + random.randint(-fw//2 - 3, fw//2 + 3)
                        sy = py + random.randint(-fh//2 - 2, fh//2 + 2)
                        self.trail[(sx, sy)] = {
                            'life': random.uniform(0.6, 1.0),
                            'hue': (self.trail_hue + random.uniform(-0.2, 0.2)) % 1.0,
                            'seed': random.random()
                        }
            
            self.prev_center = (cx, cy)
        
        # Decay all trail pixels — SLOWER decay for longer trails
        dead = []
        for k, v in self.trail.items():
            v['life'] -= 0.025  # Was 0.06 — now 2.4× slower decay
            if v['life'] <= 0:
                dead.append(k)
        for k in dead:
            del self.trail[k]
        
        # Cap max trail entries to prevent performance issues
        if len(self.trail) > 8000:
            # Remove oldest entries (FIFO)
            for _ in range(len(self.trail) - 6000):
                self.trail.popitem(last=False)
    
    def get_trail(self, x, y):
        """Returns (has_trail, char, r, g, b) for this pixel."""
        if (x, y) in self.trail:
            t = self.trail[(x, y)]
            # Brightness boost — trails stay bright longer
            vis_life = min(1.0, t['life'] * 1.8)
            r, g, b = colorsys.hsv_to_rgb(t['hue'], 1.0, vis_life)  # Full saturation
            
            # Choose character based on life remaining AND seed (deterministic)
            seed = t.get('seed', 0)
            if t['life'] > 0.6:
                char = self.TRAIL_BRIGHT_CHARS[int(seed * len(self.TRAIL_BRIGHT_CHARS))]
            elif t['life'] > 0.3:
                char = self.TRAIL_FADE_CHARS[int(seed * len(self.TRAIL_FADE_CHARS))]
            else:
                char = self.TRAIL_DIM_CHARS[int(seed * len(self.TRAIL_DIM_CHARS))]
            
            return True, char, int(r * 255), int(g * 255), int(b * 255)
        return False, ' ', 0, 0, 0


# ═══════════════════════════════════════════════════════════════
#  BODY MOTION TRAIL
# ═══════════════════════════════════════════════════════════════

class BodyTrail:
    """Leaves neon trails when the body moves — less intense than face trails."""
    
    TRAIL_BRIGHT_CHARS = list('█▓▒░★✦⚡●◆♦♢✧')
    TRAIL_FADE_CHARS = list('▒░·•✧○◇')
    TRAIL_DIM_CHARS = list('·.,:;')
    
    def __init__(self):
        self.trail = OrderedDict()  # (x, y) -> {'life': float, 'hue': float}
        self.prev_centroid = None
        self.frame_count = 0
        self.trail_hue = 0.0
    
    def update(self, body_mask, is_body):
        """Update body trail based on body mask centroid movement."""
        self.frame_count += 1
        self.trail_hue = (self.trail_hue + 0.015) % 1.0  # Slower hue cycle than face
        
        # Find body centroid from mask
        ys, xs = np.where(is_body)
        has_body = len(xs) > 20  # Need minimum body pixels
        
        if has_body:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            
            # Get body bounding box
            bx_min, bx_max = int(np.min(xs)), int(np.max(xs))
            by_min, by_max = int(np.min(ys)), int(np.max(ys))
            bw = max(bx_max - bx_min, 1)
            bh = max(by_max - by_min, 1)
            
            if self.prev_centroid is not None:
                px, py = self.prev_centroid
                dx = abs(cx - px)
                dy = abs(cy - py)
                
                # Only trail if body is actually moving
                if dx > 2 or dy > 2:
                    speed = math.sqrt(dx**2 + dy**2)
                    trail_life = min(1.0, 0.7 + speed * 0.05)  # Brighter start (matched face)
                    
                    # Sample edge pixels from the body boundary
                    # Use a stride to keep it performant but dense enough for prominence
                    h, w = body_mask.shape
                    stride = max(2, bh // 30)  # Denser stride (was max(3, bh//15))
                    
                    for row in range(by_min, by_max, stride):
                        row_pixels = np.where(is_body[row, :])[0]
                        if len(row_pixels) > 0:
                            # Left and right edges of body at this row
                            left_edge = int(row_pixels[0])
                            right_edge = int(row_pixels[-1])
                            
                            # Add some thickness (2 pixels)
                            for off in range(2):
                                if left_edge + off < w:
                                    self.trail[(left_edge + off, row)] = {
                                        'life': trail_life,
                                        'hue': (self.trail_hue + row * 0.005) % 1.0,
                                        'seed': random.random()
                                    }
                                if right_edge - off >= 0:
                                    self.trail[(right_edge - off, row)] = {
                                        'life': trail_life,
                                        'hue': (self.trail_hue + 0.4 + row * 0.005) % 1.0,
                                        'seed': random.random()
                                    }
                    
                    # Top and bottom edges
                    for col in range(bx_min, bx_max, stride):
                        col_pixels = np.where(is_body[:, col])[0]
                        if len(col_pixels) > 0:
                            top_edge = int(col_pixels[0])
                            bot_edge = int(col_pixels[-1])
                            
                            for off in range(2):
                                if top_edge + off < h:
                                    self.trail[(col, top_edge + off)] = {
                                        'life': trail_life,
                                        'hue': (self.trail_hue + 0.2) % 1.0,
                                        'seed': random.random()
                                    }
                                if bot_edge - off >= 0:
                                    self.trail[(col, bot_edge - off)] = {
                                        'life': trail_life,
                                        'hue': (self.trail_hue + 0.6) % 1.0,
                                        'seed': random.random()
                                    }
                    
                    # Scatter sparkles — match face intensity
                    num_sparkles = 20 + int(speed * 3)
                    for _ in range(num_sparkles):
                        sx = px + random.randint(-bw//4, bw//4)
                        sy = py + random.randint(-bh//4, bh//4)
                        self.trail[(sx, sy)] = {
                            'life': random.uniform(0.4, 0.75),
                            'hue': (self.trail_hue + random.uniform(-0.15, 0.15)) % 1.0,
                            'seed': random.random()
                        }
            
            self.prev_centroid = (cx, cy)
        
        # Decay — faster than face trails for subtler effect
        dead = []
        for k, v in self.trail.items():
            v['life'] -= 0.025  # Slower decay (matched face)
            if v['life'] <= 0:
                dead.append(k)
        for k in dead:
            del self.trail[k]
        
        if len(self.trail) > 5000:
             # Remove oldest entries (FIFO)
            for _ in range(len(self.trail) - 3500):
                self.trail.popitem(last=False)
    
    def get_trail(self, x, y):
        """Returns (has_trail, char, r, g, b) for this pixel."""
        if (x, y) in self.trail:
            t = self.trail[(x, y)]
            # Boost brightness to match face trails
            vis_life = min(1.0, t['life'] * 1.8)  # 1.8x multiplier
            r, g, b = colorsys.hsv_to_rgb(t['hue'], 1.0, vis_life)  # Full saturation
            
            # Choose character based on life remaining
            seed = t.get('seed', 0)
            if t['life'] > 0.5:
                # Deterministic char based on seed
                char = self.TRAIL_BRIGHT_CHARS[int(seed * len(self.TRAIL_BRIGHT_CHARS))]
            elif t['life'] > 0.25:
                char = self.TRAIL_FADE_CHARS[int(seed * len(self.TRAIL_FADE_CHARS))]
            else:
                char = self.TRAIL_DIM_CHARS[int(seed * len(self.TRAIL_DIM_CHARS))]
            
            return True, char, int(r * 255), int(g * 255), int(b * 255)
        return False, ' ', 0, 0, 0


# ═══════════════════════════════════════════════════════════════
#  MUSIC-REACTIVE PSYCHEDELIC DISTORTION 
# ═══════════════════════════════════════════════════════════════

class MusicDistortion:
    """Creates psychedelic rainbow heat-map / flowing neon liquid distortions
    driven by audio levels. Inspired by thermal-vision rainbow overlays 
    and organic cyan/magenta flowing patterns."""
    
    def __init__(self):
        self.frame_count = 0
        self.phase = 0.0
        self.beat_flash = 0.0
        self.prev_audio = 0.0
        self.beat_accumulator = 0.0
        self.flow_offset_x = 0.0
        self.flow_offset_y = 0.0
        # Smooth audio for less jittery response
        self.smooth_audio = 0.0
    
    def update(self, rms_norm, bass_norm, flow=(0,0)):
        self.frame_count += 1
        
        # Smooth RMS for flowing movement
        self.smooth_audio = self.smooth_audio * 0.9 + rms_norm * 0.1
        
        # Phase advances with overall energy
        self.phase += 0.05 + self.smooth_audio * 0.2
        
        # Beat detection — use BASS energy
        # If bass is significantly higher than recent average or absolute threshold
        if bass_norm > 0.4 and bass_norm > self.prev_audio * 1.3:
            self.beat_flash = min(1.0, self.beat_flash + 0.6)
            self.beat_accumulator += 1.0
            
        self.beat_flash *= 0.85  # Decay beat flash
        self.prev_audio = self.prev_audio * 0.95 + bass_norm * 0.05  # Rolling average of bass
        
        # Flow movement — organic drifting + Optical Flow influence
        # flow is (dx, dy) in pixels (small scale)
        fx, fy = flow
        self.flow_offset_x += 0.03 + self.smooth_audio * 0.1 - fx * 0.1
        self.flow_offset_y += 0.02 + self.smooth_audio * 0.08 - fy * 0.1
    
    def get_body_distortion(self, x, y, w, h, pixel_rgb, brightness):
        """Rainbow heat-map distortion over the body — like thermal/psychedelic vision.
        Returns (r, g, b) with rainbow overlay blended based on audio."""
        audio = self.smooth_audio
        if audio < 0.05:
            return int(pixel_rgb[0]), int(pixel_rgb[1]), int(pixel_rgb[2])
        
        strength = audio * MUSIC_DISTORTION_STRENGTH * AUDIO_REACT_BODY
        
        # Multi-frequency sine waves for rainbow heat-map effect
        nx = x / max(w, 1)
        ny = y / max(h, 1)
        
        # Create flowing rainbow pattern (like thermal vision from Image 1)
        wave1 = math.sin(nx * 8.0 + self.phase * 1.3)
        wave2 = math.sin(ny * 6.0 - self.phase * 0.9)
        wave3 = math.sin((nx + ny) * 5.0 + self.phase * 1.1)
        wave4 = math.sin(math.sqrt(max(0.01, (nx - 0.5)**2 + (ny - 0.5)**2)) * 12.0 - self.phase * 1.5)
        
        # Combine waves — creates complex flowing rainbow
        combined = (wave1 + wave2 + wave3 + wave4) / 4.0
        
        # Map to full rainbow hue — this creates the psychedelic rainbow heat-map
        hue = (combined * 0.5 + 0.5 + self.frame_count * 0.01) % 1.0
        
        # Brightness follows original image brightness for "heat-map" look
        norm_b = brightness / 255.0
        val = 0.5 + norm_b * 0.5
        
        # Saturation high for vivid rainbow
        sat = 0.85 + self.beat_flash * 0.15
        
        r_dist, g_dist, b_dist = colorsys.hsv_to_rgb(hue, sat, val)
        r_dist = int(r_dist * 255)
        g_dist = int(g_dist * 255)
        b_dist = int(b_dist * 255)
        
        # Blend: original camera + rainbow distortion based on audio strength
        r_orig, g_orig, b_orig = int(pixel_rgb[0]), int(pixel_rgb[1]), int(pixel_rgb[2])
        r = int(r_orig * (1 - strength) + r_dist * strength)
        g = int(g_orig * (1 - strength) + g_dist * strength)
        b = int(b_orig * (1 - strength) + b_dist * strength)
        
        # Beat flash — white burst on beats
        if self.beat_flash > 0.3:
            flash = self.beat_flash * 0.4
            r = min(255, int(r + 255 * flash * strength))
            g = min(255, int(g + 255 * flash * strength))
            b = min(255, int(b + 255 * flash * strength))
        
        return min(255, r), min(255, g), min(255, b)
    
    def get_bg_flow(self, x, y, w, h):
        """Flowing neon liquid patterns for background — cyan/magenta organic shapes.
        Returns (r, g, b) intensity to add to background. Like Image 2."""
        audio = self.smooth_audio
        if audio < 0.03:
            return 0, 0, 0
        
        strength = audio * MUSIC_BG_FLOW_STRENGTH
        
        nx = x / max(w, 1)
        ny = y / max(h, 1)
        
        # Organic flowing pattern — multiple overlapping sine waves
        # This creates the liquid/blob effect from Image 2
        flow1 = math.sin(nx * 6.0 + self.flow_offset_x + math.sin(ny * 3.0 + self.flow_offset_y) * 2.0)
        flow2 = math.sin(ny * 7.0 - self.flow_offset_y * 1.3 + math.sin(nx * 4.0 - self.flow_offset_x) * 1.5)
        flow3 = math.sin((nx * 3.0 + ny * 4.0) + self.flow_offset_x * 0.7)
        flow4 = math.cos(nx * 5.0 - ny * 3.0 + self.flow_offset_y * 0.9)
        
        # Combine into organic blob shape
        blob = (flow1 + flow2 + flow3 + flow4) / 4.0
        
        # Create two-tone effect: cyan and magenta (like Image 2)
        # blob ranges -1 to 1, map to color choice
        if blob > 0.0:
            # Cyan channel — bright neon cyan
            intensity = blob * strength
            r = int(0 * intensity * 255)
            g = int(0.9 * intensity * 255)
            b = int(1.0 * intensity * 255)
        else:
            # Magenta channel — vivid magenta/pink
            intensity = abs(blob) * strength
            r = int(1.0 * intensity * 255)
            g = int(0.0 * intensity * 255)
            b = int(0.8 * intensity * 255)
        
        # Beat pulse — expand intensity on beats
        if self.beat_flash > 0.2:
            mult = 1.0 + self.beat_flash * 0.8
            r = int(r * mult)
            g = int(g * mult)
            b = int(b * mult)
        
        return min(255, r), min(255, g), min(255, b)
    
    def get_face_glow(self, x, y, w, h, audio_norm):
        """Very subtle rainbow edge glow for face on loud beats.
        Returns (r_add, g_add, b_add) to add to face color."""
        if audio_norm < 0.4 or self.beat_flash < 0.2:
            return 0, 0, 0
        
        strength = audio_norm * AUDIO_REACT_FACE * self.beat_flash
        
        nx = x / max(w, 1)
        ny = y / max(h, 1)
        
        hue = (nx * 2.0 + ny * 2.0 + self.phase * 0.5) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, strength * 0.3)
        
        return int(r * 255), int(g * 255), int(b * 255)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Setup Audio ──
    p = pyaudio.PyAudio()
    audio_streams = []
    
    try:
        if AUDIO_SOURCE in ("mic", "both"):
            mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                input=True, frames_per_buffer=CHUNK)
            audio_streams.append(mic_stream)
            print("✓ Microphone audio stream opened.")
        
        if AUDIO_SOURCE in ("system", "both"):
            monitor_idx, monitor_name = find_monitor_device(p)
            if monitor_idx is not None:
                sys_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                    input=True, input_device_index=monitor_idx,
                                    frames_per_buffer=CHUNK)
                audio_streams.append(sys_stream)
                print(f"✓ System audio stream opened: {monitor_name}")
            else:
                print("⚠ No monitor device found! Falling back to microphone.")
                print("  Tip: On PulseAudio, ensure a monitor source exists.")
                print("  Available input devices:")
                for i in range(p.get_device_count()):
                    try:
                        info = p.get_device_info_by_index(i)
                        if info.get('maxInputChannels', 0) > 0:
                            print(f"    [{i}] {info['name']}")
                    except Exception:
                        pass
                mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                    input=True, frames_per_buffer=CHUNK)
                audio_streams.append(mic_stream)
        
        if not audio_streams:
            print("Error: No audio streams could be opened.")
            return
            
    except Exception as e:
        print(f"Error opening audio: {e}")
        return

    # Start threaded audio capture (prevents render flicker)
    audio_capture = AudioCapture(audio_streams)
    # ── Setup Video ──
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Could not open video source: {CAMERA_SOURCE}")
        return

    # ── Setup MediaPipe Tasks ──
    # Resolve model paths relative to this script's directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SEG_MODEL = os.path.join(SCRIPT_DIR, "selfie_segmenter.tflite")
    FACE_MODEL = os.path.join(SCRIPT_DIR, "face_landmarker.task")
    
    segmenter = None
    face_landmarker = None
    
    try:
        # Auto-download models if missing
        import urllib.request
        MODELS = {
            SEG_MODEL: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        }
        for path, url in MODELS.items():
            if not os.path.exists(path):
                print(f"Downloading {os.path.basename(path)}...")
                try:
                    urllib.request.urlretrieve(url, path)
                    print(f"  ✓ Saved to {path}")
                except Exception as e:
                    print(f"  ⚠ Download failed: {e}")

        # 1. Selfie Segmentation
        if os.path.exists(SEG_MODEL):
            seg_base = python.BaseOptions(model_asset_path=SEG_MODEL)
            seg_opts = vision.ImageSegmenterOptions(base_options=seg_base,
                                                    output_category_mask=True)
            segmenter = vision.ImageSegmenter.create_from_options(seg_opts)
        else:
            print("⚠ Segmentation model not found. Running without body segmentation.")

        # 2. Face Landmark Detection
        if os.path.exists(FACE_MODEL):
            mesh_base = python.BaseOptions(model_asset_path=FACE_MODEL)
            mesh_opts = vision.FaceLandmarkerOptions(base_options=mesh_base,
                                                     num_faces=1,
                                                     min_face_detection_confidence=0.5,
                                                     min_face_presence_confidence=0.5,
                                                     min_tracking_confidence=0.5)
            face_landmarker = vision.FaceLandmarker.create_from_options(mesh_opts)
        else:
             print("⚠ Face model not found. Running without face detection.")
             
    except Exception as e:
        print(f"⚠ ML Initialization Error: {e}")
        print("Running in fallback mode (No ML).")


    # Clear screen and hide cursor
    sys.stdout.write("\033[2J\033[?25l")
    sys.stdout.flush()

    # Will be initialized on first frame
    scifi_bg = None
    body_fx = BodyEffects()
    face_fx = FaceEffects()
    face_trail = FaceTrail()
    body_trail = BodyTrail()
    music_dist = MusicDistortion()
    
    # ─── Frame Rate Control ───
    fps_limit = 30
    prev_time = time.time()
    
    # ─── Audio Processing ───
    rms_envelope = Envelope(attack=0.1, decay=0.5)
    bass_envelope = Envelope(attack=0.05, decay=0.3)
    
    # Auto-calibration for audio levels
    max_rms = 1000.0
    max_bass = 1000.0
    
    # ─── ML Rate Control ───
    DETECT_HZ = 10
    detect_interval = 1.0 / DETECT_HZ
    last_detect_time = 0
    
    # Cache for ML results
    cached_seg_mask = None
    cached_face_result = None
    
    
    print("Starting Sci-Fi ASCII Camera (Tasks API)... Press Ctrl+C to stop.")
    time.sleep(0.5)
    sys.stdout.write("\033[2J")
    
    # ─── Optical Flow State ───
    prev_gray_small = None
    flow_mean = (0, 0)
    
    def analyze_face(landmarks):
        """Analyze face landmarks for expressions."""
        # MediaPipe Mesh Landmarks:
        # 13, 14 = Upper/Lower lip center
        # 159, 145 = Left eye top/bottom
        # 386, 374 = Right eye top/bottom
        
        # Mouth open
        upper = landmarks[13]
        lower = landmarks[14]
        mouth_dist = math.sqrt((upper.x - lower.x)**2 + (upper.y - lower.y)**2)
        mouth_open = min(1.0, max(0.0, (mouth_dist - 0.01) * 20.0)) # Threshold and scale
        
        # Blink (simple aspect ratio check could go here, for now just mouth)
        return {'mouth_open': mouth_open}

    try:
        while True:
            # Yield CPU time if we are too fast
            now = time.time()
            elapsed = now - prev_time
            if elapsed < 1.0 / fps_limit:
                 time.sleep((1.0 / fps_limit) - elapsed)
                 
            prev_time = time.time()

            # ── AUDIO (non-blocking, from background thread) ──
            raw_rms, raw_bass = audio_capture.get_levels()
            
            # Update auto-calibration (slowly decay max to adapt to volume changes)
            max_rms = max(max_rms * 0.999, raw_rms)
            max_bass = max(max_bass * 0.999, raw_bass)
            
            # Normalize (avoid divide by zero)
            start_elapsed = time.time() # Start of processing for dt
            dt = 1.0 / fps_limit # Approximate dt
            
            norm_rms = min(1.0, raw_rms / max(1.0, max_rms))
            norm_bass = min(1.0, raw_bass / max(1.0, max_bass))
            
            # Smooth with envelopes
            smooth_rms = rms_envelope.update(norm_rms, dt)
            smooth_bass = bass_envelope.update(norm_bass, dt)
            
            # Use smooth_rms for general reactivity
            audio_norm = smooth_rms 
            reactivity = int(audio_norm * 20)

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

            # ── Compute ASCII dimensions (auto-fill terminal) ──
            term_cols, term_rows = shutil.get_terminal_size((WIDTH, 50))
            if AUTO_FULLSCREEN:
                ascii_w = term_cols
                ascii_h = term_rows - 1  # -1 to avoid scrolling
            else:
                h_orig, w_orig = gray.shape
                aspect = w_orig / h_orig
                ascii_w = WIDTH
                ascii_h = int(WIDTH / aspect / 0.55)

            # ── SEGMENTATION (Tasks API) ──
            # Run at reduced rate
            run_ml = (time.time() - last_detect_time) > detect_interval
            
            if segmenter and run_ml:
                try:
                    seg_result = segmenter.segment(mp_image)
                    category_mask = seg_result.category_mask.numpy_view()
                    cached_seg_mask = category_mask
                except Exception:
                    pass
            
            if cached_seg_mask is not None:
                category_mask = cached_seg_mask
            else:
                 # Fallback if no result yet
                 category_mask = np.zeros((h_orig if not AUTO_FULLSCREEN else 480, 
                                          w_orig if not AUTO_FULLSCREEN else 640), dtype=np.uint8)
            
            # Resize mask to ASCII dimensions
            seg_mask_resized = cv2.resize(category_mask, (ascii_w, ascii_h), interpolation=cv2.INTER_NEAREST)
            
            # MediaPipe selfie segmenter: 0 = person, >0 = background
            body_mask = seg_mask_resized == 0
            
            # Create strict boolean masks
            aura_body_mask = seg_mask_resized == 0
            
            # Simple erosion for "inner body" vs "aura"
            kernel = np.ones((3,3), np.uint8)
            body_uint8 = (body_mask.astype(np.uint8)) * 255
            eroded_body = cv2.erode(body_uint8, kernel, iterations=1)
            
            is_body = eroded_body > 0
            is_aura = (aura_body_mask) & (~is_body)
            

            # ── FACE DETECTION (Tasks API) ──
            if face_landmarker and run_ml:
                try:
                    face_result = face_landmarker.detect(mp_image)
                    cached_face_result = face_result
                    last_detect_time = time.time()
                except Exception:
                    pass
            
            if cached_face_result is not None:
                face_result = cached_face_result
            else:
                # Dummy empty result object
                class EmptyResult:
                    face_landmarks = []
                face_result = EmptyResult()

            face_top, face_bottom, face_left, face_right = 0, 0, 0, 0
            has_face = False

            face_mask = np.zeros((ascii_h, ascii_w), dtype=np.uint8)
            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                
                face_left = int(min(xs) * ascii_w)
                face_right = int(max(xs) * ascii_w)
                face_top = int(min(ys) * ascii_h)
                face_bottom = int(max(ys) * ascii_h)
                
                
                # Padding
                pad_x = int((face_right - face_left) * 0.15)
                pad_y = int((face_bottom - face_top) * 0.15)
                face_left = max(0, min(ascii_w - 1, face_left - pad_x))
                face_right = max(0, min(ascii_w - 1, face_right + pad_x))
                face_top = max(0, min(ascii_h - 1, face_top - pad_y))
                face_bottom = max(0, min(ascii_h - 1, face_bottom + pad_y))
                
                face_mask[face_top:face_bottom + 1, face_left:face_right + 1] = 255
                has_face = True

            face_mask = face_mask > 0  # Convert to boolean

            # ── Resize grayscale ──
            # Used for ASCII and Optical Flow
            flow_w, flow_h = 80, 45 # Low res for flow performance
            gray_small = cv2.resize(gray, (flow_w, flow_h))
            
            # Compute Optical Flow
            if prev_gray_small is not None:
                try:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray_small, gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    fx, fy = np.mean(flow[..., 0]), np.mean(flow[..., 1])
                    flow_mean = (flow_mean[0] * 0.7 + fx * 0.3, flow_mean[1] * 0.7 + fy * 0.3)
                except Exception:
                    pass
            prev_gray_small = gray_small
            
            gray_resized = cv2.resize(gray, (ascii_w, ascii_h))
            
            # ── Contrast Enhancement (CLAHE) ──
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_enhanced = clahe.apply(gray_resized)
            
            # Audio boost
            gray_boosted = np.clip(gray_enhanced.astype(int) + int(reactivity * 2 * AUDIO_REACT_BODY), 0, 255).astype(np.uint8)
            
            # Resize color frame for sampling (RGB format)
            frame_small_rgb = cv2.resize(frame_rgb, (ascii_w, ascii_h))

            # ── Initialize/Update Effects ──
            if scifi_bg is None or scifi_bg.w != ascii_w or scifi_bg.h != ascii_h:
                scifi_bg = SciFiBackground(ascii_w, ascii_h)

            scifi_bg.update(raw_rms, audio_norm) # Pass audio_norm (smooth RMS)
            body_fx.update(audio_norm)
            music_dist.update(audio_norm, smooth_bass, flow_mean) # Pass RMS, Bass, Flow
            face_rect = (face_top, face_bottom, face_left, face_right)
            face_features = {'mouth_open': 0.0}
            
            if has_face:
                if face_result.face_landmarks:
                    face_features = analyze_face(face_result.face_landmarks[0])
                # We need to update face_fx signature first or pass it separately
                # For now, let's just pass it to update if feasible, but update only takes top/bottom
                face_fx.update(face_top, face_bottom) # Existing call
                
            # face_features will optionally be used in get_color later
            pass # Just a placeholderline to anchor the replacement context logic
            face_trail.update(face_rect, has_face)
            body_trail.update(body_mask, is_body)

            # ── RENDER ──
            output_lines = []

            for y in range(ascii_h):
                line_chars = []
                for x in range(ascii_w):
                    brightness = int(gray_boosted[y, x])
                    pixel_rgb = frame_small_rgb[y, x]  # (R, G, B) tuple

                    # Check face trail first (overlays everything)
                    has_trail, trail_char, trail_r, trail_g, trail_b = face_trail.get_trail(x, y)
                    # Check body trail
                    has_btrail, btrail_char, btrail_r, btrail_g, btrail_b = body_trail.get_trail(x, y)

                    if face_mask[y, x]:
                        # FACE — bright and crisp + subtle music glow
                        # Note: we need to pass face_features to get_color or handle it here.
                        # Since I can't easily change get_color signature in multiple places without risk, 
                        # I'll modify the result here.
                        char, r, g, b = face_fx.get_color(x, y, brightness, face_rect, audio_norm, pixel_rgb)
                        
                        # Add mouth flare
                        if face_features.get('mouth_open', 0) > 0.05:
                             # Gold/White flare 
                             flare = min(1.0, face_features['mouth_open'] * 0.8)
                             r = min(255, int(r + 255 * flare))
                             g = min(255, int(g + 200 * flare))
                             b = min(255, int(b + 100 * flare))
                             
                        # Add subtle rainbow glow on beats
                        gr, gg, gb = music_dist.get_face_glow(x, y, ascii_w, ascii_h, audio_norm)
                        r = min(255, r + gr)
                        g = min(255, g + gg)
                        b = min(255, b + gb)
                    
                    elif has_trail and not is_body[y, x]:
                        # FACE TRAIL — trippy neon sparkles
                        char, r, g, b = trail_char, trail_r, trail_g, trail_b
                    
                    elif is_body[y, x]:
                        # BODY — real color + neon tint + MUSIC RAINBOW DISTORTION
                        char, r, g, b = body_fx.get_color(x, y, brightness, ascii_w, ascii_h, pixel_rgb, is_aura=False)
                        # Apply psychedelic rainbow heat-map from music
                        dr, dg, db = music_dist.get_body_distortion(x, y, ascii_w, ascii_h, pixel_rgb, brightness)
                        # Blend body effect with music distortion
                        music_blend = min(1.0, audio_norm * AUDIO_REACT_BODY)
                        r = int(r * (1 - music_blend) + dr * music_blend)
                        g = int(g * (1 - music_blend) + dg * music_blend)
                        b = int(b * (1 - music_blend) + db * music_blend)
                        
                    elif is_aura[y, x]:
                        # AURA — also gets music distortion
                        char, r, g, b = body_fx.get_color(x, y, brightness, ascii_w, ascii_h, pixel_rgb, is_aura=True)
                        dr, dg, db = music_dist.get_body_distortion(x, y, ascii_w, ascii_h, pixel_rgb, brightness)
                        music_blend = min(1.0, audio_norm * AUDIO_REACT_BODY * 0.5)
                        r = int(r * (1 - music_blend) + dr * music_blend)
                        g = int(g * (1 - music_blend) + dg * music_blend)
                        b = int(b * (1 - music_blend) + db * music_blend)
                        
                    else:
                        # BACKGROUND — full trippy mode + FLOWING NEON MUSIC PATTERNS
                        # Check body trail first — shows where body WAS
                        if has_btrail:
                            char, r, g, b = btrail_char, btrail_r, btrail_g, btrail_b
                        elif has_trail:
                            # Face trail in background
                            char, r, g, b = trail_char, trail_r, trail_g, trail_b
                        else:
                            bg_char, bg_r, bg_g, bg_b = scifi_bg.get_effect(x, y, brightness)
                            r_cam, g_cam, b_cam = int(pixel_rgb[0]), int(pixel_rgb[1]), int(pixel_rgb[2])
                            r = int(r_cam * BG_CAMERA_BLEND + bg_r * BG_EFFECT_BLEND)
                            g = int(g_cam * BG_CAMERA_BLEND + bg_g * BG_EFFECT_BLEND)
                            b = int(b_cam * BG_CAMERA_BLEND + bg_b * BG_EFFECT_BLEND)
                            char = bg_char
                            # Add flowing neon music patterns (cyan/magenta)
                            fr, fg, fb = music_dist.get_bg_flow(x, y, ascii_w, ascii_h)
                            r = min(255, r + fr)
                            g = min(255, g + fg)
                            b = min(255, b + fb)

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
    if 'segmenter' in locals():
        try:
            segmenter.close()
        except Exception:
            pass
            
    if 'face_landmarker' in locals():
        try:
            face_landmarker.close()
        except Exception:
            pass
            
    if 'cap' in locals():
        cap.release()
        
    if 'audio_capture' in locals():
        audio_capture.stop()
        
    if 'audio_streams' in locals():
        for s in audio_streams:
            try:
                s.stop_stream()
                s.close()
            except Exception:
                pass
                
    if 'p' in locals():
        p.terminate()


if __name__ == "__main__":
    main()