import cv2
import numpy as np
import pyaudio
import sys
import os

# --- CONFIGURATION ---
WIDTH = 200  # Width of the ASCII output in characters
CHUNK = 1024 # Audio chunk size
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 # Audio sampling rate

# The Gradient: From dark pixels to light pixels
# The "HD" Gradient: 70 levels of shading instead of 10
ASCII_CHARS = list(" `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@")

def get_audio_level(stream):
    """Reads a chunk of audio and calculates the RMS (volume) amplitude."""
    try:
        # Read raw data
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Convert to numpy array
        data = np.frombuffer(raw_data, dtype=np.int16)
        
        # --- THE FIX IS HERE ---
        # We convert the integers to floats BEFORE squaring them.
        # This prevents the numbers from overflowing and turning negative.
        data = data.astype(np.float64) 
        
        # Calculate Root Mean Square (volume)
        rms = np.sqrt(np.mean(data**2))
        return rms
    except Exception as e:
        # It's good to see audio errors if they happen
        # print(e) 
        return 0

def pixel_to_ascii(image, intensity_shift=0):
    """Converts a grayscale image to a string of ASCII characters."""
    
    # --- FIX 1: UPGRADE THE CONTAINER ---
    # Convert uint8 (0-255) to a standard int so it can hold large numbers
    image = image.astype(int) 
    
    # --- FIX 2: ADD BRIGHTNESS AND CLIP ---
    # Now we can safely add the loud audio volume
    # Then we clip it back down to 0-255
    image = np.clip(image + intensity_shift, 0, 255)
    
    # --- FIX 3: SAFE INDEXING ---
    # Map 0-255 strictly to the number of characters we have
    n_chars = len(ASCII_CHARS)
    indices = (image * (n_chars - 1)) // 255
    
    # Ensure indices are integers for the lookup
    indices = indices.astype(int)
    
    # Look up the ASCII characters
    ascii_arr = np.array(ASCII_CHARS)[indices]
    
    # Join them into a string
    return "\n".join("".join(row) for row in ascii_arr)

def main():
    # 1. Setup Audio
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                        input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Error opening audio: {e}")
        return

    # 2. Setup Video
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Starting Audioreactive Camera... Press Ctrl+C to stop.")

    try:
        while True:
            # --- AUDIO PROCESSING ---
            vol = get_audio_level(stream)
            
            # Normalize volume to a factor (0 to 100ish)
            # You might need to tweak the divisor '100' depending on your mic sensitivity
            reactivity = int(vol / 50) 
            
            # --- VIDEO PROCESSING ---
            ret, frame = cap.read()
            if not ret: break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize to maintain aspect ratio (terminal characters are tall)
            height, width = gray.shape
            aspect_ratio = width / height
            # Font aspect ratio correction (chars are usually 2x as high as wide)
            new_height = int(WIDTH / aspect_ratio / 0.55) 
            resized_gray = cv2.resize(gray, (WIDTH, new_height))

            # --- RENDERING ---
            # Generate ASCII string with audio reactivity applied to brightness
            ascii_art = pixel_to_ascii(resized_gray, intensity_shift=reactivity*5)
            
            # Add Color Reactivity using ANSI Escape Codes
            # Quiet = Cyan, Loud = Red/Bold
            if reactivity < 5:
                color = "\033[96m" # Cyan
            elif reactivity < 15:
                color = "\033[93m" # Yellow
            else:
                color = "\033[91;1m" # Red Bold!

            # Move cursor to top-left (H) so we overwrite, not scroll
            sys.stdout.write("\033[H") 
            sys.stdout.write(color + ascii_art + "\033[0m") # Print art and reset color
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nExiting...")
    
    # Cleanup
    cap.release()
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()