#pragma once
#include <opencv2/opencv.hpp>
#include <portaudio.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <sys/ioctl.h>
#include <unistd.h>
#include <csignal>
#include <fstream>
#include <sstream>
#include <functional>
#include <numeric>

// ═══════════════════════════════════════════════════════════════
//  CONFIGURATION
// ═══════════════════════════════════════════════════════════════
constexpr int CFG_WIDTH = 200;
constexpr bool AUTO_FULLSCREEN = true;
constexpr int CAMERA_SOURCE = 0;
constexpr int CHUNK = 1024;
constexpr int CHANNELS = 1;
constexpr int RATE = 44100;

constexpr double BG_EFFECT_BLEND = 1.0;
constexpr double BG_CAMERA_BLEND = 0.0;
constexpr double BODY_EFFECT_BLEND = 0.2;
constexpr double BODY_CAMERA_BLEND = 0.8;
constexpr double FACE_EFFECT_BLEND = 0.350;
constexpr double FACE_CAMERA_BLEND = 0.75;
constexpr double FACE_BRIGHTNESS_BOOST = 1.5;
constexpr double BODY_BRIGHTNESS_BOOST = 2.3;
constexpr double AUDIO_REACT_BODY = 0.1;
constexpr double AUDIO_REACT_FACE = 0.1;
constexpr double BG_INTENSITY = 1.0;
constexpr double MUSIC_DISTORTION_STRENGTH = 1.0;
constexpr double MUSIC_BG_FLOW_STRENGTH = 2.0;

using Color3 = std::array<int,3>;

// ═══════════════════════════════════════════════════════════════
//  GLOBAL DATA
// ═══════════════════════════════════════════════════════════════
extern const std::string ASCII_RAMP;
extern const std::vector<std::string> MATRIX_CHARS;
extern const std::vector<std::string> GLITCH_CHARS;
extern const std::vector<std::string> TRIPPY_CHARS;
extern const std::vector<std::string> CYBER_WORDS;
extern const std::vector<std::string> HUD_CHARS;
extern const std::map<std::string, std::vector<Color3>> PALETTES;
extern const std::vector<Color3> FACE_COLORS;

// ═══════════════════════════════════════════════════════════════
//  UTILITY
// ═══════════════════════════════════════════════════════════════
struct PixelResult { std::string ch; int r, g, b; };

Color3 lerp_color(const Color3& c1, const Color3& c2, double t);
Color3 sample_gradient(const std::vector<Color3>& colors, double t);
void hsv_to_rgb(double h, double s, double v, double& r, double& g, double& b);
std::pair<int,int> get_terminal_size();

// thread-safe RNG
thread_local inline std::mt19937 tl_rng{std::random_device{}()};
inline int randint(int lo, int hi) { return std::uniform_int_distribution<int>(lo,hi)(tl_rng); }
inline double randf() { return std::uniform_real_distribution<double>(0.0,1.0)(tl_rng); }
inline double randf(double lo, double hi) { return std::uniform_real_distribution<double>(lo,hi)(tl_rng); }

template<typename T>
const T& rand_choice(const std::vector<T>& v) { return v[randint(0,(int)v.size()-1)]; }
inline char rand_ramp_char(const std::string& s) { return s[randint(0,(int)s.size()-1)]; }

// ═══════════════════════════════════════════════════════════════
//  AUDIO CAPTURE (PortAudio)
// ═══════════════════════════════════════════════════════════════
class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();
    bool open_mic();
    bool open_monitor();
    double get_volume() const;
    void stop();
private:
    static int pa_callback(const void* in, void*, unsigned long frames,
                           const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void* ud);
    PaStream* stream_ = nullptr;
    std::atomic<double> volume_{0.0};
    std::atomic<bool> running_{true};
    bool initialized_ = false;
};

// ═══════════════════════════════════════════════════════════════
//  SCIFI BACKGROUND
// ═══════════════════════════════════════════════════════════════
struct RainCol { int y; int speed; int len; bool active; double hue; };
struct GlitchBlock { int x,y,w,h; double life, hue; std::string ch; };
struct HTrail { int y,x; std::string word; int speed; double hue; };
struct VTrail { int x,y; std::string word; int speed; double hue; };

class SciFiBackground {
public:
    int w, h, frame_count=0;
    double audio_norm=0, plasma_phase;
    std::vector<RainCol> rain_cols;
    std::vector<GlitchBlock> glitch_blocks;
    std::map<std::pair<int,int>, double> sparks;
    std::vector<HTrail> h_trails;
    std::vector<VTrail> v_trails;
    int scan_y=0, scan_speed=2;

    SciFiBackground(int w, int h);
    void update(double audio_level, double audio_norm);
    PixelResult get_effect(int x, int y, int brightness);
};

// ═══════════════════════════════════════════════════════════════
//  BODY EFFECTS
// ═══════════════════════════════════════════════════════════════
class BodyEffects {
public:
    int frame_count=0;
    double pattern_offset=0.0;
    std::string current_palette = "CYBERPUNK";
    void update(double audio_norm);
    PixelResult get_color(int x, int y, int brightness, int w, int h, const cv::Vec3b& pixel_rgb, bool is_aura);
};

// ═══════════════════════════════════════════════════════════════
//  FACE EFFECTS
// ═══════════════════════════════════════════════════════════════
class FaceEffects {
public:
    int scan_line_y=0, scan_speed=2, frame_count=0;
    void update(int face_top, int face_bottom);
    PixelResult get_color(int x, int y, int brightness, int ft, int fb, int fl, int fr,
                          double audio_norm, const cv::Vec3b& pixel_rgb);
};

// ═══════════════════════════════════════════════════════════════
//  FACE TRAIL
// ═══════════════════════════════════════════════════════════════
struct TrailPixel { double life; double hue; };

class FaceTrail {
public:
    std::unordered_map<long long, TrailPixel> trail;
    int prev_cx=-1, prev_cy=-1;
    bool has_prev=false;
    int frame_count=0;
    double trail_hue=0.0;

    static const std::vector<std::string> BRIGHT_CHARS, FADE_CHARS, DIM_CHARS;

    void update(int ft, int fb, int fl, int fr, bool has_face);
    std::tuple<bool,std::string,int,int,int> get_trail(int x, int y);
private:
    long long key(int x, int y) const { return ((long long)x << 32) | (unsigned int)y; }
};

// ═══════════════════════════════════════════════════════════════
//  BODY TRAIL
// ═══════════════════════════════════════════════════════════════
class BodyTrail {
public:
    std::unordered_map<long long, TrailPixel> trail;
    int prev_cx=-1, prev_cy=-1;
    bool has_prev=false;
    int frame_count=0;
    double trail_hue=0.0;

    static const std::vector<std::string> BRIGHT_CHARS, FADE_CHARS, DIM_CHARS;

    void update(const cv::Mat& body_mask, const cv::Mat& is_body);
    std::tuple<bool,std::string,int,int,int> get_trail(int x, int y);
private:
    long long key(int x, int y) const { return ((long long)x << 32) | (unsigned int)y; }
};

// ═══════════════════════════════════════════════════════════════
//  MUSIC DISTORTION
// ═══════════════════════════════════════════════════════════════
class MusicDistortion {
public:
    int frame_count=0;
    double phase=0, beat_flash=0, prev_audio=0, beat_accumulator=0;
    double flow_offset_x=0, flow_offset_y=0, smooth_audio=0;

    void update(double audio_norm);
    Color3 get_body_distortion(int x, int y, int w, int h, const cv::Vec3b& pixel_rgb, int brightness);
    Color3 get_bg_flow(int x, int y, int w, int h);
    Color3 get_face_glow(int x, int y, int w, int h, double audio_norm);
};
