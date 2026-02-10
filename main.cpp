#include "ascii_cam.h"

// ═══════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════

static volatile bool g_running = true;
static void sig_handler(int) { g_running = false; }

int main() {
    signal(SIGINT, sig_handler);

    // ── Setup Audio ──
    AudioCapture audio;
    if (!audio.open_mic()) {
        fprintf(stderr, "Error: Could not open any audio stream.\n");
        return 1;
    }

    // ── Setup Video ──
    cv::VideoCapture cap(CAMERA_SOURCE);
    if (!cap.isOpened()) {
        fprintf(stderr, "Could not open video source: %d\n", CAMERA_SOURCE);
        return 1;
    }

    // ── Setup Face Detection (Haar Cascade) ──
    cv::CascadeClassifier face_cascade;
    std::vector<std::string> cascade_paths = {
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    };
    bool cascade_loaded = false;
    for (auto& p : cascade_paths) {
        if (face_cascade.load(p)) { cascade_loaded = true; break; }
    }
    if (!cascade_loaded) {
        fprintf(stderr, "⚠ Could not load Haar cascade. Face detection disabled.\n");
        fprintf(stderr, "  Install: sudo apt-get install libopencv-dev\n");
    }

    // ── Background Subtractor for body segmentation ──
    auto bg_sub = cv::createBackgroundSubtractorMOG2(500, 50, true);

    // Clear screen, hide cursor
    printf("\033[2J\033[?25l");
    fflush(stdout);

    SciFiBackground* scifi_bg = nullptr;
    BodyEffects body_fx;
    FaceEffects face_fx;
    FaceTrail face_trail;
    BodyTrail body_trail;
    MusicDistortion music_dist;

    constexpr int fps_limit = 30;
    auto prev_time = std::chrono::steady_clock::now();

    printf("Starting Sci-Fi ASCII Camera (C++)... Press Ctrl+C to stop.\n");
    fflush(stdout);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    printf("\033[2J");
    fflush(stdout);

    std::string output_buf;
    output_buf.reserve(1024 * 1024);

    cv::Mat frame, frame_rgb, gray, fg_mask;

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - prev_time).count();
        if (elapsed < 1.0 / fps_limit) continue;
        prev_time = now;

        // ── AUDIO ──
        double vol = audio.get_volume();
        double audio_norm = std::min(vol / 5000.0, 1.0);
        int reactivity = (int)(vol / 50);

        // ── VIDEO ──
        if (!cap.read(frame)) break;
        cv::flip(frame, frame, 1);
        cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // ── Terminal size ──
        auto [term_cols, term_rows] = get_terminal_size();
        int ascii_w, ascii_h;
        if (AUTO_FULLSCREEN) {
            ascii_w = term_cols;
            ascii_h = term_rows - 1;
        } else {
            double aspect = (double)frame.cols / frame.rows;
            ascii_w = CFG_WIDTH;
            ascii_h = (int)(CFG_WIDTH / aspect / 0.55);
        }
        if (ascii_w < 10 || ascii_h < 5) continue;

        // ── BODY SEGMENTATION ──
        cv::Mat frame_small;
        cv::resize(frame, frame_small, cv::Size(ascii_w, ascii_h));
        bg_sub->apply(frame_small, fg_mask, 0.01);
        cv::threshold(fg_mask, fg_mask, 200, 255, cv::THRESH_BINARY);
        cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
        cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_OPEN, morph_kernel);
        cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_CLOSE, morph_kernel);

        cv::Mat body_mask = fg_mask.clone();
        cv::Mat eroded_body;
        cv::erode(body_mask, eroded_body, morph_kernel, cv::Point(-1,-1), 1);
        cv::Mat is_body = eroded_body;
        cv::Mat is_aura_mat;
        cv::Mat not_eroded;
        cv::bitwise_not(eroded_body, not_eroded);
        cv::bitwise_and(body_mask, not_eroded, is_aura_mat);

        // ── FACE DETECTION ──
        int face_top=0, face_bottom=0, face_left=0, face_right=0;
        bool has_face = false;
        cv::Mat face_mask_mat = cv::Mat::zeros(ascii_h, ascii_w, CV_8UC1);

        if (cascade_loaded) {
            cv::Mat gray_small;
            cv::resize(gray, gray_small, cv::Size(ascii_w, ascii_h));
            std::vector<cv::Rect> faces;
            face_cascade.detectMultiScale(gray_small, faces, 1.1, 4,
                                          cv::CASCADE_SCALE_IMAGE, cv::Size(ascii_w/10, ascii_h/10));
            if (!faces.empty()) {
                auto& f = faces[0];
                face_left = f.x;
                face_right = f.x + f.width;
                face_top = f.y;
                face_bottom = f.y + f.height;
                int pad_x = (int)((face_right - face_left) * 0.15);
                int pad_y = (int)((face_bottom - face_top) * 0.15);
                face_left = std::max(0, face_left - pad_x);
                face_right = std::min(ascii_w - 1, face_right + pad_x);
                face_top = std::max(0, face_top - pad_y);
                face_bottom = std::min(ascii_h - 1, face_bottom + pad_y);
                face_mask_mat(cv::Rect(face_left, face_top,
                    face_right - face_left + 1, face_bottom - face_top + 1)) = 255;
                has_face = true;
            }
        }

        // ── Resize grayscale + CLAHE ──
        cv::Mat gray_resized;
        cv::resize(gray, gray_resized, cv::Size(ascii_w, ascii_h));
        auto clahe = cv::createCLAHE(2.0, cv::Size(8,8));
        cv::Mat gray_enhanced;
        clahe->apply(gray_resized, gray_enhanced);
        cv::Mat gray_boosted;
        gray_enhanced.convertTo(gray_boosted, CV_8UC1, 1.0, reactivity * 2 * AUDIO_REACT_BODY);

        // Resize color frame (RGB)
        cv::Mat frame_small_rgb;
        cv::resize(frame_rgb, frame_small_rgb, cv::Size(ascii_w, ascii_h));

        // ── Update Effects ──
        if (!scifi_bg || scifi_bg->w != ascii_w || scifi_bg->h != ascii_h) {
            delete scifi_bg;
            scifi_bg = new SciFiBackground(ascii_w, ascii_h);
        }
        scifi_bg->update(vol, audio_norm);
        body_fx.update(audio_norm);
        music_dist.update(audio_norm);
        if (has_face) face_fx.update(face_top, face_bottom);
        face_trail.update(face_top, face_bottom, face_left, face_right, has_face);
        body_trail.update(body_mask, is_body);

        // ── RENDER ──
        output_buf.clear();
        char ansi_buf[64];

        for (int y = 0; y < ascii_h; y++) {
            for (int x = 0; x < ascii_w; x++) {
                int brightness = (int)gray_boosted.at<uchar>(y, x);
                cv::Vec3b pixel_rgb = frame_small_rgb.at<cv::Vec3b>(y, x);

                auto [has_ft, ft_ch, ft_r, ft_g, ft_b] = face_trail.get_trail(x, y);
                auto [has_bt, bt_ch, bt_r, bt_g, bt_b] = body_trail.get_trail(x, y);

                std::string ch;
                int r, g, b;

                bool px_face = face_mask_mat.at<uchar>(y, x) > 0;
                bool px_body = is_body.at<uchar>(y, x) > 0;
                bool px_aura = is_aura_mat.at<uchar>(y, x) > 0;

                if (px_face) {
                    auto res = face_fx.get_color(x, y, brightness,
                        face_top, face_bottom, face_left, face_right, audio_norm, pixel_rgb);
                    ch = res.ch; r = res.r; g = res.g; b = res.b;
                    auto glow = music_dist.get_face_glow(x, y, ascii_w, ascii_h, audio_norm);
                    r = std::min(255, r + glow[0]);
                    g = std::min(255, g + glow[1]);
                    b = std::min(255, b + glow[2]);
                } else if (has_ft && !px_body) {
                    ch = ft_ch; r = ft_r; g = ft_g; b = ft_b;
                } else if (px_body) {
                    auto res = body_fx.get_color(x, y, brightness, ascii_w, ascii_h, pixel_rgb, false);
                    ch = res.ch; r = res.r; g = res.g; b = res.b;
                    auto dist = music_dist.get_body_distortion(x, y, ascii_w, ascii_h, pixel_rgb, brightness);
                    double mb = std::min(1.0, audio_norm * AUDIO_REACT_BODY);
                    r = (int)(r * (1 - mb) + dist[0] * mb);
                    g = (int)(g * (1 - mb) + dist[1] * mb);
                    b = (int)(b * (1 - mb) + dist[2] * mb);
                } else if (px_aura) {
                    auto res = body_fx.get_color(x, y, brightness, ascii_w, ascii_h, pixel_rgb, true);
                    ch = res.ch; r = res.r; g = res.g; b = res.b;
                    auto dist = music_dist.get_body_distortion(x, y, ascii_w, ascii_h, pixel_rgb, brightness);
                    double mb = std::min(1.0, audio_norm * AUDIO_REACT_BODY * 0.5);
                    r = (int)(r * (1 - mb) + dist[0] * mb);
                    g = (int)(g * (1 - mb) + dist[1] * mb);
                    b = (int)(b * (1 - mb) + dist[2] * mb);
                } else {
                    if (has_bt) {
                        ch = bt_ch; r = bt_r; g = bt_g; b = bt_b;
                    } else if (has_ft) {
                        ch = ft_ch; r = ft_r; g = ft_g; b = ft_b;
                    } else {
                        auto res = scifi_bg->get_effect(x, y, brightness);
                        int r_cam = pixel_rgb[0], g_cam = pixel_rgb[1], b_cam = pixel_rgb[2];
                        r = (int)(r_cam * BG_CAMERA_BLEND + res.r * BG_EFFECT_BLEND);
                        g = (int)(g_cam * BG_CAMERA_BLEND + res.g * BG_EFFECT_BLEND);
                        b = (int)(b_cam * BG_CAMERA_BLEND + res.b * BG_EFFECT_BLEND);
                        ch = res.ch;
                        auto flow = music_dist.get_bg_flow(x, y, ascii_w, ascii_h);
                        r = std::min(255, r + flow[0]);
                        g = std::min(255, g + flow[1]);
                        b = std::min(255, b + flow[2]);
                    }
                }

                r = std::clamp(r, 0, 255);
                g = std::clamp(g, 0, 255);
                b = std::clamp(b, 0, 255);

                int len = snprintf(ansi_buf, sizeof(ansi_buf), "\033[38;2;%d;%d;%dm", r, g, b);
                output_buf.append(ansi_buf, len);
                output_buf.append(ch);
            }
            if (y < ascii_h - 1) output_buf.push_back('\n');
        }

        printf("\033[H");
        fwrite(output_buf.data(), 1, output_buf.size(), stdout);
        printf("\033[0m");
        fflush(stdout);
    }

    // ── Cleanup ──
    printf("\033[?25h\033[0m\n");
    printf("Exiting Sci-Fi ASCII Camera...\n");
    fflush(stdout);
    delete scifi_bg;
    cap.release();
    audio.stop();
    return 0;
}
