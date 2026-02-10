#include "ascii_cam.h"

// ═══════════════════════════════════════════════════════════════
//  GLOBAL DATA DEFINITIONS
// ═══════════════════════════════════════════════════════════════
const std::string ASCII_RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkha*#MW&8%B@$";

static std::vector<std::string> make_char_vec(const std::string& s) {
    std::vector<std::string> v;
    // Handle multi-byte UTF-8 chars
    size_t i = 0;
    while (i < s.size()) {
        int len = 1;
        unsigned char c = s[i];
        if (c >= 0xC0 && c < 0xE0) len = 2;
        else if (c >= 0xE0 && c < 0xF0) len = 3;
        else if (c >= 0xF0) len = 4;
        v.push_back(s.substr(i, len));
        i += len;
    }
    return v;
}

const std::vector<std::string> MATRIX_CHARS = make_char_vec("01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン");
const std::vector<std::string> GLITCH_CHARS = make_char_vec("!@#$%^&*<>{}[]|/\\~`?=+");
const std::vector<std::string> TRIPPY_CHARS = make_char_vec("░▒▓█╬╠╣╦╩┼┤├┬┴│─┌┐└┘◆◇○●∞≈≡±×÷∆∇∂∫Σπφψω");
const std::vector<std::string> CYBER_WORDS = {"CYBER","HACK","NEURAL","SYNC","VOID","FLUX","GLITCH","NEON",
    "DATA","CODE","PULSE","WAVE","NODE","GRID","LINK","CORE","ZERO","ROOT","SYS","NET","BIT","HEX","BOOT","INIT"};
const std::vector<std::string> HUD_CHARS = make_char_vec("╔╗╚╝═║╠╣");

const std::map<std::string, std::vector<Color3>> PALETTES = {
    {"CYBERPUNK", {{0,255,255},{255,0,255},{0,0,255},{255,255,0}}},
    {"MATRIX",    {{0,255,0},{0,128,0},{150,255,150},{0,50,0}}},
    {"PLASMA",    {{255,0,0},{255,100,0},{255,0,100},{100,0,255}}}
};

const std::vector<Color3> FACE_COLORS = {{255,215,0},{255,165,0},{255,69,0},{255,255,100}};

const std::vector<std::string> FaceTrail::BRIGHT_CHARS = make_char_vec("█▓▒░★✦⚡●◆♦♢✧");
const std::vector<std::string> FaceTrail::FADE_CHARS = make_char_vec("▒░·•✧○◇");
const std::vector<std::string> FaceTrail::DIM_CHARS = make_char_vec("·.,:;");

const std::vector<std::string> BodyTrail::BRIGHT_CHARS = make_char_vec("▓▒░✦●◆♦");
const std::vector<std::string> BodyTrail::FADE_CHARS = make_char_vec("░·•○◇");
const std::vector<std::string> BodyTrail::DIM_CHARS = make_char_vec("·.:");

// ═══════════════════════════════════════════════════════════════
//  UTILITY IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════
Color3 lerp_color(const Color3& c1, const Color3& c2, double t) {
    t = std::max(0.0, std::min(1.0, t));
    return {(int)(c1[0]+(c2[0]-c1[0])*t), (int)(c1[1]+(c2[1]-c1[1])*t), (int)(c1[2]+(c2[2]-c1[2])*t)};
}

Color3 sample_gradient(const std::vector<Color3>& colors, double t) {
    t = fmod(t, 1.0); if(t<0) t+=1.0;
    int n = (int)colors.size()-1;
    double idx = t*n;
    int i = std::min((int)idx, n-1);
    return lerp_color(colors[i], colors[i+1], idx-i);
}

void hsv_to_rgb(double h, double s, double v, double& ro, double& go, double& bo) {
    if(s<=0.0){ro=go=bo=v;return;}
    h=fmod(h,1.0); if(h<0)h+=1.0;
    h*=6.0;
    int i=(int)h;
    double f=h-i, p=v*(1-s), q=v*(1-s*f), t2=v*(1-s*(1-f));
    switch(i%6){
        case 0:ro=v;go=t2;bo=p;break; case 1:ro=q;go=v;bo=p;break;
        case 2:ro=p;go=v;bo=t2;break; case 3:ro=p;go=q;bo=v;break;
        case 4:ro=t2;go=p;bo=v;break; default:ro=v;go=p;bo=q;break;
    }
}

std::pair<int,int> get_terminal_size() {
    struct winsize ws;
    if(ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws)==0) return {ws.ws_col, ws.ws_row};
    return {CFG_WIDTH, 50};
}

// ═══════════════════════════════════════════════════════════════
//  AUDIO CAPTURE
// ═══════════════════════════════════════════════════════════════
AudioCapture::AudioCapture() {
    PaError err = Pa_Initialize();
    initialized_ = (err == paNoError);
    if(!initialized_) fprintf(stderr, "PortAudio init failed: %s\n", Pa_GetErrorText(err));
}

AudioCapture::~AudioCapture() { stop(); }

int AudioCapture::pa_callback(const void* input, void*, unsigned long frames,
                               const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void* ud) {
    auto* self = (AudioCapture*)ud;
    if(!input) return paContinue;
    const short* data = (const short*)input;
    double sum = 0;
    for(unsigned long i=0; i<frames; i++) sum += (double)data[i]*data[i];
    double rms = sqrt(sum/frames);
    // Exponential smoothing
    double prev = self->volume_.load();
    self->volume_.store(prev*0.7 + rms*0.3);
    return paContinue;
}

bool AudioCapture::open_mic() {
    if(!initialized_) return false;
    PaStreamParameters inp;
    inp.device = Pa_GetDefaultInputDevice();
    if(inp.device == paNoDevice){fprintf(stderr,"No mic\n");return false;}
    inp.channelCount = CHANNELS;
    inp.sampleFormat = paInt16;
    inp.suggestedLatency = Pa_GetDeviceInfo(inp.device)->defaultLowInputLatency;
    inp.hostApiSpecificStreamInfo = nullptr;
    PaError err = Pa_OpenStream(&stream_,&inp,nullptr,RATE,CHUNK,paClipOff,pa_callback,this);
    if(err!=paNoError){fprintf(stderr,"Mic open fail: %s\n",Pa_GetErrorText(err));return false;}
    Pa_StartStream(stream_);
    printf("✓ Microphone audio stream opened.\n");
    return true;
}

bool AudioCapture::open_monitor() {
    if(!initialized_) return false;
    int numDev = Pa_GetDeviceCount();
    for(int i=0;i<numDev;i++){
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if(!info || info->maxInputChannels<=0) continue;
        std::string name = info->name;
        // lowercase search for "monitor"
        std::string lower = name;
        std::transform(lower.begin(),lower.end(),lower.begin(),::tolower);
        if(lower.find("monitor")!=std::string::npos){
            PaStreamParameters inp;
            inp.device = i; inp.channelCount = CHANNELS;
            inp.sampleFormat = paInt16;
            inp.suggestedLatency = info->defaultLowInputLatency;
            inp.hostApiSpecificStreamInfo = nullptr;
            PaError err = Pa_OpenStream(&stream_,&inp,nullptr,RATE,CHUNK,paClipOff,pa_callback,this);
            if(err==paNoError){Pa_StartStream(stream_);printf("✓ System audio: %s\n",name.c_str());return true;}
        }
    }
    fprintf(stderr,"⚠ No monitor device found, falling back to mic\n");
    return open_mic();
}

double AudioCapture::get_volume() const { return volume_.load(); }

void AudioCapture::stop() {
    running_ = false;
    if(stream_){Pa_StopStream(stream_);Pa_CloseStream(stream_);stream_=nullptr;}
    if(initialized_){Pa_Terminate();initialized_=false;}
}

// ═══════════════════════════════════════════════════════════════
//  SCIFI BACKGROUND
// ═══════════════════════════════════════════════════════════════
SciFiBackground::SciFiBackground(int w_, int h_) : w(w_), h(h_), plasma_phase(randf()*6.28) {
    rain_cols.resize(w);
    for(int x=0;x<w;x++){
        rain_cols[x]={randint(-h,h), randint(1,5), randint(3,12), randf()<0.35, randf()};
    }
    for(int i=0;i<5;i++){
        h_trails.push_back({randint(0,h-1), randint(-30,w), rand_choice(CYBER_WORDS), randint(2,6), randf()});
        v_trails.push_back({randint(0,w-1), randint(-20,h), rand_choice(CYBER_WORDS), randint(1,4), randf()});
    }
}

void SciFiBackground::update(double audio_level, double anorm) {
    frame_count++; audio_norm=anorm;
    plasma_phase += 0.08 + anorm*0.15;
    for(auto& col:rain_cols){
        if(col.active){col.y+=col.speed;if(col.y>h+col.len){col.y=-randint(3,15);col.speed=randint(1,5);col.hue=randf();col.active=randf()<0.4;}}
        else{if(randf()<0.03+anorm*0.05){col.active=true;col.y=-col.len;}}
    }
    if(anorm>0.3 && randf()<0.4){
        int bw=randint(3,20),bh=randint(2,8);
        glitch_blocks.push_back({randint(0,std::max(1,w-bw)),randint(0,std::max(1,h-bh)),bw,bh,randf(0.3,1.0),randf(),rand_choice(TRIPPY_CHARS)});
    }
    glitch_blocks.erase(std::remove_if(glitch_blocks.begin(),glitch_blocks.end(),[](GlitchBlock&g){g.life-=0.08;return g.life<=0;}),glitch_blocks.end());
    int ns=3+(int)(anorm*15);
    for(int i=0;i<ns;i++) sparks[{randint(0,w-1),randint(0,h-1)}]=randf(0.3,1.0);
    std::vector<std::pair<int,int>> dead;
    for(auto& [k,v]:sparks){v-=0.15;if(v<=0)dead.push_back(k);}
    for(auto& k:dead) sparks.erase(k);
    for(auto& ht:h_trails){ht.x+=ht.speed;if(ht.x>w+10){ht.x=-(int)ht.word.size()*2;ht.y=randint(0,h-1);ht.word=rand_choice(CYBER_WORDS);ht.hue=randf();}}
    for(auto& vt:v_trails){vt.y+=vt.speed;if(vt.y>h+10){vt.y=-(int)vt.word.size()*2;vt.x=randint(0,w-1);vt.word=rand_choice(CYBER_WORDS);vt.hue=randf();}}
    scan_y=(scan_y+scan_speed)%h;
    if(anorm>0.5&&randf()<0.2) h_trails.push_back({randint(0,h-1),-10,rand_choice(CYBER_WORDS),randint(3,8),randf()});
    if((int)h_trails.size()>12) h_trails.erase(h_trails.begin(),h_trails.begin()+((int)h_trails.size()-12));
    if((int)v_trails.size()>10) v_trails.erase(v_trails.begin(),v_trails.begin()+((int)v_trails.size()-10));
}

PixelResult SciFiBackground::get_effect(int x, int y, int brightness) {
    double inten=BG_INTENSITY;
    double ro,go,bo;
    // Scan line
    if(abs(y-scan_y)<1){
        double hue=fmod(frame_count*0.02,1.0);
        hsv_to_rgb(hue,0.5,0.8*inten,ro,go,bo);
        return {"─",(int)(ro*255),(int)(go*255),(int)(bo*255)};
    }
    // H trails
    for(auto& ht:h_trails){
        if(y==ht.y){int ci=x-ht.x;if(ci>=0&&ci<(int)ht.word.size()){
            hsv_to_rgb(ht.hue,0.9,0.9*inten,ro,go,bo);
            return {std::string(1,ht.word[ci]),(int)(ro*255),(int)(go*255),(int)(bo*255)};
        }}
    }
    // V trails
    for(auto& vt:v_trails){
        if(x==vt.x){int ci=y-vt.y;if(ci>=0&&ci<(int)vt.word.size()){
            hsv_to_rgb(vt.hue,0.9,0.85*inten,ro,go,bo);
            return {std::string(1,vt.word[ci]),(int)(ro*255),(int)(go*255),(int)(bo*255)};
        }}
    }
    // Sparks
    auto it=sparks.find({x,y});
    if(it!=sparks.end()){
        double hue=fmod(x*0.01+y*0.01+frame_count*0.1,1.0);
        hsv_to_rgb(hue,0.8,it->second*inten,ro,go,bo);
        return {rand_choice(GLITCH_CHARS),(int)(ro*255),(int)(go*255),(int)(bo*255)};
    }
    // Glitch blocks
    for(auto& gb:glitch_blocks){
        if(x>=gb.x&&x<gb.x+gb.w&&y>=gb.y&&y<gb.y+gb.h){
            hsv_to_rgb(gb.hue,0.9,gb.life*0.6*inten,ro,go,bo);
            return {gb.ch,(int)(ro*255),(int)(go*255),(int)(bo*255)};
        }
    }
    // Matrix rain
    auto& col=rain_cols[x];
    if(col.active){
        int dist=y-col.y;
        if(dist>=0&&dist<col.len){
            double nd=(double)dist/col.len, intensity=1.0-nd;
            double hue=fmod(col.hue+frame_count*0.005,1.0);
            if(dist<1){hsv_to_rgb(hue,0.3,0.9*inten,ro,go,bo);return{rand_choice(MATRIX_CHARS),(int)(ro*255),(int)(go*255),(int)(bo*255)};}
            hsv_to_rgb(hue,0.7,intensity*0.5*inten,ro,go,bo);
            return{rand_choice(MATRIX_CHARS),(int)(ro*255),(int)(go*255),(int)(bo*255)};
        }
    }
    // Plasma
    double t=plasma_phase;
    double v1=sin(x*0.04+t),v2=sin(y*0.06-t*0.7),v3=sin(x*0.03+y*0.04+t*0.5);
    double v4=sin(sqrt(std::max(0.01,(double)(x-w/2)*(x-w/2)+(double)(y-h/2)*(y-h/2)))*0.08-t);
    double plasma=(v1+v2+v3+v4)/4.0;
    double hue=fmod((plasma+1.0)/2.0+frame_count*0.008,1.0);
    double sat=0.6+audio_norm*0.3;
    double val=(0.08+fabs(plasma)*0.12+audio_norm*0.06)*inten;
    hsv_to_rgb(hue,sat,val,ro,go,bo);
    std::string ch;
    if(randf()<0.02+audio_norm*0.03) ch=rand_choice(TRIPPY_CHARS);
    else{int n=(int)ASCII_RAMP.size();int idx=std::min((int)((fabs(plasma)+0.1)*(n-1)),n-1);ch=std::string(1,ASCII_RAMP[idx]);}
    return{ch,(int)(ro*255),(int)(go*255),(int)(bo*255)};
}

// ═══════════════════════════════════════════════════════════════
//  BODY EFFECTS
// ═══════════════════════════════════════════════════════════════
void BodyEffects::update(double audio_norm) {
    frame_count++;
    double ba = audio_norm*AUDIO_REACT_BODY;
    pattern_offset += 0.1+ba*0.2;
    if(ba>0.8) current_palette="PLASMA";
    else if(ba>0.4) current_palette="CYBERPUNK";
    else current_palette="MATRIX";
}

PixelResult BodyEffects::get_color(int x, int y, int brightness, int w, int h, const cv::Vec3b& px, bool is_aura) {
    if(is_aura){if(randf()<0.4) return{":",(int)100,(int)200,(int)255}; return{" ",0,0,0};}
    int n=(int)ASCII_RAMP.size();
    double nb=brightness/255.0;
    int idx=std::min((int)(nb*(n-1)),n-1);
    std::string ch(1,ASCII_RAMP[idx]);
    int r_orig=px[0],g_orig=px[1],b_orig=px[2];
    double nx=(double)x/w, ny=(double)y/h;
    double wave=sin(nx*10+ny*10-pattern_offset), wave2=cos(nx*20-ny*5+pattern_offset*0.5);
    double pattern=(wave+wave2)/2.0;
    auto pit=PALETTES.find(current_palette);
    auto& palette=pit->second;
    double t=(pattern+1.0)/2.0;
    Color3 neon=sample_gradient(palette,t);
    int r=(int)(r_orig*BODY_CAMERA_BLEND+neon[0]*BODY_EFFECT_BLEND*nb);
    int g=(int)(g_orig*BODY_CAMERA_BLEND+neon[1]*BODY_EFFECT_BLEND*nb);
    int b=(int)(b_orig*BODY_CAMERA_BLEND+neon[2]*BODY_EFFECT_BLEND*nb);
    r=std::min(255,(int)(r*BODY_BRIGHTNESS_BOOST));
    g=std::min(255,(int)(g*BODY_BRIGHTNESS_BOOST));
    b=std::min(255,(int)(b*BODY_BRIGHTNESS_BOOST));
    return{ch,r,g,b};
}

// ═══════════════════════════════════════════════════════════════
//  FACE EFFECTS
// ═══════════════════════════════════════════════════════════════
void FaceEffects::update(int face_top, int face_bottom) {
    frame_count++;
    int h=face_bottom-face_top;
    if(h>0) scan_line_y=face_top+(frame_count*scan_speed)%h;
}

PixelResult FaceEffects::get_color(int x, int y, int brightness, int ft, int fb, int fl, int fr,
                                    double audio_norm, const cv::Vec3b& px) {
    if(x==fl&&y==ft) return{"╔",255,255,0};
    if(x==fr&&y==ft) return{"╗",255,255,0};
    if(x==fl&&y==fb) return{"╚",255,255,0};
    if(x==fr&&y==fb) return{"╝",255,255,0};
    int n=(int)ASCII_RAMP.size();
    double nb=brightness/255.0;
    int idx=std::min((int)(nb*(n-1)),n-1);
    std::string ch(1,ASCII_RAMP[idx]);
    auto now=std::chrono::steady_clock::now();
    double t_sec=std::chrono::duration<double>(now.time_since_epoch()).count()*0.5;
    Color3 base=sample_gradient(FACE_COLORS,fmod(t_sec,1.0));
    int r=(int)(px[0]*FACE_CAMERA_BLEND+base[0]*FACE_EFFECT_BLEND);
    int g=(int)(px[1]*FACE_CAMERA_BLEND+base[1]*FACE_EFFECT_BLEND);
    int b=(int)(px[2]*FACE_CAMERA_BLEND+base[2]*FACE_EFFECT_BLEND);
    r=std::min(255,(int)(r*FACE_BRIGHTNESS_BOOST));
    g=std::min(255,(int)(g*FACE_BRIGHTNESS_BOOST));
    b=std::min(255,(int)(b*FACE_BRIGHTNESS_BOOST));
    if(audio_norm*AUDIO_REACT_FACE>0.7&&randf()<0.1){
        r=g=b=255;
        if(randf()<0.5) ch=rand_choice(MATRIX_CHARS);
    }
    return{ch,r,g,b};
}

// ═══════════════════════════════════════════════════════════════
//  FACE TRAIL
// ═══════════════════════════════════════════════════════════════
void FaceTrail::update(int ft, int fb, int fl, int fr, bool has_face) {
    frame_count++;
    trail_hue=fmod(trail_hue+0.025,1.0);
    if(has_face){
        int cx=(fl+fr)/2, cy=(ft+fb)/2, fw=std::max(fr-fl,1), fh=std::max(fb-ft,1);
        if(has_prev){
            int dx=abs(cx-prev_cx), dy=abs(cy-prev_cy);
            if(dx>1||dy>1){
                double speed=sqrt((double)(dx*dx+dy*dy));
                double tlife=std::min(1.0,0.7+speed*0.05);
                for(int i=0;i<std::max(fw,fh);i++){
                    int tx=fl+(i%fw);
                    for(int off=0;off<2;off++){
                        trail[key(tx,ft+off)]={tlife,trail_hue};
                        trail[key(tx,fb-off)]={tlife,fmod(trail_hue+0.3,1.0)};
                    }
                    int ty=ft+(i%fh);
                    for(int off=0;off<2;off++){
                        trail[key(fl+off,ty)]={tlife,fmod(trail_hue+0.5,1.0)};
                        trail[key(fr-off,ty)]={tlife,fmod(trail_hue+0.7,1.0)};
                    }
                }
                int steps=std::max((int)speed,2);
                for(int s=0;s<steps;s++){
                    double t=(double)s/steps;
                    int ix=(int)(prev_cx+(cx-prev_cx)*t), iy=(int)(prev_cy+(cy-prev_cy)*t);
                    for(int ox=-2;ox<=2;ox++) for(int oy=-1;oy<=1;oy++)
                        trail[key(ix+ox,iy+oy)]={tlife*0.8,fmod(trail_hue+t*0.5,1.0)};
                }
                int nsp=20+(int)(speed*3);
                for(int i=0;i<nsp;i++){
                    int sx=prev_cx+randint(-fw/2-3,fw/2+3), sy=prev_cy+randint(-fh/2-2,fh/2+2);
                    trail[key(sx,sy)]={randf(0.6,1.0),fmod(trail_hue+randf(-0.2,0.2),1.0)};
                }
            }
        }
        prev_cx=cx; prev_cy=cy; has_prev=true;
    }
    std::vector<long long> dead;
    for(auto& [k,v]:trail){v.life-=0.025;if(v.life<=0)dead.push_back(k);}
    for(auto k:dead) trail.erase(k);
    if((int)trail.size()>8000){
        std::vector<std::pair<long long,double>> sorted;
        for(auto&[k,v]:trail) sorted.push_back({k,v.life});
        std::sort(sorted.begin(),sorted.end(),[](auto&a,auto&b){return a.second<b.second;});
        for(int i=0;i<2000&&i<(int)sorted.size();i++) trail.erase(sorted[i].first);
    }
}

std::tuple<bool,std::string,int,int,int> FaceTrail::get_trail(int x, int y) {
    auto it=trail.find(key(x,y));
    if(it!=trail.end()){
        auto& t=it->second;
        double vis=std::min(1.0,t.life*1.8);
        double ro,go,bo;
        hsv_to_rgb(t.hue,1.0,vis,ro,go,bo);
        std::string ch;
        if(t.life>0.6) ch=rand_choice(BRIGHT_CHARS);
        else if(t.life>0.3) ch=rand_choice(FADE_CHARS);
        else ch=rand_choice(DIM_CHARS);
        return{true,ch,(int)(ro*255),(int)(go*255),(int)(bo*255)};
    }
    return{false," ",0,0,0};
}

// ═══════════════════════════════════════════════════════════════
//  BODY TRAIL
// ═══════════════════════════════════════════════════════════════
void BodyTrail::update(const cv::Mat& body_mask, const cv::Mat& is_body_mat) {
    frame_count++;
    trail_hue=fmod(trail_hue+0.015,1.0);
    // Find body centroid
    std::vector<cv::Point> pts;
    cv::findNonZero(is_body_mat, pts);
    bool has_body = (int)pts.size()>20;
    if(has_body){
        double sx=0,sy=0;
        for(auto&p:pts){sx+=p.x;sy+=p.y;}
        int cx=(int)(sx/pts.size()), cy=(int)(sy/pts.size());
        int bx_min=INT_MAX,bx_max=0,by_min=INT_MAX,by_max=0;
        for(auto&p:pts){bx_min=std::min(bx_min,p.x);bx_max=std::max(bx_max,p.x);by_min=std::min(by_min,p.y);by_max=std::max(by_max,p.y);}
        int bw=std::max(bx_max-bx_min,1), bh=std::max(by_max-by_min,1);
        if(has_prev){
            int dx=abs(cx-prev_cx),dy=abs(cy-prev_cy);
            if(dx>2||dy>2){
                double speed=sqrt((double)(dx*dx+dy*dy));
                double tlife=std::min(0.85,0.5+speed*0.03);
                int h=body_mask.rows, w=body_mask.cols;
                int stride=std::max(3,bh/15);
                for(int row=by_min;row<by_max;row+=stride){
                    const uchar* r=is_body_mat.ptr<uchar>(row);
                    int left=-1,right=-1;
                    for(int c=bx_min;c<=bx_max;c++){
                        if(r[c]>0){if(left<0)left=c;right=c;}
                    }
                    if(left>=0){
                        trail[key(left,row)]={tlife,fmod(trail_hue+row*0.005,1.0)};
                        trail[key(right,row)]={tlife,fmod(trail_hue+0.4+row*0.005,1.0)};
                    }
                }
                for(int col=bx_min;col<bx_max;col+=stride){
                    int top=-1,bot=-1;
                    for(int row=by_min;row<=by_max;row++){
                        if(is_body_mat.at<uchar>(row,col)>0){if(top<0)top=row;bot=row;}
                    }
                    if(top>=0){
                        trail[key(col,top)]={tlife,fmod(trail_hue+0.2,1.0)};
                        trail[key(col,bot)]={tlife,fmod(trail_hue+0.6,1.0)};
                    }
                }
                int nsp=10+(int)(speed*1.5);
                for(int i=0;i<nsp;i++){
                    int sx2=prev_cx+randint(-bw/4,bw/4), sy2=prev_cy+randint(-bh/4,bh/4);
                    trail[key(sx2,sy2)]={randf(0.4,0.75),fmod(trail_hue+randf(-0.15,0.15),1.0)};
                }
            }
        }
        prev_cx=cx;prev_cy=cy;has_prev=true;
    }
    std::vector<long long> dead;
    for(auto&[k,v]:trail){v.life-=0.035;if(v.life<=0)dead.push_back(k);}
    for(auto k:dead) trail.erase(k);
    if((int)trail.size()>5000){
        std::vector<std::pair<long long,double>> sorted;
        for(auto&[k,v]:trail) sorted.push_back({k,v.life});
        std::sort(sorted.begin(),sorted.end(),[](auto&a,auto&b){return a.second<b.second;});
        for(int i=0;i<1500&&i<(int)sorted.size();i++) trail.erase(sorted[i].first);
    }
}

std::tuple<bool,std::string,int,int,int> BodyTrail::get_trail(int x, int y) {
    auto it=trail.find(key(x,y));
    if(it!=trail.end()){
        auto& t=it->second;
        double vis=std::min(1.0,t.life*1.4);
        double ro,go,bo;
        hsv_to_rgb(t.hue,0.85,vis,ro,go,bo);
        std::string ch;
        if(t.life>0.5) ch=rand_choice(BRIGHT_CHARS);
        else if(t.life>0.25) ch=rand_choice(FADE_CHARS);
        else ch=rand_choice(DIM_CHARS);
        return{true,ch,(int)(ro*255),(int)(go*255),(int)(bo*255)};
    }
    return{false," ",0,0,0};
}

// ═══════════════════════════════════════════════════════════════
//  MUSIC DISTORTION
// ═══════════════════════════════════════════════════════════════
void MusicDistortion::update(double audio_norm) {
    frame_count++;
    smooth_audio=smooth_audio*0.7+audio_norm*0.3;
    phase+=0.05+smooth_audio*0.2;
    double ad=audio_norm-prev_audio;
    if(ad>0.15){beat_flash=std::min(1.0,beat_flash+ad*2.0);beat_accumulator+=1.0;}
    beat_flash*=0.85;
    prev_audio=audio_norm;
    flow_offset_x+=0.03+smooth_audio*0.1;
    flow_offset_y+=0.02+smooth_audio*0.08;
}

Color3 MusicDistortion::get_body_distortion(int x, int y, int w, int h, const cv::Vec3b& px, int brightness) {
    double audio=smooth_audio;
    if(audio<0.05) return{(int)px[0],(int)px[1],(int)px[2]};
    double strength=audio*MUSIC_DISTORTION_STRENGTH*AUDIO_REACT_BODY;
    double nx=(double)x/std::max(w,1), ny=(double)y/std::max(h,1);
    double w1=sin(nx*8.0+phase*1.3),w2=sin(ny*6.0-phase*0.9);
    double w3=sin((nx+ny)*5.0+phase*1.1);
    double w4=sin(sqrt(std::max(0.01,(nx-0.5)*(nx-0.5)+(ny-0.5)*(ny-0.5)))*12.0-phase*1.5);
    double combined=(w1+w2+w3+w4)/4.0;
    double hue=fmod(combined*0.5+0.5+frame_count*0.01,1.0);
    double nb=brightness/255.0, val=0.5+nb*0.5, sat=0.85+beat_flash*0.15;
    double ro,go,bo;
    hsv_to_rgb(hue,sat,val,ro,go,bo);
    int rd=(int)(ro*255),gd=(int)(go*255),bd=(int)(bo*255);
    int r=(int)(px[0]*(1-strength)+rd*strength);
    int g=(int)(px[1]*(1-strength)+gd*strength);
    int b=(int)(px[2]*(1-strength)+bd*strength);
    if(beat_flash>0.3){double fl=beat_flash*0.4;r=std::min(255,(int)(r+255*fl*strength));g=std::min(255,(int)(g+255*fl*strength));b=std::min(255,(int)(b+255*fl*strength));}
    return{std::min(255,r),std::min(255,g),std::min(255,b)};
}

Color3 MusicDistortion::get_bg_flow(int x, int y, int w, int h) {
    if(smooth_audio<0.03) return{0,0,0};
    double strength=smooth_audio*MUSIC_BG_FLOW_STRENGTH;
    double nx=(double)x/std::max(w,1), ny=(double)y/std::max(h,1);
    double f1=sin(nx*6.0+flow_offset_x+sin(ny*3.0+flow_offset_y)*2.0);
    double f2=sin(ny*7.0-flow_offset_y*1.3+sin(nx*4.0-flow_offset_x)*1.5);
    double f3=sin((nx*3.0+ny*4.0)+flow_offset_x*0.7);
    double f4=cos(nx*5.0-ny*3.0+flow_offset_y*0.9);
    double blob=(f1+f2+f3+f4)/4.0;
    int r,g,b;
    if(blob>0){double i=blob*strength;r=(int)(0*i*255);g=(int)(0.9*i*255);b=(int)(1.0*i*255);}
    else{double i=fabs(blob)*strength;r=(int)(1.0*i*255);g=0;b=(int)(0.8*i*255);}
    if(beat_flash>0.2){double m=1.0+beat_flash*0.8;r=(int)(r*m);g=(int)(g*m);b=(int)(b*m);}
    return{std::min(255,r),std::min(255,g),std::min(255,b)};
}

Color3 MusicDistortion::get_face_glow(int x, int y, int w, int h, double audio_norm) {
    if(audio_norm<0.4||beat_flash<0.2) return{0,0,0};
    double strength=audio_norm*AUDIO_REACT_FACE*beat_flash;
    double nx=(double)x/std::max(w,1), ny=(double)y/std::max(h,1);
    double hue=fmod(nx*2.0+ny*2.0+phase*0.5,1.0);
    double ro,go,bo;
    hsv_to_rgb(hue,0.8,strength*0.3,ro,go,bo);
    return{(int)(ro*255),(int)(go*255),(int)(bo*255)};
}
