// segment.py: ellipse_roi, _clamp_roi, ring_weight, ellipse_points, sample_map.
#include "pmw.h"

#include <opencv2/imgproc.hpp>

namespace pmw {

Roi clamp_roi(double x, double y, double w, double h, int H, int W, int pad) {
    // `int(round(v))` in Python: half to even.
    int xi = (int)pyround(x), yi = (int)pyround(y), wi = (int)pyround(w), hi = (int)pyround(h);
    int x0 = std::max(0, xi - pad), y0 = std::max(0, yi - pad);
    int x1 = std::min(W, xi + wi + pad), y1 = std::min(H, yi + hi + pad);
    return {x0, y0, x1 - x0, y1 - y0};
}

std::optional<Roi> ellipse_roi(const std::optional<Ellipse>& e, int H, int W, double margin) {
    if (!e) return std::nullopt;
    double r = 0.5 * std::max(e->major, e->minor) * margin;
    if (!(r > 0) || !(std::isfinite(e->cx) && std::isfinite(e->cy) && std::isfinite(r)))
        return std::nullopt;
    return clamp_roi(e->cx - r, e->cy - r, 2 * r, 2 * r, H, W, 0);
}

// Top-hat in float so the plate subtraction can go negative (segment.ring_weight).
// `img` must own its pixels: OpenCV filters a submatrix using pixels outside it as the
// border, while the Python reference hands cv2 a numpy slice, which has no parent.
static cv::Mat response(const cv::Mat& img, const cv::Mat& kernel) {
    cv::Mat m, mf, f, out;
    cv::morphologyEx(img, m, cv::MORPH_OPEN, kernel);
    m.convertTo(mf, CV_32F);
    img.convertTo(f, CV_32F);
    cv::subtract(f, mf, out);
    return out;
}

// `patch -= pw * pr` on float32 arrays: numpy keeps the product in float32.
static void subtract_scaled(cv::Mat& patch, const cv::Mat& pr, float pw) {
    for (int r = 0; r < patch.rows; ++r) {
        float* p = patch.ptr<float>(r);
        const float* q = pr.ptr<float>(r);
        for (int c = 0; c < patch.cols; ++c) p[c] -= pw * q[c];
    }
}

cv::Mat ring_weight(const cv::Mat& gray, const cv::Mat* plate, int k, double sigma,
                    double plate_weight, const std::optional<Roi>& roi, PlateCache* cache,
                    long long version) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
    bool usable = plate != nullptr && plate->size() == gray.size();
    float pw = (float)plate_weight;
    // A version < 0 (a saved plate) is never cached: Python keys those on id(img) in
    // one slot shared by both cameras, so they evict each other every frame.
    auto plate_response = [&](const std::optional<Roi>& box) {
        bool use = cache && version >= 0;
        if (use && cache->valid && cache->version == version && cache->has_box == box.has_value() &&
            (!box || (cache->box.x == box->x && cache->box.y == box->y && cache->box.w == box->w &&
                      cache->box.h == box->h)))
            return cache->response;
        cv::Mat r = box ? response((*plate)(cv::Rect(box->x, box->y, box->w, box->h)).clone(), kernel)
                        : response(*plate, kernel);
        if (use) {
            cache->valid = true;
            cache->version = version;
            cache->has_box = box.has_value();
            if (box) cache->box = *box;
            cache->response = r;
        }
        return r;
    };

    if (!roi) {
        cv::Mat w = response(gray, kernel);
        if (usable && plate_weight != 0.0) subtract_scaled(w, plate_response(std::nullopt), pw);
        cv::Mat out;
        cv::GaussianBlur(w, out, cv::Size(0, 0), sigma);
        return out;
    }
    Roi box = clamp_roi(roi->x, roi->y, roi->w, roi->h, gray.rows, gray.cols, k);
    cv::Mat out = cv::Mat::zeros(gray.size(), CV_32F);
    if (box.w <= 0 || box.h <= 0) return out;
    cv::Rect rect(box.x, box.y, box.w, box.h);
    cv::Mat patch = response(gray(rect).clone(), kernel);
    if (usable && plate_weight != 0.0) subtract_scaled(patch, plate_response(box), pw);
    cv::Mat blurred;
    cv::GaussianBlur(patch, blurred, cv::Size(0, 0), sigma);
    blurred.copyTo(out(rect));
    return out;
}

std::vector<Vector2d> ellipse_points(const Ellipse& e, int n) {
    // np.linspace(0, 2pi, n, endpoint=False): k * (2pi / n).
    double step = 2.0 * M_PI / n;
    double a = e.major / 2.0, b = e.minor / 2.0, th = e.ang * M_PI / 180.0;
    double ct = std::cos(th), st = std::sin(th);
    std::vector<Vector2d> out(n);
    for (int i = 0; i < n; ++i) {
        double t = i * step;
        double c = std::cos(t), s = std::sin(t);
        // Same association as the numpy expression, term by term.
        out[i] = Vector2d(e.cx + a * c * ct - b * s * st, e.cy + a * c * st + b * s * ct);
    }
    return out;
}

void sample_map(const cv::Mat& w, const double* xy, int n, double* out) {
    // Exact bilinear in float32, per tap, zero outside the frame. NOT cv::remap: the
    // cv2 wheel is OpenCV 5.0, whose remap reads float maps exactly, while Homebrew's
    // 4.11 quantises the coordinate to 1/32 px (the behaviour theory.md 19.12 describes).
    // The reference is the wheel, so this reproduces its arithmetic: float32 coordinates,
    // float32 weights, float32 accumulation.
    const int H = w.rows, W = w.cols;
    for (int i = 0; i < n; ++i) {
        float x = (float)xy[2 * i], y = (float)xy[2 * i + 1];
        float xf = std::floor(x), yf = std::floor(y);
        int x0 = (int)xf, y0 = (int)yf;
        float fx = x - xf, fy = y - yf;
        auto tap = [&](int r, int c) -> float {
            return (r >= 0 && r < H && c >= 0 && c < W) ? w.at<float>(r, c) : 0.0f;
        };
        // OpenCV 5's kernel, found by matching it bit for bit: a lerp of lerps, each a
        // fused multiply-add in float. Any other association differs in the last bit.
        float p00 = tap(y0, x0), p01 = tap(y0, x0 + 1), p10 = tap(y0 + 1, x0), p11 = tap(y0 + 1, x0 + 1);
        float r0 = std::fmaf(fx, p01 - p00, p00);
        float r1 = std::fmaf(fx, p11 - p10, p10);
        float v = std::fmaf(fy, r1 - r0, r0);
        out[i] = v;
    }
}

}  // namespace pmw
