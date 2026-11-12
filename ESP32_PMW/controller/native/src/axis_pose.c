#include "axis_pose.h"

#include <math.h>

static const double PI = 3.14159265358979323846;   /* M_PI is POSIX, not C11 */

static double dot(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

static void cross(const double a[3], const double b[3], double o[3]) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

int axis_view(const uint8_t *restrict mask, int stride, int x0, int y0, int w, int h,
              const struct axis_cfg *cfg, struct axis_view *out) {
    /* Exact integer sums; the inner loop is branch-free so it vectorises. */
    int64_t n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (int r = 0; r < h; ++r) {
        const uint8_t *row = mask + (int64_t)r * stride;
        int64_t rn = 0, rx = 0, rxx = 0;
        for (int c = 0; c < w; ++c) {
            const int64_t m = row[c] != 0, x = x0 + c;
            rn += m;
            rx += m * x;
            rxx += m * x * x;
        }
        const int64_t y = y0 + r;
        n += rn;
        sx += rx;
        sxx += rxx;
        sy += rn * y;
        syy += rn * y * y;
        sxy += rx * y;
    }
    if (n < cfg->min_px) return -1;
    const double dn = (double)n;
    const double cxx = ((double)sxx - (double)sx * (double)sx / dn) / dn;
    const double cyy = ((double)syy - (double)sy * (double)sy / dn) / dn;
    const double cxy = ((double)sxy - (double)sx * (double)sy / dn) / dn;
    const double tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
    const double disc = sqrt(fmax(tr * tr / 4 - det, 0.0));
    if (tr / 2 - disc > cfg->max_round * (tr / 2 + disc)) return -1;
    const double t = 0.5 * atan2(2 * cxy, cxx - cyy) + PI / 2;
    *out = (struct axis_view){n, (double)sx / dn, (double)sy / dn, cos(t), sin(t)};
    return 0;
}

/* World unit ray through a distorted pixel: OpenCV's undistortPoints fixed point. */
static void ray(const struct axis_cam *k, double u, double v, int iters, double o[3]) {
    const double k1 = k->dist[0], k2 = k->dist[1], p1 = k->dist[2], p2 = k->dist[3], k3 = k->dist[4];
    const double x0 = (u - k->cx) / k->fx, y0 = (v - k->cy) / k->fy;
    double x = x0, y = y0;
    for (int i = 0; i < iters; ++i) {
        const double r2 = x * x + y * y;
        const double ic = 1.0 / (1 + ((k3 * r2 + k2) * r2 + k1) * r2);
        const double dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        const double dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
        x = (x0 - dx) * ic;
        y = (y0 - dy) * ic;
    }
    const double *R = k->R;
    for (int i = 0; i < 3; ++i) o[i] = R[3 * i] * x + R[3 * i + 1] * y + R[3 * i + 2];
    const double s = sqrt(dot(o, o));
    for (int i = 0; i < 3; ++i) o[i] /= s;
}

int axis_pose(const struct axis_cam *a, const struct axis_cam *b, const struct axis_view *va,
              const struct axis_view *vb, const double ref[3], const struct axis_cfg *cfg,
              struct axis_pose *out) {
    const struct axis_cam *cam[2] = {a, b};
    const struct axis_view *v[2] = {va, vb};
    double d[2][3], pl[2][2][3], e[3];
    for (int i = 0; i < 2; ++i) {
        /* Each eigenvector's line and the lens span a plane; the axis lies in one of the two. */
        ray(cam[i], v[i]->cx, v[i]->cy, cfg->iters, d[i]);
        const double u[2][2] = {{v[i]->ux, v[i]->uy}, {-v[i]->uy, v[i]->ux}};
        for (int k = 0; k < 2; ++k) {
            ray(cam[i], v[i]->cx + u[k][0], v[i]->cy + u[k][1], cfg->iters, e);
            cross(d[i], e, pl[i][k]);
        }
    }
    /* Of the four plane pairs, the axis nearest ``ref`` (the last frame's) wins. */
    double ax[3] = {0, 0, 0}, best = -1;
    for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 2; ++k) {
            double c[3];
            cross(pl[0][j], pl[1][k], c);
            const double s = sqrt(dot(c, c));
            if (s / (sqrt(dot(pl[0][j], pl[0][j])) * sqrt(dot(pl[1][k], pl[1][k]))) < cfg->min_sin) continue;
            for (int i = 0; i < 3; ++i) c[i] /= s;
            if (fabs(dot(c, ref)) > best) {
                best = fabs(dot(c, ref));
                for (int i = 0; i < 3; ++i) ax[i] = c[i];
            }
        }
    if (best < 0) return -1;
    const double sign = dot(ax, ref) < 0 ? -1.0 : 1.0;
    /* Closest approach of the two centroid rays; a wide miss means two different objects. */
    double w[3];
    for (int i = 0; i < 3; ++i) w[i] = a->pos[i] - b->pos[i];
    const double c = dot(d[0], d[1]), den = 1 - c * c;
    if (den < 1e-12) return -1;
    const double dw0 = dot(d[0], w), dw1 = dot(d[1], w);
    const double s1 = (c * dw1 - dw0) / den, s2 = (dw1 - c * dw0) / den;
    double g2 = 0;
    for (int i = 0; i < 3; ++i) {
        const double p = a->pos[i] + s1 * d[0][i], q = b->pos[i] + s2 * d[1][i];
        out->xyz[i] = 0.5 * (p + q);
        out->axis[i] = sign * ax[i];
        g2 += (p - q) * (p - q);
    }
    out->gap = sqrt(g2);
    return out->gap > cfg->max_gap_mm ? -1 : 0;
}
