/* 5-DOF pose from two silhouettes' moments. C twin of controller/pose/axis_pose.py;
 * native_parity.py --stage axis holds the two together. pose/theory.md 23. */
#ifndef PMW_AXIS_POSE_H
#define PMW_AXIS_POSE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tuning lives in axis_pose.py; the caller passes it in, so there is one home. */
struct axis_cfg {
    int64_t min_px;   /* fewer mask pixels than this: no view */
    double max_round; /* lambda_min / lambda_max above this: no minor axis */
    double min_sin;   /* sine of the two planes' angle below this: no axis */
    int iters;        /* undistortion fixed-point steps */
    double max_gap_mm;/* centroid rays missing by more than this: not the same object */
};

struct axis_cam {
    double fx, fy, cx, cy;
    double dist[5];   /* OpenCV k1 k2 p1 p2 k3 */
    double R[9];      /* camera -> world, row-major */
    double pos[3];    /* lens in world, mm */
};

struct axis_view { int64_t n; double cx, cy, ux, uy; };   /* centroid, unit minor axis */
struct axis_pose { double xyz[3], axis[3], gap; };

/* Moments of the nonzero pixels of a mask ROI; ``x0, y0`` place it in the frame.
 * Returns 0, or -1 when the view is refused. */
int axis_view(const uint8_t *mask, int stride, int x0, int y0, int w, int h,
              const struct axis_cfg *cfg, struct axis_view *out);

/* Two views -> pose. ``ref`` is the last frame's axis (or up): it picks which eigenvector
 * is the axis and fixes the sign. Returns 0, or -1 when refused. */
int axis_pose(const struct axis_cam *a, const struct axis_cam *b, const struct axis_view *va,
              const struct axis_view *vb, const double ref[3], const struct axis_cfg *cfg,
              struct axis_pose *out);

#ifdef __cplusplus
}
#endif

#endif
