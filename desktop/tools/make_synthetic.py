#!/usr/bin/env python3
"""Generate a synthetic capture in exactly the format the Android app uploads.

Ray-traces a small scene (two textured spheres on a checkerboard disc) from a
ring of cameras and writes

    <out>/transforms.json   nerfstudio style, OpenGL camera-to-world matrices (Y up)
    <out>/images/*.jpg
    <out>/points3d.ply      sparse, uncoloured points (like ARCore feature points)

Usage: python3 make_synthetic.py <out_dir> [--views 48] [--width 320] [--height 240]
Requires numpy and Pillow.
"""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image

SPHERES = [  # centre, radius, base colour
    (np.array([0.0, 0.15, 0.0]), 0.15, np.array([0.9, 0.3, 0.2])),
    (np.array([0.22, 0.08, 0.12]), 0.08, np.array([0.2, 0.5, 0.95])),
]
DISC_RADIUS = 0.45


def shade_sphere(p, c, base):
    n = (p - c) / np.linalg.norm(p - c, axis=-1, keepdims=True)
    # latitude / longitude stripes as texture
    lat = np.arcsin(np.clip(n[..., 1], -1, 1))
    lon = np.arctan2(n[..., 2], n[..., 0])
    stripes = 0.6 + 0.4 * (np.sin(lat * 10) * np.sin(lon * 8) > 0)
    light = 0.35 + 0.65 * np.clip(n @ np.array([0.4, 0.8, 0.45]) / 1.0, 0, 1)
    return base[None, :] * (stripes * light)[:, None]


def render(eye, target, w, h, f):
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cx, cy = w / 2.0, h / 2.0
    j, i = np.mgrid[0:h, 0:w]
    x = (i - cx) / f
    y = (j - cy) / f  # OpenCV: y down
    d = x[..., None] * right - y[..., None] * up + fwd
    d = d.reshape(-1, 3)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    n_px = d.shape[0]
    t_best = np.full(n_px, np.inf)
    col = np.zeros((n_px, 3))
    # background gradient (sky)
    col[:] = (0.15 + 0.25 * np.clip(d[:, 1:2], 0, 1)) * np.array([0.6, 0.7, 1.0])

    for c, r, base in SPHERES:
        oc = eye - c
        b = d @ oc
        cc = oc @ oc - r * r
        disc = b * b - cc
        hit = disc > 0
        t = np.where(hit, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
        hit &= (t > 0) & (t < t_best)
        if hit.any():
            p = eye + d[hit] * t[hit, None]
            col[hit] = shade_sphere(p, c, base)
            t_best[hit] = t[hit]

    # checkerboard disc at y = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -eye[1] / d[:, 1]
    p = eye + d * t[:, None]
    hit = (t > 0) & (t < t_best) & (p[:, 0] ** 2 + p[:, 2] ** 2 < DISC_RADIUS ** 2)
    if hit.any():
        q = p[hit]
        chk = ((np.floor(q[:, 0] * 10) + np.floor(q[:, 2] * 10)) % 2 == 0)
        col[hit] = np.where(chk[:, None], [0.85, 0.85, 0.8], [0.25, 0.3, 0.25])
        t_best[hit] = t[hit]

    img = (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(h, w, 3)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -fwd
    c2w[:3, 3] = eye
    return img, c2w


def sample_points(n, rng):
    pts = []
    for k in range(n):
        u = rng.random()
        if u < 0.55:
            c, r, _ = SPHERES[0 if rng.random() < 0.75 else 1]
            v = rng.normal(size=3)
            pts.append(c + r * v / np.linalg.norm(v))
        else:
            a = rng.random() * 2 * math.pi
            rr = DISC_RADIUS * math.sqrt(rng.random())
            pts.append(np.array([rr * math.cos(a), 0.0, rr * math.sin(a)]))
    pts = np.array(pts) + rng.normal(scale=0.004, size=(n, 3))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--views", type=int, default=48)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--points", type=int, default=1500)
    args = ap.parse_args()

    rng = np.random.default_rng(1)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    f = 0.9 * args.width
    frames = []
    target = np.array([0.05, 0.1, 0.03])
    for k in range(args.views):
        a = 2 * math.pi * k / args.views
        height = 0.25 + 0.35 * (k % 3) / 2.0
        eye = np.array([0.9 * math.cos(a), height, 0.9 * math.sin(a)])
        img, c2w = render(eye, target, args.width, args.height, f)
        name = "images/frame_%05d.jpg" % k
        Image.fromarray(img).save(os.path.join(args.out, name), quality=92)
        frames.append({
            "file_path": name,
            "transform_matrix": c2w.tolist(),
            "fl_x": f, "fl_y": f, "cx": args.width / 2.0, "cy": args.height / 2.0,
            "w": args.width, "h": args.height,
        })
    with open(os.path.join(args.out, "transforms.json"), "w") as fh:
        json.dump({"camera_model": "OPENCV", "ply_file_path": "points3d.ply",
                   "source": "make_synthetic.py", "frames": frames}, fh, indent=1)

    pts = sample_points(args.points, rng)
    with open(os.path.join(args.out, "points3d.ply"), "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(pts))
        fh.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            fh.write("%.6f %.6f %.6f\n" % tuple(p))
    print("wrote %d views to %s" % (args.views, args.out))


if __name__ == "__main__":
    main()
