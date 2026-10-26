/*
 * dataset.c - loads a capture produced by the Android app.
 *
 * Expected layout (nerfstudio compatible):
 *   transforms.json   camera intrinsics + per frame camera-to-world matrices
 *                     (OpenGL convention: +X right, +Y up, camera looks down -Z)
 *   images/<n>.jpg    the photos
 *   points3d.ply      sparse point cloud used for initialisation (optional)
 */
#include "dataset.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "platform.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "../third_party/stb_image.h"

/* Area-averaging resize of an RGBA image. */
static unsigned char *resize_rgba(const unsigned char *src, int sw, int sh, int dw, int dh)
{
    unsigned char *dst = (unsigned char *)malloc((size_t)dw * dh * 4);
    float sx = (float)sw / dw, sy = (float)sh / dh;
    int x, y;
    if (!dst)
        return NULL;
    for (y = 0; y < dh; y++) {
        int y0 = (int)(y * sy), y1 = (int)((y + 1) * sy);
        if (y1 <= y0)
            y1 = y0 + 1;
        if (y1 > sh)
            y1 = sh;
        for (x = 0; x < dw; x++) {
            int x0 = (int)(x * sx), x1 = (int)((x + 1) * sx);
            unsigned acc[4] = {0, 0, 0, 0};
            unsigned n = 0;
            int yy, xx, c;
            if (x1 <= x0)
                x1 = x0 + 1;
            if (x1 > sw)
                x1 = sw;
            for (yy = y0; yy < y1; yy++)
                for (xx = x0; xx < x1; xx++) {
                    const unsigned char *p = src + ((size_t)yy * sw + xx) * 4;
                    for (c = 0; c < 4; c++)
                        acc[c] += p[c];
                    n++;
                }
            for (c = 0; c < 4; c++)
                dst[((size_t)y * dw + x) * 4 + c] = (unsigned char)((acc[c] + n / 2) / (n ? n : 1));
        }
    }
    return dst;
}

/* ---------------------------------------------------------------------- */
/* PLY point loader                                                        */
/* ---------------------------------------------------------------------- */

typedef enum { P_CHAR, P_UCHAR, P_SHORT, P_USHORT, P_INT, P_UINT, P_FLOAT, P_DOUBLE } ply_type;

static int ply_type_from_name(const char *n, ply_type *t, int *size)
{
    static const struct {
        const char *name;
        ply_type t;
        int size;
    } map[] = {{"char", P_CHAR, 1},    {"int8", P_CHAR, 1},     {"uchar", P_UCHAR, 1},
               {"uint8", P_UCHAR, 1},  {"short", P_SHORT, 2},   {"int16", P_SHORT, 2},
               {"ushort", P_USHORT, 2}, {"uint16", P_USHORT, 2}, {"int", P_INT, 4},
               {"int32", P_INT, 4},    {"uint", P_UINT, 4},     {"uint32", P_UINT, 4},
               {"float", P_FLOAT, 4},  {"float32", P_FLOAT, 4}, {"double", P_DOUBLE, 8},
               {"float64", P_DOUBLE, 8}};
    size_t i;
    for (i = 0; i < sizeof(map) / sizeof(map[0]); i++)
        if (strcmp(map[i].name, n) == 0) {
            *t = map[i].t;
            *size = map[i].size;
            return 0;
        }
    return -1;
}

static double ply_read_bin(const unsigned char *p, ply_type t)
{
    switch (t) {
    case P_CHAR: return (double)*(const signed char *)p;
    case P_UCHAR: return (double)*p;
    case P_SHORT: { short v; memcpy(&v, p, 2); return v; }
    case P_USHORT: { unsigned short v; memcpy(&v, p, 2); return v; }
    case P_INT: { int v; memcpy(&v, p, 4); return v; }
    case P_UINT: { unsigned v; memcpy(&v, p, 4); return v; }
    case P_FLOAT: { float v; memcpy(&v, p, 4); return v; }
    case P_DOUBLE: { double v; memcpy(&v, p, 8); return v; }
    }
    return 0;
}

int gs_load_point_ply(const char *path, float **pts_out, float **cols_out, int *count_out,
                      int *has_colors_out)
{
    size_t len;
    char *data = gs_read_file(path, &len);
    char *hdr_end, *line;
    int binary = 0, nvert = -1, in_vertex = 0, nprops = 0;
    int idx[6] = {-1, -1, -1, -1, -1, -1}; /* x y z r g b */
    ply_type types[64];
    int offsets[64];
    int stride = 0, i;
    float *pts, *cols;
    int color_is_float = 0;
    if (!data)
        return -1;
    hdr_end = strstr(data, "end_header");
    if (strncmp(data, "ply", 3) != 0 || !hdr_end) {
        free(data);
        return -1;
    }
    *hdr_end = 0;
    for (line = strtok(data, "\r\n"); line; line = strtok(NULL, "\r\n")) {
        char a[64], b[64], c[64];
        if (sscanf(line, "format %63s", a) == 1) {
            if (strcmp(a, "binary_little_endian") == 0)
                binary = 1;
            else if (strcmp(a, "ascii") != 0) {
                free(data);
                return -1; /* big endian not supported */
            }
        } else if (sscanf(line, "element %63s %63s", a, b) == 2) {
            in_vertex = strcmp(a, "vertex") == 0;
            if (in_vertex)
                nvert = atoi(b);
            else if (nvert >= 0)
                break; /* only the leading vertex element is used */
        } else if (in_vertex && sscanf(line, "property %63s %63s", a, b) == 2) {
            ply_type t;
            int sz;
            if (strcmp(a, "list") == 0 || nprops >= 64 || ply_type_from_name(a, &t, &sz) != 0) {
                free(data);
                return -1;
            }
            (void)c;
            types[nprops] = t;
            offsets[nprops] = stride;
            stride += sz;
            if (!strcmp(b, "x")) idx[0] = nprops;
            if (!strcmp(b, "y")) idx[1] = nprops;
            if (!strcmp(b, "z")) idx[2] = nprops;
            if (!strcmp(b, "red") || !strcmp(b, "r")) idx[3] = nprops;
            if (!strcmp(b, "green") || !strcmp(b, "g")) idx[4] = nprops;
            if (!strcmp(b, "blue") || !strcmp(b, "b")) idx[5] = nprops;
            nprops++;
        }
    }
    if (nvert < 0 || idx[0] < 0 || idx[1] < 0 || idx[2] < 0) {
        free(data);
        return -1;
    }
    if (idx[3] >= 0)
        color_is_float = types[idx[3]] == P_FLOAT || types[idx[3]] == P_DOUBLE;
    pts = (float *)malloc(sizeof(float) * 3 * (size_t)(nvert ? nvert : 1));
    cols = (float *)malloc(sizeof(float) * 3 * (size_t)(nvert ? nvert : 1));
    {
        char *body = hdr_end + strlen("end_header");
        if (*body == '\r')
            body++;
        if (*body == '\n')
            body++;
        if (binary) {
            size_t avail = len - (size_t)(body - data);
            if ((size_t)nvert * stride > avail)
                nvert = (int)(avail / (size_t)stride);
            for (i = 0; i < nvert; i++) {
                const unsigned char *rec = (const unsigned char *)body + (size_t)i * stride;
                int k;
                for (k = 0; k < 3; k++)
                    pts[i * 3 + k] = (float)ply_read_bin(rec + offsets[idx[k]], types[idx[k]]);
                for (k = 0; k < 3; k++) {
                    double v = idx[3 + k] >= 0 ? ply_read_bin(rec + offsets[idx[3 + k]], types[idx[3 + k]]) : 128;
                    cols[i * 3 + k] = (float)(color_is_float ? v : v / 255.0);
                }
            }
        } else {
            char *p = body;
            double vals[64];
            int n = 0;
            for (i = 0; i < nvert; i++) {
                int k;
                for (k = 0; k < nprops; k++) {
                    char *end;
                    vals[k] = strtod(p, &end);
                    if (end == p)
                        break;
                    p = end;
                }
                if (k < nprops)
                    break;
                for (k = 0; k < 3; k++)
                    pts[n * 3 + k] = (float)vals[idx[k]];
                for (k = 0; k < 3; k++) {
                    double v = idx[3 + k] >= 0 ? vals[idx[3 + k]] : 128;
                    cols[n * 3 + k] = (float)(color_is_float ? v : v / 255.0);
                }
                n++;
            }
            nvert = n;
        }
    }
    free(data);
    *pts_out = pts;
    *cols_out = cols;
    *count_out = nvert;
    *has_colors_out = idx[3] >= 0 && idx[4] >= 0 && idx[5] >= 0;
    return 0;
}

/* ---------------------------------------------------------------------- */

static void set_err(char *err, size_t err_len, const char *msg, const char *detail)
{
    if (err && err_len)
        snprintf(err, err_len, "%s%s%s", msg, detail ? ": " : "", detail ? detail : "");
}

static int read_matrix(const json_value *m, float out[16])
{
    int r, c;
    if (!m || json_len(m) != 4)
        return -1;
    for (r = 0; r < 4; r++) {
        const json_value *row = json_at(m, (size_t)r);
        if (json_len(row) != 4)
            return -1;
        for (c = 0; c < 4; c++)
            out[r * 4 + c] = (float)json_num(json_at(row, (size_t)c), 0);
    }
    return 0;
}

int gs_dataset_load(gs_dataset *ds, const char *dir, int max_dim, char *err, size_t err_len)
{
    char path[1100];
    char jerr[256];
    char *text;
    json_value *root;
    const json_value *frames;
    size_t i, nframes;
    double g_fx, g_fy, g_cx, g_cy, g_w, g_h;
    int ncam = 0;

    memset(ds, 0, sizeof(*ds));
    snprintf(path, sizeof(path), "%s/transforms.json", dir);
    text = gs_read_file(path, NULL);
    if (!text) {
        set_err(err, err_len, "cannot read", path);
        return -1;
    }
    root = json_parse(text, jerr, sizeof(jerr));
    free(text);
    if (!root) {
        set_err(err, err_len, "transforms.json", jerr);
        return -1;
    }
    frames = json_get(root, "frames");
    nframes = json_len(frames);
    if (nframes == 0) {
        json_free(root);
        set_err(err, err_len, "transforms.json has no frames", NULL);
        return -1;
    }
    g_fx = json_num(json_get(root, "fl_x"), 0);
    g_fy = json_num(json_get(root, "fl_y"), g_fx);
    g_cx = json_num(json_get(root, "cx"), -1);
    g_cy = json_num(json_get(root, "cy"), -1);
    g_w = json_num(json_get(root, "w"), 0);
    g_h = json_num(json_get(root, "h"), 0);

    ds->cams = (gs_camera *)calloc(nframes, sizeof(gs_camera));
    for (i = 0; i < nframes; i++) {
        const json_value *f = json_at(frames, i);
        const char *fp = json_str(json_get(f, "file_path"), NULL);
        gs_camera *cam = &ds->cams[ncam];
        float m[16];
        int w, h, comp, k;
        double fx, fy, cx, cy, dw, dh, sx, sy;
        unsigned char *img;
        if (!fp || read_matrix(json_get(f, "transform_matrix"), m) != 0) {
            fprintf(stderr, "[dataset] skipping frame %u: missing file_path/transform_matrix\n", (unsigned)i);
            continue;
        }
        snprintf(path, sizeof(path), "%s/%s", dir, fp);
        img = stbi_load(path, &w, &h, &comp, 4);
        if (!img) {
            fprintf(stderr, "[dataset] skipping %s: %s\n", path, stbi_failure_reason());
            continue;
        }
        fx = json_num(json_get(f, "fl_x"), g_fx);
        fy = json_num(json_get(f, "fl_y"), json_num(json_get(f, "fl_x"), g_fy));
        cx = json_num(json_get(f, "cx"), g_cx);
        cy = json_num(json_get(f, "cy"), g_cy);
        dw = json_num(json_get(f, "w"), g_w);
        dh = json_num(json_get(f, "h"), g_h);
        if (dw <= 0) dw = w;
        if (dh <= 0) dh = h;
        if (cx < 0) cx = dw * 0.5;
        if (cy < 0) cy = dh * 0.5;
        if (fx <= 0) {
            /* fall back to camera_angle_x (instant-ngp style) */
            double ax = json_num(json_get(root, "camera_angle_x"), 1.0);
            fx = fy = 0.5 * dw / tan(0.5 * ax);
        }
        /* intrinsics refer to (dw, dh); rescale to the decoded image and target size */
        {
            int tw = w, th = h;
            if (max_dim > 0 && (w > max_dim || h > max_dim)) {
                double s = (double)max_dim / (w > h ? w : h);
                tw = (int)(w * s + 0.5);
                th = (int)(h * s + 0.5);
            }
            sx = tw / dw;
            sy = th / dh;
            if (tw != w || th != h) {
                unsigned char *r = resize_rgba(img, w, h, tw, th);
                stbi_image_free(img);
                img = r;
                w = tw;
                h = th;
            } else {
                unsigned char *copy = (unsigned char *)malloc((size_t)w * h * 4);
                memcpy(copy, img, (size_t)w * h * 4);
                stbi_image_free(img);
                img = copy;
            }
        }
        cam->width = w;
        cam->height = h;
        cam->fx = (float)(fx * sx);
        cam->fy = (float)(fy * sy);
        cam->cx = (float)(cx * sx);
        cam->cy = (float)(cy * sy);
        cam->rgba = img;
        snprintf(cam->name, sizeof(cam->name), "%s", fp);
        /* c2w (OpenGL) -> c2w (OpenCV): flip camera Y and Z axes, then invert */
        {
            float Rc[9];
            int r;
            for (r = 0; r < 3; r++) {
                Rc[r * 3 + 0] = m[r * 4 + 0];
                Rc[r * 3 + 1] = -m[r * 4 + 1];
                Rc[r * 3 + 2] = -m[r * 4 + 2];
                cam->campos[r] = m[r * 4 + 3];
            }
            for (r = 0; r < 3; r++)
                for (k = 0; k < 3; k++)
                    cam->R[r * 3 + k] = Rc[k * 3 + r];
            for (r = 0; r < 3; r++)
                cam->t[r] = -(cam->R[r * 3 + 0] * cam->campos[0] + cam->R[r * 3 + 1] * cam->campos[1] +
                              cam->R[r * 3 + 2] * cam->campos[2]);
        }
        ncam++;
    }
    ds->num_cams = ncam;
    if (ncam == 0) {
        json_free(root);
        set_err(err, err_len, "no usable frames/images found", dir);
        return -1;
    }

    /* scene normalisation (camera rig centre and radius) */
    {
        double c[3] = {0, 0, 0}, maxd = 0;
        int k, j;
        for (j = 0; j < ncam; j++)
            for (k = 0; k < 3; k++)
                c[k] += ds->cams[j].campos[k] / ncam;
        for (j = 0; j < ncam; j++) {
            double d = 0;
            for (k = 0; k < 3; k++)
                d += (ds->cams[j].campos[k] - c[k]) * (ds->cams[j].campos[k] - c[k]);
            if (d > maxd)
                maxd = d;
        }
        for (k = 0; k < 3; k++)
            ds->center[k] = (float)c[k];
        ds->extent = (float)(sqrt(maxd) * 1.1);
        if (ds->extent < 1e-3f)
            ds->extent = 1.0f;
    }

    /* initial point cloud */
    {
        const char *ply = json_str(json_get(root, "ply_file_path"), "points3d.ply");
        snprintf(path, sizeof(path), "%s/%s", dir, ply);
        if (gs_file_exists(path)) {
            if (gs_load_point_ply(path, &ds->points, &ds->colors, &ds->num_points, &ds->has_colors) != 0)
                fprintf(stderr, "[dataset] could not parse %s\n", path);
        }
    }
    json_free(root);
    return 0;
}

void gs_dataset_free(gs_dataset *ds)
{
    int i;
    for (i = 0; i < ds->num_cams; i++)
        free(ds->cams[i].rgba);
    free(ds->cams);
    free(ds->points);
    free(ds->colors);
    memset(ds, 0, sizeof(*ds));
}
