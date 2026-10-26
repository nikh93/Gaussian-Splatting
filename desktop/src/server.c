/*
 * server.c - job management and REST API used by the Android app.
 *
 *   GET    /api/info                         server + GPU info
 *   GET    /api/jobs                         list jobs
 *   POST   /api/jobs                         create job  {"name", "iterations", "max_dim", "sh_degree"}
 *   GET    /api/jobs/{id}                    job status
 *   PUT    /api/jobs/{id}/files/{path}       upload a capture file (transforms.json, images/..., points3d.ply)
 *   POST   /api/jobs/{id}/start              queue the job for training
 *   POST   /api/jobs/{id}/cancel             stop training
 *   DELETE /api/jobs/{id}                    delete job and its files
 *   GET    /api/jobs/{id}/model.ply          trained model (standard 3DGS PLY)
 *   GET    /api/jobs/{id}/model.splat        trained model (compact .splat)
 */
#include "server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dataset.h"
#include "http.h"
#include "json.h"
#include "platform.h"

#define SERVER_VERSION "1.0.0"
#define MAX_UPLOAD (512LL << 20)

typedef struct {
    char id[48];
    char name[128];
    char state[16]; /* uploading, queued, training, done, failed, cancelled */
    long long created, updated;
    int iteration, total;
    float loss, psnr;
    int gaussians;
    int model_iteration;
    char error[256];
    int iterations, max_dim, sh_degree;
    int files;
    long long bytes;
    volatile int cancel;
    int deleted;
    double last_save;
} job;

static job **g_jobs;
static int g_njobs, g_cap;
static gs_mutex g_lock;
static const gs_server_config *g_cfg;
static unsigned g_counter;
static gs_train_control *g_running_ctl; /* control block of the job being trained */

/* ---------------------------------------------------------------------- */

static void job_dir(const job *j, char *out, size_t len, const char *sub)
{
    if (sub)
        snprintf(out, len, "%s/jobs/%s/%s", g_cfg->data_dir, j->id, sub);
    else
        snprintf(out, len, "%s/jobs/%s", g_cfg->data_dir, j->id);
}

static void job_json(sbuf *b, const job *j)
{
    char p[1100];
    int has_model;
    job_dir(j, p, sizeof(p), "output/model.ply");
    has_model = gs_file_exists(p);
    sb_append(b, "{\"id\":");
    sb_json_string(b, j->id);
    sb_append(b, ",\"name\":");
    sb_json_string(b, j->name);
    sb_append(b, ",\"state\":");
    sb_json_string(b, j->state);
    sb_printf(b,
              ",\"created\":%lld,\"updated\":%lld,\"iteration\":%d,\"total_iterations\":%d,"
              "\"progress\":%.4f,\"loss\":%.5f,\"psnr\":%.3f,\"gaussians\":%d,\"model_available\":%s,"
              "\"model_iteration\":%d,\"files\":%d,\"bytes\":%lld,\"iterations\":%d,\"max_dim\":%d,"
              "\"sh_degree\":%d,\"error\":",
              j->created, j->updated, j->iteration, j->total, j->total > 0 ? (double)j->iteration / j->total : 0.0,
              j->loss, j->psnr, j->gaussians, has_model ? "true" : "false", j->model_iteration, j->files, j->bytes,
              j->iterations, j->max_dim, j->sh_degree);
    sb_json_string(b, j->error);
    sb_append(b, "}");
}

/* must hold g_lock */
static void job_save(job *j)
{
    char p[1100];
    sbuf b;
    if (j->deleted)
        return;
    sb_init(&b);
    job_json(&b, j);
    job_dir(j, p, sizeof(p), "job.json");
    gs_write_file(p, b.data, b.len);
    sb_free(&b);
    j->last_save = gs_now_seconds();
}

static void set_state(job *j, const char *state)
{
    snprintf(j->state, sizeof(j->state), "%s", state);
    j->updated = gs_unix_time();
    job_save(j);
}

static job *job_find(const char *id)
{
    int i;
    for (i = 0; i < g_njobs; i++)
        if (!g_jobs[i]->deleted && strcmp(g_jobs[i]->id, id) == 0)
            return g_jobs[i];
    return NULL;
}

static void job_add(job *j)
{
    if (g_njobs == g_cap) {
        g_cap = g_cap ? g_cap * 2 : 16;
        g_jobs = (job **)realloc(g_jobs, sizeof(job *) * (size_t)g_cap);
    }
    g_jobs[g_njobs++] = j;
}

static int valid_id(const char *s)
{
    size_t n = strlen(s), i;
    if (n == 0 || n >= 48)
        return 0;
    for (i = 0; i < n; i++) {
        char c = s[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_'))
            return 0;
    }
    return 1;
}

/* relative upload path: [A-Za-z0-9._-] segments separated by '/', no "..", no empty segment */
static int valid_rel_path(const char *s)
{
    size_t n = strlen(s), i;
    if (n == 0 || n > 200 || s[0] == '/' || s[0] == '.' || strstr(s, "..") || strstr(s, "//"))
        return 0;
    for (i = 0; i < n; i++) {
        char c = s[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' ||
              c == '.' || c == '/'))
            return 0;
    }
    return s[n - 1] != '/';
}

static int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ---------------------------------------------------------------------- */
/* persistence                                                             */
/* ---------------------------------------------------------------------- */

static void load_job_cb(const char *name, void *user)
{
    char p[1100];
    char *text;
    json_value *v;
    job *j;
    (void)user;
    if (!valid_id(name))
        return;
    snprintf(p, sizeof(p), "%s/jobs/%s/job.json", g_cfg->data_dir, name);
    text = gs_read_file(p, NULL);
    if (!text)
        return;
    v = json_parse(text, NULL, 0);
    free(text);
    if (!v)
        return;
    j = (job *)calloc(1, sizeof(job));
    snprintf(j->id, sizeof(j->id), "%s", name);
    snprintf(j->name, sizeof(j->name), "%s", json_str(json_get(v, "name"), name));
    snprintf(j->state, sizeof(j->state), "%s", json_str(json_get(v, "state"), "uploading"));
    snprintf(j->error, sizeof(j->error), "%s", json_str(json_get(v, "error"), ""));
    j->created = (long long)json_num(json_get(v, "created"), 0);
    j->updated = (long long)json_num(json_get(v, "updated"), 0);
    j->iteration = (int)json_num(json_get(v, "iteration"), 0);
    j->total = (int)json_num(json_get(v, "total_iterations"), 0);
    j->loss = (float)json_num(json_get(v, "loss"), 0);
    j->psnr = (float)json_num(json_get(v, "psnr"), 0);
    j->gaussians = (int)json_num(json_get(v, "gaussians"), 0);
    j->model_iteration = (int)json_num(json_get(v, "model_iteration"), 0);
    j->files = (int)json_num(json_get(v, "files"), 0);
    j->bytes = (long long)json_num(json_get(v, "bytes"), 0);
    j->iterations = (int)json_num(json_get(v, "iterations"), g_cfg->defaults.iterations);
    j->max_dim = (int)json_num(json_get(v, "max_dim"), g_cfg->defaults.max_dim);
    j->sh_degree = (int)json_num(json_get(v, "sh_degree"), g_cfg->defaults.sh_degree);
    json_free(v);
    if (strcmp(j->state, "training") == 0) {
        /* server was stopped during training: train again */
        snprintf(j->state, sizeof(j->state), "queued");
    }
    job_add(j);
}

/* ---------------------------------------------------------------------- */
/* training worker                                                         */
/* ---------------------------------------------------------------------- */

static void on_progress(void *user, int iter, int total, float loss, float psnr, int n)
{
    job *j = (job *)user;
    gs_mutex_lock(&g_lock);
    j->iteration = iter;
    j->total = total;
    j->loss = loss;
    j->psnr = psnr;
    j->gaussians = n;
    j->updated = gs_unix_time();
    if (gs_now_seconds() - j->last_save > 5.0)
        job_save(j);
    gs_mutex_unlock(&g_lock);
}

static void on_snapshot(void *user, int iter)
{
    job *j = (job *)user;
    gs_mutex_lock(&g_lock);
    j->model_iteration = iter;
    job_save(j);
    gs_mutex_unlock(&g_lock);
}

static void worker(void *arg)
{
    (void)arg;
    for (;;) {
        job *j = NULL;
        int i;
        gs_mutex_lock(&g_lock);
        for (i = 0; i < g_njobs; i++) {
            job *c = g_jobs[i];
            if (!c->deleted && strcmp(c->state, "queued") == 0 && (!j || c->created < j->created))
                j = c;
        }
        if (j) {
            j->cancel = 0;
            j->iteration = 0;
            j->model_iteration = 0;
            j->error[0] = 0;
            set_state(j, "training");
        }
        gs_mutex_unlock(&g_lock);
        if (!j) {
            gs_sleep_ms(500);
            continue;
        }

        {
            char cap_dir[1100], out_dir[1100], err[512];
            gs_dataset ds;
            gs_train_params p = g_cfg->defaults;
            gs_train_control ctl;
            int r;
            job_dir(j, cap_dir, sizeof(cap_dir), "capture");
            job_dir(j, out_dir, sizeof(out_dir), "output");
            gs_mkdirs(out_dir);
            p.iterations = j->iterations;
            p.max_dim = j->max_dim;
            p.sh_degree = j->sh_degree;
            if (p.iterations < 2 * p.densify_until)
                p.densify_until = p.iterations / 2;
            memset(&ctl, 0, sizeof(ctl));
            ctl.progress = on_progress;
            ctl.snapshot = on_snapshot;
            ctl.user = j;
            err[0] = 0;
            printf("[job %s] loading capture\n", j->id);
            if (gs_dataset_load(&ds, cap_dir, p.max_dim, err, sizeof(err)) != 0) {
                r = -1;
            } else {
                /* cancel/delete requests set ctl.cancel through g_running_ctl */
                gs_mutex_lock(&g_lock);
                j->total = p.iterations;
                g_running_ctl = &ctl;
                ctl.cancel = j->cancel;
                gs_mutex_unlock(&g_lock);
                r = gs_train(g_cfg->cl, &ds, &p, out_dir, &ctl, err, sizeof(err));
                gs_mutex_lock(&g_lock);
                g_running_ctl = NULL;
                gs_mutex_unlock(&g_lock);
                gs_dataset_free(&ds);
            }
            gs_mutex_lock(&g_lock);
            if (j->deleted) {
                char d[1100];
                job_dir(j, d, sizeof(d), NULL);
                gs_remove_tree(d);
            } else if (r == 0) {
                j->model_iteration = j->total;
                j->iteration = j->total;
                set_state(j, "done");
            } else if (r == 1) {
                set_state(j, "cancelled");
            } else {
                snprintf(j->error, sizeof(j->error), "%s", err[0] ? err : "training failed");
                set_state(j, "failed");
            }
            gs_mutex_unlock(&g_lock);
            printf("[job %s] finished: %s %s\n", j->id, j->state, j->error);
            fflush(stdout);
        }
    }
}

/* ---------------------------------------------------------------------- */
/* HTTP handlers                                                           */
/* ---------------------------------------------------------------------- */

static void respond_job(http_request *req, int status, job *j)
{
    sbuf b;
    sb_init(&b);
    job_json(&b, j);
    http_respond_json(req, status, b.data);
    sb_free(&b);
}

static void handle_info(http_request *req)
{
    sbuf b;
    sb_init(&b);
    sb_append(&b, "{\"name\":\"gsplat-server\",\"version\":\"" SERVER_VERSION "\",\"device\":");
    sb_json_string(&b, g_cfg->cl->device_name);
    sb_append(&b, ",\"platform\":");
    sb_json_string(&b, g_cfg->cl->platform_name);
    sb_printf(&b, ",\"gpu_memory_mb\":%llu,\"default_iterations\":%d,\"default_max_dim\":%d,\"auth\":%s}",
              (unsigned long long)(g_cfg->cl->global_mem >> 20), g_cfg->defaults.iterations, g_cfg->defaults.max_dim,
              g_cfg->token ? "true" : "false");
    http_respond_json(req, 200, b.data);
    sb_free(&b);
}

static void handle_list(http_request *req)
{
    sbuf b;
    int i, first = 1;
    sb_init(&b);
    sb_append(&b, "{\"jobs\":[");
    gs_mutex_lock(&g_lock);
    for (i = g_njobs - 1; i >= 0; i--) {
        if (g_jobs[i]->deleted)
            continue;
        if (!first)
            sb_append(&b, ",");
        first = 0;
        job_json(&b, g_jobs[i]);
    }
    gs_mutex_unlock(&g_lock);
    sb_append(&b, "]}");
    http_respond_json(req, 200, b.data);
    sb_free(&b);
}

static void handle_create(http_request *req)
{
    size_t len = 0;
    char *body = http_read_body(req, 1 << 16, &len);
    json_value *v = NULL;
    job *j;
    char d[1100];
    if (!body) {
        http_respond_error(req, 413, "request too large");
        return;
    }
    if (len)
        v = json_parse(body, NULL, 0);
    free(body);
    j = (job *)calloc(1, sizeof(job));
    gs_mutex_lock(&g_lock);
    {
        time_t now = time(NULL);
        struct tm *tm = gmtime(&now);
        char stamp[32];
        strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", tm);
        g_counter++;
        snprintf(j->id, sizeof(j->id), "%s-%04x", stamp, (unsigned)((g_counter * 2654435761u) >> 16) & 0xffff);
    }
    snprintf(j->name, sizeof(j->name), "%s", json_str(json_get(v, "name"), j->id));
    j->iterations = clampi((int)json_num(json_get(v, "iterations"), g_cfg->defaults.iterations), 100, 200000);
    j->max_dim = clampi((int)json_num(json_get(v, "max_dim"), g_cfg->defaults.max_dim), 128, 4096);
    j->sh_degree = clampi((int)json_num(json_get(v, "sh_degree"), g_cfg->defaults.sh_degree), 0, 3);
    j->created = j->updated = gs_unix_time();
    snprintf(j->state, sizeof(j->state), "uploading");
    job_dir(j, d, sizeof(d), "capture/images");
    gs_mkdirs(d);
    job_add(j);
    job_save(j);
    gs_mutex_unlock(&g_lock);
    json_free(v);
    printf("[job %s] created (%s) by %s\n", j->id, j->name, req->peer);
    gs_mutex_lock(&g_lock);
    respond_job(req, 201, j);
    gs_mutex_unlock(&g_lock);
}

static void handle_upload(http_request *req, job *j, const char *rel)
{
    char path[1400], dir[1400];
    char *slash;
    long long n;
    gs_mutex_lock(&g_lock);
    if (strcmp(j->state, "training") == 0 || strcmp(j->state, "queued") == 0) {
        gs_mutex_unlock(&g_lock);
        http_respond_error(req, 409, "job is queued or training");
        return;
    }
    gs_mutex_unlock(&g_lock);
    if (!valid_rel_path(rel)) {
        http_respond_error(req, 400, "invalid file path");
        return;
    }
    snprintf(path, sizeof(path), "%s/jobs/%s/capture/%s", g_cfg->data_dir, j->id, rel);
    snprintf(dir, sizeof(dir), "%s", path);
    slash = strrchr(dir, '/');
    if (slash) {
        *slash = 0;
        gs_mkdirs(dir);
    }
    n = http_body_to_file(req, path, MAX_UPLOAD);
    if (n < 0) {
        http_respond_error(req, 400, "upload failed or Content-Length missing / too large");
        return;
    }
    gs_mutex_lock(&g_lock);
    j->files++;
    j->bytes += n;
    j->updated = gs_unix_time();
    if (gs_now_seconds() - j->last_save > 2.0)
        job_save(j);
    gs_mutex_unlock(&g_lock);
    {
        char msg[256];
        snprintf(msg, sizeof(msg), "{\"ok\":true,\"bytes\":%lld}", n);
        http_respond_json(req, 200, msg);
    }
}

static void handle_start(http_request *req, job *j)
{
    char p[1100];
    size_t len = 0;
    char *body = http_read_body(req, 1 << 16, &len);
    json_value *v = body && len ? json_parse(body, NULL, 0) : NULL;
    free(body);
    job_dir(j, p, sizeof(p), "capture/transforms.json");
    if (!gs_file_exists(p)) {
        json_free(v);
        http_respond_error(req, 400, "transforms.json has not been uploaded");
        return;
    }
    gs_mutex_lock(&g_lock);
    if (strcmp(j->state, "training") == 0 || strcmp(j->state, "queued") == 0) {
        gs_mutex_unlock(&g_lock);
        json_free(v);
        http_respond_error(req, 409, "job is already queued or training");
        return;
    }
    if (v) {
        j->iterations = clampi((int)json_num(json_get(v, "iterations"), j->iterations), 100, 200000);
        j->max_dim = clampi((int)json_num(json_get(v, "max_dim"), j->max_dim), 128, 4096);
        j->sh_degree = clampi((int)json_num(json_get(v, "sh_degree"), j->sh_degree), 0, 3);
    }
    j->total = j->iterations;
    j->iteration = 0;
    j->error[0] = 0;
    set_state(j, "queued");
    respond_job(req, 200, j);
    gs_mutex_unlock(&g_lock);
    json_free(v);
}

static void handle_cancel(http_request *req, job *j)
{
    gs_mutex_lock(&g_lock);
    if (strcmp(j->state, "training") == 0) {
        j->cancel = 1;
        if (g_running_ctl)
            g_running_ctl->cancel = 1;
    } else if (strcmp(j->state, "queued") == 0) {
        set_state(j, "cancelled");
    }
    respond_job(req, 200, j);
    gs_mutex_unlock(&g_lock);
}

static void handle_delete(http_request *req, job *j)
{
    char d[1100];
    gs_mutex_lock(&g_lock);
    j->deleted = 1;
    if (strcmp(j->state, "training") == 0) {
        j->cancel = 1;
        if (g_running_ctl)
            g_running_ctl->cancel = 1; /* worker removes the files afterwards */
    } else {
        job_dir(j, d, sizeof(d), NULL);
        gs_remove_tree(d);
    }
    gs_mutex_unlock(&g_lock);
    http_respond_json(req, 200, "{\"ok\":true}");
}

static void handle_index(http_request *req)
{
    sbuf b;
    int i;
    sb_init(&b);
    sb_append(&b, "<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width'>"
                  "<meta http-equiv=refresh content=5><title>gsplat-server</title>"
                  "<style>body{font-family:sans-serif;margin:2em;background:#111;color:#eee}"
                  "td,th{padding:4px 10px;text-align:left}a{color:#8cf}</style></head><body>"
                  "<h2>Gaussian Splatting server</h2><p>Device: ");
    sb_append(&b, g_cfg->cl->device_name);
    sb_append(&b, "</p><table><tr><th>Job</th><th>Name</th><th>State</th><th>Progress</th><th>PSNR</th>"
                  "<th>Gaussians</th><th>Model</th></tr>");
    gs_mutex_lock(&g_lock);
    for (i = g_njobs - 1; i >= 0; i--) {
        job *j = g_jobs[i];
        if (j->deleted)
            continue;
        sb_printf(&b, "<tr><td>%s</td><td>", j->id);
        {
            /* minimal HTML escaping of the user supplied name */
            const char *c;
            for (c = j->name; *c; c++) {
                if (*c == '<') sb_append(&b, "&lt;");
                else if (*c == '>') sb_append(&b, "&gt;");
                else if (*c == '&') sb_append(&b, "&amp;");
                else sb_appendn(&b, c, 1);
            }
        }
        sb_printf(&b, "</td><td>%s</td><td>%d / %d</td><td>%.2f</td><td>%d</td><td>", j->state, j->iteration,
                  j->total, j->psnr, j->gaussians);
        sb_printf(&b, "<a href='/api/jobs/%s/model.ply'>ply</a> <a href='/api/jobs/%s/model.splat'>splat</a></td></tr>",
                  j->id, j->id);
    }
    gs_mutex_unlock(&g_lock);
    sb_append(&b, "</table></body></html>");
    http_respond(req, 200, "text/html; charset=utf-8", b.data, b.len);
    sb_free(&b);
}

static void handler(http_request *req, void *user)
{
    const char *path = req->path;
    const char *m = req->method;
    (void)user;

    if (strcmp(path, "/") == 0 && strcmp(m, "GET") == 0) {
        handle_index(req);
        return;
    }
    if (strncmp(path, "/api/", 5) != 0) {
        http_respond_error(req, 404, "not found");
        return;
    }
    if (g_cfg->token && strcmp(req->auth, g_cfg->token) != 0) {
        char tok[256];
        if (!http_query_param(req, "token", tok, sizeof(tok)) || strcmp(tok, g_cfg->token) != 0) {
            http_respond_error(req, 401, "invalid or missing token");
            return;
        }
    }
    path += 5;
    if (strcmp(path, "info") == 0) {
        handle_info(req);
        return;
    }
    if (strcmp(path, "jobs") == 0 || strcmp(path, "jobs/") == 0) {
        if (strcmp(m, "GET") == 0)
            handle_list(req);
        else if (strcmp(m, "POST") == 0)
            handle_create(req);
        else
            http_respond_error(req, 405, "method not allowed");
        return;
    }
    if (strncmp(path, "jobs/", 5) == 0) {
        char id[64];
        const char *rest = path + 5;
        const char *slash = strchr(rest, '/');
        size_t idlen = slash ? (size_t)(slash - rest) : strlen(rest);
        job *j;
        if (idlen >= sizeof(id)) {
            http_respond_error(req, 404, "unknown job");
            return;
        }
        memcpy(id, rest, idlen);
        id[idlen] = 0;
        rest = slash ? slash + 1 : "";
        gs_mutex_lock(&g_lock);
        j = valid_id(id) ? job_find(id) : NULL;
        gs_mutex_unlock(&g_lock);
        if (!j) {
            http_respond_error(req, 404, "unknown job");
            return;
        }
        if (*rest == 0) {
            if (strcmp(m, "GET") == 0) {
                gs_mutex_lock(&g_lock);
                respond_job(req, 200, j);
                gs_mutex_unlock(&g_lock);
            } else if (strcmp(m, "DELETE") == 0) {
                handle_delete(req, j);
            } else {
                http_respond_error(req, 405, "method not allowed");
            }
            return;
        }
        if (strncmp(rest, "files/", 6) == 0 && (strcmp(m, "PUT") == 0 || strcmp(m, "POST") == 0)) {
            handle_upload(req, j, rest + 6);
            return;
        }
        if (strcmp(rest, "start") == 0 && strcmp(m, "POST") == 0) {
            handle_start(req, j);
            return;
        }
        if (strcmp(rest, "cancel") == 0 && strcmp(m, "POST") == 0) {
            handle_cancel(req, j);
            return;
        }
        if ((strcmp(rest, "model.ply") == 0 || strcmp(rest, "model.splat") == 0) && strcmp(m, "GET") == 0) {
            char p[1100], fname[128];
            snprintf(p, sizeof(p), "%s/jobs/%s/output/%s", g_cfg->data_dir, j->id, rest);
            snprintf(fname, sizeof(fname), "%s-%s", j->id, rest);
            http_respond_file(req, p, "application/octet-stream", fname);
            return;
        }
    }
    http_respond_error(req, 404, "not found");
}

int gs_server_run(const gs_server_config *cfg)
{
    char d[1100];
    g_cfg = cfg;
    gs_mutex_init(&g_lock);
    snprintf(d, sizeof(d), "%s/jobs", cfg->data_dir);
    if (gs_mkdirs(d) != 0) {
        fprintf(stderr, "cannot create data directory %s\n", d);
        return -1;
    }
    gs_list_dir(d, load_job_cb, NULL);
    printf("[server] data directory %s (%d existing jobs)\n", cfg->data_dir, g_njobs);
    if (gs_thread_start(NULL, worker, NULL, 1) != 0) {
        fprintf(stderr, "cannot start worker thread\n");
        return -1;
    }
    return http_serve(cfg->bind_addr, cfg->port, handler, NULL);
}
