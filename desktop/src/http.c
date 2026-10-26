/*
 * http.c - tiny blocking HTTP/1.1 server.
 *
 * Every connection is served by its own detached thread and closed after one
 * request ("Connection: close"); that is plenty for a handful of phones on a
 * tailnet and keeps the implementation small.
 */
#include "http.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_HEADER 16384

typedef struct {
    gs_socket sock;
    char peer[64];
    http_handler handler;
    void *user;
} conn_info;

static void url_decode(const char *in, size_t n, char *out, size_t out_len)
{
    size_t i, o = 0;
    for (i = 0; i < n && o + 1 < out_len; i++) {
        if (in[i] == '%' && i + 2 < n && isxdigit((unsigned char)in[i + 1]) && isxdigit((unsigned char)in[i + 2])) {
            char hex[3];
            hex[0] = in[i + 1];
            hex[1] = in[i + 2];
            hex[2] = 0;
            out[o++] = (char)strtol(hex, NULL, 16);
            i += 2;
        } else if (in[i] == '+') {
            out[o++] = ' ';
        } else {
            out[o++] = in[i];
        }
    }
    out[o] = 0;
}

static int header_is(const char *line, const char *name)
{
    size_t n = strlen(name);
    size_t i;
    for (i = 0; i < n; i++)
        if (tolower((unsigned char)line[i]) != tolower((unsigned char)name[i]))
            return 0;
    return line[n] == ':';
}

static const char *header_value(const char *line)
{
    const char *v = strchr(line, ':');
    if (!v)
        return "";
    v++;
    while (*v == ' ' || *v == '\t')
        v++;
    return v;
}

static const char *status_text(int s)
{
    switch (s) {
    case 200: return "OK";
    case 201: return "Created";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Status";
    }
}

static void conn_thread(void *arg)
{
    conn_info ci = *(conn_info *)arg;
    char *hdr = (char *)malloc(MAX_HEADER + 1);
    size_t got = 0;
    char *end = NULL;
    http_request req;
    free(arg);
    memset(&req, 0, sizeof(req));
    req.sock = ci.sock;
    snprintf(req.peer, sizeof(req.peer), "%s", ci.peer);
    req.content_length = -1;
    gs_set_socket_timeout(ci.sock, 120);

    while (got < MAX_HEADER) {
        long r = gs_recv(ci.sock, hdr + got, MAX_HEADER - got);
        if (r <= 0)
            break;
        got += (size_t)r;
        hdr[got] = 0;
        end = strstr(hdr, "\r\n\r\n");
        if (end)
            break;
    }
    if (!end) {
        free(hdr);
        gs_close_socket(ci.sock);
        return;
    }
    *end = 0;
    req.pre = end + 4;
    req.pre_len = got - (size_t)(req.pre - hdr);

    {
        char *line = hdr, *next;
        int first = 1;
        while (line && *line) {
            next = strstr(line, "\r\n");
            if (next) {
                *next = 0;
                next += 2;
            }
            if (first) {
                char target[2048];
                char *q;
                first = 0;
                if (sscanf(line, "%15s %2047s", req.method, target) != 2) {
                    req.method[0] = 0;
                    break;
                }
                q = strchr(target, '?');
                if (q) {
                    snprintf(req.query, sizeof(req.query), "%s", q + 1);
                    *q = 0;
                }
                url_decode(target, strlen(target), req.path, sizeof(req.path));
            } else if (header_is(line, "Content-Length")) {
                req.content_length = atoll(header_value(line));
            } else if (header_is(line, "Content-Type")) {
                snprintf(req.content_type, sizeof(req.content_type), "%s", header_value(line));
            } else if (header_is(line, "Authorization") || header_is(line, "X-Auth-Token")) {
                const char *v = header_value(line);
                if (strncmp(v, "Bearer ", 7) == 0)
                    v += 7;
                snprintf(req.auth, sizeof(req.auth), "%s", v);
            } else if (header_is(line, "Expect")) {
                /* "Expect: 100-continue" - acknowledge so clients start sending */
                if (strstr(header_value(line), "100"))
                    gs_send_all(ci.sock, "HTTP/1.1 100 Continue\r\n\r\n", 25);
            }
            line = next;
        }
    }

    if (req.method[0]) {
        if (strcmp(req.method, "OPTIONS") == 0) {
            http_respond(&req, 204, "text/plain", "", 0);
        } else {
            ci.handler(&req, ci.user);
            if (!req.responded)
                http_respond_error(&req, 500, "no response");
        }
    }
    free(hdr);
    gs_close_socket(ci.sock);
}

int http_serve(const char *bind_addr, int port, http_handler handler, void *user)
{
    gs_socket ls = gs_listen(bind_addr, port);
    if (ls == GS_INVALID_SOCKET) {
        fprintf(stderr, "[http] cannot listen on %s:%d\n", bind_addr ? bind_addr : "0.0.0.0", port);
        return -1;
    }
    printf("[http] listening on %s:%d\n", bind_addr && *bind_addr ? bind_addr : "0.0.0.0", port);
    fflush(stdout);
    for (;;) {
        conn_info *ci = (conn_info *)malloc(sizeof(conn_info));
        ci->sock = gs_accept(ls, ci->peer, sizeof(ci->peer));
        if (ci->sock == GS_INVALID_SOCKET) {
            free(ci);
            gs_sleep_ms(10);
            continue;
        }
        ci->handler = handler;
        ci->user = user;
        if (gs_thread_start(NULL, conn_thread, ci, 1) != 0) {
            gs_close_socket(ci->sock);
            free(ci);
        }
    }
    return 0;
}

/* reads body bytes, first from the header buffer, then from the socket */
static long body_recv(http_request *req, char *dst, size_t len)
{
    long long remaining;
    if (req->content_length < 0)
        return 0;
    remaining = req->content_length - req->body_read;
    if (remaining <= 0)
        return 0;
    if ((long long)len > remaining)
        len = (size_t)remaining;
    if (req->pre_pos < req->pre_len) {
        size_t n = req->pre_len - req->pre_pos;
        if (n > len)
            n = len;
        memcpy(dst, req->pre + req->pre_pos, n);
        req->pre_pos += n;
        req->body_read += (long long)n;
        return (long)n;
    }
    {
        long r = gs_recv(req->sock, dst, len);
        if (r > 0)
            req->body_read += r;
        return r;
    }
}

char *http_read_body(http_request *req, size_t max, size_t *out_len)
{
    size_t cap = req->content_length > 0 ? (size_t)req->content_length : 0;
    size_t len = 0;
    char *buf;
    if (cap > max)
        return NULL;
    buf = (char *)malloc(cap + 1);
    if (!buf)
        return NULL;
    while (len < cap) {
        long r = body_recv(req, buf + len, cap - len);
        if (r <= 0)
            break;
        len += (size_t)r;
    }
    buf[len] = 0;
    if (out_len)
        *out_len = len;
    return buf;
}

long long http_body_to_file(http_request *req, const char *path, long long max)
{
    char tmp[1100];
    char *buf;
    FILE *f;
    long long total = 0;
    if (req->content_length < 0 || req->content_length > max)
        return -1;
    snprintf(tmp, sizeof(tmp), "%s.part", path);
    f = fopen(tmp, "wb");
    if (!f)
        return -1;
    buf = (char *)malloc(1 << 16);
    while (total < req->content_length) {
        long r = body_recv(req, buf, 1 << 16);
        if (r <= 0)
            break;
        if (fwrite(buf, 1, (size_t)r, f) != (size_t)r) {
            total = -1;
            break;
        }
        total += r;
    }
    free(buf);
    fclose(f);
    if (total != req->content_length) {
        remove(tmp);
        return -1;
    }
    if (gs_rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return total;
}

static void send_head(http_request *req, int status, const char *content_type, long long len, const char *extra)
{
    char head[1024];
    int n = snprintf(head, sizeof(head),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: %s\r\n"
                     "Content-Length: %lld\r\n"
                     "Connection: close\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
                     "Access-Control-Allow-Headers: Content-Type, Authorization, X-Auth-Token\r\n"
                     "%s"
                     "\r\n",
                     status, status_text(status), content_type, len, extra ? extra : "");
    gs_send_all(req->sock, head, (size_t)n);
    req->responded = 1;
}

/* Drain an unread request body so the client sees our response instead of a reset. */
static void drain_body(http_request *req)
{
    char buf[8192];
    long long limit = 64LL << 20;
    while (req->content_length > 0 && req->body_read < req->content_length && req->body_read < limit) {
        if (body_recv(req, buf, sizeof(buf)) <= 0)
            break;
    }
}

void http_respond(http_request *req, int status, const char *content_type, const void *body, size_t len)
{
    drain_body(req);
    send_head(req, status, content_type, (long long)len, NULL);
    if (len)
        gs_send_all(req->sock, body, len);
}

void http_respond_json(http_request *req, int status, const char *json)
{
    http_respond(req, status, "application/json; charset=utf-8", json, strlen(json));
}

void http_respond_error(http_request *req, int status, const char *message)
{
    sbuf b;
    sb_init(&b);
    sb_append(&b, "{\"error\":");
    sb_json_string(&b, message);
    sb_append(&b, "}");
    http_respond_json(req, status, b.data);
    sb_free(&b);
}

void http_respond_file(http_request *req, const char *path, const char *content_type, const char *download_name)
{
    FILE *f;
    long long size = gs_file_size(path);
    char extra[512];
    char *buf;
    if (size < 0 || !(f = fopen(path, "rb"))) {
        http_respond_error(req, 404, "file not found");
        return;
    }
    drain_body(req);
    extra[0] = 0;
    if (download_name)
        snprintf(extra, sizeof(extra), "Content-Disposition: attachment; filename=\"%s\"\r\n", download_name);
    send_head(req, 200, content_type, size, extra);
    buf = (char *)malloc(1 << 16);
    for (;;) {
        size_t n = fread(buf, 1, 1 << 16, f);
        if (n == 0)
            break;
        if (gs_send_all(req->sock, buf, n) != 0)
            break;
    }
    free(buf);
    fclose(f);
}

int http_query_param(const http_request *req, const char *key, char *out, size_t out_len)
{
    const char *p = req->query;
    size_t klen = strlen(key);
    while (p && *p) {
        const char *amp = strchr(p, '&');
        size_t seg = amp ? (size_t)(amp - p) : strlen(p);
        if (seg > klen && strncmp(p, key, klen) == 0 && p[klen] == '=') {
            url_decode(p + klen + 1, seg - klen - 1, out, out_len);
            return 1;
        }
        p = amp ? amp + 1 : NULL;
    }
    return 0;
}
