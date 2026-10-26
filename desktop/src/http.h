/*
 * http.h - tiny blocking HTTP/1.1 server (one thread per connection).
 */
#ifndef GS_HTTP_H
#define GS_HTTP_H

#include <stddef.h>

#include "json.h"
#include "platform.h"

typedef struct {
    gs_socket sock;
    char peer[64];
    char method[16];
    char path[1024];  /* URL decoded path without query */
    char query[1024]; /* raw query string */
    char content_type[128];
    char auth[256];   /* Authorization or X-Auth-Token header */
    long long content_length; /* -1 if absent */
    /* bytes of the body that were read together with the header */
    char *pre;
    size_t pre_len, pre_pos;
    long long body_read;
    int responded;
} http_request;

typedef void (*http_handler)(http_request *req, void *user);

/* Runs the accept loop forever (returns only on error). */
int http_serve(const char *bind_addr, int port, http_handler handler, void *user);

/* Reads up to `max` body bytes into a malloc'd NUL terminated buffer. */
char *http_read_body(http_request *req, size_t max, size_t *out_len);
/* Streams the whole body into `path`. Returns bytes written or -1. */
long long http_body_to_file(http_request *req, const char *path, long long max);

void http_respond(http_request *req, int status, const char *content_type, const void *body, size_t len);
void http_respond_json(http_request *req, int status, const char *json);
void http_respond_error(http_request *req, int status, const char *message);
/* Streams a file; 404 if missing. */
void http_respond_file(http_request *req, const char *path, const char *content_type, const char *download_name);

/* Query parameter lookup (URL decoded). Returns 1 if found. */
int http_query_param(const http_request *req, const char *key, char *out, size_t out_len);

#endif
