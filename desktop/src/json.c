/*
 * json.c - minimal recursive descent JSON parser.
 */
#include "json.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p;
    const char *err;
    int depth;
} parser;

static json_value *parse_value(parser *ps);

static void skip_ws(parser *ps)
{
    while (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')
        ps->p++;
}

static json_value *new_value(json_type t)
{
    json_value *v = (json_value *)calloc(1, sizeof(json_value));
    if (v)
        v->type = t;
    return v;
}

static int push_item(json_value *parent, json_value *child)
{
    json_value **n = (json_value **)realloc(parent->items, (parent->count + 1) * sizeof(*n));
    if (!n)
        return -1;
    parent->items = n;
    parent->items[parent->count++] = child;
    return 0;
}

static void encode_utf8(sbuf *b, unsigned cp)
{
    char out[4];
    if (cp < 0x80) {
        out[0] = (char)cp;
        sb_appendn(b, out, 1);
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        sb_appendn(b, out, 2);
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        sb_appendn(b, out, 3);
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        sb_appendn(b, out, 4);
    }
}

static int parse_hex4(const char *p, unsigned *out)
{
    unsigned v = 0;
    int i;
    for (i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9')
            v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f')
            v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            v |= (unsigned)(c - 'A' + 10);
        else
            return -1;
    }
    *out = v;
    return 0;
}

static char *parse_string_raw(parser *ps)
{
    sbuf b;
    if (*ps->p != '"') {
        ps->err = "expected string";
        return NULL;
    }
    ps->p++;
    sb_init(&b);
    sb_append(&b, "");
    while (*ps->p && *ps->p != '"') {
        if (*ps->p == '\\') {
            char c = ps->p[1];
            ps->p += 2;
            switch (c) {
            case '"': sb_appendn(&b, "\"", 1); break;
            case '\\': sb_appendn(&b, "\\", 1); break;
            case '/': sb_appendn(&b, "/", 1); break;
            case 'b': sb_appendn(&b, "\b", 1); break;
            case 'f': sb_appendn(&b, "\f", 1); break;
            case 'n': sb_appendn(&b, "\n", 1); break;
            case 'r': sb_appendn(&b, "\r", 1); break;
            case 't': sb_appendn(&b, "\t", 1); break;
            case 'u': {
                unsigned cp;
                if (parse_hex4(ps->p, &cp) != 0) {
                    ps->err = "bad \\u escape";
                    sb_free(&b);
                    return NULL;
                }
                ps->p += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && ps->p[0] == '\\' && ps->p[1] == 'u') {
                    unsigned lo;
                    if (parse_hex4(ps->p + 2, &lo) == 0 && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        ps->p += 6;
                    }
                }
                encode_utf8(&b, cp);
                break;
            }
            default:
                ps->err = "bad escape";
                sb_free(&b);
                return NULL;
            }
        } else {
            sb_appendn(&b, ps->p, 1);
            ps->p++;
        }
    }
    if (*ps->p != '"') {
        ps->err = "unterminated string";
        sb_free(&b);
        return NULL;
    }
    ps->p++;
    return b.data;
}

static json_value *parse_value(parser *ps)
{
    json_value *v;
    skip_ws(ps);
    if (++ps->depth > 256) {
        ps->err = "nesting too deep";
        return NULL;
    }
    switch (*ps->p) {
    case '{':
        v = new_value(JSON_OBJECT);
        ps->p++;
        skip_ws(ps);
        if (*ps->p == '}') {
            ps->p++;
            break;
        }
        for (;;) {
            char *key;
            json_value *child;
            skip_ws(ps);
            key = parse_string_raw(ps);
            if (!key) {
                json_free(v);
                return NULL;
            }
            skip_ws(ps);
            if (*ps->p != ':') {
                free(key);
                json_free(v);
                ps->err = "expected ':'";
                return NULL;
            }
            ps->p++;
            child = parse_value(ps);
            if (!child) {
                free(key);
                json_free(v);
                return NULL;
            }
            child->key = key;
            push_item(v, child);
            skip_ws(ps);
            if (*ps->p == ',') {
                ps->p++;
                continue;
            }
            if (*ps->p == '}') {
                ps->p++;
                break;
            }
            json_free(v);
            ps->err = "expected ',' or '}'";
            return NULL;
        }
        break;
    case '[':
        v = new_value(JSON_ARRAY);
        ps->p++;
        skip_ws(ps);
        if (*ps->p == ']') {
            ps->p++;
            break;
        }
        for (;;) {
            json_value *child = parse_value(ps);
            if (!child) {
                json_free(v);
                return NULL;
            }
            push_item(v, child);
            skip_ws(ps);
            if (*ps->p == ',') {
                ps->p++;
                continue;
            }
            if (*ps->p == ']') {
                ps->p++;
                break;
            }
            json_free(v);
            ps->err = "expected ',' or ']'";
            return NULL;
        }
        break;
    case '"':
        v = new_value(JSON_STRING);
        v->string = parse_string_raw(ps);
        if (!v->string) {
            json_free(v);
            return NULL;
        }
        break;
    case 't':
        if (strncmp(ps->p, "true", 4) != 0)
            goto bad;
        v = new_value(JSON_BOOL);
        v->number = 1;
        ps->p += 4;
        break;
    case 'f':
        if (strncmp(ps->p, "false", 5) != 0)
            goto bad;
        v = new_value(JSON_BOOL);
        ps->p += 5;
        break;
    case 'n':
        if (strncmp(ps->p, "null", 4) != 0)
            goto bad;
        v = new_value(JSON_NULL);
        ps->p += 4;
        break;
    default: {
        char *end;
        double d = strtod(ps->p, &end);
        if (end == ps->p)
            goto bad;
        v = new_value(JSON_NUMBER);
        v->number = d;
        ps->p = end;
        break;
    }
    }
    ps->depth--;
    return v;
bad:
    ps->err = "unexpected character";
    return NULL;
}

json_value *json_parse(const char *text, char *err, size_t err_len)
{
    parser ps;
    json_value *v;
    ps.p = text;
    ps.err = NULL;
    ps.depth = 0;
    v = parse_value(&ps);
    if (v) {
        skip_ws(&ps);
        if (*ps.p) {
            json_free(v);
            v = NULL;
            ps.err = "trailing characters";
        }
    }
    if (!v && err && err_len)
        snprintf(err, err_len, "JSON error at offset %ld: %s", (long)(ps.p - text),
                 ps.err ? ps.err : "unknown");
    return v;
}

void json_free(json_value *v)
{
    size_t i;
    if (!v)
        return;
    for (i = 0; i < v->count; i++)
        json_free(v->items[i]);
    free(v->items);
    free(v->string);
    free(v->key);
    free(v);
}

const json_value *json_get(const json_value *obj, const char *key)
{
    size_t i;
    if (!obj || obj->type != JSON_OBJECT)
        return NULL;
    for (i = 0; i < obj->count; i++)
        if (obj->items[i]->key && strcmp(obj->items[i]->key, key) == 0)
            return obj->items[i];
    return NULL;
}

const json_value *json_at(const json_value *arr, size_t index)
{
    if (!arr || (arr->type != JSON_ARRAY && arr->type != JSON_OBJECT) || index >= arr->count)
        return NULL;
    return arr->items[index];
}

size_t json_len(const json_value *v)
{
    return v ? v->count : 0;
}

double json_num(const json_value *v, double def)
{
    if (!v || (v->type != JSON_NUMBER && v->type != JSON_BOOL))
        return def;
    return v->number;
}

const char *json_str(const json_value *v, const char *def)
{
    if (!v || v->type != JSON_STRING)
        return def;
    return v->string;
}

/* ===================================================================== */
/* string builder                                                        */
/* ===================================================================== */

void sb_init(sbuf *b)
{
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
}

void sb_free(sbuf *b)
{
    free(b->data);
    sb_init(b);
}

static void sb_reserve(sbuf *b, size_t extra)
{
    size_t need = b->len + extra + 1;
    if (need > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        char *n;
        while (cap < need)
            cap *= 2;
        n = (char *)realloc(b->data, cap);
        if (!n)
            abort();
        b->data = n;
        b->cap = cap;
    }
}

void sb_appendn(sbuf *b, const char *s, size_t n)
{
    sb_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = 0;
}

void sb_append(sbuf *b, const char *s)
{
    sb_appendn(b, s, strlen(s));
}

void sb_printf(sbuf *b, const char *fmt, ...)
{
    va_list ap;
    int n;
    va_start(ap, fmt);
    n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0)
        return;
    sb_reserve(b, (size_t)n);
    va_start(ap, fmt);
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, ap);
    va_end(ap);
    b->len += (size_t)n;
}

void sb_json_string(sbuf *b, const char *s)
{
    sb_append(b, "\"");
    for (; s && *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"': sb_append(b, "\\\""); break;
        case '\\': sb_append(b, "\\\\"); break;
        case '\n': sb_append(b, "\\n"); break;
        case '\r': sb_append(b, "\\r"); break;
        case '\t': sb_append(b, "\\t"); break;
        default:
            if (c < 0x20)
                sb_printf(b, "\\u%04x", c);
            else
                sb_appendn(b, (const char *)&c, 1);
        }
    }
    sb_append(b, "\"");
}
