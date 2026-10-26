/*
 * json.h - minimal JSON DOM parser and a growable string builder.
 */
#ifndef GS_JSON_H
#define GS_JSON_H

#include <stddef.h>

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type;

typedef struct json_value json_value;
struct json_value {
    json_type type;
    double number;      /* JSON_NUMBER, JSON_BOOL (0/1) */
    char *string;       /* JSON_STRING, and key of object members */
    char *key;          /* set when this value is an object member */
    json_value **items; /* JSON_ARRAY / JSON_OBJECT children */
    size_t count;
};

json_value *json_parse(const char *text, char *err, size_t err_len);
void json_free(json_value *v);

const json_value *json_get(const json_value *obj, const char *key);
const json_value *json_at(const json_value *arr, size_t index);
size_t json_len(const json_value *v);
double json_num(const json_value *v, double def);
const char *json_str(const json_value *v, const char *def);

/* ---- string builder -------------------------------------------------- */
typedef struct {
    char *data;
    size_t len;
    size_t cap;
} sbuf;

void sb_init(sbuf *b);
void sb_free(sbuf *b);
void sb_append(sbuf *b, const char *s);
void sb_appendn(sbuf *b, const char *s, size_t n);
void sb_printf(sbuf *b, const char *fmt, ...);
void sb_json_string(sbuf *b, const char *s); /* appends a quoted, escaped string */

#endif
