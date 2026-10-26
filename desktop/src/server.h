/*
 * server.h - job management + REST API.
 */
#ifndef GS_SERVER_H
#define GS_SERVER_H

#include "cl_util.h"
#include "trainer.h"

typedef struct {
    const char *bind_addr;
    int port;
    const char *data_dir;
    const char *token;          /* optional shared secret, NULL = none */
    gs_train_params defaults;   /* defaults for new jobs */
    gs_cl *cl;
} gs_server_config;

/* Starts the training worker and serves HTTP forever. */
int gs_server_run(const gs_server_config *cfg);

#endif
