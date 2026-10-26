/*
 * platform.h - thin portability layer (sockets, threads, files, time).
 * Supports POSIX (Linux, macOS) and Windows (MSVC / MinGW).
 */
#ifndef GS_PLATFORM_H
#define GS_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
typedef SOCKET gs_socket;
#  define GS_INVALID_SOCKET INVALID_SOCKET
typedef HANDLE gs_thread;
typedef CRITICAL_SECTION gs_mutex;
#else
#  include <pthread.h>
typedef int gs_socket;
#  define GS_INVALID_SOCKET (-1)
typedef pthread_t gs_thread;
typedef pthread_mutex_t gs_mutex;
#endif

/* --- networking ------------------------------------------------------- */
int gs_net_init(void);
gs_socket gs_listen(const char *bind_addr, int port);
gs_socket gs_accept(gs_socket s, char *peer, size_t peer_len);
long gs_recv(gs_socket s, void *buf, size_t len);
int gs_send_all(gs_socket s, const void *buf, size_t len);
void gs_close_socket(gs_socket s);
void gs_set_socket_timeout(gs_socket s, int seconds);

/* --- threads ---------------------------------------------------------- */
typedef void (*gs_thread_fn)(void *arg);
int gs_thread_start(gs_thread *t, gs_thread_fn fn, void *arg, int detached);
void gs_thread_join(gs_thread t);
void gs_mutex_init(gs_mutex *m);
void gs_mutex_lock(gs_mutex *m);
void gs_mutex_unlock(gs_mutex *m);
void gs_sleep_ms(int ms);

/* --- files ------------------------------------------------------------ */
int gs_mkdir(const char *path);          /* 0 on success or if it exists */
int gs_mkdirs(const char *path);         /* recursive mkdir -p */
int gs_file_exists(const char *path);
int gs_is_dir(const char *path);
long long gs_file_size(const char *path);
char *gs_read_file(const char *path, size_t *out_len); /* malloc'd, NUL terminated */
int gs_write_file(const char *path, const void *data, size_t len);
int gs_rename(const char *from, const char *to);
int gs_remove_tree(const char *path);
/* Calls cb(name) for every entry of a directory (except . and ..). */
int gs_list_dir(const char *path, void (*cb)(const char *name, void *user), void *user);
/* Directory of the running executable (with trailing separator). */
void gs_exe_dir(char *out, size_t len);

/* --- time ------------------------------------------------------------- */
double gs_now_seconds(void);   /* monotonic */
long long gs_unix_time(void);  /* wall clock */

#endif
