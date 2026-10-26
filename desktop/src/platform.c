/*
 * platform.c - implementation of the portability layer.
 */
#ifndef _WIN32
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE
#endif

#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#  include <direct.h>
#  include <io.h>
#  include <sys/stat.h>
#  ifdef _MSC_VER
#    pragma comment(lib, "ws2_32.lib")
#  endif
#else
#  include <arpa/inet.h>
#  include <dirent.h>
#  include <errno.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <signal.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <sys/time.h>
#  include <unistd.h>
#  ifdef __APPLE__
#    include <mach-o/dyld.h>
#  endif
#endif

/* ===================================================================== */
/* networking                                                            */
/* ===================================================================== */

int gs_net_init(void)
{
#ifdef _WIN32
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0 ? 0 : -1;
#else
    signal(SIGPIPE, SIG_IGN);
    return 0;
#endif
}

gs_socket gs_listen(const char *bind_addr, int port)
{
    struct sockaddr_in addr;
    gs_socket s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int yes = 1;
    if (s == GS_INVALID_SOCKET)
        return GS_INVALID_SOCKET;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    if (!bind_addr || !*bind_addr || strcmp(bind_addr, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, bind_addr, &addr.sin_addr) != 1) {
        gs_close_socket(s);
        return GS_INVALID_SOCKET;
    }
    if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(s, 32) != 0) {
        gs_close_socket(s);
        return GS_INVALID_SOCKET;
    }
    return s;
}

gs_socket gs_accept(gs_socket s, char *peer, size_t peer_len)
{
    struct sockaddr_in addr;
#ifdef _WIN32
    int alen = sizeof(addr);
#else
    socklen_t alen = sizeof(addr);
#endif
    gs_socket c = accept(s, (struct sockaddr *)&addr, &alen);
    if (c != GS_INVALID_SOCKET && peer && peer_len) {
        if (!inet_ntop(AF_INET, &addr.sin_addr, peer, (socklen_t)peer_len))
            snprintf(peer, peer_len, "?");
    }
    if (c != GS_INVALID_SOCKET) {
        int yes = 1;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char *)&yes, sizeof(yes));
    }
    return c;
}

long gs_recv(gs_socket s, void *buf, size_t len)
{
#ifdef _WIN32
    return recv(s, (char *)buf, (int)(len > 0x7fffffff ? 0x7fffffff : len), 0);
#else
    ssize_t r;
    do {
        r = recv(s, buf, len, 0);
    } while (r < 0 && errno == EINTR);
    return (long)r;
#endif
}

int gs_send_all(gs_socket s, const void *buf, size_t len)
{
    const char *p = (const char *)buf;
    while (len > 0) {
#ifdef _WIN32
        int n = send(s, p, (int)(len > 1 << 20 ? 1 << 20 : len), 0);
#else
        ssize_t n = send(s, p, len, 0);
        if (n < 0 && errno == EINTR)
            continue;
#endif
        if (n <= 0)
            return -1;
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

void gs_close_socket(gs_socket s)
{
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

void gs_set_socket_timeout(gs_socket s, int seconds)
{
#ifdef _WIN32
    DWORD ms = (DWORD)seconds * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&ms, sizeof(ms));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&ms, sizeof(ms));
#else
    struct timeval tv;
    tv.tv_sec = seconds;
    tv.tv_usec = 0;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

/* ===================================================================== */
/* threads                                                               */
/* ===================================================================== */

typedef struct {
    gs_thread_fn fn;
    void *arg;
} thread_start_info;

#ifdef _WIN32
static DWORD WINAPI thread_trampoline(LPVOID p)
#else
static void *thread_trampoline(void *p)
#endif
{
    thread_start_info info = *(thread_start_info *)p;
    free(p);
    info.fn(info.arg);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

int gs_thread_start(gs_thread *t, gs_thread_fn fn, void *arg, int detached)
{
    thread_start_info *info = (thread_start_info *)malloc(sizeof(*info));
    gs_thread tmp;
    if (!info)
        return -1;
    info->fn = fn;
    info->arg = arg;
    if (!t)
        t = &tmp;
#ifdef _WIN32
    *t = CreateThread(NULL, 0, thread_trampoline, info, 0, NULL);
    if (!*t) {
        free(info);
        return -1;
    }
    if (detached)
        CloseHandle(*t);
#else
    if (pthread_create(t, NULL, thread_trampoline, info) != 0) {
        free(info);
        return -1;
    }
    if (detached)
        pthread_detach(*t);
#endif
    return 0;
}

void gs_thread_join(gs_thread t)
{
#ifdef _WIN32
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
#else
    pthread_join(t, NULL);
#endif
}

void gs_mutex_init(gs_mutex *m)
{
#ifdef _WIN32
    InitializeCriticalSection(m);
#else
    pthread_mutex_init(m, NULL);
#endif
}

void gs_mutex_lock(gs_mutex *m)
{
#ifdef _WIN32
    EnterCriticalSection(m);
#else
    pthread_mutex_lock(m);
#endif
}

void gs_mutex_unlock(gs_mutex *m)
{
#ifdef _WIN32
    LeaveCriticalSection(m);
#else
    pthread_mutex_unlock(m);
#endif
}

void gs_sleep_ms(int ms)
{
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

/* ===================================================================== */
/* files                                                                 */
/* ===================================================================== */

int gs_mkdir(const char *path)
{
#ifdef _WIN32
    if (_mkdir(path) == 0)
        return 0;
#else
    if (mkdir(path, 0755) == 0)
        return 0;
#endif
    return gs_is_dir(path) ? 0 : -1;
}

int gs_mkdirs(const char *path)
{
    char tmp[1024];
    size_t i, n = strlen(path);
    if (n >= sizeof(tmp))
        return -1;
    memcpy(tmp, path, n + 1);
    for (i = 1; i < n; i++) {
        if (tmp[i] == '/' || tmp[i] == '\\') {
            char c = tmp[i];
            if (i > 0 && tmp[i - 1] == ':')
                continue; /* "C:\" */
            tmp[i] = 0;
            gs_mkdir(tmp);
            tmp[i] = c;
        }
    }
    return gs_mkdir(tmp);
}

int gs_file_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

int gs_is_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        return 0;
#ifdef _WIN32
    return (st.st_mode & _S_IFDIR) != 0;
#else
    return S_ISDIR(st.st_mode);
#endif
}

long long gs_file_size(const char *path)
{
#ifdef _WIN32
    struct _stat64 st;
    if (_stat64(path, &st) != 0)
        return -1;
    return (long long)st.st_size;
#else
    struct stat st;
    if (stat(path, &st) != 0)
        return -1;
    return (long long)st.st_size;
#endif
}

char *gs_read_file(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    long long size;
    char *buf;
    if (!f)
        return NULL;
    size = gs_file_size(path);
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    buf = (char *)malloc((size_t)size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (size > 0 && fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[size] = 0;
    if (out_len)
        *out_len = (size_t)size;
    return buf;
}

int gs_write_file(const char *path, const void *data, size_t len)
{
    char tmp[1100];
    FILE *f;
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    f = fopen(tmp, "wb");
    if (!f)
        return -1;
    if (len && fwrite(data, 1, len, f) != len) {
        fclose(f);
        remove(tmp);
        return -1;
    }
    fclose(f);
    return gs_rename(tmp, path);
}

int gs_rename(const char *from, const char *to)
{
#ifdef _WIN32
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING) ? 0 : -1;
#else
    return rename(from, to);
#endif
}

typedef struct {
    const char *base;
} remove_ctx;

static void remove_cb(const char *name, void *user)
{
    remove_ctx *ctx = (remove_ctx *)user;
    char p[1024];
    snprintf(p, sizeof(p), "%s/%s", ctx->base, name);
    gs_remove_tree(p);
}

int gs_remove_tree(const char *path)
{
    if (gs_is_dir(path)) {
        remove_ctx ctx;
        ctx.base = path;
        gs_list_dir(path, remove_cb, &ctx);
#ifdef _WIN32
        return _rmdir(path);
#else
        return rmdir(path);
#endif
    }
    return remove(path);
}

int gs_list_dir(const char *path, void (*cb)(const char *name, void *user), void *user)
{
#ifdef _WIN32
    char pattern[1024];
    WIN32_FIND_DATAA fd;
    HANDLE h;
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE)
        return -1;
    do {
        if (strcmp(fd.cFileName, ".") && strcmp(fd.cFileName, ".."))
            cb(fd.cFileName, user);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    return 0;
#else
    DIR *d = opendir(path);
    struct dirent *e;
    if (!d)
        return -1;
    while ((e = readdir(d)) != NULL) {
        if (strcmp(e->d_name, ".") && strcmp(e->d_name, ".."))
            cb(e->d_name, user);
    }
    closedir(d);
    return 0;
#endif
}

void gs_exe_dir(char *out, size_t len)
{
    char buf[1024];
    size_t n = 0;
    char *slash;
    buf[0] = 0;
#ifdef _WIN32
    n = GetModuleFileNameA(NULL, buf, sizeof(buf));
#elif defined(__APPLE__)
    {
        uint32_t sz = sizeof(buf);
        if (_NSGetExecutablePath(buf, &sz) == 0)
            n = strlen(buf);
    }
#else
    {
        ssize_t r = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (r > 0)
            n = (size_t)r;
    }
#endif
    buf[n < sizeof(buf) ? n : sizeof(buf) - 1] = 0;
    slash = strrchr(buf, '/');
#ifdef _WIN32
    {
        char *bs = strrchr(buf, '\\');
        if (bs && (!slash || bs > slash))
            slash = bs;
    }
#endif
    if (slash)
        slash[1] = 0;
    else
        snprintf(buf, sizeof(buf), "./");
    snprintf(out, len, "%s", buf);
}

/* ===================================================================== */
/* time                                                                  */
/* ===================================================================== */

double gs_now_seconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

long long gs_unix_time(void)
{
    return (long long)time(NULL);
}
