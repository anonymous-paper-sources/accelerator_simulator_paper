#ifndef STREAM_H
#define STREAM_H

#include <stddef.h>

#ifndef NDEBUG
#include <stdio.h>
#define stream_check(expr, fmt, ...) do                                             \
{                                                                                   \
    if (!(expr))                                                                    \
    {                                                                               \
        fprintf(stderr,"Line %d, File %s:" fmt, __LINE__, __FILE__, ##__VA_ARGS__); \
        __builtin_trap();                                                           \
    }                                                                               \
} while (0)
#else
#define stream_check(expr, fmt, ...) ((void)0)
#endif

typedef struct
{
    char *buf;
    ptrdiff_t buf_len, prod, cons, elem_size;
    int fd;
    int last_err;   // errno
} ostream;

typedef struct
{
    char *buf;
    ptrdiff_t buf_len, prod, cons, elem_size;
    int fd;
    int last_err;   // errno
    char eof;       // end-of-file
    char eos;       // end-of-stream
} istream;

#define ostream_open(base, len, fd, type)   \
ostream_open_((base), (len), (fd), sizeof(type), _Alignof(type))
#define ostream_push(s, type)                   \
({                                              \
    stream_check(sizeof(type) == s->elem_size,  \
        "ostream_push type mismatch\n");        \
    (type *)ostream_push_((s));                 \
})

#define istream_open(base, len, fd, type)   \
istream_open_((base), (len), (fd), sizeof(type), _Alignof(type))

ostream *ostream_open_(char *base, ptrdiff_t len, int fd, ptrdiff_t size, ptrdiff_t align);
void *ostream_push_(ostream *s);
int ostream_flush(ostream *s);
int ostream_close(ostream *s);

istream *istream_open_(char *base, ptrdiff_t len, int fd, ptrdiff_t size, ptrdiff_t align);
void *istream_pop(istream *s, ptrdiff_t *count);

#endif