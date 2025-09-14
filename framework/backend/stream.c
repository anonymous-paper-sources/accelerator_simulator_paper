/*  The idea is having a stream abstraction over an IO descriptor
    to expose the user an interface for allocating bytes in the stream
    and handling under the hood non-blocking writes to the descriptor. */

#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include "stream.h"

static inline int make_nonblocking(int fd)
{
    int flags = fcntl(fd, F_GETFL);
    if (flags == -1) return -1;
    
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1)
        return -1;
    
    return 0;
}

static inline int write_all(int fd, const void *buf, ptrdiff_t count)
{
    char *curr = (char *)buf;
    const char *end = curr + count;
    ssize_t written;
    while (curr < end)
    {
        written = write(fd, curr, end - curr);
        if (written > 0) curr += written;
        else if (written < 0 &&
            errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
            return -1;
    }

    return 0;
}

ostream *ostream_open_(char *base, ptrdiff_t len, int fd, ptrdiff_t size, ptrdiff_t align)
{
    if (!base || size <= 0 || align & (align - 1)) return NULL;

    const char *end = base + len;
    ptrdiff_t padding = -(uintptr_t)base & (_Alignof(ostream) - 1);
    ostream *s = (ostream *)(base + padding);
    padding = -(uintptr_t)(s + 1) & (align - 1);
    
    s->buf = (char *)(s + 1) + padding;
    s->buf_len = end - s->buf;
    // Modulo because size can be any value
    s->buf_len -= s->buf_len % size;
    s->prod = s->cons = 0;
    s->fd = fd;
    // The size in bytes of a single element
    // on the stream
    s->elem_size = size;
    s->last_err = 0;

    if (s->buf_len <= 0 || make_nonblocking(fd))
        return NULL;

    return s;
}

void *ostream_push_(ostream *s)
{
    ptrdiff_t nused = s->prod - s->cons;
    ptrdiff_t nwrtbl = nused - 1;
    if (nused < 0)
    {
        nused += s->buf_len;
        nwrtbl = s->buf_len - s->cons - !s->prod;
    }

    nwrtbl *= !s->last_err;

    if (nwrtbl > 0 && nused >= s->buf_len / 2)
    {
        ssize_t r = write(s->fd, s->buf + s->cons, nwrtbl);
        if (r > 0)
        {
            s->cons = (s->cons + r) % s->buf_len;
            nused -= r;
        }
        else if (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
            s->last_err = errno;
    }

    void  *p = NULL;
    ptrdiff_t navlbl = s->buf_len - nused - s->elem_size;
    if (navlbl <= 0) return p;
    
    p = s->buf + s->prod;
    s->prod = (s->prod + s->elem_size) % s->buf_len;

    return p;
}

int ostream_flush(ostream *s)
{
    int r = -1;
    if (s->last_err) return r;
    
    if (s->prod >= s->cons)
        r = write_all(s->fd, s->buf + s->cons, s->prod - s->cons);
    else
    {
        r = write_all(s->fd, s->buf + s->cons, s->buf_len - s->cons);
        r |= write_all(s->fd, s->buf, s->prod);
    }

    return r;
}

int ostream_close(ostream *s)
{
    return ostream_flush(s);
}

istream *istream_open_(char *base, ptrdiff_t len, int fd, ptrdiff_t size, ptrdiff_t align)
{
    if (!base || size <= 0 || align & (align - 1)) return NULL;

    const char *end = base + len;
    ptrdiff_t padding = -(uintptr_t)base & (_Alignof(istream) - 1);
    istream *s = (istream *)(base + padding);
    padding = -(uintptr_t)(s + 1) & (align - 1);
    
    s->buf = (char *)(s + 1) + padding;
    s->buf_len = end - s->buf;
    // Modulo because size can be any value
    s->buf_len -= s->buf_len % size;
    s->prod = s->cons = 0;
    s->fd = fd;
    // The size in bytes of a single element
    // on the stream
    s->elem_size = size;
    s->eof = s->eos = s->last_err = 0;

    if (s->buf_len <= 0 || make_nonblocking(fd))
        return NULL;
    
    ssize_t r = read(s->fd, s->buf, s->buf_len - 1);
    if (r > 0) s->prod = (s->prod + r) % s->buf_len;

    return s;
}

void *istream_pop(istream *s, ptrdiff_t *count)
{
    stream_check(s && count, "istream_pop failed: s = %p, count = %p", s, count);
    // The total number of bytes available to be
    // consumed in the stream
    ptrdiff_t ncnsmbl = s->prod - s->cons;
    // The number of consecutive bytes available
    // to be popped by the consumer in one go.
    ptrdiff_t npopbl = ncnsmbl;
    // The number of consecutive bytes available
    // to be filled by the producer in one read
    ptrdiff_t nrdbl = s->buf_len - s->prod - !s->cons;
    if (ncnsmbl < 0)
    {
        ncnsmbl += s->buf_len;
        npopbl = s->buf_len - s->cons;
        nrdbl = s->cons - s->prod - 1;
    }

    nrdbl *= !s->eof;

    /* printf("eof = %d,  prod = %ld, cons = %ld, nrdbl = %ld, npopbl = %ld\n",
        s->eof, s->prod , s->cons, nrdbl, npopbl); */

    if (nrdbl > 0 && ncnsmbl <= s->buf_len / 2)
    {
        ssize_t r = read(s->fd, s->buf + s->prod, nrdbl);
        if (r > 0)
        {
            s->prod = (s->prod + r) % s->buf_len;
            ncnsmbl += r;
        }
        else if (r == 0) s->eof = 1;
        else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
            s->last_err = errno;
    }

    // count will be zero if there's no bytes available
    // to supply for one element
    ptrdiff_t cnt = npopbl / s->elem_size;
    *count = *count < cnt ? *count : cnt;
    ptrdiff_t to_consume = s->elem_size * (*count);
    
    void *c = s->buf + s->cons;
    s->cons = (s->cons + to_consume) % s->buf_len;    

    if (s->eof && ncnsmbl - to_consume < s->elem_size) s->eos = 1;

    return c;
}
