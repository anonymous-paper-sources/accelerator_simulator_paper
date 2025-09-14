#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>
#include "stream.h"

#define R   0
#define W   1

static inline uint64_t htonll(uint64_t val)
{
#if __BYTE_ORDER == __LITTLE_ENDIAN
    return ((uint64_t)htonl(val & 0xFFFFFFFF) << 32) | htonl(val >> 32);
#else
    return val;
#endif
}

typedef struct __attribute__((__packed__))
{
    uint32_t be_clock;
    uint64_t be_addr_38_r_w_1;
} mem_access_log_t;

typedef struct __attribute__((__packed__))
{
    uint32_t be_clock;
    uint64_t be_addr_38_r_w_1_hit_1;
} cache_access_log_t;

typedef struct __attribute__((__packed__))
{
    uint32_t be_clock;
    uint64_t be_addr;
} cache_eviction_log_t;

typedef struct __attribute__((__packed__))
{
    uint32_t be_clock;
    uint64_t be_rng_start;
    uint64_t be_rng_end;
    int64_t be_vn;
} vnstore_eviction_log_t;

#define MEM_ACCESS_LOG(s, c, a, rw)     \
*ostream_push(s, mem_access_log_t) =    \
    (mem_access_log_t){.be_clock = htonl(c), .be_addr_38_r_w_1 = htonll(((a) << 1) | (rw))}

#ifdef VERBOSE_LOGGING

#define CACHE_ACCESS_LOG(s, c, a, rw, h) \
*ostream_push(s, cache_access_log_t) =  \
        (cache_access_log_t){.be_clock = htonl(c), .be_addr_38_r_w_1_hit_1 = htonll(((a) << 2) | ((rw) << 1) | (h))}

#define CACHE_EVICTION_LOG(s, c, a)         \
*ostream_push(s, cache_eviction_log_t) =    \
        (cache_eviction_log_t){.be_clock = htonl(c), .be_addr = htonll(a)}

#define VNSTORE_ACCESS_LOG(s, c, a, rw, h)  \
*ostream_push(s, cache_access_log_t) =      \
    (cache_access_log_t){.be_clock = htonl(c), .be_addr_38_r_w_1_hit_1 = htonll(((a) << 2) | ((rw) << 1) | (h))}

#define VNSTORE_EVICTION_LOG(s, c, rngs, rnge, v_n) \
*ostream_push(s, vnstore_eviction_log_t) =          \
    (vnstore_eviction_log_t){.be_clock = htonl(c), .be_rng_start = htonll(rngs), .be_rng_end = htonll(rnge), .be_vn = htonll(v_n)}

#else

#define CACHE_ACCESS_LOG(s, c, a, rw, h)            ((void)0)
#define CACHE_EVICTION_LOG(s, c, a)                 ((void)0)
#define VNSTORE_ACCESS_LOG(s, c, a, rw, h)          ((void)0)
#define VNSTORE_EVICTION_LOG(s, c, rngs, rnge, v_n) ((void)0)

#endif

#endif