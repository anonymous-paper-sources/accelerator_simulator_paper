/*
 *  The Memory controller interface
 */

#ifndef MC_H
#define MC_H

#ifndef NDEBUG
#include <stdio.h>
#define mc_check(expr, fmt, ...) do                                                    \
{                                                                                   \
    if (!(expr))                                                                    \
    {                                                                               \
        fprintf(stderr,"Line %d, File %s:" fmt, __LINE__, __FILE__, ##__VA_ARGS__); \
        __builtin_trap();                                                           \
    }                                                                               \
} while (0)
#else
#define mc_check(expr, fmt, ...) ((void)0)
#endif

#define GLOBAL_ARENA_SIZE   (8*1024*1024*1024ull)
#define STREAM_BUF_SIZE     (512*1024)

#define PZ_START  0x0
#define MZ_START  0x3E00000000
#define TZ2_START 0x3F00000000
#define TZ1_START 0x3FC8000000
#define TZ0_START 0x3FCA800000

int mc_elaborate_trace(char *name, int trace_fd);

#endif