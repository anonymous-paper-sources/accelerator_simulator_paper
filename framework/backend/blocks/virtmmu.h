/*
 *  The thing with NVBit is that is produces virtual addresses, but
 *  we want to simulate physical behaviur so we need physcal
 *  addresses. The naive thing would be to find the minimum in the
 *  virtual address space and subtract that minimum from all the
 *  addresses of the input trace. This is good as long as we have
 *  a single contiguos range of virtual memory used in the whole
 *  application. This is not the case when using torch or tensorflow.
 *  Using high level ML frameworks makes CUDA use different chunk
 *  of virtual address space. For example, 0x7D0000000000 to
 *  0x7D1000000000 for the application and 0x000000010000 to
 *  0x000000020000 for their own structures. By doing the subtraction
 *  thing, the resulting physical address space goes beyond the
 *  38 bit (256 GB of space) available.
 *  This module overcomes this problem by scanning the trace file
 *  before the execution and identifying all the ranges so that
 *  any virtual address can be subtracted with the right "base"
 *  address. The result will be a physical address that has no
 *  holes and is less than 256 GB.
 */

#ifndef VIRTMMU_H
#define VIRTMMU_H

#include <stddef.h>
#include <arpa/inet.h>
#include "stream.h"
#include "arena.h"

#define RANGE_MIN_LEN (1ULL << 12)
#define RANGE_MAX_LEN (1ULL << 32)

struct ranges
{
    ptrdiff_t len;
    struct range
    {
        unsigned long long start;   // Included
        unsigned long long end;     // Not included
        unsigned long long offset;
    } data[];
};

static uint64_t ntohll(uint64_t val)
{
#if __BYTE_ORDER == __LITTLE_ENDIAN
    return ((uint64_t)ntohl(val & 0xFFFFFFFF) << 32) | ntohl(val >> 32);
#else
    return val;
#endif
}

/*
 *  Returns 1 if the address is inside the range or the range
 *  can be extended linearly to contain it on one of the ends,
 *  otherwise returns 0.
 */
static inline int try_extend_range_(struct range *r, unsigned long long addr)
{
    if (addr >= r->start && addr < r->end) return 1;

    if (addr >= r->end && addr - r->start < RANGE_MAX_LEN)
    {
        r->end = addr + RANGE_MIN_LEN;
        return 1;
    }

    if (addr < r->start && r->end - addr <= RANGE_MAX_LEN)
    {
        r->start = addr;
        return 1;
    }
    
    return 0;
}

static void sort_range_arr_(struct range *data, ptrdiff_t len)
{
    for (ptrdiff_t j = 0; j < len - 1; j++)
    {
        for (ptrdiff_t i = 0; i < len - 1 - j; i++)
        {
            if (data[i].start > data[i+1].start)
            {
                struct range d = data[i];
                data[i] = data[i + 1];
                data[i + 1] = d;
            }
            
        }
    }
}

static long long virt_to_phy(struct ranges *r, unsigned long long addr)
{
    if (!r) return -1;
    
    ptrdiff_t max = r->len - 1;
    ptrdiff_t min = 0;
    ptrdiff_t i;
    while (min <= max)
    {
        i = min + (max - min) / 2;
        if (addr < r->data[i].start)
            max = i - 1;
        else if (addr >= r->data[i].end)
            min = i + 1;
        else return addr - r->data[i].offset;
    }

    return -1;
}

static struct ranges *ranges_from_trace(istream *trace, struct arena *arn)
{
    if (!trace || !arn) return NULL;
    // Only when we are done we will report the result to arn
    struct arena a = *arn;

    struct ranges *r = new(&a, struct ranges, 1);
    if (!r) return NULL;

    r->len = 0;
    
    uint64_t *be_addr_49_r_w_1;
    uint64_t addr;
    while (!trace->eos)
    {
        ptrdiff_t count = 256;
        be_addr_49_r_w_1 = istream_pop(trace, &count);
        for (ptrdiff_t i = 0; i < count; i++)
        {
            be_addr_49_r_w_1[i] = ntohll(be_addr_49_r_w_1[i]);
            // Align back address to RANGE_MIN_LEN bytes
            addr = (be_addr_49_r_w_1[i] >> 1) & -RANGE_MIN_LEN;

            for (ptrdiff_t j = 0; j < r->len; j++)
                if (try_extend_range_(r->data + j, addr))
                    goto next;
            
            struct range *x = new(&a, struct range, 1);
            if (!x) return NULL;
            x->start = addr;
            x->end = addr + RANGE_MIN_LEN;
            r->len++;
            next: (void)0;
        }
    }
    sort_range_arr_(r->data, r->len);

    if (r->len > 0) r->data[0].offset = r->data[0].start;
    for (ptrdiff_t i = 1; i < r->len; i++)
        r->data[i].offset = r->data[i].start - (r->data[i-1].end - r->data[i-1].offset);

    *arn = a;
    return r;
}

#endif