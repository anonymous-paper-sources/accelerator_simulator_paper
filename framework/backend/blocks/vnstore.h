/*
 *  VNSTORE: Th structure that caches VNs on-chip for non-overlappping ranges
 *  of Protected Zone cache lines that use the same VN value. Each line has
 *  the following structure:
 *  ---------------------------------
 *  | V | rng_start | rng_end | ctr |
 *  ---------------------------------
 *  
 *  The table is initialized as follows at device boot:
 *  ----------------------------------
 *  | V | rng_start | rng_end  | ctr |
 *  ----------------------------------
 *  | 1 | PZ_START  | MZ_START | 0   |
 *  ----------------------------------
 * 
 *  On each get from this cache, the cache line address is checked with all
 *  the ranges and if a valid line exists whose range contains the address,
 *  then the corresponding ctr is returned, otherwise -1 is returned.
 * 
 *  On each set to this cache, the user provides with the cache line
 *  address and the new value for the ctr. If a valid line exists
 *  whose range contains the address, then the VNSTORE is update accordingly.
 *  See the documentation of vns_set for a complete explaination of the set.
 */

#ifndef VNSTORE_H
#define VNSTORE_H

#include "arena.h"

struct vnsl
{
    unsigned long long rng_start;
    unsigned long long rng_end;
    long long ctr;
};

struct vnstore
{
    unsigned long long n_lines;
    unsigned long long width;
    struct _vnsl
    {
        unsigned long long use_ctr;
        struct vnsl l;
    } lines[];
};

static struct vnstore *vns_make(unsigned char n_lines_lg2, unsigned char width_lg2,
    unsigned long long frst_rng_start, unsigned long long frst_rng_end, struct arena *a)
{
    if (n_lines_lg2 > 63 || width_lg2 > 63) return NULL;
    struct vnstore *v = alloc(a, sizeof(struct vnstore) +
        (1 << n_lines_lg2) * sizeof(struct _vnsl),
        _Alignof(struct vnstore));
    if (!v) return NULL;

    v->n_lines = 1 << n_lines_lg2;
    v->width = 1 << width_lg2;

    v->lines[0].use_ctr = v->n_lines - 1;
    v->lines[0].l.rng_start = frst_rng_start & -v->width;
    v->lines[0].l.rng_end = frst_rng_end & -v->width;
    // ctr = 0 is used as the counter for never-written memory
    // ctr = -1 is used as an invalid line
    v->lines[0].l.ctr = 0;

    for (unsigned long long i = 1; i < v->n_lines; i++)
    {
        v->lines[i].use_ctr = i - 1;
        v->lines[i].l.ctr = -1;
    }

    return v;
}

/*
 *  Searches the VNSTORE for the line whose range contains the
 *  provided address and returns the index of the line if found,
 *  otherwise returns the number of lines in the VNSTORE.
 */
static inline unsigned long long
search_line_(struct vnstore *v, unsigned long long addr)
{
    for (unsigned long long i = 0; i < v->n_lines; i++)
        if (v->lines[i].l.ctr >= 0            &&
            addr >= v->lines[i].l.rng_start   &&
            addr < v->lines[i].l.rng_end)
            return i;
    
    return v->n_lines;
}

/*
 *  Marks the line at the provided position as the most recently used
 *  line in the VNSTORE.
 */
static inline void
mark_line_lru_(struct vnstore *v, unsigned long long pos)
{
    unsigned long long svd_use_ctr = v->lines[pos].use_ctr;
    for (unsigned long long i = 0; i < v->n_lines; i++)
        if (v->lines[i].use_ctr > svd_use_ctr) v->lines[i].use_ctr--;
    v->lines[pos].use_ctr = v->n_lines - 1;
}

/*
 *  Creates a new range in the VNSTORE with the provided
 *  start and end addresses and the provided counter and
 *  marks it as the most recently used. The least recently
 *  used line is evicted and the new range is placed in
 *  its position. If the evicted line is not valid, then
 *  it has a counter of -1. Returns the position of the
 *  new range in the VNSTORE.
 */
static inline unsigned long long
new_range_(struct vnstore *v, unsigned long long rng_start,
    unsigned long long rng_end, long long ctr, struct vnsl *evicted)
{
    unsigned long long pos;

    for (unsigned long long i = 0; i < v->n_lines; i++)
    {
        if (v->lines[i].use_ctr == 0)
        {
            pos = i;

            evicted->rng_start = v->lines[i].l.rng_start;
            evicted->rng_end = v->lines[i].l.rng_end;
            evicted->ctr = v->lines[i].l.ctr;

            v->lines[i].l.rng_start = rng_start;
            v->lines[i].l.rng_end = rng_end;
            v->lines[i].l.ctr = ctr;
            v->lines[i].use_ctr = v->n_lines - 1;
        }
        else v->lines[i].use_ctr--;
    }

    return pos;
}

/*
 *  Searches the VNSTORE for the line whose range contains the
 *  provided address and returns the corresponding counter if found,
 *  otherwise returns -1.
 */
static long long vns_get(struct vnstore *v, unsigned long long addr)
{
    if (!v) return -1;
    
    addr &= -v->width;
    unsigned long long pos = search_line_(v, addr);
    if (pos >= v->n_lines) return -1;
    
    mark_line_lru_(v, pos);
    
    return v->lines[pos].l.ctr;
}

/*
 *  Given the cache line address corresponding to addr, search the
 *  corresponding range and update it as follows if it is found:
 *  - If addr is in the middle of the range:
 *      1) Create two ranges [rng_start, addr) and [addr + v->width, rng_end)
 *      with the same counter as before
 *      2) Create a new range [addr, addr + v->width) with the input ctr
 *  - If addr is at the start of the range:
 *      1) Update the range to [rng_start + v->width, rng_end)
 *      2) Search a range with addr - v->width and, if found and with the
 *      same counter as the input ctr, extend that range, othewise create
 *      a new range [addr, addr + v->width) with the input ctr
 *  - If addr is at the end of the range:
 *      1) Update the range to [rng_start, rng_end - v->width)
 *      2) Search a range with addr + v->width and, if found and with the
 *      same counter as the input ctr, extend that range, othewise create
 *      a new range [addr, addr + v->width) with the input ctr
 *  - If addr is the only cacheline of the range:
 *      1) Update the counter to the input ctr
 *      2) Search a range with addr + v->width, if found and with the
 *      same counter as the input ctr, merge the two ranges and free
 *      one of them
 *      3) Search a range with addr - v->width, if found and with the
 *      same counter as the input ctr, merge again this range to the
 *      current and free one of them
 *  
 *  If no range is found, then:
 *  - If a range exists on the right that has the same counter as the
 *  input ctr: Update the range to [rng_start - v->width, rng_end).
 *  - If a range exists on the left that has the same counter as the
 *  input ctr: Update the range to [rng_start, rng_end + v->width).
 *  - Otherwise: create a new range [addr, addr + v->width) with the input ctr.
 * 
 *  Sets (*evicted)[0] and (*evicted)[1] with the evicted line(s) during the process
 */
static void
vns_set(struct vnstore *v, unsigned long long addr, long long ctr, struct vnsl (*evicted)[2])
{
    if (!v || !evicted) return;
    
    (*evicted)[0].ctr = (*evicted)[1].ctr = -1;

    addr &= -v->width;
    unsigned long long curr = search_line_(v, addr);
    unsigned long long next = search_line_(v, addr + v->width);
    unsigned long long prev = search_line_(v, addr - v->width);

    if (curr < v->n_lines)
    {
        // To avoid useless modifications
        if (v->lines[curr].l.ctr == ctr) return;

        if (addr == v->lines[curr].l.rng_start &&
            addr == v->lines[curr].l.rng_end - v->width)
        {
            v->lines[curr].l.ctr = ctr;
            if (next < v->n_lines && v->lines[next].l.ctr == ctr)
            {
                v->lines[curr].l.rng_end = v->lines[next].l.rng_end;
                v->lines[next].l.ctr = -1;
            }
            if (prev < v->n_lines && v->lines[prev].l.ctr == ctr)
            {
                v->lines[curr].l.rng_start = v->lines[prev].l.rng_start;
                v->lines[prev].l.ctr = -1;
            }
            mark_line_lru_(v, curr);
        }
        else if (addr == v->lines[curr].l.rng_start)
        {
            v->lines[curr].l.rng_start += v->width;
            if (prev < v->n_lines && v->lines[prev].l.ctr == ctr)
            {
                v->lines[prev].l.rng_end = v->lines[curr].l.rng_start;
                mark_line_lru_(v, prev);
            }
            else new_range_(v, addr, addr + v->width, ctr, &(*evicted)[0]);
        }
        else if (addr == v->lines[curr].l.rng_end - v->width)
        {
            v->lines[curr].l.rng_end -= v->width;
            if (next < v->n_lines && v->lines[next].l.ctr == ctr)
            {
                v->lines[next].l.rng_start = v->lines[curr].l.rng_end;
                mark_line_lru_(v, next);
            }
            else new_range_(v, addr, addr + v->width, ctr, &(*evicted)[0]);
        }
        else
        {
            unsigned long long rng_end = v->lines[curr].l.rng_end;
            v->lines[curr].l.rng_end = addr;
            mark_line_lru_(v, curr);
            new_range_(v, addr + v->width, rng_end, v->lines[curr].l.ctr, &(*evicted)[0]);
            new_range_(v, addr, addr + v->width, ctr, &(*evicted)[1]);
        }
    }
    else
    {
        if (next < v->n_lines && v->lines[next].l.ctr == ctr)
        {
            v->lines[next].l.rng_start = addr;
            mark_line_lru_(v, curr);
        }
        else if (prev < v->n_lines && v->lines[prev].l.ctr == ctr)
        {
            v->lines[prev].l.rng_end = addr;
            mark_line_lru_(v, prev);
        }
        else new_range_(v, addr, addr + v->width, ctr, &(*evicted)[0]);
    }
}

#ifndef NDEBUG
#include <stdio.h>
static void vns_fprint_state(FILE *f, struct vnstore *v)
{
    if (!v || !f) return;
    fputs("VNSTORE valid lines:\n", stderr);
    for (unsigned long long i = 0; i < v->n_lines; i++)
    {        
        if (v->lines[i].l.ctr < 0) continue;
        fprintf(f,
            "Line %llu: use_ctr = %llu, rng_start = %#llX, rng_end = %#llX, ctr = %lld\n",
            i, v->lines[i].use_ctr, v->lines[i].l.rng_start, v->lines[i].l.rng_end,
            v->lines[i].l.ctr);
    }
}
#else
#define vns_fprint_state(f, v) ((void)0)
#endif

#endif