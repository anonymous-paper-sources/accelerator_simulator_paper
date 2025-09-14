#ifndef CACHE_H
#define CACHE_H

#include <stdbool.h>
#include "arena.h"

struct line
{
    bool dirty;
    unsigned long long addr;
};

struct cache
{
    unsigned char width_lg2;
    unsigned char assoc_lg2;
    unsigned char n_lines_lg2;
    unsigned char n_set_lg2;
    struct _l
    {
        bool valid;
        bool dirty;
        unsigned long long use_ctr;
        unsigned long long tag;
    } lines[];
};

static struct cache *
cache_make(unsigned char width_lg2, unsigned char n_lines_lg2, unsigned char assoc_lg2, struct arena *a)
{
    if(width_lg2 > 63 || assoc_lg2 > 63 || n_lines_lg2 > 63 ||
    // Can't have a cache with a set bigger than the ammount of lines
        assoc_lg2 > n_lines_lg2) return NULL;
    
    unsigned long long n_lines = 1 << n_lines_lg2;
    struct cache *c = alloc(a, sizeof(struct cache) +
        sizeof(struct _l) * n_lines, _Alignof(struct cache));
    if (!c) return NULL;
    
    c->n_lines_lg2 = n_lines_lg2;
    c->assoc_lg2 = assoc_lg2;
    c->width_lg2 = width_lg2;
    c->n_set_lg2 = n_lines_lg2 - assoc_lg2;

    unsigned long long as = 1 << c->assoc_lg2;
    
    for (unsigned long long i = 0; i < n_lines; i++)
    {
        c->lines[i].valid = false;
        c->lines[i].use_ctr = i & (as - 1);
    }

    return c;
}

static bool cache_read(struct cache *c, unsigned long long addr)
{
    if (!c) return false;
    
    unsigned long long block = addr >> c->width_lg2;
    unsigned long long tag = block >> c->n_set_lg2;
    unsigned long long set = block & ((1 << c->n_set_lg2) - 1);
    unsigned long long a = 1 << c->assoc_lg2;

    unsigned long long ctr;

    for (unsigned long long i = set * a; i < a * (set + 1); i++)
        if (c->lines[i].valid && c->lines[i].tag == tag)
        {
            ctr = c->lines[i].use_ctr;
            for (unsigned long long j = set * a; j < a * (set + 1); j++)
                if (c->lines[j].use_ctr > ctr) c->lines[j].use_ctr--;
            c->lines[i].use_ctr = a - 1;
            
            return true;
        }
    
    return false;
}

static bool cache_write(struct cache *c, unsigned long long addr)
{
    if (!c) return false;

    unsigned long long block = addr >> c->width_lg2;
    unsigned long long tag = block >> c->n_set_lg2;
    unsigned long long set = block & ((1 << c->n_set_lg2) - 1);
    unsigned long long a = 1 << c->assoc_lg2;

    unsigned long long ctr;

    for (unsigned long long i = set * a; i < a * (set + 1); i++)
        if (c->lines[i].valid && c->lines[i].tag == tag)
        {
            c->lines[i].dirty = true;
            ctr = c->lines[i].use_ctr;
            for (unsigned long long j = set * a; j < a * (set + 1); j++)
                if (c->lines[j].use_ctr > ctr) c->lines[j].use_ctr--;
            c->lines[i].use_ctr = a - 1;
            
            return true;
        }
    
    return false;
}

static struct line cache_lru_set(struct cache *c, unsigned long long addr)
{
    struct line evicted = {.dirty = false, .addr = 0};
    if (!c) return evicted;

    unsigned long long block = addr >> c->width_lg2;
    unsigned long long tag = block >> c->n_set_lg2;
    unsigned long long set = block & ((1 << c->n_set_lg2) - 1);
    unsigned long long a = 1 << c->assoc_lg2;

    // Return immediately if setting a cacheline that is already
    // setted
    for (unsigned long long i = set * a; i < a * (set + 1); i++)
        if (c->lines[i].valid && c->lines[i].tag == tag)
            return evicted;

    for (unsigned long long i = set * a; i < a * (set + 1); i++)
    {
        if (c->lines[i].use_ctr == 0)
        {
            evicted.dirty = c->lines[i].dirty;
            evicted.addr = (c->lines[i].tag << (c->n_set_lg2 + c->width_lg2)) |
                (set << c->width_lg2);
            
            c->lines[i].valid = true;
            c->lines[i].dirty = false;
            c->lines[i].use_ctr = a - 1;
            c->lines[i].tag = tag;
        }
        else c->lines[i].use_ctr--;
    }

    return evicted;
}

#endif