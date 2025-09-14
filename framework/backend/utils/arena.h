#ifndef ARENA_H
#define ARENA_H

#include <stdint.h>
#include <stddef.h>

#define new(arena, type, number) (type *)alloc((arena), sizeof(type)*(number), _Alignof(type))

struct arena
{
    char *beg;
    char *end;
};

static void *alloc(struct arena *a, ptrdiff_t size, ptrdiff_t align)
{
    if (!a) return NULL;
    ptrdiff_t padding = -(uintptr_t)a->beg & (align - 1);
    ptrdiff_t available = a->end - a->beg - padding - size;
    if (available < 0) return NULL;
    void *p = a->beg + padding;
    a->beg += padding + size;
    return p;
}

#endif