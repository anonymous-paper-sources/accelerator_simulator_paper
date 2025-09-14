#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "logger.h"
#include "mc.h"
#include "stream.h"
#include "arena.h"
#include "cache.h"
#include "virtmmu.h"

struct mc_base
{
    unsigned long long clock;
    istream *trace_s;
    ostream *L2_evct_log;
    struct cache *L2;
    struct ranges *r;
    struct arena a;
};

static struct mc_base mc = {0};

int mc_arch_init(const char *name, struct arena *a, struct ranges *r);
void mc_arch_free(void);
void mc_arch_dram_access(unsigned long long clock, unsigned long long virt_addr,
    unsigned long long phy_addr, unsigned char r_w);

static int mc_init(char *name, int trace_fd)
{
    mc.a.beg = malloc(GLOBAL_ARENA_SIZE);
    mc.a.end = mc.a.beg + GLOBAL_ARENA_SIZE;
    if (!mc.a.beg)
    {
        fputs("Error while allocating arena memory\n", stderr);
        return -1;
    }

    name[0] = 'l'; name[1] = 'o'; name[2] = 'g';
    int dir_fd;
    if ((mkdir(name, 0777) && errno != EEXIST) ||
        (dir_fd = open(name, O_RDONLY | O_DIRECTORY)) < 0)
    {
        fprintf(stderr, "Error while creating output directory %s\n", name);
        return -1;
    }

    mc.L2_evct_log = ostream_open(alloc(&mc.a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "L2_eviction.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), cache_eviction_log_t);

    close(dir_fd);

    if (!mc.L2_evct_log)
    {
        fputs("Error while creating output trace files\n", stderr);
        return -1;
    }

    if (posix_fadvise(mc.L2_evct_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL))
    {
        fputs("posix_fadvise(s) failed\n", stderr);
    }

    // 64 MB L2 cache: 16-way cache of 128 K lines of 512 Byte
    mc.L2 = cache_make(9, 17, 4, &mc.a);
    if (!mc.L2)
    {
        fputs("Error while creating L2 cache\n", stderr);
        return -1;
    }

    if (posix_fadvise(trace_fd, 0, 0, POSIX_FADV_SEQUENTIAL))
    {
        fputs("posix_fadvise failed\n", stderr);
    }
    void *trace_s_buf = alloc(&mc.a, STREAM_BUF_SIZE, 1);
    mc.trace_s = istream_open(trace_s_buf, STREAM_BUF_SIZE, trace_fd, uint64_t);
    if (!mc.trace_s)
    {
        fputs("Can't open istream on trace file\n", stderr);
        return -1;
    }

    mc.r = ranges_from_trace(mc.trace_s, &mc.a);
    if (!mc.r)
    {
        fputs("ranges_from_trace returned NULL\n", stderr);
        return -1;
    }
    
    if (lseek(trace_fd, 0, SEEK_SET) == -1)
    {
        fputs("Error repositioning the trace file offset\n", stderr);
        return -1;
    }

    // NOTE: istream_close(trace_s_buf) will be needed if multithreaded stream
    // implementation is used
    mc.trace_s = istream_open(trace_s_buf, STREAM_BUF_SIZE, trace_fd, uint64_t);
    if (!mc.trace_s)
    {
        fputs("Can't open istream on trace file\n", stderr);
        return -1;
    }

    return mc_arch_init(name, &mc.a, mc.r);
}

static void mc_free(void)
{
    mc_arch_free();
    ostream_close(mc.L2_evct_log);
    close(mc.L2_evct_log->fd);
    printf("[ACCSIM] Total used memory: %llu Bytes\n", GLOBAL_ARENA_SIZE - (mc.a.end - mc.a.beg));
    free(mc.a.end - GLOBAL_ARENA_SIZE);
}

static int mc_memop(unsigned long long virt_addr, unsigned char r_w)
{
    uint64_t phy_addr = virt_to_phy(mc.r, virt_addr);
    mc_check(phy_addr != (uint64_t)-1, "phy_addr was -1\n");
    if (phy_addr >= MZ_START) return -1;
    
    unsigned long long clock = mc.clock++;
    // It's useless to model L1 given that:
    // Accesses to DRAM (the real thing we are interested in)
    // are unchanged with/without a L1
    bool hit;
    struct line evicted;

    if (r_w == 0) hit = cache_read(mc.L2, virt_addr);
    else hit = cache_write(mc.L2, virt_addr);
    if (!hit)
    {
        // L2 miss, MEM access to bring cache line in L2
        mc_arch_dram_access(clock, virt_addr, phy_addr, 0);
        evicted = cache_lru_set(mc.L2, virt_addr);
        if (r_w == 1) cache_write(mc.L2, virt_addr);
        if (evicted.dirty)
        {
            // L2 writes the evicted line to MEM to make space for the new one
            // NOTE: For the evicted line we need to calculate the physical address
            phy_addr = virt_to_phy(mc.r, evicted.addr);
            mc_check(phy_addr != (uint64_t)-1, "phy_addr was -1\n");
            CACHE_EVICTION_LOG(mc.L2_evct_log, clock, evicted.addr);
            mc_arch_dram_access(clock, evicted.addr, phy_addr, 1);
        }
    }

    return 0;
}

int mc_elaborate_trace(char *name, int trace_fd)
{
    if (mc_init(name, trace_fd)) return -1;

    uint64_t *be_addr_49_r_w_1;
    while (!mc.trace_s->eos)
    {
        ptrdiff_t count = 128;
        be_addr_49_r_w_1 = istream_pop(mc.trace_s, &count);
        for (ptrdiff_t i = 0; i < count; i++)
        {
            be_addr_49_r_w_1[i] = ntohll(be_addr_49_r_w_1[i]);
            uint64_t virt_addr = be_addr_49_r_w_1[i] >> 1;
            unsigned char r_w = be_addr_49_r_w_1[i] & 1;
            if (mc_memop(virt_addr, r_w))
            {
                fputs("Error in mc_do_memop\n", stderr);
                return -1;
            }
        }
    }

    mc_free();
    return 0;
}