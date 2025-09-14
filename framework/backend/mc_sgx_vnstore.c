#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include "logger.h"
#include "cache.h"
#include "vnstore.h"
#include "mc.h"
#include "virtmmu.h"

#ifdef VNSTORE512
#define IS_512 1
#else
#define IS_512 0
#endif

static ostream *mem_acc_log, *MAC_acc_log, *MAC_evct_log,
    *VNCACHE_acc_log, *VNCACHE_evct_log,
    *VNSTORE_acc_log, *VNSTORE_evct_log;
static struct cache *MACs, *VNCACHE;
static struct vnstore *VNSTORE;
static struct ranges *map;
static uint64_t (*VN_virt_storage)[6501172][80];

int mc_arch_init(const char *name, struct arena *a, struct ranges *r)
{
    map = r;
    int dir_fd;
    if ((mkdir(name, 0777) && errno != EEXIST) ||
        (dir_fd = open(name, O_RDONLY | O_DIRECTORY)) < 0)
    {
        fprintf(stderr, "Error while creating output directory %s\n", name);
        return -1;
    }

    VN_virt_storage = new(a, typeof(*VN_virt_storage), 1);
    if (!VN_virt_storage)
    {
        fputs("Error while allocating VN backing storage\n", stderr);
        return 1;
    }

    mem_acc_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "mem_access.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), mem_access_log_t);
    MAC_acc_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "MAC_access.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), cache_access_log_t);
    MAC_evct_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "MAC_eviction.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), cache_eviction_log_t);
    VNCACHE_acc_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "VN_access.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), cache_access_log_t);
    VNCACHE_evct_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "VN_eviction.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), cache_eviction_log_t);
    VNSTORE_acc_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "VNSTORE_access.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), cache_access_log_t);
    VNSTORE_evct_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "VNSTORE_eviction.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), vnstore_eviction_log_t);

    close(dir_fd);

    if (!mem_acc_log        || !MAC_acc_log     ||
        !MAC_evct_log       || !VNCACHE_acc_log ||
        !VNCACHE_evct_log   || !VNSTORE_acc_log ||
        !VNSTORE_evct_log)
    {
        fputs("Error while creating output trace files\n", stderr);
        return -1;
    }

    if (posix_fadvise(mem_acc_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL)      ||
        posix_fadvise(MAC_acc_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL)      ||
        posix_fadvise(MAC_evct_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL)     ||
        posix_fadvise(VNCACHE_acc_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL)  ||
        posix_fadvise(VNCACHE_evct_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL) ||
        posix_fadvise(VNSTORE_acc_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL)  ||
        posix_fadvise(VNSTORE_evct_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL))
    {
        fputs("posix_fadvise(s) failed\n", stderr);
    }

    // 64 KB MACs cache: 2-way cache of 128 lines of 512 Byte
    MACs = cache_make(9, 7, 1, a);
    if (!MACs)
    {
        fputs("Error while creating MACs cache\n", stderr);
        return -1;
    }

    // 64 KB VN/CTR cache: 4-way cache of 128 lines of 512 Byte
    VNCACHE = cache_make(9, 7, 2, a);
    if (!VNCACHE)
    {
        fputs("Error while creating VNCACHE\n", stderr);
        return -1;
    }

    // VNSTORE: 256 or 512 lines depending on VNSTORE512 being defined
    VNSTORE = vns_make(8 + IS_512, 9, PZ_START, MZ_START, a);
    if (!VNSTORE)
    {
        fputs("Error while creating VNSTORE\n", stderr);
        return -1;
    }

    return 0;
}

void mc_arch_free(void)
{
    ostream_close(mem_acc_log);
    ostream_close(MAC_acc_log);
    ostream_close(MAC_evct_log);
    ostream_close(VNCACHE_acc_log);
    ostream_close(VNCACHE_evct_log);
    ostream_close(VNSTORE_acc_log);
    ostream_close(VNSTORE_evct_log);
    close(mem_acc_log->fd);
    close(MAC_acc_log->fd);
    close(MAC_evct_log->fd);
    close(VNCACHE_acc_log->fd);
    close(VNCACHE_evct_log->fd);
    close(VNSTORE_acc_log->fd);
    close(VNSTORE_evct_log->fd);
}

static inline uint64_t read_vn_from_virt_storage(unsigned long long pz_phy_addr)
{
    unsigned long long pz_cl_num = pz_phy_addr >> 9;

    return (*VN_virt_storage)[pz_cl_num / 80][pz_cl_num % 80];
}

static inline void write_vn_to_virt_storage(unsigned long long pz_phy_addr, uint64_t vn)
{
    unsigned long long pz_cl_num = pz_phy_addr >> 9;
    (*VN_virt_storage)[pz_cl_num / 80][pz_cl_num % 80] = vn;
}

static void
access_mac(unsigned long long clock, unsigned long long pz_addr, unsigned char r_w)
{
    const unsigned long long pz_cl_num = pz_addr >> 9;
    const unsigned long long mz_cl_num = pz_cl_num >> 6;
    const unsigned long long mz_cl_addr = MZ_START | (mz_cl_num << 9);
    bool hit;
    if (r_w == 0) hit = cache_read(MACs, mz_cl_addr);
    else hit = cache_write(MACs, mz_cl_addr);
    CACHE_ACCESS_LOG(MAC_acc_log, clock, mz_cl_addr, r_w == 0 ? R : W, hit);
    if (!hit)
    {
        // MAC cache miss, MEM access to bring cache line in MAC cache
        MEM_ACCESS_LOG(mem_acc_log, clock, mz_cl_addr, R);
        struct line evicted = cache_lru_set(MACs, mz_cl_addr);
        if (r_w == 1) cache_write(MACs, mz_cl_addr);
        if (evicted.dirty)
        {
            // MAC cache writes the evicted line to MEM to make space for the new one
            CACHE_EVICTION_LOG(MAC_evct_log, clock, evicted.addr);
            MEM_ACCESS_LOG(mem_acc_log, clock, evicted.addr, W);
        }
    }
}

static void
update_vn_tree_path_to_vncache(unsigned long long clock, unsigned long long pz_addr)
{
    static const unsigned long long tz_start_addrs[3] = {TZ2_START, TZ1_START, TZ0_START};
    const unsigned long long pz_cl_num = pz_addr >> 9;
    unsigned long long tz_cl_num = pz_cl_num / 80;
    unsigned long long tz_cl_addr;
    bool hit;
    for (int i = 0; i < 3; i++)
    {
        tz_cl_addr = tz_start_addrs[i] | (tz_cl_num << 9);
        hit = cache_write(VNCACHE, tz_cl_addr);
        CACHE_ACCESS_LOG(VNCACHE_acc_log, clock, tz_cl_addr, W, hit);
        tz_cl_num = tz_cl_num / 80;
        // VN cache hit, jump to next CTR
        if (hit) continue;
        // VN cache miss, MEM access to bring cache line in VN cache
        MEM_ACCESS_LOG(mem_acc_log, clock, tz_cl_addr, R);
        struct line evicted = cache_lru_set(VNCACHE, tz_cl_addr);
        cache_write(VNCACHE, tz_cl_addr);
        if (evicted.dirty)
        {
            // VN cache writes the evicted line to MEM to make space for the new one
            CACHE_EVICTION_LOG(VNCACHE_evct_log, clock, evicted.addr);
            MEM_ACCESS_LOG(mem_acc_log, clock, evicted.addr, W);
        }
    }
    // One more access with respect to the last tz_cl_num in the trusted on-chip storage
    // We don't log it since it's on-chip
}

static void
read_and_verify_vn_tree_path_to_vncache(unsigned long long clock, unsigned long long pz_addr)
{
    static const unsigned long long tz_start_addrs[3] = {TZ2_START, TZ1_START, TZ0_START};
    const unsigned long long pz_cl_num = pz_addr >> 9;
    unsigned long long tz_cl_num = pz_cl_num / 80;
    unsigned long long tz_cl_addr;
    bool hit;
    for (int i = 0; i < 3; i++)
    {
        tz_cl_addr = tz_start_addrs[i] | (tz_cl_num << 9);
        hit = cache_read(VNCACHE, tz_cl_addr);
        CACHE_ACCESS_LOG(VNCACHE_acc_log, clock, tz_cl_addr, R, hit);
        tz_cl_num = tz_cl_num / 80;
        // VN cache hit, tree walk ends
        if (hit) return;
        // VN cache miss, MEM access to bring cache line in VN cache
        MEM_ACCESS_LOG(mem_acc_log, clock, tz_cl_addr, R);
        struct line evicted = cache_lru_set(VNCACHE, tz_cl_addr);
        if (evicted.dirty)
        {
            // VN cache writes the evicted line to MEM to make space for the new one
            CACHE_EVICTION_LOG(VNCACHE_evct_log, clock, evicted.addr);
            MEM_ACCESS_LOG(mem_acc_log, clock, evicted.addr, W);
        }
    }
    // One more access with respect to the last tz_cl_num in the trusted on-chip storage
    // We don't log it since it's on-chip
}

static long long
read_and_verify_vn_tree_path(unsigned long long clock, unsigned long long pz_virt_addr,
    unsigned long long pz_phy_addr)
{
    // If VN is in the VNSTORE, just return it
    // NOTE: VNSTORE works with virtual addresses
    long long vn = vns_get(VNSTORE, pz_virt_addr);
    VNSTORE_ACCESS_LOG(VNSTORE_acc_log, clock, pz_virt_addr, R, vn >= 0 ? 1 : 0);
    if (vn >= 0) return vn;

    // If not in the VNSTORE, read from the VNCACHE
    // NOTE: VNCACHE works with physical addresses
    read_and_verify_vn_tree_path_to_vncache(clock, pz_phy_addr);

    // Actually obtain the vn from the virtual storage
    vn = read_vn_from_virt_storage(pz_phy_addr);

    // Here we set the VNSTORE with the PZ cache line retrieved VN
    struct vnsl evicted[2];
    vns_set(VNSTORE, pz_virt_addr, vn, (struct vnsl (*)[2])evicted);
    VNSTORE_ACCESS_LOG(VNSTORE_acc_log, clock, pz_virt_addr, W, 1);
    for (int i = 0; i < 2; i++)
        // Invalid or clean lines are not written to MEM
        if (evicted[i].ctr > 0)
        {
            VNSTORE_EVICTION_LOG(VNSTORE_evct_log, clock,
                evicted[i].rng_start, evicted[i].rng_end, evicted[i].ctr);
            long long phy_addr;
            while (evicted[i].rng_start < evicted[i].rng_end)
            {
                phy_addr = virt_to_phy(map, evicted[i].rng_start);
                mc_check(phy_addr != -1, "%#llX generated a phy_addr of -1\n", evicted[i].rng_start);
                write_vn_to_virt_storage(phy_addr, evicted[i].ctr);
                update_vn_tree_path_to_vncache(clock, phy_addr);
                evicted[i].rng_start += 512;
            }
        }

    return vn;
}

static void
update_vn_tree_path(unsigned long long clock, unsigned long long pz_virt_addr, 
    unsigned long long pz_phy_addr, long long vn)
{
    struct vnsl evicted[2];
    vns_set(VNSTORE, pz_virt_addr, vn, (struct vnsl (*)[2])evicted);
    VNSTORE_ACCESS_LOG(VNSTORE_acc_log, clock, pz_virt_addr, W, 1);
    // Soft update of the VNCACHE: misses are ignored not
    // to augment pressure on DRAM
    static const unsigned long long tz_start_addrs[3] = {TZ2_START, TZ1_START, TZ0_START};
    const unsigned long long pz_cl_num = pz_phy_addr >> 9;
    unsigned long long tz_cl_num = pz_cl_num / 80;
    unsigned long long tz_cl_addr;
    bool hit;
    for (int i = 0; i < 3; i++)
    {
        tz_cl_addr = tz_start_addrs[i] | (tz_cl_num << 9);
        hit = cache_write(VNCACHE, tz_cl_addr);
        CACHE_ACCESS_LOG(VNCACHE_acc_log, clock, tz_cl_addr, W, hit);
        tz_cl_num = tz_cl_num / 80;
    }

    for (int i = 0; i < 2; i++)
        // Invalid or clean lines are not written to MEM
        if (evicted[i].ctr > 0)
        {
            VNSTORE_EVICTION_LOG(VNSTORE_evct_log, clock,
                evicted[i].rng_start, evicted[i].rng_end, evicted[i].ctr);
            long long phy_addr;
            while (evicted[i].rng_start < evicted[i].rng_end)
            {
                phy_addr = virt_to_phy(map, evicted[i].rng_start);
                mc_check(phy_addr != -1, "phy_addr was -1\n");
                write_vn_to_virt_storage(phy_addr, evicted[i].ctr);
                update_vn_tree_path_to_vncache(clock, phy_addr);
                evicted[i].rng_start += 512;
            }
        }
}

void mc_arch_dram_access(unsigned long long clock, unsigned long long virt_addr,
    unsigned long long phy_addr, unsigned char r_w)
{
    MEM_ACCESS_LOG(mem_acc_log, clock, phy_addr, r_w == 0 ? R : W);
    access_mac(clock, phy_addr, 0);
    long long vn = read_and_verify_vn_tree_path(clock, virt_addr, phy_addr);
    mc_check(vn >= 0, "read_and_verify_vn_tree_path failed\n");
    if (r_w == 0) return;
    
    // Update MAC with new one and update the VN tree
    access_mac(clock, phy_addr, 1);
    update_vn_tree_path(clock, virt_addr, phy_addr, vn + 1);
}