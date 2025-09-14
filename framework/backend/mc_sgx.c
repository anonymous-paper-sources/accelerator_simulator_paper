#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include "logger.h"
#include "cache.h"
#include "virtmmu.h"
#include "mc.h"

static ostream *mem_acc_log, *MAC_acc_log, *MAC_evct_log, *VN_acc_log, *VN_evct_log;
static struct cache *MACs, *VNs;

int mc_arch_init(const char *name, struct arena *a, struct ranges *r)
{
    (void)r;
    int dir_fd;
    if ((mkdir(name, 0777) && errno != EEXIST) ||
        (dir_fd = open(name, O_RDONLY | O_DIRECTORY)) < 0)
    {
        fprintf(stderr, "Error while creating output directory %s\n", name);
        return -1;
    }

    mem_acc_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "mem_access.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), mem_access_log_t);
    MAC_acc_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "MAC_access.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), cache_access_log_t);
    MAC_evct_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "MAC_eviction.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), cache_eviction_log_t);
    VN_acc_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "VN_access.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), cache_access_log_t);
    VN_evct_log = ostream_open(alloc(a, STREAM_BUF_SIZE, 1), STREAM_BUF_SIZE,
        openat(dir_fd, "VN_eviction.log", O_CREAT | O_WRONLY | O_TRUNC, 0666), cache_eviction_log_t);

    close(dir_fd);

    if (!mem_acc_log || !MAC_acc_log || !MAC_evct_log ||
        !VN_acc_log  || !VN_evct_log)
    {
        fputs("Error while creating output trace files\n", stderr);
        return -1;
    }

    if (posix_fadvise(mem_acc_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL)  ||
        posix_fadvise(MAC_acc_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL)  ||
        posix_fadvise(MAC_evct_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL) ||
        posix_fadvise(VN_acc_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL)   ||
        posix_fadvise(VN_evct_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL))
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
    VNs = cache_make(9, 7, 2, a);
    if (!VNs)
    {
        fputs("Error while creating VNs cache\n", stderr);
        return -1;
    }

    return 0;
}

void mc_arch_free(void)
{
    ostream_close(mem_acc_log);
    ostream_close(MAC_acc_log);
    ostream_close(MAC_evct_log);
    ostream_close(VN_acc_log);
    ostream_close(VN_evct_log);
    close(mem_acc_log->fd);
    close(MAC_acc_log->fd);
    close(MAC_evct_log->fd);
    close(VN_acc_log->fd);
    close(VN_evct_log->fd);
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
read_and_verify_vn_tree_path(unsigned long long clock, unsigned long long pz_addr)
{
    static const unsigned long long tz_start_addrs[3] = {TZ2_START, TZ1_START, TZ0_START};
    const unsigned long long pz_cl_num = pz_addr >> 9;
    unsigned long long tz_cl_num = pz_cl_num / 80;
    unsigned long long tz_cl_addr;
    bool hit;
    for (int i = 0; i < 3; i++)
    {
        tz_cl_addr = tz_start_addrs[i] | (tz_cl_num << 9);
        hit = cache_read(VNs, tz_cl_addr);
        CACHE_ACCESS_LOG(VN_acc_log, clock, tz_cl_addr, R, hit);
        tz_cl_num = tz_cl_num / 80;
        // VN cache hit, tree walk ends
        if (hit) return;
        // VN cache miss, MEM access to bring cache line in VN cache
        MEM_ACCESS_LOG(mem_acc_log, clock, tz_cl_addr, R);
        struct line evicted = cache_lru_set(VNs, tz_cl_addr);
        if (evicted.dirty)
        {
            // VN cache writes the evicted line to MEM to make space for the new one
            CACHE_EVICTION_LOG(VN_evct_log, clock, evicted.addr);
            MEM_ACCESS_LOG(mem_acc_log, clock, evicted.addr, W);
        }
    }
    // One more access with respect to the last tz_cl_num in the trusted on-chip storage
    // We don't log it since it's on-chip
}

static void
update_vn_tree_path(unsigned long long clock, unsigned long long pz_addr)
{
    static const unsigned long long tz_start_addrs[3] = {TZ2_START, TZ1_START, TZ0_START};
    const unsigned long long pz_cl_num = pz_addr >> 9;
    unsigned long long tz_cl_num = pz_cl_num / 80;
    unsigned long long tz_cl_addr;
    bool hit;
    for (int i = 0; i < 3; i++)
    {
        tz_cl_addr = tz_start_addrs[i] | (tz_cl_num << 9);
        hit = cache_write(VNs, tz_cl_addr);
        CACHE_ACCESS_LOG(VN_acc_log, clock, tz_cl_addr, W, hit);
        tz_cl_num = tz_cl_num / 80;
        // VN cache hit, jump to next CTR
        if (hit) continue;
        // VN cache miss, MEM access to bring cache line in VN cache
        MEM_ACCESS_LOG(mem_acc_log, clock, tz_cl_addr, R);
        struct line evicted = cache_lru_set(VNs, tz_cl_addr);
        cache_write(VNs, tz_cl_addr);
        if (evicted.dirty)
        {
            // VN cache writes the evicted line to MEM to make space for the new one
            CACHE_EVICTION_LOG(VN_evct_log, clock, evicted.addr);
            MEM_ACCESS_LOG(mem_acc_log, clock, evicted.addr, W);
        }
    }
    // One more access with respect to the last tz_cl_num in the trusted on-chip storage
    // We don't log it since it's on-chip
}

void mc_arch_dram_access(unsigned long long clock, unsigned long long virt_addr,
    unsigned long long phy_addr, unsigned char r_w)
{
    (void)virt_addr;
    MEM_ACCESS_LOG(mem_acc_log, clock, phy_addr, r_w == 0 ? R : W);
    // For each memory read we need to:
    // 1) Read the MAC from the MAC cache or from MEM if missing
    // 2) If read from MEM, also set it in the cache
    // 3) Read the VN from the VN cache or start the recursive
    //      read of the VN integrity tree from MEM up to
    //      the first trusted verification counter
    //      (VN cache or on chip VN root)
    // 4) For each VN read from MEM, set it in the cache
    // 
    // For each memory write we need to:
    // 1) Execute the steps of a read to validate the current memory line in MEM
    // 2) Calculate the new MAC with the updated VN
    // 4) Store the updated MAC in MEM
    // 5) Store the updated memory line in MEM
    // 6) Update the VN and all the verification counters
    //      in the VN cache if present or in MEM if
    //      in case of a VN cache miss
    access_mac(clock, phy_addr, 0);
    read_and_verify_vn_tree_path(clock, phy_addr);

    if (r_w == 0) return;
    
    // Update MAC with new one and update the VN tree
    access_mac(clock, phy_addr, 1);
    update_vn_tree_path(clock, phy_addr);
}
