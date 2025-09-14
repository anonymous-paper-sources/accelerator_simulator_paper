#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include "logger.h"
#include "cache.h"
#include "virtmmu.h"
#include "mc.h"

static ostream *mem_acc_log, *MAC_acc_log, *MAC_evct_log;
static struct cache *MACs;

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

    close(dir_fd);

    if (!mem_acc_log || !MAC_acc_log || !MAC_evct_log)
    {
        fputs("Error while creating output trace files\n", stderr);
        return -1;
    }

    if (posix_fadvise(mem_acc_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL)  ||
        posix_fadvise(MAC_acc_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL)  ||
        posix_fadvise(MAC_evct_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL))
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

    return 0;
}

void mc_arch_free(void)
{
    ostream_close(mem_acc_log);
    ostream_close(MAC_acc_log);
    ostream_close(MAC_evct_log);
    close(mem_acc_log->fd);
    close(MAC_acc_log->fd);
    close(MAC_evct_log->fd);
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

void mc_arch_dram_access(unsigned long long clock, unsigned long long virt_addr,
    unsigned long long phy_addr, unsigned char r_w)
{
    (void)virt_addr;
    MEM_ACCESS_LOG(mem_acc_log, clock, phy_addr, r_w == 0 ? R : W);
    // For each memory read we need to:
    // 1) Read the MAC from the MAC cache or from MEM if missing
    // 2) If read from MEM, also set it in the cache
    // 
    // For each memory write we need to:
    // 1) Execute the steps of a read to validate the current memory line in MEM
    // 2) Calculate the new MAC with the updated VN
    // 4) Store the updated MAC in MEM
    // 5) Store the updated memory line in MEM
    access_mac(clock, phy_addr, 0);

    if (r_w == 0) return;
    
    // Update MAC with new one
    access_mac(clock, phy_addr, 1);
}