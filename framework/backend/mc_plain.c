#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include "virtmmu.h"
#include "mc.h"
#include "logger.h"
#include "arena.h"
#include "stream.h"

static ostream *mem_acc_log = 0;

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

    close(dir_fd);

    if (!mem_acc_log)
    {
        fputs("Error while creating output trace files\n", stderr);
        return -1;
    }

    if (posix_fadvise(mem_acc_log->fd, 0, 0, POSIX_FADV_SEQUENTIAL))
    {
        fputs("posix_fadvise(s) failed\n", stderr);
    }

    return 0;
}

void mc_arch_free(void)
{
    ostream_close(mem_acc_log);
    close(mem_acc_log->fd);
}

void mc_arch_dram_access(unsigned long long clock, unsigned long long virt_addr,
    unsigned long long phy_addr, unsigned char r_w)
{
    (void)virt_addr;
    MEM_ACCESS_LOG(mem_acc_log, clock, phy_addr, r_w == 0 ? R : W);
}