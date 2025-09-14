#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include "mc.h"

static char *exec_name_from_path(char *path)
{
    char *e = path;
    while (*path)
        if (*path++ == '/') e = path;
    
    return e;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fputs("Input memory trace filename missing\n", stderr);
        return 1;
    }

    int trace_fd = open(argv[1], O_RDONLY, 0666);
    if (trace_fd < 0)
    {
        fputs("Can't open memory trace file\n", stderr);
        return 1;
    }

    int res = mc_elaborate_trace(exec_name_from_path(argv[0]), trace_fd);
    close(trace_fd);

    return res;
}
