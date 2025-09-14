import struct
import math
import sys
import os
from typing import BinaryIO
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PZ_START  = 0x0
MZ_START  = 0x3E00000000
TZ2_START = 0x3F00000000
TZ1_START = 0x3FC8000000
TZ0_START = 0x3FCA800000

LA_WIN_SIZE = 32
NUM_STRIDES = 256

def open_log_files(folder_path: str) -> list[tuple[str, BinaryIO]] | None:
    fds = []
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            file = open(file_path, "rb")
            fds.append((filename, file))
    except Exception as e:
        print(f"Error: {e}")
        for _, f in fds:
            try:
                f.close()
            except:
                pass
        return None
    return fds

def close_log_files(fds: list[tuple[str, BinaryIO]]):
    for _, f in fds:
        try:
            f.close()
        except:
            pass

def read_last_12_bytes(f: BinaryIO) -> bytes:
    current_pos = f.tell()
    f.seek(-12, 2)
    data = f.read(12)
    f.seek(current_pos)
    return data

def plot_stats_l2_access(acc_name: str, fd: BinaryIO):
    pass

def plot_stats_l2_eviction(acc_name: str, fd: BinaryIO):
    pass

def plot_stats_mac_access(acc_name: str, fd: BinaryIO):
    pass

def plot_stats_mac_eviction(acc_name: str, fd: BinaryIO):
    pass

def plot_stats_vn_access(acc_name: str, fd: BinaryIO):
    pass

def plot_stats_vn_eviction(acc_name: str, fd: BinaryIO):
    pass

def plot_stats_mem_access(acc_name: str, fd: BinaryIO):
    
    # LA_WIN_SIZE must be a power of 2
    assert((LA_WIN_SIZE & (LA_WIN_SIZE-1)) == 0)

    # The list of the ratios of accesses per stride found in the trace
    strides = [0] * NUM_STRIDES
    la_window = [-1] * LA_WIN_SIZE
    la_head = 0
    
    last_clock, _ = struct.unpack(">IQ", read_last_12_bytes(fd))
    EVNTS_MAX_LEN = 1 << 6
    evnt_len = min(EVNTS_MAX_LEN, last_clock + 1)
    bucket_size = ceil((last_clock + 1) / evnt_len)
    events_per_bucket = [0] * evnt_len
    accesses_pz = [0] * 2
    accesses_mz = [0] * 2
    accesses_tz0 = [0] * 2
    accesses_tz1 = [0] * 2
    accesses_tz2 = [0] * 2
    avg_addr_per_bucket = [0] * evnt_len
    addresses_per_bucket = [ [] for _ in range(evnt_len)]
    while True:
        data = fd.read(12)
        if len(data) == 0:
            break
        if len(data) < 12:
            raise EOFError("mem_access.log ended uncorrectly")
        clock, addr_38_r_w_1 = struct.unpack(">IQ", data)
        r_w = addr_38_r_w_1 & 0x1
        addr = addr_38_r_w_1 >> 1

        events_per_bucket[clock // bucket_size] += 1
        if addr < MZ_START:
            accesses_pz[r_w] += 1
        elif addr < TZ2_START:
            accesses_mz[r_w] += 1
        elif addr < TZ1_START:
            accesses_tz2[r_w] += 1
        elif addr < TZ0_START:
            accesses_tz1[r_w] += 1
        else: accesses_tz0[r_w] += 1

        avg_addr_per_bucket[clock // bucket_size] += addr
        addresses_per_bucket[clock // bucket_size].append(addr)
        
        cl = addr >> 9
        distances = [abs(cl - prev_cl) for prev_cl in la_window if prev_cl != -1 and prev_cl != cl]
        if len(distances) > 0:
            cl_dist = min(distances)
            if cl_dist > 0 and cl_dist < NUM_STRIDES:
                strides[cl_dist - 1] += 1

        la_window[la_head & (LA_WIN_SIZE-1)] = cl # To store the cache lines number
        la_head += 1

    avg_addr_per_bucket = [ a / (n+1) for a, n in zip(avg_addr_per_bucket, events_per_bucket) ]
    clocks = [bucket_size * x for x in range(len(events_per_bucket))]
    total = sum(accesses_pz) + sum(accesses_mz) + sum(accesses_tz0) + sum(accesses_tz1) + sum(accesses_tz2)
    avg_acc = total / last_clock

    sp_locality = sum(num/(i+1) for i, num in enumerate(strides)) / total
    print(f"Spatial locality index {sp_locality}")

    plt.figure(figsize=(12, 6))
    plt.bar(clocks, events_per_bucket, width = bucket_size)
    plt.title(f"{acc_name} - HBM accesses")
    plt.xlabel(f"Buckets of {bucket_size} clocks")
    plt.ylabel("Number of accesses per bucket")
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    info_text = (
        f"Num. of clock cycles: $\\bf{{{last_clock:,}}}$\n"
        f"Num. of total accesses to the HBM: $\\bf{{{total:,}}}$\n"
        f"Num. of accesses to the Protected Zone: $\\bf{{{sum(accesses_pz):,}}}$ "
        f"(reads: {accesses_pz[0]:,}, writes: {accesses_pz[1]:,})\n"
        f"Num. of accesses to the MAC Zone: $\\bf{{{sum(accesses_mz):,}}}$ "
        f"(reads: {accesses_mz[0]:,}, writes: {accesses_mz[1]:,})\n"
        f"Num. of accesses to the Tree Zone lvl2: $\\bf{{{sum(accesses_tz2):,}}}$ "
        f"(reads: {accesses_tz2[0]:,}, writes: {accesses_tz2[1]:,})\n"
        f"Num. of accesses to the Tree Zone lvl1: $\\bf{{{sum(accesses_tz1):,}}}$ "
        f"(reads: {accesses_tz1[0]:,}, writes: {accesses_tz1[1]:,})\n"
        f"Num. of accesses to the Tree Zone lvl0: $\\bf{{{sum(accesses_tz0):,}}}$ "
        f"(reads: {accesses_tz0[0]:,}, writes: {accesses_tz0[1]:,})\n"
        f"Avg. num. of accesses per clock cycle: $\\bf{{{avg_acc:.10f}}}$"
    )
    ax = plt.gca()
    max_y = max(events_per_bucket)
    x_padding = 0.01 * (clocks[-1] - clocks[0])
    x_pos = clocks[-1] - x_padding

    # Stima altezza post-it
    info_lines = info_text.count('\n') + 1
    line_height = 0.04 * max_y  # Empirico: 4% del max per riga
    postit_height = info_lines * line_height

    # Posizione del post-it sopra le barre
    y_pos = max_y + postit_height # Inizia leggermente sopra

    # Aggiusta y-limits per includere il post-it
    ax.set_ylim(top=max_y + postit_height * 1.2)

    # Aggiungi testo
    ax.text(
        x_pos, y_pos, info_text,
        ha='right', va='top',
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9)
    )
    
    plt.savefig(f"{acc_name}_mem_access.pdf", format="pdf", bbox_inches="tight")

    plt.figure(figsize=(12, 6))

    # Formatter per asse Y in esadecimale
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"log10(1 + 0x{int(10**x - 1):X})"))

    plt.grid(True, axis='both')
    plt.tight_layout()

    bpdata = [[math.log10(1 + elem) for elem in bucket] for bucket in addresses_per_bucket]
    # Disegno del boxplot
    bp = plt.boxplot(
        bpdata,
        positions = clocks,
        widths = bucket_size * 0.8,
        vert = True,
        whis = (0, 100),  # usiamo i valori estremi come whiskers
        patch_artist = True
    )
    plt.ylim(-1, math.log10(274877906943) + 1)
    plt.xticks(ticks=clocks, labels=[str(i+1) for i in range(len(clocks))])
    
    plt.title(f"{acc_name} - Box plot of accessed addresses per bucket")
    plt.xlabel(f"Buckets of {bucket_size} clocks")
    plt.ylabel("Addresses in log scale")
    plt.grid(True, axis='y')
    plt.tight_layout()

    plt.savefig(f"{acc_name}_mem_access_range.pdf", format="pdf", bbox_inches="tight")

def plot_stats(acc_name: str, filename: str, fd: BinaryIO):
    if filename == "L2_access.log":
        plot_stats_l2_access(acc_name, fd)
    elif filename == "L2_eviction.log":
        plot_stats_l2_eviction(acc_name, fd)
    elif filename == "MAC_access.log":
        plot_stats_mac_access(acc_name, fd)
    elif filename == "MAC_eviction.log":
        plot_stats_mac_eviction(acc_name, fd)
    elif filename == "VN_access.log":
        plot_stats_vn_access(acc_name, fd)
    elif filename == "VN_eviction.log":
        plot_stats_vn_eviction(acc_name, fd)
    elif filename == "mem_access.log":
        plot_stats_mem_access(acc_name, fd)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <backend_output_folder_path>")
        return

    fds = open_log_files(sys.argv[1])
    if fds == None:
        return

    acc_name = "acc" + os.path.basename(os.path.normpath(sys.argv[1]))[3:]
    
    for filename, fd in fds:
        plot_stats(acc_name, filename, fd)
    plt.show()
    close_log_files(fds)

if __name__ == "__main__":
    main()
