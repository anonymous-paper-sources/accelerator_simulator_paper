CC      = gcc
CFLAGS  += -Iblocks -Iutils -Wall -Wextra -Wno-unused-function -Wno-unused-but-set-variable
HEADERS	:= $(wildcard *.h) $(wildcard blocks/*.h) $(wildcard utils/*.h)
MKDIR	= @mkdir -p $(@D)

all: CFLAGS += -O2 -DNDEBUG
all: acc_plain acc_mac acc_sgx acc_sgx_vnstore_256 acc_sgx_vnstore_512

verbose: CFLAGS += -DVERBOSE_LOGGING
verbose: all

debug: CFLAGS += -O0 -g
debug: acc_plain acc_mac acc_sgx acc_sgx_vnstore_256 acc_sgx_vnstore_512

acc_plain: obj/main.o obj/mc.o obj/mc_plain.o obj/stream.o
	$(CC) $^ -o $@

acc_mac: obj/main.o obj/mc.o obj/mc_mac.o obj/stream.o
	$(CC) $^ -o $@

acc_sgx: obj/main.o obj/mc.o obj/mc_sgx.o obj/stream.o
	$(CC) $^ -o $@

acc_sgx_vnstore_256: obj/main.o obj/mc.o obj/mc_sgx_vnstore.o obj/stream.o
	$(CC) $^ -o $@

acc_sgx_vnstore_512: CFLAGS += -DVNSTORE512
acc_sgx_vnstore_512: obj/main.o obj/mc.o obj/mc_sgx_vnstore_512.o obj/stream.o
	$(CC) $^ -o $@

obj/%_512.o: %.c $(HEADERS)
	$(MKDIR)
	$(CC) $(CFLAGS) -c $< -o $@

obj/%.o: %.c $(HEADERS)
	$(MKDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj acc_* log_*