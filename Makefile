CC ?= gcc
CFLAGS ?= -O3
build: hrp_bench
hrp_bench: bench.c
	$(CC) $(CFLAGS) -o hrp_bench bench.c -lm
run: hrp_bench
	./hrp_bench
clean:
	rm -f hrp_bench
.PHONY: build run clean
