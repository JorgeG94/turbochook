# =============================================================================
# spikes/Makefile — deliberately NOT CMake.
#
# These run on someone else's GPU box, possibly behind a module system, possibly
# with a CMake older than doctest wants. A flat Makefile with one target per
# toolchain question has no configure step to go wrong, and every command it runs
# is visible in the output so a failure can be pasted back verbatim.
#
#   make s04              host-only, no GPU        (run this first, costs nothing)
#   make s04_negative     MUST FAIL to compile
#   make s01              nvc++ -stdpar=gpu, managed pool + prefetch
#   make s02              nvc++ -stdpar=gpu -gpu=mem:separate, cudaMalloc pool
#   make s03              nvc++ -stdpar=gpu -cuda, one pool + two launchers
#   make s03b             g++ host + nvcc device layer
#
#   make all              everything (s02 is allowed to fail — that IS a result)
#   make run              build + run each spike that built
#
# Override as needed:  make s01 ARCH=cc90 NVCXX=nvc++ N=4096 ITERS=500
# =============================================================================

# Make variables are case-sensitive, so `make run arch=cc90` would silently build
# for the DEFAULT arch and hand you a binary for the wrong card. Fail instead.
ifdef arch
$(error you passed arch=$(arch); the variable is ARCH (upper case). Use: make run ARCH=$(arch))
endif

ARCH   ?= cc80
NVCXX  ?= nvc++
NVCC   ?= nvcc
CXX    ?= g++
STD    ?= -std=c++23
OPT    ?= -O2
BIN    ?= build

N      ?= 2048
ITERS  ?= 200

$(shell mkdir -p $(BIN))

.PHONY: all run clean s01 s02 s03 s03b s04 s04_negative

# ── 04: host only. No GPU, no CUDA, no NVHPC. ────────────────────────────────
s04: $(BIN)/s04
$(BIN)/s04: 04_space_tags.cpp
	$(CXX) $(STD) $(OPT) $< -o $@

# This target must FAIL. If it links, the compile-time gate is not live.
s04_negative: 04_space_tags.cpp
	@echo "--- expecting a static_assert failure below ---"
	@if $(CXX) $(STD) $(OPT) -DTC_NEGATIVE $< -o $(BIN)/s04_neg 2>$(BIN)/s04_neg.log; then \
	  echo "*** FAIL: the negative test COMPILED — host access to Space::Device is not gated"; \
	  exit 1; \
	else \
	  echo "PASS: rejected at compile time. First diagnostic:"; \
	  grep -m1 -i "static_assert\|static assertion" $(BIN)/s04_neg.log || head -3 $(BIN)/s04_neg.log; \
	fi

# ── 01: managed pool + prefetch under stdpar (the expected load-bearing one) ──
s01: $(BIN)/s01
$(BIN)/s01: 01_managed_pool_stdpar.cpp common.hpp
	$(NVCXX) $(STD) $(OPT) -stdpar=gpu -gpu=$(ARCH) -cuda -DTC_SPIKE_CUDA $< -o $@

# ── 02: device-only pool under stdpar. A COMPILE FAILURE HERE IS A RESULT. ────
#    Capture the exact message; it answers whether -stdpar=gpu tolerates
#    mem:separate at all.
s02: $(BIN)/s02
$(BIN)/s02: 02_device_pool_stdpar.cpp common.hpp
	-$(NVCXX) $(STD) $(OPT) -stdpar=gpu -gpu=$(ARCH),mem:separate -cuda \
	    -DTC_SPIKE_CUDA $< -o $@ 2>&1 | tee $(BIN)/s02_build.log

# ── 03: one pool, two launchers, one binary ──────────────────────────────────
s03: $(BIN)/s03
$(BIN)/s03: 03_one_pool_two_launchers.cpp common.hpp
	$(NVCXX) $(STD) $(OPT) -stdpar=gpu -gpu=$(ARCH) -cuda -DTC_SPIKE_CUDA $< -o $@

# ── 03b: g++ host + nvcc device layer. Note the host TU never sees CUDA. ─────
s03b: $(BIN)/s03b
$(BIN)/s03b: 03b_main.cpp 03b_device_layer.cu
	$(NVCC) -O2 -arch=sm_$(subst cc,,$(ARCH)) -c 03b_device_layer.cu -o $(BIN)/03b_device.o
	$(CXX) $(STD) $(OPT) 03b_main.cpp $(BIN)/03b_device.o -o $@ \
	    -L$(CUDA_HOME)/lib64 -lcudart -lstdc++

# ── host dry-run: 01/02/04 with no GPU and no NVHPC at all. ──────────────────
#    The pool falls back to aligned_alloc and staging to memcpy, so the LOGIC is
#    exercised on a laptop before anything reaches a GPU box. Spike 01 should
#    report a ~1.0x ratio here — that is the coherent-memory control, and seeing
#    it makes a large ratio on the real card meaningful rather than mysterious.
#    (libc++ needs -fexperimental-library to expose the PSTL policies; harmless
#    elsewhere, so pass HOSTFLAGS= to drop it on libstdc++.)
HOSTFLAGS ?= -fexperimental-library
.PHONY: host
host: 04_space_tags.cpp 01_managed_pool_stdpar.cpp 02_device_pool_stdpar.cpp common.hpp
	$(CXX) $(STD) $(OPT) -w 04_space_tags.cpp -o $(BIN)/s04_host
	$(CXX) $(STD) $(OPT) -w $(HOSTFLAGS) 01_managed_pool_stdpar.cpp -o $(BIN)/s01_host
	$(CXX) $(STD) $(OPT) -w $(HOSTFLAGS) 02_device_pool_stdpar.cpp -o $(BIN)/s02_host
	@echo; $(BIN)/s04_host; $(BIN)/s01_host 512 50; $(BIN)/s02_host 512 50

all: s04 s01 s02 s03 s03b

run: all
	@echo; echo "################ spike 04 (host) ################"
	@-$(BIN)/s04
	@echo;  echo "################ spike 04 negative ##############"
	@-$(MAKE) --no-print-directory s04_negative
	@echo;  echo "################ spike 01 ########################"
	@-test -x $(BIN)/s01  && $(BIN)/s01  $(N) $(ITERS) || echo "  (s01 did not build)"
	@echo;  echo "################ spike 02 ########################"
	@-test -x $(BIN)/s02  && $(BIN)/s02  1024 100      || echo "  (s02 did not build — see $(BIN)/s02_build.log; that is a RESULT)"
	@echo;  echo "################ spike 03 ########################"
	@-test -x $(BIN)/s03  && $(BIN)/s03  $(N) $(ITERS) || echo "  (s03 did not build)"
	@echo;  echo "################ spike 03b #######################"
	@-test -x $(BIN)/s03b && $(BIN)/s03b $(N) $(ITERS) || echo "  (s03b did not build)"

clean:
	rm -rf $(BIN)
