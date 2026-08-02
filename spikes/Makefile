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

# ── 05 / 05b: SYCL on Aurora (PVC). Third backend = the honest test of the seam. ──
#    JIT by default (simplest). For ahead-of-time on PVC, which is what ALCF
#    generally recommends to avoid startup JIT cost:
#        make s05 SYCL_TARGET='-fsycl-targets=spir64_gen -Xs "-device pvc"'
ICPX        ?= icpx
SYCL_TARGET ?=
PSTL_OFFLOAD ?= -fsycl-pstl-offload=gpu

s05: $(BIN)/s05
$(BIN)/s05: 05_sycl_native.cpp common.hpp
	$(ICPX) $(STD) $(OPT) -fsycl $(SYCL_TARGET) -DTC_SPIKE_SYCL $< -o $@

# The PSTL-offload flag may not exist / may be spelled differently under this
# icpx. A build failure here IS the answer, so do not let it kill the run.
s05b: $(BIN)/s05b
$(BIN)/s05b: 05b_sycl_stdpar.cpp common.hpp
	-$(ICPX) $(STD) $(OPT) -fsycl $(PSTL_OFFLOAD) $(SYCL_TARGET) \
	    -DTC_SPIKE_SYCL $< -o $@ 2>&1 | tee $(BIN)/s05b_build.log

# 05c: is a stdpar-SHAPED launcher reachable on Intel at all? Built TWICE — the
# std:: spellings need the pstl-offload flag, the oneDPL ones do not. Either build
# may fail; each is a separate result.
s05c: $(BIN)/s05c
$(BIN)/s05c: 05c_sycl_stdpar_variants.cpp common.hpp
	-$(ICPX) $(STD) $(OPT) -fsycl $(SYCL_TARGET) -DTC_SPIKE_SYCL $< -o $@ \
	    2>&1 | tee $(BIN)/s05c_build.log

s05c_pstl: $(BIN)/s05c_pstl
$(BIN)/s05c_pstl: 05c_sycl_stdpar_variants.cpp common.hpp
	-$(ICPX) $(STD) $(OPT) -fsycl $(PSTL_OFFLOAD) $(SYCL_TARGET) \
	    -DTC_SPIKE_SYCL -DTC_TRY_PSTL $< -o $@ 2>&1 | tee $(BIN)/s05c_pstl_build.log

# 05d: why only ~392 GB/s on a ~3.2 TB/s part?
s05d: $(BIN)/s05d
$(BIN)/s05d: 05d_sycl_perf_probe.cpp common.hpp
	$(ICPX) $(STD) $(OPT) -fsycl $(SYCL_TARGET) -DTC_SPIKE_SYCL $< -o $@

.PHONY: sycl syclx
sycl: s05 s05b
	@echo; echo "################ spike 05 (native SYCL) ##########"
	@-test -x $(BIN)/s05  && $(BIN)/s05  $(N) $(ITERS) || echo "  (s05 did not build)"
	@echo;  echo "################ spike 05b (stdpar) ##############"
	@-test -x $(BIN)/s05b && $(BIN)/s05b $(N) $(ITERS) || \
	   echo "  (s05b did not build — see $(BIN)/s05b_build.log; that is a RESULT)"

syclx: s05c s05c_pstl s05d
	@echo; echo "################ 05c  oneDPL dialect (no pstl flag) ########"
	@-test -x $(BIN)/s05c && $(BIN)/s05c $(N) $(ITERS) || \
	   echo "  (did not build — see $(BIN)/s05c_build.log)"
	@echo;  echo "################ 05c  std:: spellings (pstl-offload) #######"
	@-test -x $(BIN)/s05c_pstl && $(BIN)/s05c_pstl $(N) $(ITERS) || \
	   echo "  (did not build — see $(BIN)/s05c_pstl_build.log; that is a RESULT)"
	@echo;  echo "################ 05d  performance probe ####################"
	@-test -x $(BIN)/s05d && $(BIN)/s05d $(N) $(ITERS) || echo "  (s05d did not build)"
	@echo;  echo "  Then, for the tile question, with no rebuild:"
	@echo   "     ZE_FLAT_DEVICE_HIERARCHY=FLAT $(BIN)/s05d $(N) $(ITERS)"

# ── 06: is std::mdspan + a CUSTOM LAYOUT viable as the kernel currency? ──────
#    Built four ways. A build failure on any one is a result, not a bug — the
#    availability answer is exactly what decides whether to ask ALCF for a newer
#    libstdc++ or to keep owning the type.
s06:      $(BIN)/s06
s06_gpu:  $(BIN)/s06_gpu
s06_sycl: $(BIN)/s06_sycl

$(BIN)/s06: 06_mdspan_layout.cpp common.hpp                    # host / any compiler
	-$(CXX) $(STD) $(OPT) -w $(HOSTFLAGS) $< -o $@ 2>&1 | tee $(BIN)/s06_build.log

$(BIN)/s06_gpu: 06_mdspan_layout.cpp common.hpp                # nvc++ -stdpar=gpu
	-$(NVCXX) $(STD) $(OPT) -stdpar=gpu -gpu=$(ARCH) -cuda -DTC_SPIKE_CUDA $< -o $@ \
	    2>&1 | tee $(BIN)/s06_gpu_build.log

$(BIN)/s06_sycl: 06_mdspan_layout.cpp common.hpp               # icpx -fsycl
	-$(ICPX) $(STD) $(OPT) -fsycl $(SYCL_TARGET) -DTC_SPIKE_SYCL $< -o $@ \
	    2>&1 | tee $(BIN)/s06_sycl_build.log

.PHONY: mdspan
mdspan: s06
	@echo; echo "################ spike 06 (host) #########################"
	@-test -x $(BIN)/s06 && $(BIN)/s06 1024 50 || \
	   echo "  (did not build -- see $(BIN)/s06_build.log; that is a RESULT)"

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
