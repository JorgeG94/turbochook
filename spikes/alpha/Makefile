# =============================================================================
# spikes/alpha/Makefile — one target per vendor, no configure step.
#
#   make host              no GPU needed. Run first; costs nothing.
#   make nvidia  ARCH=cc90 nvc++ -stdpar=gpu -gpu=mem:separate + cudaMalloc
#   make amd     GFX=gfx90a  hipcc --hipstdpar + hipMalloc      <-- the unknown
#   make intel             icpx -fsycl + oneDPL + malloc_device
#
#   make run-<target>      build and run it
#   make probe             which of these toolchains exist on this box
#
# Paste the whole output back -- the banner lines identify the box and the
# compiler, which is most of what makes a result interpretable later.
# =============================================================================

# Make variables are case-sensitive; `make nvidia arch=cc90` would silently
# build for the DEFAULT arch and hand you a binary for the wrong card.
ifdef arch
$(error you passed arch=$(arch); the variable is ARCH (upper case). Use: make nvidia ARCH=$(arch))
endif
ifdef gfx
$(error you passed gfx=$(gfx); the variable is GFX (upper case). Use: make amd GFX=$(gfx))
endif

ARCH   ?= cc80
GFX    ?= gfx90a
NVCXX  ?= nvc++
HIPCXX ?= hipcc
SYCLCXX?= icpx
CXX    ?= g++
STD    ?= -std=c++20
OPT    ?= -O2
BIN    ?= build
NX     ?= 512
NY     ?= 256

$(shell mkdir -p $(BIN))

.PHONY: all probe clean host nvidia amd intel \
        run-host run-nvidia run-amd run-intel

all: host

# ── host: plain par_unseq over ordinary memory. Proves the shape compiles. ────
# Not evidence about offload -- see the comment at the top of main.cpp.
host: $(BIN)/alpha_host
$(BIN)/alpha_host: main.cpp device.hpp
	$(CXX) $(STD) $(OPT) main.cpp -o $@ $(HOST_LIBS)
	@echo "built $@"

# libstdc++ par_unseq wants TBB. If your g++ does not, drop HOST_LIBS=.
HOST_LIBS ?= -ltbb

# ── nvidia: the combination spike 02 already proved; here as the control. ─────
# mem:separate is the whole point -- NO managed memory, so a host-side execution
# of the loop body would fault instead of quietly working.
nvidia: $(BIN)/alpha_nvidia
$(BIN)/alpha_nvidia: main.cpp device.hpp
	$(NVCXX) $(STD) $(OPT) -stdpar=gpu -gpu=$(ARCH),mem:separate \
	  -DTC_BACKEND_CUDA main.cpp -o $@ -cudalib=cudart
	@echo "built $@ for $(ARCH)"

# ── amd: THE UNKNOWN CELL. ───────────────────────────────────────────────────
# hipstdpar is not hipified CUDA: it leans on HMM/XNACK to make ordinary host
# allocations device-reachable, so its design assumption is that you did NOT
# hand-manage memory. Whether it accepts hipMalloc'd pointers is the question.
#
# If it needs the rocThrust/rocPRIM headers explicitly:
#     make amd HIPSTDPAR_EXTRA="--hipstdpar-path=/opt/rocm/include/thrust"
# XNACK may need enabling in the environment:  export HSA_XNACK=1
HIPSTDPAR_EXTRA ?=
amd: $(BIN)/alpha_amd
$(BIN)/alpha_amd: main.cpp device.hpp
	$(HIPCXX) $(STD) $(OPT) --hipstdpar --offload-arch=$(GFX) $(HIPSTDPAR_EXTRA) \
	  -DTC_BACKEND_HIP main.cpp -o $@
	@echo "built $@ for $(GFX)"

# ── intel: oneDPL. NOTE the deliberate absence of -fsycl-pstl-offload, ────────
# which was measured to double binary size for no gain (spike 05).
intel: $(BIN)/alpha_intel
$(BIN)/alpha_intel: main.cpp device.hpp
	$(SYCLCXX) $(STD) $(OPT) -fsycl -DTC_BACKEND_SYCL main.cpp -o $@
	@echo "built $@"
	@echo "NOTE: on a multi-tile card select ONE TILE -- a composite root device"
	@echo "      measured 6.3x slower. e.g. ZE_FLAT_DEVICE_HIERARCHY=FLAT"

run-host:   host   ; @echo; ./$(BIN)/alpha_host   $(NX) $(NY)
run-nvidia: nvidia ; @echo; ./$(BIN)/alpha_nvidia $(NX) $(NY)
run-amd:    amd    ; @echo; ./$(BIN)/alpha_amd    $(NX) $(NY)
run-intel:  intel  ; @echo; ./$(BIN)/alpha_intel  $(NX) $(NY)

probe:
	@echo "--- toolchains visible on this box ---"
	@for c in $(NVCXX) $(HIPCXX) $(SYCLCXX) $(CXX); do \
	  if command -v $$c >/dev/null 2>&1; then \
	    printf "  %-8s FOUND  %s\n" "$$c" "$$(command -v $$c)"; \
	    $$c --version 2>/dev/null | head -1 | sed 's/^/           /'; \
	  else \
	    printf "  %-8s absent\n" "$$c"; \
	  fi; \
	done

clean:
	rm -rf $(BIN)
