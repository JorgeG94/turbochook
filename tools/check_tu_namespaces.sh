#!/usr/bin/env bash
# =============================================================================
# tools/check_tu_namespaces.sh — every test TU must have its own namespace.
#
# WHY. A SYCL unnamed-lambda kernel is identified by the MANGLED TYPE of its
# closure, and that mangling embeds the enclosing function's name. C++ mangling
# does not encode the translation unit. doctest's TEST_CASE expands to
#
#     static void DOCTEST_ANON_FUNC_<n>()
#
# off a counter that RESTARTS IN EVERY FILE, so a lambda in the third TEST_CASE
# of any two files mangles to the same kernel name. On Aurora that was 76
# collisions across 18 test TUs: the runtime launches whichever image it finds
# first, giving UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE where the captures
# differ and SILENTLY WRONG NUMBERS where they happen to match. 24 of 67 tests.
#
# No compiler flag fixes it -- not -fsycl-unnamed-lambda, -fsycl-unique-prefix,
# nor -funique-internal-linkage-names. It is a source-level property, so this is
# a source-level check.
#
# Wrapping each file in a NAMED namespace puts the file's identity into the
# mangling. That is the whole fix, and it is one line per file.
#
# This check needs no GPU, no SYCL toolchain and no build -- which matters,
# because the bug it prevents is invisible on the machine most work happens on.
# (The symptom is only detectable in a SYCL build, where duplicate kernel names
# show up as duplicate _ZTS strings in the objects. That is the confirmation;
# this is the prevention.)
# =============================================================================
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

for f in "$root"/tests/test_*.cpp; do
    [ -e "$f" ] || continue
    base="$(basename "$f")"

    # Line of the first TEST_CASE, and of the first named-namespace opening.
    tc_line=$(grep -n '^\s*TEST_CASE' "$f" | head -1 | cut -d: -f1)
    ns_line=$(grep -n '^\s*namespace\s\+[A-Za-z_][A-Za-z0-9_]*\s*{' "$f" | head -1 | cut -d: -f1)

    [ -z "${tc_line:-}" ] && continue          # no test cases: nothing to protect

    if [ -z "${ns_line:-}" ]; then
        echo "  FAIL $base: TEST_CASE at line $tc_line is not inside a named namespace"
        fail=1
    elif [ "$ns_line" -gt "$tc_line" ]; then
        echo "  FAIL $base: named namespace opens at $ns_line, AFTER the first TEST_CASE at $tc_line"
        fail=1
    fi

    # A NESTED anonymous namespace is fine -- the enclosing named one already
    # puts this file's identity into the mangling. Only an anonymous namespace
    # with no named one around it is a trap: _GLOBAL__N_1 is the same string in
    # every TU, so it looks like it isolates and does not. That case is already
    # caught above, because such a file has no qualifying named namespace.
done

if [ "$fail" -eq 0 ]; then
    echo "  all test TUs carry a unique named namespace"
else
    echo
    echo "  Fix: wrap the file's contents in 'namespace tu_<filename> { ... }'."
    echo "  See docs/GPU_STDPAR_NOTES.md."
fi
exit "$fail"
