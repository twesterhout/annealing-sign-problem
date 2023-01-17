from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef(
    """
    typedef struct ls_bits512 {
      uint64_t words[8];
    } ls_bits512;
    
    uint64_t build_matrix(uint64_t num_spins, ls_bits512 const spins[],
                          int64_t const *counts, double const *psi,
                          ls_bits512 const *other_spins, double const *other_coeffs,
                          int64_t const *other_counts, double const *other_psi,
                          uint32_t *row_indices, uint32_t *col_indices,
                          double *elements, double *field);
    
    void extract_signs(uint64_t num_spins, double const *psi, uint64_t *signs);
"""
)

ffibuilder.set_source(
    "_build_matrix",
    """
    #include "build_matrix.h"
""",
    sources=["cbits/build_matrix.c"],
    include_dirs=["cbits"],
    libraries=["m"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
