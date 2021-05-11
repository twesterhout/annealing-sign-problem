#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ls_bits512 {
  uint64_t words[8];
} ls_bits512;

static int ls_bits512_cmp(void const *_a, void const *_b) {
  ls_bits512 const *const a = _a;
  ls_bits512 const *const b = _b;

  for (int i = 0; i < 8; ++i) {
    if (a->words[i] < b->words[i]) {
      return -1;
    }
    if (a->words[i] > b->words[i]) {
      return 1;
    }
  }
  return 0;
}

__attribute__((visibility("default"))) uint64_t build_matrix(
    uint64_t const num_spins, ls_bits512 const spins[],
    int64_t const *restrict counts, double const *restrict psi,
    ls_bits512 const *restrict other_spins, double const *restrict other_coeffs,
    int64_t const *restrict other_counts, double const *restrict other_psi,
    uint32_t *restrict row_indices, uint32_t *restrict col_indices,
    double *restrict elements, double *restrict field) {
  memset(field, 0, sizeof(double) * num_spins);
  uint64_t size = 0;
  double t_max = 0.0;
  for (uint64_t row_index = 0; row_index < num_spins;
       ++row_index, ++psi, ++counts, ++other_counts) {
    for (int64_t j = 0; j < *other_counts;
         ++j, ++other_spins, ++other_coeffs, ++other_psi) {
      ls_bits512 const *const p = bsearch(other_spins, spins, num_spins,
                                          sizeof(ls_bits512), &ls_bits512_cmp);
      if (p != NULL) {
        double const t =
            (*counts) * (*other_coeffs) * fabs(*psi) * fabs(*other_psi);
        row_indices[size] = row_index;
        col_indices[size] = p - spins;
        elements[size] = t;
        if (fabs(t) > t_max) {
          t_max = fabs(t);
        }
        ++size;
      } else {
        field[row_index] += *counts * *other_coeffs * fabs(*psi) * *other_psi;
      }
    }
  }
  uint64_t new_size = 0;
  double const cutoff = 1e-8 * t_max;
  for (uint64_t i = 0; i < size; ++i) {
    if (fabs(elements[i]) > cutoff) {
      row_indices[new_size] = row_indices[i];
      col_indices[new_size] = col_indices[i];
      elements[new_size] = elements[i];
      ++new_size;
    }
  }
  return new_size;
}

__attribute__((visibility("default"))) void
extract_signs(uint64_t const num_spins, double const *restrict psi,
              uint64_t *restrict signs) {
  uint64_t num_words = (num_spins + (64 - 1)) / 64;
  memset(signs, 0, sizeof(uint64_t) * num_words);
  for (uint64_t i = 0; i < num_spins; ++i) {
    if (psi[i] > 0) {
      signs[i / 64U] |= ((uint64_t)1) << (i % 64U);
    }
  }
}
