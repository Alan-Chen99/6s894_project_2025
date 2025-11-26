#pragma once

#include "dtype.h"

template <DType dtype>
void run_fht(void* a, void* out, uint32_t numel, uint32_t had_size, cudaStream_t stream);
