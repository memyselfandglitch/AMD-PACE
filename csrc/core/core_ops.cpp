/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

// Core helpers exposed as torch dispatcher ops. Replaces the previous pybind11
// `pace._C` module: registering through the dispatcher decouples libpace_cpp.so
// from CPython's ABI, so the same .so loads on any CPython 3.x via
// `torch.ops.load_library(...)`.

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include <core/logging.h>
#include <core/threading.h>
#include <graph/register.h>

namespace pace {
namespace {

void thread_bind_op(at::IntArrayRef cores) {
  std::vector<int32_t> cores_i32(cores.begin(), cores.end());
  thread_bind(cores_i32);
}

void log_op(int64_t level, const std::string& message) {
  pace_logger(level, message);
}

void enable_fusion_op(bool enabled) {
  PACEPassOptimize::enable_pace_fusion(enabled);
}

} // namespace
} // namespace pace

// `m.def(schema, fn)` registers a CompositeImplicitAutograd implementation,
// which is the right choice for tensor-free utility ops -- the dispatcher
// can't pick a backend by tensor argument because there are none.
TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def("thread_bind(int[] cores) -> ()", &pace::thread_bind_op);
  m.def("log(int level, str message) -> ()", &pace::log_op);
  m.def("enable_fusion(bool enabled) -> ()", &pace::enable_fusion_op);
}
