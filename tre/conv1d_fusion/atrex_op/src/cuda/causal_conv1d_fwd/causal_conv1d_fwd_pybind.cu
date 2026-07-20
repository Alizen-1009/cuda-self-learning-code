#include <torch/all.h>
#include <torch/python.h>

#include "causal_conv1d_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_causal_conv1d_fwd_kernel",
        &causal_conv1d_fwd,
        "Width-4 causal depthwise conv1d + SiLU, channel-contiguous bf16 (SM100)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("state"),
        py::arg("out"),
        py::arg("has_init"));
}
