load("//:rules.bzl", "pyx_library")

pyx_library(
  name = "cbfs",
  srcs = ["cbfs.pyx"],
)

py_library(
  name = "graph",
  srcs = ["graph.py"],
  deps = [":cbfs"],
)

py_library(
  name = "fwgrad",
  srcs = ["fwgrad.py", "maxpool_gradgrad.py"],
  deps = [":graph"],
)

py_test(
  name = "fwgrad_tests",
  srcs = ["fwgrad_tests.py"],
  deps = [":fwgrad"],
)

py_library(
  name = "second_order",
  srcs = ["second_order.py"],
  deps = [":fwgrad"],
)

py_test(
  name = "second_order_tests",
  srcs = ["second_order_tests.py"],
  deps = [":second_order"],
)
