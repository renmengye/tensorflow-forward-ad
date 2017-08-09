load("//:rules.bzl", "pyx_library")

pyx_library(
  name = "cbfs",
  srcs = ["cbfs.pyx"],
)

py_library(
  name = "logger",
  srcs = ["logger.py"],
  deps = [],
)

py_test(
  name = "bfs_tests",
  srcs = ["bfs_tests.py"],
  deps = [":cbfs"],
)

py_library(
  name = "graph",
  srcs = ["graph.py"],
  deps = [":cbfs"],
)

py_library(
  name = "maxpool_gradgrad",
  srcs = ["maxpool_gradgrad.py"],
  deps = [],
)

py_test(
  name = "maxpool_gradgrad_tests",
  srcs = ["maxpool_gradgrad_tests.py"],
  deps = [":maxpool_gradgrad"],
)

py_library(
  name = "fwgrad",
  srcs = ["fwgrad.py"],
  deps = [":logger", ":graph", ":maxpool_gradgrad"],
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
