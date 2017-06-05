# Description:
#   Kernels for GRPC ops.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "grpc_kernels",
    srcs = ["grpc_kernels.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        # "//tensorflow/core:protos_all_cc",
        # "//tensorflow/core/kernels:concat_lib_hdrs",
        "//tensorflow/core/kernels:ops_util_hdrs",
        # "//tensorflow/core/kernels:split_lib_hdrs",
        "@grpc//:grpc++_unsecure",
    ],
    alwayslink = 1,
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
