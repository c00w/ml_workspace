load("@io_bazel_rules_go//proto:go_proto_library.bzl", "go_proto_library")

go_proto_library(
    name = "tensorflow_example",
    srcs = [
        "example.proto",
        "feature.proto",
        "model.proto",
    ],
    deps = [
        "@com_github_golang_protobuf//ptypes/any:go_default_library",
    ],
    visibility = ["//visibility:public"],
)
