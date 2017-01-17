load("@io_bazel_rules_go//go:def.bzl", "go_binary")

go_binary(
    name = "generate_data",
    srcs = ["generate_data.go"],
    deps = [
            "//protos:tensorflow_example",
            "@com_github_golang_protobuf//proto:go_default_library",
    ],
)

genrule(
    name = "make_test_data",
    cmd = "$(locations :generate_data) $(OUTS)",
    tools= [
            ":generate_data",
    ],
    outs = ["data.record"],
)

py_binary(
    name="rnn2",
    main="rnn2.py",
    srcs = ["rnn2.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
