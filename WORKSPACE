git_repository(
    name = "io_bazel_rules_go",
    remote = "https://github.com/bazelbuild/rules_go.git",
    tag = "0.3.2",
)
load("@io_bazel_rules_go//go:def.bzl", "go_repositories")
go_repositories()

git_repository(
    name = "org_protobuf",
    remote = "https://github.com/google/protobuf/",
    tag = "v3.1.0",
)

git_repository(
  name = "org_pubref_rules_protobuf",
  remote = "https://github.com/pubref/rules_protobuf",
  tag = "v0.7.1",
)
load("@org_pubref_rules_protobuf//go:rules.bzl", "go_proto_repositories")

local_repository(
    name = "org_tensorflow",
    path = "tensorflow",
)
load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace()
go_proto_repositories()

