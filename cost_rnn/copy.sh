#!/bin/bash

bazel build :all
cp -f ../bazel-out/local-fastbuild/genfiles/cost_rnn/data.record .
