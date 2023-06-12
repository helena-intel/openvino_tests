# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME latency_benchmark
              SOURCES "main.cpp"
              DEPENDENCIES ie_samples_utils)
