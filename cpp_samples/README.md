This directory contains OpenVINO sample code for asynchronous inference in
latency mode.  It is based on OpenVINO's [throughput
sample](https://github.com/openvinotoolkit/openvino/tree/master/samples/cpp/benchmark/throughput_benchmark)
and includes input and output data.

Latency is shown in microseconds.

To build the sample, run `./build_samples.sh` after initializing OpenVINO with `source setupvars.sh` from the OpenVINO install directory.

The executable `latency_benchmark` will be created in the OpenVINO samples directory (the name of this directory will be displayed after building the samples).

Usage : `./latency_benchmark <path_to_model> <batch_size> <streams> <infer_requests> <device>`

By default `batch_size` is 1 and `device` GPU. `streams` and `infer_requests` are taken from OpenVINO Runtime.
