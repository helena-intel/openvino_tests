// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <condition_variable>
#include <string>
#include <vector>
#include "stdlib.h"

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/latency_metrics.hpp"
#include "samples/slog.hpp"
// clang-format on

using Us = std::chrono::duration<double, std::ratio<1, 1000000>>;

int main(int argc, char* argv[]) {
    try {
        slog::info << "OpenVINO:" << slog::endl;
        slog::info << ov::get_openvino_version();
        if (argc < 2) {
            slog::info << "Usage : " << argv[0] << " <path_to_model> <batch_size> <streams> <infer_requests> <device>"<< slog::endl;
            return EXIT_FAILURE;
        }

        char unit[4] = "us";
        auto batch_size = argc < 3 ? 1 : atoi(argv[2]);
        auto streams = argc < 4 ? -1 : atoi(argv[3]);
        char device[4] = "GPU";
        if (argc >= 6) {
            strcpy(device, argv[5]);
        }

        // Optimize for latency
        ov::AnyMap config{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY}};
        if (streams > 0) {
            config.emplace(ov::num_streams.name(), streams);
        }

        // Create ov::Core and use it to compile a model.
        // Pick a device by replacing CPU, for example MULTI:CPU(4),GPU(8).
        ov::Core core;
        auto model = core.read_model(argv[1]);

        // Set batch size
        ov::Shape inputShape = model->input().get_shape();
        auto batchId = 0;
        inputShape[batchId] = batch_size;
        model->reshape(inputShape);

        ov::CompiledModel compiled_model = core.compile_model(model, device, config);
        // Create optimal number of ov::InferRequest instances
        uint32_t nireq = argc < 5 ? compiled_model.get_property(ov::optimal_number_of_infer_requests) : atoi(argv[4]);
        std::vector<ov::InferRequest> ireqs(nireq);
        std::generate(ireqs.begin(), ireqs.end(), [&] {
            return compiled_model.create_infer_request();
        });

        // Create and initialize input data (2 * nireq)
        std::vector<std::vector<std::shared_ptr<float[]>>> data;
        std::vector<ov::element::Type> input_types;
        std::vector<ov::Shape> input_shapes;
        for (const ov::Output<const ov::Node>& model_input : compiled_model.inputs()) {
            auto input_data = generate_random_data<float>(ireqs[0].get_tensor(model_input), nireq);
            input_types.push_back(ireqs[0].get_tensor(model_input).get_element_type());
            input_shapes.push_back(ireqs[0].get_tensor(model_input).get_shape());
            data.push_back(input_data);
        }
        auto data_size = data[0].size();

        // Create ouput data storage
        std::vector<std::shared_ptr<float[]>> result;
        for (size_t i = 0; i < nireq; i++) {
            std::shared_ptr<float[]> res(new float[ireqs[0].get_output_tensor(0).get_size()]);
            result.push_back(res);
        }

        // Warm up
        for (ov::InferRequest& ireq : ireqs) {
            ireq.start_async();
        }
        for (ov::InferRequest& ireq : ireqs) {
            ireq.wait();
        }
        // Benchmark for seconds_to_run seconds and at least niter iterations
        std::chrono::seconds seconds_to_run{10};
        size_t niter = 10;
        std::vector<double> latencies;
        std::mutex mutex;
        std::condition_variable cv;
        std::exception_ptr callback_exception;
        struct TimedIreq {
            ov::InferRequest& ireq;  // ref
            std::chrono::steady_clock::time_point start;
            bool has_start_time;
        };
        std::deque<TimedIreq> finished_ireqs;
        for (ov::InferRequest& ireq : ireqs) {
            finished_ireqs.push_back({ireq, std::chrono::steady_clock::time_point{}, false});
        }
        auto start = std::chrono::steady_clock::now();
        auto time_point_to_finish = start + seconds_to_run;
        size_t data_index = 0;

        // Once there’s a finished ireq wake up main thread.
        // Compute and save latency for that ireq and prepare for next inference by setting up callback.
        // Callback pushes that ireq again to finished ireqs when infrence is completed.
        // Start asynchronous infer with updated callback
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex);
            while (!callback_exception && finished_ireqs.empty()) {
                cv.wait(lock);
            }
            if (callback_exception) {
                std::rethrow_exception(callback_exception);
            }
            if (!finished_ireqs.empty()) {
                auto time_point = std::chrono::steady_clock::now();
                if (time_point > time_point_to_finish && latencies.size() > niter) {
                    break;
                }
                TimedIreq timedIreq = finished_ireqs.front();
                finished_ireqs.pop_front();
                lock.unlock();
                ov::InferRequest& ireq = timedIreq.ireq;
                if (timedIreq.has_start_time) {
                    latencies.push_back(std::chrono::duration_cast<Us>(time_point - timedIreq.start).count());
                }
                ireq.set_callback(
                [&ireq, time_point, &mutex, &finished_ireqs, &callback_exception, &cv, &result, &data_index](std::exception_ptr ex) {
                    // Keep callback small. This improves performance for fast (tens of thousands FPS) models
                    std::unique_lock<std::mutex> lock(mutex);


                    {
                        try {
                            if (ex) {
                                std::rethrow_exception(ex);
                            }
                            finished_ireqs.push_back({ireq, time_point, true});
                        } catch (const std::exception&) {
                            if (!callback_exception) {
                                callback_exception = std::current_exception();
                            }
                        }
                    }
                    cv.notify_one();

                    // Copy result
                    auto output = ireq.get_output_tensor(0);
                    auto tensor_data = output.data<float>();
                    auto result_data = result[0].get();
                    std::memcpy(result_data, tensor_data, output.get_byte_size());
                });


                // Fill model inputs
                for (size_t i = 0; i < data.size(); i++) {
                    auto input = ireq.get_input_tensor(i);
                    auto tensor_data = input.data<float>(); // TODO: float is used explicitely
                    auto user_data = data[i][data_index].get();


                    std::memcpy(tensor_data, user_data, input.get_byte_size());
                }
                data_index = data_index == data_size - 1 ? 0 : data_index++;
                ireq.start_async();
            }
        }
        auto end = std::chrono::steady_clock::now();
        double duration = std::chrono::duration_cast<Us>(end - start).count();
        // Report results
        slog::info << "Device:     " << device << slog::endl
                   << "Count:      " << latencies.size() << " iterations" << slog::endl
                   << "Duration:   " << duration << " " << unit << slog::endl
                   << "Latency:" << slog::endl;
        size_t percent = 50;
        LatencyMetrics{latencies, "", percent}.write_to_slog(unit);

    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
