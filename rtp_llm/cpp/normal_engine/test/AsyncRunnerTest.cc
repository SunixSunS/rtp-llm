#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <ATen/record_function.h>
#include "gtest/gtest.h"
#include "torch/torch.h"

#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/DecodeBufferRetirement.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

namespace rtp_llm {
namespace {

std::atomic<int>* g_record_function_callback_count = nullptr;

std::unique_ptr<at::ObserverContext> countRecordFunctionCallback(const at::RecordFunction&) {
    if (g_record_function_callback_count != nullptr) {
        g_record_function_callback_count->fetch_add(1, std::memory_order_relaxed);
    }
    return nullptr;
}

class HostCallbackGate {
public:
    static void wait(void* data) {
        auto&                        gate = *static_cast<HostCallbackGate*>(data);
        std::unique_lock<std::mutex> lock(gate.mutex_);
        gate.entered.set_value();
        gate.cv_.wait(lock, [&gate] { return gate.released_; });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        cv_.notify_one();
    }

    std::promise<void> entered;

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    released_ = false;
};

struct ReleaseHostCallbackGate {
    HostCallbackGate& gate;
    ~ReleaseHostCallbackGate() {
        gate.release();
    }
};

void enqueueHostCallback(const torch::Stream& stream, void (*callback)(void*), void* data) {
#if USING_ROCM
    const auto result = hipLaunchHostFunc(cuda_graph::toGraphStream(stream).stream(), callback, data);
    if (result != hipSuccess) {
        throw std::runtime_error(std::string("hipLaunchHostFunc failed: ") + hipGetErrorString(result));
    }
#else
    const auto result = cudaLaunchHostFunc(cuda_graph::toGraphStream(stream).stream(), callback, data);
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("cudaLaunchHostFunc failed: ") + cudaGetErrorString(result));
    }
#endif
}

void markCopyComplete(void* data) {
    static_cast<std::promise<void>*>(data)->set_value();
}

}  // namespace

class AsyncRunnerTest: public ::testing::Test {
protected:
    torch::Stream makeStream() {
        return cuda_graph::graphGetStreamFromPool(/*is_high_priority=*/true);
    }

    torch::Stream currentStream() {
        return cuda_graph::graphGetCurrentStream();
    }

    void expectGpuDrainedBeforeException(bool observe_from_next_launch) {
        auto stream      = makeStream();
        auto main_stream = currentStream();
        auto cpu_options = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
        auto source      = torch::full({64}, 37, cpu_options);
        auto copied      = torch::zeros({64}, cpu_options);
        auto device      = torch::empty({64}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        HostCallbackGate   gate;
        auto               gate_entered = gate.entered.get_future();
        std::promise<void> copy_complete;
        auto               copy_completed = copy_complete.get_future().share();
        std::promise<void> task_submitted;
        auto               submitted = task_submitted.get_future();
        AsyncRunner        runner(stream);
        // Also unblock the callback if a fatal assertion returns early. The
        // gate and copy promises outlive the runner's worker-thread join.
        ReleaseHostCallbackGate release_gate{gate};

        runner.launch([&, source, copied, device] {
            try {
                enqueueHostCallback(stream, &HostCallbackGate::wait, &gate);
                device.copy_(source, /*non_blocking=*/true);
                copied.copy_(device, /*non_blocking=*/true);
                enqueueHostCallback(stream, &markCopyComplete, &copy_complete);
            } catch (...) {
                task_submitted.set_exception(std::current_exception());
                throw;
            }
            task_submitted.set_value();
            throw std::runtime_error("failure after queued GPU copy");
        });

        ASSERT_EQ(submitted.wait_for(std::chrono::seconds(10)), std::future_status::ready);
        ASSERT_NO_THROW(submitted.get());
        ASSERT_EQ(gate_entered.wait_for(std::chrono::seconds(10)), std::future_status::ready);

        std::atomic<bool>  rejected_task_ran{false};
        std::promise<void> observer_started;
        auto               started  = observer_started.get_future();
        auto               observed = std::async(std::launch::async, [&] {
            try {
                cuda_graph::setDevice(static_cast<int>(stream.device_index()));
            } catch (...) {
                observer_started.set_exception(std::current_exception());
                throw;
            }
            observer_started.set_value();
            try {
                if (observe_from_next_launch) {
                    runner.launch([&] { rejected_task_ran.store(true); });
                } else {
                    runner.sync(main_stream);
                }
            } catch (const std::runtime_error& e) {
                return std::make_pair(std::string(e.what()),
                                      copy_completed.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
            }
            return std::make_pair(std::string(), false);
        });
        ASSERT_NO_THROW(started.get());
        // The callback prevents either copy from executing. This bounded
        // future wait verifies that neither exception entry point can return
        // while the GPU still owns the task's input/output storage.
        EXPECT_EQ(observed.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
        gate.release();

        auto result = observed.get();
        EXPECT_EQ(result.first, "failure after queued GPU copy");
        EXPECT_TRUE(result.second);
        EXPECT_FALSE(rejected_task_ran.load());
        ASSERT_EQ(copy_completed.wait_for(std::chrono::seconds(10)), std::future_status::ready);
        EXPECT_TRUE(torch::equal(copied, source));

        // The original exception is delivered once. A later successful task
        // records a fresh event and remains usable through the normal path.
        auto recovery_source = torch::full({64}, 91, cpu_options);
        runner.launch([device, recovery_source] { device.copy_(recovery_source, /*non_blocking=*/true); });
        runner.sync(main_stream);
        EXPECT_TRUE(torch::equal(device.cpu(), recovery_source));
    }
};

TEST_F(AsyncRunnerTest, LaunchAndSyncBasic) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int> counter{0};
    runner.launch([&counter] { counter.fetch_add(1); });
    runner.sync(currentStream());

    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, SequentialLaunches) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int> counter{0};
    constexpr int    kIterations = 10;
    for (int i = 0; i < kIterations; i++) {
        runner.launch([&counter] { counter.fetch_add(1); });
    }
    runner.sync(currentStream());

    EXPECT_EQ(counter.load(), kIterations);
}

TEST_F(AsyncRunnerTest, ExecutionOrder) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::vector<int> order;
    std::mutex       mu;

    for (int i = 0; i < 5; i++) {
        runner.launch([&order, &mu, i] {
            std::lock_guard<std::mutex> lk(mu);
            order.push_back(i);
        });
    }
    runner.sync(currentStream());

    ASSERT_EQ(order.size(), 5u);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(order[i], i);
    }
}

TEST_F(AsyncRunnerTest, CudaTensorWork) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    auto src = torch::ones({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto dst = torch::zeros({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto input_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    input_ready_event->record(currentStream());
    runner.launch([&src, &dst, input_ready_event] {
        input_ready_event->block(cuda_graph::graphGetCurrentStream());
        dst.copy_(src);
    });
    runner.sync(currentStream());

    auto result = dst.cpu();
    EXPECT_TRUE(torch::equal(result, torch::ones({64}, torch::kFloat32)));
}

TEST_F(AsyncRunnerTest, DoesNotPropagateRecordFunctionCallbacksToWorker) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int>  callback_count{0};
    std::atomic<bool> allow_worker_ops{false};
    std::atomic<bool> worker_task_started{false};
    g_record_function_callback_count = &callback_count;
    auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(&countRecordFunctionCallback));

    {
        at::RecordFunction main_record(at::RecordScope::FUNCTION);
        main_record.before("async_runner.main_thread_positive_control");
        main_record.end();
    }
    EXPECT_GT(callback_count.load(std::memory_order_relaxed), 0);
    callback_count.store(0, std::memory_order_relaxed);

    runner.launch([&allow_worker_ops, &worker_task_started] {
        worker_task_started.store(true, std::memory_order_release);
        while (!allow_worker_ops.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        auto src = torch::ones({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto dst = src.narrow(0, 0, 32).clone();
        (void)dst;
    });

    while (!worker_task_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    callback_count.store(0, std::memory_order_relaxed);
    at::removeCallback(handle);
    allow_worker_ops.store(true, std::memory_order_release);
    runner.sync(currentStream());

    g_record_function_callback_count = nullptr;
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 0);
}

TEST_F(AsyncRunnerTest, DestructorJoinsThread) {
    auto             stream = makeStream();
    std::atomic<int> counter{0};

    {
        AsyncRunner runner(stream);
        runner.launch([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            counter.fetch_add(1);
        });
    }
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, SyncWithoutLaunch) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    runner.sync(currentStream());
}

TEST_F(AsyncRunnerTest, RethrowsWorkerExceptionFromSync) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    runner.launch([] { throw std::runtime_error("async failure"); });
    EXPECT_THROW(runner.sync(currentStream()), std::runtime_error);

    std::atomic<int> counter{0};
    runner.launch([&counter] { counter.fetch_add(1); });
    runner.sync(currentStream());
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, RethrowsWorkerExceptionFromNextLaunch) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    runner.launch([] { throw std::runtime_error("async failure"); });
    EXPECT_THROW(runner.launch([] {}), std::runtime_error);

    std::atomic<int> counter{0};
    runner.launch([&counter] { counter.fetch_add(1); });
    runner.sync(currentStream());
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, DrainsGpuWorkBeforeRethrowingFromSync) {
    expectGpuDrainedBeforeException(/*observe_from_next_launch=*/false);
}

TEST_F(AsyncRunnerTest, DrainsGpuWorkBeforeRethrowingFromNextLaunch) {
    expectGpuDrainedBeforeException(/*observe_from_next_launch=*/true);
}

TEST_F(AsyncRunnerTest, LaunchSyncInterleavedStress) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int> counter{0};
    constexpr int    kIterations = 100;

    for (int i = 0; i < kIterations; i++) {
        runner.launch([&counter] { counter.fetch_add(1); });
        runner.sync(currentStream());
        EXPECT_EQ(counter.load(), i + 1);
    }
}

TEST_F(AsyncRunnerTest, RetirementRetainsCopySourcesUntilGpuCompletion) {
    auto         stream = makeStream();
    auto         source = torch::full({64}, 37, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    auto         result = torch::zeros({64}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    auto         device = torch::empty({64}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    TensorHolder holder;
    holder.hold(source);
    holder.release();
    holder.hold(device);
    auto                                      owner = std::make_shared<std::vector<torch::Tensor>>(holder.takeAll());
    std::weak_ptr<std::vector<torch::Tensor>> weak  = owner;
    EXPECT_EQ(owner->size(), 2);
    EXPECT_TRUE(holder.tensors.empty());
    EXPECT_TRUE(holder.clear_tensors.empty());

    HostCallbackGate        gate;
    auto                    entered = gate.entered.get_future();
    DecodeBufferRetirement  retirement;
    ReleaseHostCallbackGate release_gate{gate};
    auto                    completion = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    {
        c10::StreamGuard stream_guard(stream);
        enqueueHostCallback(stream, &HostCallbackGate::wait, &gate);
        device.copy_(source, true);
        result.copy_(device, true);
        completion->record(stream);
    }
    ASSERT_EQ(entered.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    retirement.retire(completion, owner);
    owner.reset();
    source = torch::Tensor();
    device = torch::Tensor();
    retirement.reap();
    EXPECT_EQ(retirement.size(), 1);
    EXPECT_FALSE(weak.expired());
    gate.release();
    retirement.drain();
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(retirement.size(), 0);
    EXPECT_TRUE(torch::equal(result, torch::full_like(result, 37)));
}

TEST_F(AsyncRunnerTest, RetirementBoundsSubmissionAndDestructorDrains) {
    auto                    stream = makeStream();
    HostCallbackGate        gate;
    auto                    entered    = gate.entered.get_future();
    auto                    retirement = std::make_unique<DecodeBufferRetirement>();
    ReleaseHostCallbackGate release_gate{gate};
    auto                    owner      = std::make_shared<int>(42);
    std::weak_ptr<int>      weak       = owner;
    auto                    completion = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    enqueueHostCallback(stream, &HostCallbackGate::wait, &gate);
    completion->record(stream);
    ASSERT_EQ(entered.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    for (size_t i = 0; i < DecodeBufferRetirement::kMaxPendingRounds; ++i) {
        retirement->retire(completion, owner);
    }
    EXPECT_EQ(retirement->size(), DecodeBufferRetirement::kMaxPendingRounds);
    auto pending = std::async(std::launch::async, [&] {
        cuda_graph::setDevice(static_cast<int>(stream.device_index()));
        retirement->retire(completion, owner);
    });
    EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_FALSE(weak.expired());
    gate.release();
    pending.get();
    retirement->drain();

    HostCallbackGate exit_gate;
    enqueueHostCallback(stream, &HostCallbackGate::wait, &exit_gate);
    completion = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    completion->record(stream);
    retirement->retire(completion, owner);
    owner.reset();
    auto exiting = std::async(std::launch::async, [&] {
        cuda_graph::setDevice(static_cast<int>(stream.device_index()));
        retirement.reset();
    });
    EXPECT_EQ(exiting.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_FALSE(weak.expired());
    exit_gate.release();
    exiting.get();
    EXPECT_TRUE(weak.expired());
}

}  // namespace rtp_llm
