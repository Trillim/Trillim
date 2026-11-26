#include "runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

using namespace trillim::cpu;

namespace {
using namespace std::chrono_literals;

#ifdef _WIN32
HANDLE pipe_handle(FILE *stream) {
    return reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stream)));
}
#endif

void configure_pipes() {
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1) {
        throw std::runtime_error("cannot configure binary runtime pipes");
    }
#else
    const int flags = fcntl(STDOUT_FILENO, F_GETFL);
    if (flags == -1 || fcntl(STDOUT_FILENO, F_SETFL, flags | O_NONBLOCK) == -1 ||
        std::signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
        throw std::runtime_error("cannot configure runtime response pipe");
    }
#endif
}

bool parent_disconnected(std::stop_token stop) {
#ifdef _WIN32
    const HANDLE input = pipe_handle(stdin);
    while (!stop.stop_requested()) {
        if (!PeekNamedPipe(input, nullptr, 0, nullptr, nullptr, nullptr)) {
            const DWORD error = GetLastError();
            return error == ERROR_BROKEN_PIPE || error == ERROR_NO_DATA;
        }
        std::this_thread::sleep_for(20ms);
    }
#else
    pollfd input{STDIN_FILENO, POLLHUP, 0};
    while (!stop.stop_requested()) {
        const int result = poll(&input, 1, 20);
        if (result > 0 && (input.revents & (POLLHUP | POLLERR | POLLNVAL))) {
            return true;
        }
        if (result < 0 && errno != EINTR) {
            return false;
        }
    }
#endif
    return false;
}

bool read_exact(void *destination, size_t bytes, bool allow_eof = false) {
    auto *target = static_cast<char *>(destination);
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("IPC payload too large");
    }
    std::cin.read(target, static_cast<std::streamsize>(bytes));
    if (std::cin.gcount() == 0 && allow_eof && std::cin.eof()) {
        return false;
    }
    if (static_cast<size_t>(std::cin.gcount()) != bytes) {
        throw std::runtime_error("truncated runtime request");
    }
    return true;
}

bool read_header(Json &header) {
    uint64_t length = 0;
    if (!read_exact(&length, sizeof(length), true)) {
        return false;
    }
    if (length == 0 || length > 64 * 1024 * 1024) {
        throw std::runtime_error("invalid runtime request frame");
    }
    std::string data(static_cast<size_t>(length), '\0');
    read_exact(data.data(), data.size());
    header = Json::parse(data);
    return true;
}

void write_exact(const void *source, size_t bytes, const Runtime *runtime) {
    const auto *data = static_cast<const uint8_t *>(source);
    while (bytes) {
        if (runtime) {
            runtime->check_cancelled();
        }
        const size_t chunk = std::min<size_t>(bytes, 65536);
#ifdef _WIN32
        DWORD count = 0;
        if (!WriteFile(pipe_handle(stdout), data, static_cast<DWORD>(chunk), &count, nullptr) || count == 0) {
            if (runtime) {
                runtime->check_cancelled();
            }
            throw std::runtime_error("runtime response pipe closed");
        }
#else
        const ssize_t count = write(STDOUT_FILENO, data, chunk);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd output{STDOUT_FILENO, POLLOUT, 0};
            if (poll(&output, 1, 10) >= 0 || errno == EINTR) {
                continue;
            }
        }
        if (count <= 0) {
            throw std::runtime_error("runtime response pipe closed");
        }
#endif
        data += count;
        bytes -= static_cast<size_t>(count);
    }
}

void write_header(const Json &header, const Runtime *runtime = nullptr) {
    const std::string data = header.dump();
    const uint64_t length = data.size();
    std::string frame(reinterpret_cast<const char *>(&length), sizeof(length));
    frame += data;
    write_exact(frame.data(), frame.size(), runtime);
}

void write_cancelled() {
#ifndef _WIN32
    pollfd output{STDOUT_FILENO, POLLOUT, 0};
    if (poll(&output, 1, 0) <= 0 || !(output.revents & POLLOUT)) {
        return;
    }
#endif
    write_header(Json{{"ok", false}, {"error", "model execution cancelled"}, {"cancelled", true}});
}

void materialize(Runtime &runtime, const std::string &name, Allocation &output,
                 Shape &shape, std::string &dtype) {
    Allocation decoded;
    const View source = runtime.logical_view(name, decoded);
    shape = source.shape;
    dtype = source.dtype == TRILLIM_FP32 ? "fp32" :
        runtime.buffer(name).at("dtype").get<std::string>();
    const size_t count = elements(shape);
    const size_t width = itemsize(source.dtype);
    if (!width || count > SIZE_MAX / width) {
        throw std::runtime_error("logical output size exceeds address space");
    }
    output = Allocation(count * width);
    runtime.copy_view(View{output.data, shape, contiguous_strides(shape), source.dtype, output.size}, source);
}

int serve(Runtime &runtime) {
    write_header(Json{{"ok", true}, {"ready", true}});
    Json request;
    while (read_header(request)) {
        if (request.at("op") == "close") {
            break;
        }
        if (request.at("op") != "run") {
            throw std::runtime_error("unknown runtime request");
        }
        bool response_incomplete = false;
        try {
            std::vector<Input> inputs;
            for (const auto &item : request.at("inputs")) {
                const size_t bytes = item.at("nbytes").get<size_t>();
                if (bytes > (uint64_t{1} << 40)) {
                    throw std::runtime_error("input payload exceeds runtime limit");
                }
                Input input{item.at("name").get<std::string>(), item.at("shape").get<Shape>(), Allocation(bytes)};
                read_exact(input.payload.data, bytes);
                inputs.push_back(std::move(input));
            }
            runtime.probe_outputs.clear();
            if (request.contains("probe_outputs")) {
                for (const auto &name : request.at("probe_outputs")) {
                    runtime.probe_outputs.insert(name.get<std::string>());
                }
            }
            runtime.profile = request.value("profile", false);
            runtime.native_ns = 0;
            runtime.native_calls = 0;
            runtime.host_copy_ns = 0;
            runtime.profiled_instruction_ns = 0;
            runtime.native_by_kernel.clear();
            runtime.instruction_ns.clear();
            runtime.copy_by_cache.clear();
            runtime.sample_times.clear();
            runtime.host_copy_bytes = 0;
            runtime.bind_inputs(std::move(inputs));
            const auto entered = trillim_cpu_enter(runtime.pool);
            if (entered != TRILLIM_OK) {
                throw std::runtime_error("another model is already executing");
            }
            const std::unique_ptr<TrillimCpuContext, decltype(&trillim_cpu_leave)> execution(
                runtime.pool, trillim_cpu_leave
            );
            Json descriptors = Json::array();
            std::vector<Allocation> outputs;
            runtime.request_start = now_ns();
            runtime.check_cancelled();
            if (request.value("stream", false)) {
                runtime.begin_token_stream([&](size_t index, const Json &token) {
                    response_incomplete = true;
                    write_header(
                        Json{{"ok", true}, {"event", "token"}, {"output", "tokens"},
                             {"index", index}, {"token", token}},
                        &runtime
                    );
                    response_incomplete = false;
                });
            }
            runtime.program(*runtime.active_program);
            runtime.finish_token_stream();
            for (const auto &[name, tensor] : runtime.base.at("outputs").items()) {
                Allocation output;
                Shape shape;
                std::string dtype;
                materialize(runtime, tensor.get<std::string>(), output, shape, dtype);
                descriptors.push_back(Json{{"name", name}, {"shape", shape}, {"dtype", dtype}, {"nbytes", output.size}});
                outputs.push_back(std::move(output));
            }
            runtime.check_cancelled();
            const uint64_t finished = now_ns();
            const auto &samples = runtime.sample_times;
            Json stats{{"execute_ns", finished - runtime.request_start},
                       {"host_copy_bytes", runtime.host_copy_bytes}, {"sample_count", samples.size()}};
            if (runtime.profile) {
                stats["native_ns"] = runtime.native_ns;
                stats["native_calls"] = runtime.native_calls;
                stats["host_copy_ns"] = runtime.host_copy_ns;
                stats["native_by_kernel"] = Json::object();
                for (const auto &[name, timing] : runtime.native_by_kernel) {
                    stats["native_by_kernel"][name] = Json::array({timing.first, timing.second});
                }
                stats["instruction_ns"] = Json::object();
                for (const auto &[name, timing] : runtime.instruction_ns) {
                    stats["instruction_ns"][name] = Json::array({timing.first, timing.second});
                }
                Json copies = Json::object();
                auto layout = [](const CacheCopyLayout &value) {
                    return Json{{"target_shape", value.target_shape}, {"target_strides", value.target_strides},
                                {"source_shape", value.source_shape}, {"source_strides", value.source_strides}, {"dtype", value.dtype}};
                };
                for (const auto &[name, value] : runtime.copy_by_cache) {
                    copies[name] = Json{{"ns", value.ns}, {"count", value.count}, {"bytes", value.bytes},
                                        {"first", layout(value.first)}, {"last", layout(value.last)}};
                }
                stats["copy_by_cache"] = std::move(copies);
            }
            if (!samples.empty()) {
                stats["first_sample_ns"] = samples.front();
                stats["last_sample_ns"] = samples.back();
                stats["decode_ns"] = samples.back() - samples.front();
            }
            response_incomplete = true;
            write_header(Json{{"ok", true}, {"outputs", descriptors}, {"stats", stats}}, &runtime);
            for (const auto &output : outputs) {
                write_exact(output.data, output.size, &runtime);
            }
        } catch (const Cancelled &) {
            if (!response_incomplete) {
                try {
                    write_cancelled();
                } catch (const std::exception &) {
                    // The exit status reports cancellation if the response pipe closed.
                }
            }
            return 130;
        } catch (const std::exception &error) {
            if (!response_incomplete) {
                write_header(Json{{"ok", false}, {"error", error.what()}, {"cancelled", false}});
            }
            break;
        }
    }
    return 0;
}
}

#ifdef _WIN32
int wmain(int argc, wchar_t **argv) {
#else
int main(int argc, char **argv) {
#endif
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    try {
        configure_pipes();
        std::filesystem::path directory;
        size_t threads = 0;
        uint64_t capabilities = 0;
        for (int i = 1; i + 1 < argc; i += 2) {
            const auto key = std::filesystem::path(argv[i]).string();
            if (key == "--bundle") {
                directory = argv[i + 1];

            } else if (key == "--threads") {
                threads = std::stoull(argv[i + 1]);
            } else if (key == "--capabilities") {
                capabilities = std::stoull(argv[i + 1]);
            } else {
                throw std::runtime_error("unknown worker option: " + key);
            }
        }
        if (argc % 2 != 1 || directory.empty() || threads < 1 || threads > 65536) {
            throw std::runtime_error("expected --bundle DIR --threads N --capabilities MASK");
        }
        std::optional<Runtime> runtime;
        std::atomic<Runtime *> active_runtime{nullptr};
#ifdef _WIN32
        HANDLE writer = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                             &writer, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
            throw std::runtime_error("cannot monitor runtime response pipe");
        }
        const std::unique_ptr<void, decltype(&CloseHandle)> writer_handle(writer, &CloseHandle);
#endif
        std::jthread watch_parent([&](std::stop_token stop) {
            if (!parent_disconnected(stop)) {
                return;
            }
            if (const auto current = active_runtime.load(std::memory_order_acquire)) {
                trillim_cpu_cancel(current->pool);
            }
            const auto deadline = std::chrono::steady_clock::now() + 100ms;
            while (!stop.stop_requested() && std::chrono::steady_clock::now() < deadline) {
#ifdef _WIN32
                CancelSynchronousIo(writer_handle.get());
#endif
                std::this_thread::sleep_for(10ms);
            }
            if (!stop.stop_requested()) {
                std::_Exit(130);
            }
        });
        runtime.emplace(directory, threads, capabilities);
        active_runtime.store(&*runtime, std::memory_order_release);
        return serve(*runtime);
    } catch (const std::exception &error) {
        try {
            write_header(Json{{"ok", false}, {"error", error.what()}, {"cancelled", false}});
        } catch (...) {
            std::cerr << error.what() << '\n';
        }
        return 1;
    }
}
