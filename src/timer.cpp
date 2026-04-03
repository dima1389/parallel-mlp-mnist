#include "timer.h"

// Capture the current high-resolution timestamp as start.
void timer_start(Timer& t) {
    t.start_time = std::chrono::high_resolution_clock::now();
}

// Capture the current high-resolution timestamp as stop.
void timer_stop(Timer& t) {
    t.stop_time = std::chrono::high_resolution_clock::now();
}

// Compute elapsed time in seconds with microsecond precision.
double timer_elapsed_sec(const Timer& t) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        t.stop_time - t.start_time);
    return static_cast<double>(duration.count()) / 1e6;
}
