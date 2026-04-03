#ifndef TIMER_H
#define TIMER_H

// High-resolution wall-clock timer for benchmarking training epochs.

#include <chrono>

// Stores start and stop timestamps for measuring elapsed time.
struct Timer {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;
};

// Record the current time as the start point.
void   timer_start(Timer& t);

// Record the current time as the stop point.
void   timer_stop(Timer& t);

// Return elapsed time between start and stop in seconds.
double timer_elapsed_sec(const Timer& t);

#endif // TIMER_H
