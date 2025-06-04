
#pragma once
#include <benchmark/benchmark.h>


// I want this to be able to select two different configurations a compile time.
// - complete mode : benchmark for all numbers for very fine benchmarking
// - fast mode : benchmark only few values to be able to read them in th terminal. 
//
#ifdef BENCHMARK_COMPLETE_MODE
constexpr int linear_increment_1 = 1 ; 
constexpr int linear_increment_2 = 16 ;
constexpr int exponential_increment = 2 ;
#else
constexpr int linear_increment_1 = 1 ;
constexpr int linear_increment_2 = 16 ;
constexpr int exponential_increment = 4 ;
#endif

inline void CustomArguments(benchmark::internal::Benchmark* b,
                            int start,
                            int end,
                            int threshold1,
                            int threshold2)
{
    // Phase linéaire (incréments de 1)
    for (int i = start; i < threshold1 && i <= end; i+= linear_increment_1)
    {
        b->Arg(i);
    }
    // Phase linéaire (incréments de 16)
    for (int i = threshold1; i <= threshold2 && i <= end; i += linear_increment_2)
    {
        b->Arg(i);
    }
    // Phase exponentielle (puissances de 2)
    for (int i = threshold2 * 2; i <= end; i *= exponential_increment)
    {
        b->Arg(i);
    }
}



#ifdef BENCHMARK_COMPLETE_MODE
inline constexpr int min = 1;
inline constexpr int max = 1024*8;
inline constexpr int threshold1 = 64;
inline constexpr int threshold2 = 1024;

#else
inline constexpr int min = 64;
inline constexpr int max = 1024;
inline constexpr int threshold1 = 64;
inline constexpr int threshold2 = 32;
#endif



inline constexpr int min_2d = 1;
inline constexpr int max_2d = 100;
inline constexpr int threshold1_2d = 8;
inline constexpr int threshold2_2d = 32;


