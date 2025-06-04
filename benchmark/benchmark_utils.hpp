
#pragma once
#include <benchmark/benchmark.h>

inline void CustomArguments(benchmark::internal::Benchmark* b,
                            int start,
                            int end,
                            int threshold1,
                            int threshold2)
{
    // Phase linéaire (incréments de 1)
    for (int i = start; i < threshold1 && i <= end; ++i)
    {
        b->Arg(i);
    }
    // Phase linéaire (incréments de 16)
    for (int i = threshold1; i <= threshold2 && i <= end; i += 16)
    {
        b->Arg(i);
    }
    // Phase exponentielle (puissances de 2)
    for (int i = threshold2 * 2; i <= end; i *= 2)
    {
        b->Arg(i);
    }
}


inline constexpr int min = 1;
inline constexpr int max = 1024*8;
inline constexpr int threshold1 = 64;
inline constexpr int threshold2 = 1024;


inline constexpr int min_2d = 1;
inline constexpr int max_2d = 100;
inline constexpr int threshold1_2d = 8;
inline constexpr int threshold2_2d = 32;


