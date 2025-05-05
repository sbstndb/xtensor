/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <benchmark/benchmark.h>

#include "benchmark_utils.hpp"

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xfixed.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/generators/xbuilder.hpp"

namespace xt
{
/**
void CustomArguments(
                benchmark::internal::Benchmark* b,
                int start ,
                int end,
                int threshold1,
                int threshold2
                ) {

  // Phase linéaire (incréments de 1)
  for (int i = start; i < threshold1 && i <= end; ++i) {
    b->Arg(i);
  }
  // Phase linéaire (incréments de 16)
  for (int i = threshold1; i <= threshold2 && i <= end; i+=16) {
    b->Arg(i);
  }
  // Phase exponentielle (puissances de 2)
  for (int i = threshold2 * 2; i <= end; i *= 2) {
    b->Arg(i);
  }
}


int min = 1 ;
int max = 100000 ;
int threshold1 = 128 ;
int threshold2 = 8096 ;


int min_s = 1 ;
int max_s = 8096 ;
int threshold1_s = 32 ;
int threshold2_s = 256 ;
**/
    void benchmark_empty(benchmark::State& state)
    {
	    const int size = state.range(0);
        for (auto _ : state)
        {
            auto e = xt::empty<double>({size, size});
        }
    }

    template <class T>
    void benchmark_from_shape(benchmark::State& state)
    {
	    const int size = state.range(0);
        for (auto _ : state)
        {
            T e = T::from_shape({static_cast<int>(size), static_cast<int>(size)});
        }
    }

    template <class T>
    void benchmark_creation(benchmark::State& state)
    {
	    const int size = state.range(0);
        for (auto _ : state)
        {
            T e(typename T::shape_type({static_cast<int>(size), static_cast<int>(size)}));
        }
    }

    void benchmark_empty_to_xtensor(benchmark::State& state)
    {
	    const int size = state.range(0);
        for (auto _ : state)
        {
            xtensor<double, 2> e = xt::empty<double>({size, size});
        }
    }

    void benchmark_empty_to_xarray(benchmark::State& state)
    {
	    const int size = state.range(0);
        for (auto _ : state)
        {
            xarray<double> e = xt::empty<double>({size, size});
        }
    }

    BENCHMARK(benchmark_empty)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK(benchmark_empty_to_xtensor)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK(benchmark_empty_to_xarray)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(benchmark_from_shape, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(benchmark_from_shape, xtensor<double, 2>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(benchmark_creation, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(benchmark_creation, xtensor<double, 2>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
}
