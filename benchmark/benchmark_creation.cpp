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

    template <class T>
    void CREATION_benchmark_from_shape(benchmark::State& state)
    {
        const int size = state.range(0);
        for (auto _ : state)
        {
            T e = T::from_shape({static_cast<int>(size), static_cast<int>(size)});
        }
    }

    template <class T>
    void CREATION_benchmark_creation(benchmark::State& state)
    {
	const int size = state.range(0);
        for (auto _ : state)
        {
            T e(typename T::shape_type({static_cast<int>(size), static_cast<int>(size)}));
        }
    }


    void CREATION_benchmark_empty(benchmark::State& state)
    {
        const int size = state.range(0);
        for (auto _ : state)
        {
            auto e = xt::empty<double>({size, size});
        }
    }

    void CREATION_benchmark_empty_to_xtensor(benchmark::State& state)
    {
	const int size = state.range(0);
        for (auto _ : state)
        {
            xtensor<double, 2> e = xt::empty<double>({size, size});
        }
    }

    void CREATION_benchmark_empty_to_xarray(benchmark::State& state)
    {
	const int size = state.range(0);
        for (auto _ : state)
        {
            xarray<double> e = xt::empty<double>({size, size});
        }
    }

    BENCHMARK(CREATION_benchmark_empty)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK(CREATION_benchmark_empty_to_xtensor)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK(CREATION_benchmark_empty_to_xarray)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});

    BENCHMARK_TEMPLATE(CREATION_benchmark_from_shape, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(CREATION_benchmark_from_shape, xtensor<double, 2>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});

    BENCHMARK_TEMPLATE(CREATION_benchmark_creation, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(CREATION_benchmark_creation, xtensor<double, 2>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});

}
