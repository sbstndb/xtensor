/***************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <benchmark/benchmark.h>

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/core/xnoalias.hpp"
#include "xtensor/generators/xbuilder.hpp"


#include "benchmark_utils.hpp"


namespace xt
{
    void EXPRESSION_lambda_cube(benchmark::State& state)
    {
        xtensor<double, 2> x = empty<double>({state.range(0), state.range(0)});
        xtensor<double, 2> res = empty<double>({state.range(0), state.range(0)});	
        for (auto _ : state)
        {
	    xt::noalias(res) = xt::cube(x);
            benchmark::DoNotOptimize(res.data());
        }
    }

    void EXPRESSION_xexpression_cube(benchmark::State& state)
    {
        xtensor<double, 2> x = empty<double>({state.range(0), state.range(0)});
        xtensor<double, 2> res = empty<double>({state.range(0), state.range(0)});	
        for (auto _ : state)
        {
            xt::noalias(res) = x * x * x;
            benchmark::DoNotOptimize(res.data());
        }
    }

    void EXPRESSION_lambda_higher_pow(benchmark::State& state)
    {
        xtensor<double, 2> x = empty<double>({state.range(0), state.range(0)});
        xtensor<double, 2> res = empty<double>({state.range(0), state.range(0)});

        for (auto _ : state)
        {
	    xt::noalias(res) = xt::pow<16>(x);
            benchmark::DoNotOptimize(res.data());
        }
    }

    void EXPRESSION_xsimd_higher_pow(benchmark::State& state)
    {
        xtensor<double, 2> x = empty<double>({state.range(0), state.range(0)});
	xtensor<double, 2> res = empty<double>({state.range(0), state.range(0)});

        for (auto _ : state)
        {
	    xt::noalias(res) = xt::pow(x, 16);
            benchmark::DoNotOptimize(res.data());
        }
    }

    void EXPRESSION_xexpression_higher_pow(benchmark::State& state)
    {
        xtensor<double, 2> x = empty<double>({state.range(0), state.range(0)});
        xtensor<double, 2> res = empty<double>({state.range(0), state.range(0)});	
        for (auto _ : state)
        {
            xt::noalias(res) = x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x;
            benchmark::DoNotOptimize(res.data());
        }
    }

    BENCHMARK(EXPRESSION_lambda_cube)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK(EXPRESSION_xexpression_cube)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK(EXPRESSION_lambda_higher_pow)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK(EXPRESSION_xsimd_higher_pow)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK(EXPRESSION_xexpression_higher_pow)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
}
