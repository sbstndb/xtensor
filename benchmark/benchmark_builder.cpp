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
#include "xtensor/core/xnoalias.hpp"

#include "benchmark_utils.hpp"

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

    template <class T>
    inline auto builder_xarange(benchmark::State& state)
    {
	const int size = state.range(0); 
        for (auto _ : state)
        {
            T res = xt::arange(0, size);
            benchmark::DoNotOptimize(res.storage().data());
        }
    }

    template <class T>
    inline auto builder_xarange_manual(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        for (auto _ : state)
        {
            T res = T::from_shape({size});
            for (std::size_t i = 0; i < size; ++i)
            {
                res.storage()[i] = i;
            }
            benchmark::DoNotOptimize(res.data());
        }
    }

    inline auto builder_iota_vector(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        for (auto _ : state)
        {
            xt::uvector<double> a{};
            a.resize(size);
            std::iota(a.begin(), a.end(), 0);
            benchmark::DoNotOptimize(a.data());
        }
    }

    template <class T>
    inline auto builder_arange_for_loop_assign(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        for (auto _ : state)
        {
            auto expr = xt::arange(0, static_cast<int>(size));
            T res = T::from_shape({size});
            for (std::size_t i = 0; i < size; ++i)
            {
                res(i) = expr(i);
            }
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class T>
    inline auto builder_arange_for_loop_iter_assign(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        for (auto _ : state)
        {
            auto expr = xt::arange<double>(0, size);
            T res = T::from_shape({size});
            auto xend = expr.cend();
            auto reit = res.begin();
            for (auto it = expr.cbegin(); it != xend; ++it)
            {
                *reit++ = *it;
            }
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class T>
    inline auto builder_arange_for_loop_iter_assign_backward(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        for (auto _ : state)
        {
            auto expr = xt::arange<double>(0, size);
            T res = T::from_shape({size});
            auto xend = expr.cend();
            auto reit = res.begin();
            auto it = expr.cbegin();
            for (ptrdiff_t n = size; n > 0; --n)
            {
                *reit = *it;
                ++it;
                ++reit;
            }
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class T>
    inline auto builder_arange_assign_iterator(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        for (auto _ : state)
        {
            auto xa = xt::arange(0, static_cast<int>(size));
            T res = T::from_shape({size});
            std::copy(xa.cbegin(), xa.cend(), res.begin());
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class T>
    inline auto builder_std_iota(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        for (auto _ : state)
        {
            T res = T::from_shape({size});
            std::iota(res.begin(), res.end(), 0);
            benchmark::DoNotOptimize(res.data());
        }
    }

    inline auto builder_ones(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        for (auto _ : state)
        {
            xt::xarray<double> res = xt::ones<double>({size, size});
            benchmark::DoNotOptimize(res.data());
        }
    }

    inline auto builder_ones_assign_iterator(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        auto xo = xt::ones<double>({size, size});
        for (auto _ : state)
        {
            xt::xarray<double> res(xt::dynamic_shape<size_t>{size, size});
            auto xo = xt::ones<double>({size, size});
            std::copy(xo.begin(), xo.end(), res.begin());
            benchmark::DoNotOptimize(res.storage().data());
        }
    }

    inline auto builder_ones_expr_for(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        auto xo = xt::ones<double>({size, size});

        for (auto _ : state)
        {
            xt::xtensor<double, 2> res(xt::static_shape<size_t, 2>({size, size}));
            auto xo = xt::ones<double>({size, size}) * 0.15;
            for (std::size_t i = 0; i < xo.shape()[0]; ++i)
            {
                for (std::size_t j = 0; j < xo.shape()[1]; ++j)
                {
                    res(i, j) = xo(i, j);
                }
            }
            benchmark::DoNotOptimize(res.storage().data());
        }
    }

    inline auto builder_ones_expr(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        auto xo = xt::ones<double>({size, size});

        for (auto _ : state)
        {
            xt::xtensor<double, 2> res = xt::ones<double>({size, size}) * 0.15;
            benchmark::DoNotOptimize(res.storage().data());
        }
    }

    inline auto builder_ones_expr_fill(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        auto xo = xt::ones<double>({size, size});

        for (auto _ : state)
        {
            xt::xtensor<double, 2> res = xt::xtensor<double, 2>::from_shape({size, size});
            std::fill(res.begin(), res.end(), 0.15);
            benchmark::DoNotOptimize(res.storage().data());
        }
    }

    inline auto builder_std_fill(benchmark::State& state)
    {
	    const std::size_t size = state.range(0);
        for (auto _ : state)
        {
            xt::xarray<double> res(xt::dynamic_shape<std::size_t>{size, size});
            std::fill(res.begin(), res.end(), 1);
            benchmark::DoNotOptimize(res.storage().data());
        }
    }

    BENCHMARK_TEMPLATE(builder_xarange, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_xarange, xtensor<double, 1>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_xarange_manual, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_xarange_manual, xtensor<double, 1>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_arange_for_loop_assign, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_arange_for_loop_assign, xtensor<double, 1>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});

    BENCHMARK_TEMPLATE(builder_arange_assign_iterator, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_arange_assign_iterator, xtensor<double, 1>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_arange_for_loop_iter_assign, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_arange_for_loop_iter_assign_backward, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_arange_for_loop_iter_assign, xtensor<double, 1>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_arange_for_loop_iter_assign_backward, xtensor<double, 1>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK_TEMPLATE(builder_std_iota, xarray<double>)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK(builder_iota_vector)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    BENCHMARK(builder_ones)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min_2d, max_2d, threshold1_2d, threshold2_2d);});
    BENCHMARK(builder_ones_assign_iterator)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min_2d, max_2d, threshold1_2d, threshold2_2d);});
    BENCHMARK(builder_ones_expr)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min_2d, max_2d, threshold1_2d, threshold2_2d);});
    BENCHMARK(builder_ones_expr_fill)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min_2d, max_2d, threshold1_2d, threshold2_2d);});
    BENCHMARK(builder_ones_expr_for)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min_2d, max_2d, threshold1_2d, threshold2_2d);});
    BENCHMARK(builder_std_fill)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min_2d, max_2d, threshold1_2d, threshold2_2d);});
}
