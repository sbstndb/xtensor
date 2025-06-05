/***************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <chrono>
#include <cstddef>
#include <string>

#include <benchmark/benchmark.h>

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xnoalias.hpp"
#include "xtensor/core/xstrides.hpp"
#include "xtensor/misc/xmanipulation.hpp"
#include "xtensor/views/xstrided_view.hpp"
#include "xtensor/views/xview.hpp"

#include "benchmark_utils.hpp"

namespace xt
{
    // Thanks to Ullrich Koethe for these benchmarks
    // https://github.com/xtensor-stack/xtensor/issues/695
    namespace view_benchmarks
    {

        template <class V>
        void VIEWS_dynamic_iterator(benchmark::State& state)
        {
            const int size = state.range(0);		
            xt::xtensor<V, 2> data = xt::ones<V>({size, size});
            xt::xtensor<V, 1> res = xt::ones<V>({size});

            auto v = xt::strided_view(data, xt::xstrided_slice_vector{xt::all(), size / 2});
            for (auto _ : state)
            {
                std::copy(v.begin(), v.end(), res.begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void VIEWS_iterator(benchmark::State& state)
        {
            const int size = state.range(0);		
            xt::xtensor<V, 2> data = xt::ones<V>({size, size});
            xt::xtensor<V, 1> res = xt::ones<V>({size});

            auto v = xt::view(data, xt::all(), size / 2);
            for (auto _ : state)
            {
                std::copy(v.begin(), v.end(), res.begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void VIEWS_loop(benchmark::State& state)
        {
            const int size = state.range(0);		
            xt::xtensor<V, 2> data = xt::ones<V>({size, size});
            xt::xtensor<V, 1> res = xt::ones<V>({size});

            auto v = xt::strided_view(data, xt::xstrided_slice_vector{xt::all(), size / 2});
            for (auto _ : state)
            {
                for (std::size_t k = 0; k < v.shape()[0]; ++k)
                {
                    res(k) = v(k);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void VIEWS_loop_view(benchmark::State& state)
        {
            const int size = state.range(0);		
            xt::xtensor<V, 2> data = xt::ones<V>({size, size});
            xt::xtensor<V, 1> res = xt::ones<V>({size});

            auto v = xt::view(data, xt::all(), size / 2);
            for (auto _ : state)
            {
                for (std::size_t k = 0; k < v.shape()[0]; ++k)
                {
                    res(k) = v(k);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void VIEWS_loop_raw(benchmark::State& state)
        {
            const int size = state.range(0);		
            xt::xtensor<V, 2> data = xt::ones<V>({size, size});
            xt::xtensor<V, 1> res = xt::ones<V>({size});

            for (auto _ : state)
            {
                std::size_t j = size / 2;
                for (std::size_t k = 0; k < size; ++k)
                {
                    res(k) = data(k, j);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void VIEWS_assign(benchmark::State& state)
        {
            const int size = state.range(0);		
            xt::xtensor<V, 2> data = xt::ones<V>({size, size});
            xt::xtensor<V, 1> res = xt::ones<V>({size});

            auto v = xt::strided_view(data, xt::xstrided_slice_vector{xt::all(), size / 2});
            for (auto _ : state)
            {
                xt::noalias(res) = v;
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void VIEWS_assign_view(benchmark::State& state)
        {
            const int size = state.range(0);		
            xt::xtensor<V, 2> data = xt::ones<V>({size, size});
            xt::xtensor<V, 1> res = xt::ones<V>({size});

            auto v = xt::view(data, xt::all(), size / 2);
            auto r = xt::view(res, xt::all());
            for (auto _ : state)
            {
                r = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void VIEWS_assign_strided_view(benchmark::State& state)
        {
            const int size = state.range(0);		
            xt::xtensor<V, 2> data = xt::ones<V>({size, size});
            xt::xtensor<V, 1> res = xt::ones<V>({size});

            auto v = xt::strided_view(data, xt::xstrided_slice_vector{xt::all(), size / 2});
            auto r = xt::strided_view(res, xt::xstrided_slice_vector{xt::all()});

            for (auto _ : state)
            {
                r = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void VIEWS_assign_view_noalias(benchmark::State& state)
        {
	    const int size = state.range(0);
            xt::xtensor<V, 2> data = xt::ones<V>({size, size});
            xt::xtensor<V, 1> res = xt::ones<V>({size});

            auto v = xt::view(data, xt::all(), size / 2);
            auto r = xt::view(res, xt::all());
            for (auto _ : state)
            {
                xt::noalias(r) = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void VIEWS_assign_strided_view_noalias(benchmark::State& state)
        {
	    const int size = state.range(0);
            xt::xtensor<V, 2> data = xt::ones<V>({size, size});
            xt::xtensor<V, 1> res = xt::ones<V>({size});

            auto v = xt::strided_view(data, xt::xstrided_slice_vector{xt::all(), size / 2});
            auto r = xt::strided_view(res, xt::xstrided_slice_vector{xt::all()});

            for (auto _ : state)
            {
                xt::noalias(r) = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        BENCHMARK_TEMPLATE(VIEWS_dynamic_iterator, float)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
        BENCHMARK_TEMPLATE(VIEWS_iterator, float)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
        BENCHMARK_TEMPLATE(VIEWS_loop, float)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
        BENCHMARK_TEMPLATE(VIEWS_loop_view, float)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
        BENCHMARK_TEMPLATE(VIEWS_loop_raw, float)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
        BENCHMARK_TEMPLATE(VIEWS_assign, float)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
        BENCHMARK_TEMPLATE(VIEWS_assign_view, float)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
        BENCHMARK_TEMPLATE(VIEWS_assign_strided_view, float)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
        BENCHMARK_TEMPLATE(VIEWS_assign_view_noalias, float)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
        BENCHMARK_TEMPLATE(VIEWS_assign_strided_view_noalias, float)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});
    }

    namespace finite_diff
    {
        inline auto VIEWS_stencil_threedirections(benchmark::State& state)
        {
	    const int size = state.range(0);
            const std::array<size_t, 3> shape = {size, size, size};
            xt::xtensor<double, 3> a(shape), b(shape);
            for (auto _ : state)
            {
                const std::array<size_t, 3> shape = {size, size, size};
                xt::xtensor<double, 3> a(shape), b(shape);
                auto core = xt::range(1, size - 1);
                xt::noalias(xt::view(b, core, core, core)
                ) = 1.0 / 7.0
                    * (xt::view(a, core, core, core) + xt::view(a, core, core, xt::range(2, size))
                       + xt::view(a, core, core, xt::range(0, size - 2))
                       + xt::view(a, core, xt::range(2, size), core)
                       + xt::view(a, core, xt::range(0, size - 2), core)
                       + xt::view(a, xt::range(2, size), core, core)
                       + xt::view(a, xt::range(0, size - 2), core, core));
                benchmark::DoNotOptimize(b);
            }
        }

        inline auto VIEWS_stencil_twodirections(benchmark::State& state)
        {
            const int size = state.range(0);		
            const std::array<size_t, 3> shape = {size, size, size};
            xt::xtensor<double, 3> a(shape), b(shape);
            for (auto _ : state)
            {
                auto core = xt::range(1, size - 1);
                xt::noalias(xt::view(b, core, core, core)
                ) = 1.0 / 7.0
                    * (xt::view(a, core, core, core) + xt::view(a, core, xt::range(2, size), core)
                       + xt::view(a, core, xt::range(0, size - 2), core)
                       + xt::view(a, xt::range(2, size), core, core)
                       + xt::view(a, xt::range(0, size - 2), core, core));
                benchmark::DoNotOptimize(b);
            }
        }

        inline auto VIEWS_stencil_onedirection(benchmark::State& state)
        {
            const int size = state.range(0);		
            const std::array<size_t, 3> shape = {size, size, size};
            xt::xtensor<double, 3> a(shape), b(shape);
            for (auto _ : state)
            {
                auto core = xt::range(1, size - 1);
                xt::noalias(xt::view(b, core, core, core)
                ) = 1.0 / 2.0
                    * (xt::view(a, xt::range(2, size), core, core)
                       - xt::view(a, xt::range(0, size - 2), core, core));
                benchmark::DoNotOptimize(b);
            }
        }

        BENCHMARK(VIEWS_stencil_threedirections)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});;
        BENCHMARK(VIEWS_stencil_twodirections)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});;
        BENCHMARK(VIEWS_stencil_onedirection)->Apply([](benchmark::internal::Benchmark* b)
                        {CustomArguments(b, min, max, threshold1, threshold2);});;

    }

    namespace stridedview
    {

        template <layout_type L1, layout_type L2>
        inline auto VIEWS_transpose_assign(benchmark::State& state, std::vector<std::size_t> shape)
        {
            xarray<double, L1> x = xt::arange<double>(compute_size(shape));
            x.resize(shape);

            xarray<double, L2> res;
            res.resize(std::vector<std::size_t>(shape.rbegin(), shape.rend()));

            for (auto _ : state)
            {
                res = transpose(x);
            }
        }

        auto VIEWS_transpose_assign_rm_rm = VIEWS_transpose_assign<layout_type::row_major, layout_type::row_major>;
        auto VIEWS_transpose_assign_cm_cm = VIEWS_transpose_assign<layout_type::column_major, layout_type::column_major>;
        auto VIEWS_transpose_assign_rm_cm = VIEWS_transpose_assign<layout_type::row_major, layout_type::column_major>;
        auto VIEWS_transpose_assign_cm_rm = VIEWS_transpose_assign<layout_type::column_major, layout_type::row_major>;

        BENCHMARK_CAPTURE(VIEWS_transpose_assign_rm_rm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(VIEWS_transpose_assign_cm_cm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(VIEWS_transpose_assign_rm_cm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(VIEWS_transpose_assign_cm_rm, 10x20x500, {10, 20, 500});
    }
}
