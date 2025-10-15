/***************************************************************************
 * Copyright (c) 2024, xtensor views2 contributors                          *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <benchmark/benchmark.h>

#include "xtensor/containers/xtensor.hpp"
#include "xtensor/xviews2.hpp"

namespace xt
{
    namespace view_benchmarks_v2
    {
        constexpr int SIZE = 2;

        // ==================== Iteration Benchmarks ====================

        template <class V>
        void view_v2_dynamic_iterator(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::views2::strided_view(data, xt::views2::all(), SIZE / 2);
            for (auto _ : state)
            {
                std::copy(v.begin(), v.end(), res.begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_v2_iterator(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::views2::view(data, xt::views2::all(), SIZE / 2);
            for (auto _ : state)
            {
                std::copy(v.begin(), v.end(), res.begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_v2_loop(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::views2::strided_view(data, xt::views2::all(), SIZE / 2);
            for (auto _ : state)
            {
                for (std::size_t k = 0; k < v.shape(0); ++k)
                {
                    res(k) = v(k);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_v2_loop_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::views2::view(data, xt::views2::all(), SIZE / 2);
            for (auto _ : state)
            {
                for (std::size_t k = 0; k < v.shape(0); ++k)
                {
                    res(k) = v(k);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_v2_loop_raw(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            for (auto _ : state)
            {
                std::size_t j = SIZE / 2;
                for (std::size_t k = 0; k < SIZE; ++k)
                {
                    res(k) = data(k, j);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        // ==================== Assignment Benchmarks ====================

        template <class V>
        void view_v2_assign(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::views2::strided_view(data, xt::views2::all(), SIZE / 2);
            for (auto _ : state)
            {
                xt::views2::noalias(res) = v;
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_v2_assign_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::views2::view(data, xt::views2::all(), SIZE / 2);
            auto r = xt::views2::view(res, xt::views2::all());
            for (auto _ : state)
            {
                r = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void view_v2_assign_strided_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::views2::strided_view(data, xt::views2::all(), SIZE / 2);
            auto r = xt::views2::strided_view(res, xt::views2::all());

            for (auto _ : state)
            {
                r = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void view_v2_assign_view_noalias(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::views2::view(data, xt::views2::all(), SIZE / 2);
            auto r = xt::views2::view(res, xt::views2::all());
            for (auto _ : state)
            {
                xt::views2::noalias(r) = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void view_v2_assign_strided_view_noalias(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::views2::strided_view(data, xt::views2::all(), SIZE / 2);
            auto r = xt::views2::strided_view(res, xt::views2::all());

            for (auto _ : state)
            {
                xt::views2::noalias(r) = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        // Register benchmarks
        BENCHMARK_TEMPLATE(view_v2_dynamic_iterator, float);
        BENCHMARK_TEMPLATE(view_v2_iterator, float);
        BENCHMARK_TEMPLATE(view_v2_loop, float);
        BENCHMARK_TEMPLATE(view_v2_loop_view, float);
        BENCHMARK_TEMPLATE(view_v2_loop_raw, float);
        BENCHMARK_TEMPLATE(view_v2_assign, float);
        BENCHMARK_TEMPLATE(view_v2_assign_view, float);
        BENCHMARK_TEMPLATE(view_v2_assign_strided_view, float);
        BENCHMARK_TEMPLATE(view_v2_assign_view_noalias, float);
        BENCHMARK_TEMPLATE(view_v2_assign_strided_view_noalias, float);
    }

    // ==================== Stencil Benchmarks ====================

    namespace stencil_v2
    {
        inline auto stencil_onedirection_v2(benchmark::State& state, size_t size)
        {
            const std::array<size_t, 3> shape = {size, size, size};
            xt::xtensor<double, 3> a(shape), b(shape);
            auto core = views2::range(std::ptrdiff_t(1), std::ptrdiff_t(size - 1));

            for (auto _ : state)
            {
                xt::views2::noalias(xt::views2::view(b, core, core, core))
                    = 0.5 * (xt::views2::view(a, views2::range(std::ptrdiff_t(2), std::ptrdiff_t(size)), core, core)
                             - xt::views2::view(a, views2::range(std::ptrdiff_t(0), std::ptrdiff_t(size - 2)), core, core));
                benchmark::DoNotOptimize(b);
            }
        }

        inline auto stencil_onedirection_v2_precomputed(benchmark::State& state, size_t size)
        {
            const std::array<size_t, 3> shape = {size, size, size};
            xt::xtensor<double, 3> a(shape), b(shape);
            auto core = views2::range(std::ptrdiff_t(1), std::ptrdiff_t(size - 1));

            // Precompute views
            auto b_view = xt::views2::view(b, core, core, core);
            auto a_view1 = xt::views2::view(a, views2::range(std::ptrdiff_t(2), std::ptrdiff_t(size)), core, core);
            auto a_view2 = xt::views2::view(a, views2::range(std::ptrdiff_t(0), std::ptrdiff_t(size - 2)), core, core);

            for (auto _ : state)
            {
                xt::views2::noalias(b_view) = 0.5 * (a_view1 - a_view2);
                benchmark::DoNotOptimize(b);
            }
        }

        inline auto stencil_twodirections_v2(benchmark::State& state, size_t size)
        {
            const std::array<size_t, 3> shape = {size, size, size};
            xt::xtensor<double, 3> a(shape), b(shape);
            auto core = views2::range(std::ptrdiff_t(1), std::ptrdiff_t(size - 1));

            for (auto _ : state)
            {
                xt::views2::noalias(xt::views2::view(b, core, core, core))
                    = (1.0 / 5.0)
                    * (xt::views2::view(a, core, core, core)
                       + xt::views2::view(a, core, views2::range(std::ptrdiff_t(2), std::ptrdiff_t(size)), core)
                       + xt::views2::view(a, core, views2::range(std::ptrdiff_t(0), std::ptrdiff_t(size - 2)), core)
                       + xt::views2::view(a, views2::range(std::ptrdiff_t(2), std::ptrdiff_t(size)), core, core)
                       + xt::views2::view(a, views2::range(std::ptrdiff_t(0), std::ptrdiff_t(size - 2)), core, core));
                benchmark::DoNotOptimize(b);
            }
        }

        inline auto stencil_threedirections_v2(benchmark::State& state, size_t size)
        {
            const std::array<size_t, 3> shape = {size, size, size};
            xt::xtensor<double, 3> a(shape), b(shape);
            auto core = views2::range(std::ptrdiff_t(1), std::ptrdiff_t(size - 1));

            for (auto _ : state)
            {
                xt::views2::noalias(xt::views2::view(b, core, core, core))
                    = (1.0 / 7.0)
                    * (xt::views2::view(a, core, core, core)
                       + xt::views2::view(a, core, core, views2::range(std::ptrdiff_t(2), std::ptrdiff_t(size)))
                       + xt::views2::view(a, core, core, views2::range(std::ptrdiff_t(0), std::ptrdiff_t(size - 2)))
                       + xt::views2::view(a, core, views2::range(std::ptrdiff_t(2), std::ptrdiff_t(size)), core)
                       + xt::views2::view(a, core, views2::range(std::ptrdiff_t(0), std::ptrdiff_t(size - 2)), core)
                       + xt::views2::view(a, views2::range(std::ptrdiff_t(2), std::ptrdiff_t(size)), core, core)
                       + xt::views2::view(a, views2::range(std::ptrdiff_t(0), std::ptrdiff_t(size - 2)), core, core));
                benchmark::DoNotOptimize(b);
            }
        }

        BENCHMARK_CAPTURE(stencil_onedirection_v2, stencil_v2_onedirection_50, 3);
        BENCHMARK_CAPTURE(stencil_onedirection_v2, stencil_v2_onedirection_100, 3);
        BENCHMARK_CAPTURE(stencil_onedirection_v2, stencil_v2_onedirection_200, 3);
        BENCHMARK_CAPTURE(stencil_onedirection_v2, stencil_v2_onedirection_300, 3);
        BENCHMARK_CAPTURE(stencil_onedirection_v2, stencil_v2_onedirection_500, 3);

        BENCHMARK_CAPTURE(stencil_onedirection_v2_precomputed, stencil_v2_onedirection_precomputed_50, 3);
        BENCHMARK_CAPTURE(stencil_onedirection_v2_precomputed, stencil_v2_onedirection_precomputed_100, 3);
        BENCHMARK_CAPTURE(stencil_onedirection_v2_precomputed, stencil_v2_onedirection_precomputed_200, 3);
        BENCHMARK_CAPTURE(stencil_onedirection_v2_precomputed, stencil_v2_onedirection_precomputed_300, 3);
        BENCHMARK_CAPTURE(stencil_onedirection_v2_precomputed, stencil_v2_onedirection_precomputed_500, 3);

        BENCHMARK_CAPTURE(stencil_twodirections_v2, stencil_v2_twodirections_50, 3);
        BENCHMARK_CAPTURE(stencil_twodirections_v2, stencil_v2_twodirections_100, 3);
        BENCHMARK_CAPTURE(stencil_twodirections_v2, stencil_v2_twodirections_200, 3);
        BENCHMARK_CAPTURE(stencil_twodirections_v2, stencil_v2_twodirections_300, 3);
        BENCHMARK_CAPTURE(stencil_twodirections_v2, stencil_v2_twodirections_500, 3);

        BENCHMARK_CAPTURE(stencil_threedirections_v2, stencil_v2_threedirections_50, 3);
        BENCHMARK_CAPTURE(stencil_threedirections_v2, stencil_v2_threedirections_100, 3);
        BENCHMARK_CAPTURE(stencil_threedirections_v2, stencil_v2_threedirections_200, 3);
        BENCHMARK_CAPTURE(stencil_threedirections_v2, stencil_v2_threedirections_300, 3);
        BENCHMARK_CAPTURE(stencil_threedirections_v2, stencil_v2_threedirections_500, 3);
    }
}
