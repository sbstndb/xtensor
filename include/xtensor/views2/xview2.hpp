/***************************************************************************
 * Copyright (c) 2024, xtensor views2 contributors                          *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_VIEWS2_HPP
#define XTENSOR_VIEWS2_HPP

#include "xview2_base.hpp"
#include "xslice2.hpp"
#include "../containers/xtensor.hpp"
#include "../containers/xarray.hpp"

namespace xt
{
    namespace views2
    {
        // ==================== Contiguity Detection ====================

        namespace detail
        {
            // Check if slices form a contiguous pattern for row-major
            // Pattern: (int*, all, all*, ...) or (int*, range(continuous), all*, ...)
            template <class... Slices>
            struct is_contiguous_pattern_impl;

            // Base case: empty
            template <>
            struct is_contiguous_pattern_impl<>
            {
                static constexpr bool value = true;
                static constexpr bool seen_all = false;
            };

            // all() slice
            template <class... Rest>
            struct is_contiguous_pattern_impl<all_tag, Rest...>
            {
                static constexpr bool value = is_contiguous_pattern_impl<Rest...>::value;
                static constexpr bool seen_all = true;
            };

            // Integer slice (before all)
            template <class T, class... Rest>
                requires std::is_integral_v<T>
            struct is_contiguous_pattern_impl<T, Rest...>
            {
                using rest = is_contiguous_pattern_impl<Rest...>;
                static constexpr bool value = !rest::seen_all && rest::value;
                static constexpr bool seen_all = rest::seen_all;
            };

            // Range slice (treated as potentially contiguous for now)
            template <class T, class... Rest>
            struct is_contiguous_pattern_impl<xrange<T>, Rest...>
            {
                using rest = is_contiguous_pattern_impl<Rest...>;
                static constexpr bool value = rest::value;
                static constexpr bool seen_all = true;
            };

            template <class... Slices>
            inline constexpr bool is_contiguous_pattern_v = is_contiguous_pattern_impl<Slices...>::value;

            // Compute new shape from slices
            template <class Container, class... Slices>
            constexpr auto compute_view_shape(const Container& c, const Slices&... slices)
            {
                constexpr std::size_t new_rank = ((is_all_slice_v<std::decay_t<Slices>> || is_range_slice_v<std::decay_t<Slices>> ? 1 : 0) + ...);

                // Create shape array
                std::array<std::size_t, new_rank> shape;
                std::size_t shape_idx = 0;
                std::size_t dim = 0;

                auto fill_shape = [&shape, &shape_idx, &dim, &c](const auto& slice)
                {
                    using slice_type = std::decay_t<decltype(slice)>;
                    if constexpr (is_all_slice_v<slice_type>)
                    {
                        shape[shape_idx++] = c.shape()[dim];
                    }
                    else if constexpr (is_range_slice_v<slice_type>)
                    {
                        shape[shape_idx++] = slice.size();
                    }
                    ++dim;
                };

                (fill_shape(slices), ...);

                return shape;
            }

            // Compute strides for view
            template <class Container, class... Slices>
            constexpr auto compute_view_strides(const Container& c, const Slices&... slices)
            {
                constexpr std::size_t new_rank = ((is_all_slice_v<std::decay_t<Slices>> || is_range_slice_v<std::decay_t<Slices>> ? 1 : 0) + ...);

                std::array<std::ptrdiff_t, new_rank> strides;
                std::size_t stride_idx = 0;
                std::size_t dim = 0;

                auto fill_strides = [&strides, &stride_idx, &dim, &c](const auto& slice)
                {
                    using slice_type = std::decay_t<decltype(slice)>;
                    if constexpr (is_all_slice_v<slice_type>)
                    {
                        strides[stride_idx++] = c.strides()[dim];
                    }
                    else if constexpr (is_range_slice_v<slice_type>)
                    {
                        strides[stride_idx++] = c.strides()[dim] * slice.step;
                    }
                    ++dim;
                };

                (fill_strides(slices), ...);

                return strides;
            }

            // Compute offset from integer slices
            template <class Container, class... Slices>
            constexpr std::size_t compute_view_offset(const Container& c, const Slices&... slices)
            {
                std::ptrdiff_t offset = 0;
                std::size_t dim = 0;

                auto add_offset = [&offset, &dim, &c](const auto& slice)
                {
                    using slice_type = std::decay_t<decltype(slice)>;
                    if constexpr (std::is_integral_v<slice_type>)
                    {
                        offset += static_cast<std::ptrdiff_t>(slice) * c.strides()[dim];
                    }
                    else if constexpr (is_range_slice_v<slice_type>)
                    {
                        offset += slice.start * c.strides()[dim];
                    }
                    ++dim;
                };

                (add_offset(slices), ...);

                return static_cast<std::size_t>(offset);
            }

        }  // namespace detail

        // ==================== View Builder ====================

        /**
         * @brief Create a view with compile-time contiguity detection
         */
        template <class Container, class... Slices>
        constexpr auto view(Container& c, Slices&&... slices)
        {
            constexpr bool is_contiguous = detail::is_contiguous_pattern_v<std::decay_t<Slices>...>;
            constexpr std::size_t new_rank = ((is_all_slice_v<std::decay_t<Slices>> || is_range_slice_v<std::decay_t<Slices>> ? 1 : 0) + ...);

            using value_type = typename std::decay_t<Container>::value_type;

            auto shape = detail::compute_view_shape(c, slices...);
            auto strides = detail::compute_view_strides(c, slices...);
            auto offset = detail::compute_view_offset(c, slices...);

            return basic_view<value_type, new_rank, is_contiguous>(
                c.data(),
                shape,
                strides,
                offset,
                c.layout()
            );
        }

        /**
         * @brief Create a strided view (always non-contiguous)
         */
        template <class Container, class... Slices>
        constexpr auto strided_view(Container& c, Slices&&... slices)
        {
            constexpr std::size_t new_rank = ((is_all_slice_v<std::decay_t<Slices>> || is_range_slice_v<std::decay_t<Slices>> ? 1 : 0) + ...);

            using value_type = typename std::decay_t<Container>::value_type;

            auto shape = detail::compute_view_shape(c, slices...);
            auto strides = detail::compute_view_strides(c, slices...);
            auto offset = detail::compute_view_offset(c, slices...);

            // Force non-contiguous
            return basic_view<value_type, new_rank, false>(
                c.data(),
                shape,
                strides,
                offset,
                layout_type::dynamic
            );
        }

    }  // namespace views2
}  // namespace xt

#endif
