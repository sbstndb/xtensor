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
        // ==================== Rank Detection ====================

        namespace detail
        {
            // Helper to detect if a type has tuple_size
            template <class T, class = void>
            struct has_tuple_size : std::false_type {};

            template <class T>
            struct has_tuple_size<T, std::void_t<std::integral_constant<std::size_t, std::tuple_size<T>::value>>>
                : std::true_type {};

            // Detect if container has static rank
            template <class Container, class = void>
            struct container_rank
            {
                static constexpr std::size_t value = dynamic_rank;
            };

            // Specialization for containers with static rank (shape_type has tuple_size)
            template <class Container>
            struct container_rank<Container, std::enable_if_t<has_tuple_size<typename std::decay_t<Container>::shape_type>::value>>
            {
                static constexpr std::size_t value = std::tuple_size_v<typename std::decay_t<Container>::shape_type>;
            };

            template <class Container>
            inline constexpr std::size_t container_rank_v = container_rank<Container>::value;

            // Check if container has dynamic rank
            template <class Container>
            inline constexpr bool is_dynamic_container_v = (container_rank_v<Container> == dynamic_rank);
        }  // namespace detail

        // ==================== Contiguity Detection ====================

        namespace detail
        {
            // Check if slices form a contiguous pattern for row-major
            // Pattern: (int*, all*, ...) or (int*, range, all*, ...)
            // An int cannot appear AFTER an all/range
            template <class... Slices>
            struct is_contiguous_pattern_impl;

            // Base case: empty
            template <>
            struct is_contiguous_pattern_impl<>
            {
                static constexpr bool value = true;
                static constexpr bool contains_int = false;
            };

            // all() slice - OK if no int follows
            template <class... Rest>
            struct is_contiguous_pattern_impl<all_tag, Rest...>
            {
                using rest = is_contiguous_pattern_impl<Rest...>;
                static constexpr bool value = !rest::contains_int && rest::value;
                static constexpr bool contains_int = rest::contains_int;
            };

            // Integer slice - always OK, marks that we have an int
            template <class T, class... Rest>
                requires std::is_integral_v<T>
            struct is_contiguous_pattern_impl<T, Rest...>
            {
                using rest = is_contiguous_pattern_impl<Rest...>;
                static constexpr bool value = rest::value;
                static constexpr bool contains_int = true;
            };

            // Range slice - OK if no int follows
            template <class T, class... Rest>
            struct is_contiguous_pattern_impl<xrange<T>, Rest...>
            {
                using rest = is_contiguous_pattern_impl<Rest...>;
                static constexpr bool value = !rest::contains_int && rest::value;
                static constexpr bool contains_int = rest::contains_int;
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

            // ==================== Dynamic Rank Helpers ====================

            // Compute shape for dynamic rank containers
            template <class Container, class... Slices>
            auto compute_view_shape_dynamic(const Container& c, const Slices&... slices)
            {
                std::vector<std::size_t> shape;
                std::size_t dim = 0;

                auto fill_shape = [&shape, &dim, &c](const auto& slice)
                {
                    using slice_type = std::decay_t<decltype(slice)>;
                    if constexpr (is_all_slice_v<slice_type>)
                    {
                        shape.push_back(c.shape()[dim]);
                    }
                    else if constexpr (is_range_slice_v<slice_type>)
                    {
                        shape.push_back(slice.size());
                    }
                    ++dim;
                };

                (fill_shape(slices), ...);

                return shape;
            }

            // Compute strides for dynamic rank containers
            template <class Container, class... Slices>
            auto compute_view_strides_dynamic(const Container& c, const Slices&... slices)
            {
                std::vector<std::ptrdiff_t> strides;
                std::size_t dim = 0;

                auto fill_strides = [&strides, &dim, &c](const auto& slice)
                {
                    using slice_type = std::decay_t<decltype(slice)>;
                    if constexpr (is_all_slice_v<slice_type>)
                    {
                        strides.push_back(c.strides()[dim]);
                    }
                    else if constexpr (is_range_slice_v<slice_type>)
                    {
                        strides.push_back(c.strides()[dim] * slice.step);
                    }
                    ++dim;
                };

                (fill_strides(slices), ...);

                return strides;
            }

        }  // namespace detail

        // ==================== View Builder ====================

        /**
         * @brief Create a view with compile-time contiguity detection
         * Supports both static rank (xtensor) and dynamic rank (xarray) containers
         */
        template <class Container, class... Slices>
        constexpr auto view(Container& c, Slices&&... slices)
        {
            constexpr bool is_contiguous = detail::is_contiguous_pattern_v<std::decay_t<Slices>...>;
            using value_type = typename std::decay_t<Container>::value_type;

            auto offset = detail::compute_view_offset(c, slices...);

            // Detect if container has dynamic rank (xarray) or static rank (xtensor)
            if constexpr (detail::is_dynamic_container_v<Container>)
            {
                // Dynamic rank path - use std::vector
                auto shape = detail::compute_view_shape_dynamic(c, slices...);
                auto strides = detail::compute_view_strides_dynamic(c, slices...);

                return basic_view<value_type, dynamic_rank, is_contiguous>(
                    c.data(),
                    shape,
                    strides,
                    offset,
                    c.layout()
                );
            }
            else
            {
                // Static rank path - use std::array
                constexpr std::size_t new_rank = ((is_all_slice_v<std::decay_t<Slices>> || is_range_slice_v<std::decay_t<Slices>> ? 1 : 0) + ...);

                auto shape = detail::compute_view_shape(c, slices...);
                auto strides = detail::compute_view_strides(c, slices...);

                return basic_view<value_type, new_rank, is_contiguous>(
                    c.data(),
                    shape,
                    strides,
                    offset,
                    c.layout()
                );
            }
        }

        /**
         * @brief Create a strided view (always non-contiguous)
         * Supports both static rank (xtensor) and dynamic rank (xarray) containers
         */
        template <class Container, class... Slices>
        constexpr auto strided_view(Container& c, Slices&&... slices)
        {
            using value_type = typename std::decay_t<Container>::value_type;

            auto offset = detail::compute_view_offset(c, slices...);

            // Detect if container has dynamic rank (xarray) or static rank (xtensor)
            if constexpr (detail::is_dynamic_container_v<Container>)
            {
                // Dynamic rank path - use std::vector
                auto shape = detail::compute_view_shape_dynamic(c, slices...);
                auto strides = detail::compute_view_strides_dynamic(c, slices...);

                return basic_view<value_type, dynamic_rank, false>(
                    c.data(),
                    shape,
                    strides,
                    offset,
                    layout_type::dynamic
                );
            }
            else
            {
                // Static rank path - use std::array
                constexpr std::size_t new_rank = ((is_all_slice_v<std::decay_t<Slices>> || is_range_slice_v<std::decay_t<Slices>> ? 1 : 0) + ...);

                auto shape = detail::compute_view_shape(c, slices...);
                auto strides = detail::compute_view_strides(c, slices...);

                return basic_view<value_type, new_rank, false>(
                    c.data(),
                    shape,
                    strides,
                    offset,
                    layout_type::dynamic
                );
            }
        }

    }  // namespace views2
}  // namespace xt

#endif
