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

            // newaxis slice - OK, doesn't affect contiguity
            template <class... Rest>
            struct is_contiguous_pattern_impl<newaxis_tag, Rest...>
            {
                using rest = is_contiguous_pattern_impl<Rest...>;
                static constexpr bool value = rest::value;
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

            // Range adaptor - OK if no int follows (same as xrange)
            template <class A, class B, class C, class... Rest>
            struct is_contiguous_pattern_impl<xrange_adaptor<A, B, C>, Rest...>
            {
                using rest = is_contiguous_pattern_impl<Rest...>;
                static constexpr bool value = !rest::contains_int && rest::value;
                static constexpr bool contains_int = rest::contains_int;
            };

            // keep_slice - NEVER contiguous
            template <class T, class... Rest>
            struct is_contiguous_pattern_impl<xkeep_slice<T>, Rest...>
            {
                static constexpr bool value = false;
                static constexpr bool contains_int = false;
            };

            // drop_slice - NEVER contiguous
            template <class T, class... Rest>
            struct is_contiguous_pattern_impl<xdrop_slice<T>, Rest...>
            {
                static constexpr bool value = false;
                static constexpr bool contains_int = false;
            };

            template <class... Slices>
            inline constexpr bool is_contiguous_pattern_v = is_contiguous_pattern_impl<Slices...>::value;

            // Compute new shape from slices
            template <class Container, class... Slices>
            constexpr auto compute_view_shape(const Container& c, Slices&... slices)
            {
                constexpr std::size_t new_rank = ((is_all_slice_v<std::decay_t<Slices>> || is_range_slice_v<std::decay_t<Slices>> || is_newaxis_v<std::decay_t<Slices>> || is_range_adaptor_v<std::decay_t<Slices>> || is_keep_slice_v<std::decay_t<Slices>> || is_drop_slice_v<std::decay_t<Slices>> ? 1 : 0) + ...);

                // Create shape array
                std::array<std::size_t, new_rank> shape;
                std::size_t shape_idx = 0;
                std::size_t dim = 0;

                auto fill_shape = [&shape, &shape_idx, &dim, &c](auto& slice)
                {
                    using slice_type = std::decay_t<decltype(slice)>;
                    if constexpr (is_newaxis_v<slice_type>)
                    {
                        shape[shape_idx++] = 1;
                        // newaxis doesn't consume a dimension, so don't increment dim
                    }
                    else if constexpr (is_all_slice_v<slice_type>)
                    {
                        shape[shape_idx++] = c.shape()[dim];
                        ++dim;
                    }
                    else if constexpr (is_range_adaptor_v<slice_type>)
                    {
                        // Convert range adaptor to concrete xrange
                        auto concrete_range = slice.template get<std::ptrdiff_t>(c.shape()[dim]);
                        auto dim_size = static_cast<std::ptrdiff_t>(c.shape()[dim]);
                        auto step = concrete_range.step;

                        // Normalize start
                        auto start = concrete_range.start < 0 ? concrete_range.start + dim_size : concrete_range.start;

                        // For stop: don't normalize -1 when step < 0 (it means "before beginning")
                        auto stop = concrete_range.stop;
                        if (stop < 0 && !(step < 0 && stop == -1))
                        {
                            stop += dim_size;
                        }

                        std::size_t size;
                        if (step > 0)
                        {
                            size = static_cast<std::size_t>((stop - start + step - 1) / step);
                        }
                        else if (step < 0)
                        {
                            size = static_cast<std::size_t>((start - stop - step - 1) / (-step));
                        }
                        else
                        {
                            size = 0;
                        }
                        shape[shape_idx++] = size;
                        ++dim;
                    }
                    else if constexpr (is_range_slice_v<slice_type>)
                    {
                        // Handle negative indices in range
                        auto dim_size = static_cast<std::ptrdiff_t>(c.shape()[dim]);
                        auto start = slice.start < 0 ? slice.start + dim_size : slice.start;
                        auto stop = slice.stop < 0 ? slice.stop + dim_size : slice.stop;
                        auto step = slice.step;

                        // Compute size with normalized start/stop
                        std::size_t size;
                        if (step > 0)
                        {
                            size = static_cast<std::size_t>((stop - start + step - 1) / step);
                        }
                        else if (step < 0)
                        {
                            size = static_cast<std::size_t>((start - stop - step - 1) / (-step));
                        }
                        else
                        {
                            size = 0;
                        }
                        shape[shape_idx++] = size;
                        ++dim;
                    }
                    else if constexpr (is_keep_slice_v<slice_type>)
                    {
                        // Normalize keep slice and get its size
                        slice.normalize(c.shape()[dim]);
                        shape[shape_idx++] = slice.size();
                        ++dim;
                    }
                    else if constexpr (is_drop_slice_v<slice_type>)
                    {
                        // Normalize drop slice and get its size
                        slice.normalize(c.shape()[dim]);
                        shape[shape_idx++] = slice.size();
                        ++dim;
                    }
                    else
                    {
                        // Integer slice - consumes dimension but doesn't add to shape
                        ++dim;
                    }
                };

                (fill_shape(slices), ...);

                return shape;
            }

            // Compute strides for view
            template <class Container, class... Slices>
            constexpr auto compute_view_strides(const Container& c, Slices&... slices)
            {
                constexpr std::size_t new_rank = ((is_all_slice_v<std::decay_t<Slices>> || is_range_slice_v<std::decay_t<Slices>> || is_newaxis_v<std::decay_t<Slices>> || is_range_adaptor_v<std::decay_t<Slices>> || is_keep_slice_v<std::decay_t<Slices>> || is_drop_slice_v<std::decay_t<Slices>> ? 1 : 0) + ...);

                std::array<std::ptrdiff_t, new_rank> strides;
                std::size_t stride_idx = 0;
                std::size_t dim = 0;

                auto fill_strides = [&strides, &stride_idx, &dim, &c](auto& slice)
                {
                    using slice_type = std::decay_t<decltype(slice)>;
                    if constexpr (is_newaxis_v<slice_type>)
                    {
                        strides[stride_idx++] = 0;  // newaxis has stride 0
                        // newaxis doesn't consume a dimension, so don't increment dim
                    }
                    else if constexpr (is_all_slice_v<slice_type>)
                    {
                        strides[stride_idx++] = c.strides()[dim];
                        ++dim;
                    }
                    else if constexpr (is_range_adaptor_v<slice_type>)
                    {
                        // Convert range adaptor to concrete xrange
                        auto concrete_range = slice.template get<std::ptrdiff_t>(c.shape()[dim]);
                        strides[stride_idx++] = c.strides()[dim] * concrete_range.step;
                        ++dim;
                    }
                    else if constexpr (is_range_slice_v<slice_type>)
                    {
                        strides[stride_idx++] = c.strides()[dim] * slice.step;
                        ++dim;
                    }
                    else if constexpr (is_keep_slice_v<slice_type> || is_drop_slice_v<slice_type>)
                    {
                        // keep/drop use base stride of underlying dimension
                        strides[stride_idx++] = c.strides()[dim];
                        ++dim;
                    }
                    else
                    {
                        // Integer slice - consumes dimension but doesn't add to strides
                        ++dim;
                    }
                };

                (fill_strides(slices), ...);

                return strides;
            }

            // Compute offset from integer slices
            template <class Container, class... Slices>
            constexpr std::size_t compute_view_offset(const Container& c, Slices&... slices)
            {
                std::ptrdiff_t offset = 0;
                std::size_t dim = 0;

                auto add_offset = [&offset, &dim, &c](auto& slice)
                {
                    using slice_type = std::decay_t<decltype(slice)>;
                    if constexpr (is_newaxis_v<slice_type>)
                    {
                        // newaxis doesn't consume a dimension or contribute to offset
                    }
                    else if constexpr (std::is_integral_v<slice_type>)
                    {
                        // Handle negative indices: -1 means last element, -2 means second to last, etc.
                        auto idx = static_cast<std::ptrdiff_t>(slice);
                        if (idx < 0)
                        {
                            idx += static_cast<std::ptrdiff_t>(c.shape()[dim]);
                        }
                        offset += idx * c.strides()[dim];
                        ++dim;
                    }
                    else if constexpr (is_range_adaptor_v<slice_type>)
                    {
                        // Convert range adaptor to concrete xrange
                        auto concrete_range = slice.template get<std::ptrdiff_t>(c.shape()[dim]);
                        auto start = concrete_range.start;
                        if (start < 0)
                        {
                            start += static_cast<std::ptrdiff_t>(c.shape()[dim]);
                        }
                        offset += start * c.strides()[dim];
                        ++dim;
                    }
                    else if constexpr (is_range_slice_v<slice_type>)
                    {
                        // Handle negative start/stop in ranges
                        auto start = slice.start;
                        if (start < 0)
                        {
                            start += static_cast<std::ptrdiff_t>(c.shape()[dim]);
                        }
                        offset += start * c.strides()[dim];
                        ++dim;
                    }
                    else if constexpr (is_keep_slice_v<slice_type> || is_drop_slice_v<slice_type>)
                    {
                        // keep/drop: offset will be handled via indirection - start at 0
                        ++dim;
                    }
                    else
                    {
                        // all() - doesn't contribute to offset but consumes dimension
                        ++dim;
                    }
                };

                (add_offset(slices), ...);

                return static_cast<std::size_t>(offset);
            }

            // ==================== Dynamic Rank Helpers ====================

            // Compute shape for dynamic rank containers
            template <class Container, class... Slices>
            auto compute_view_shape_dynamic(const Container& c, Slices&... slices)
            {
                std::vector<std::size_t> shape;
                std::size_t dim = 0;

                auto fill_shape = [&shape, &dim, &c](auto& slice)
                {
                    using slice_type = std::decay_t<decltype(slice)>;
                    if constexpr (is_newaxis_v<slice_type>)
                    {
                        shape.push_back(1);
                        // newaxis doesn't consume a dimension
                    }
                    else if constexpr (is_all_slice_v<slice_type>)
                    {
                        shape.push_back(c.shape()[dim]);
                        ++dim;
                    }
                    else if constexpr (is_range_adaptor_v<slice_type>)
                    {
                        // Convert range adaptor to concrete xrange
                        auto concrete_range = slice.template get<std::ptrdiff_t>(c.shape()[dim]);
                        auto dim_size = static_cast<std::ptrdiff_t>(c.shape()[dim]);
                        auto step = concrete_range.step;

                        // Normalize start
                        auto start = concrete_range.start < 0 ? concrete_range.start + dim_size : concrete_range.start;

                        // For stop: don't normalize -1 when step < 0 (it means "before beginning")
                        auto stop = concrete_range.stop;
                        if (stop < 0 && !(step < 0 && stop == -1))
                        {
                            stop += dim_size;
                        }

                        std::size_t size;
                        if (step > 0)
                        {
                            size = static_cast<std::size_t>((stop - start + step - 1) / step);
                        }
                        else if (step < 0)
                        {
                            size = static_cast<std::size_t>((start - stop - step - 1) / (-step));
                        }
                        else
                        {
                            size = 0;
                        }
                        shape.push_back(size);
                        ++dim;
                    }
                    else if constexpr (is_range_slice_v<slice_type>)
                    {
                        // Handle negative indices in range
                        auto dim_size = static_cast<std::ptrdiff_t>(c.shape()[dim]);
                        auto start = slice.start < 0 ? slice.start + dim_size : slice.start;
                        auto stop = slice.stop < 0 ? slice.stop + dim_size : slice.stop;
                        auto step = slice.step;

                        // Compute size with normalized start/stop
                        std::size_t size;
                        if (step > 0)
                        {
                            size = static_cast<std::size_t>((stop - start + step - 1) / step);
                        }
                        else if (step < 0)
                        {
                            size = static_cast<std::size_t>((start - stop - step - 1) / (-step));
                        }
                        else
                        {
                            size = 0;
                        }
                        shape.push_back(size);
                        ++dim;
                    }
                    else if constexpr (is_keep_slice_v<slice_type>)
                    {
                        slice.normalize(c.shape()[dim]);
                        shape.push_back(slice.size());
                        ++dim;
                    }
                    else if constexpr (is_drop_slice_v<slice_type>)
                    {
                        slice.normalize(c.shape()[dim]);
                        shape.push_back(slice.size());
                        ++dim;
                    }
                    else
                    {
                        // Integer slice - consumes dimension but doesn't add to shape
                        ++dim;
                    }
                };

                (fill_shape(slices), ...);

                return shape;
            }

            // Compute strides for dynamic rank containers
            template <class Container, class... Slices>
            auto compute_view_strides_dynamic(const Container& c, Slices&... slices)
            {
                std::vector<std::ptrdiff_t> strides;
                std::size_t dim = 0;

                auto fill_strides = [&strides, &dim, &c](auto& slice)
                {
                    using slice_type = std::decay_t<decltype(slice)>;
                    if constexpr (is_newaxis_v<slice_type>)
                    {
                        strides.push_back(0);  // newaxis has stride 0
                        // newaxis doesn't consume a dimension
                    }
                    else if constexpr (is_all_slice_v<slice_type>)
                    {
                        strides.push_back(c.strides()[dim]);
                        ++dim;
                    }
                    else if constexpr (is_range_adaptor_v<slice_type>)
                    {
                        auto concrete_range = slice.template get<std::ptrdiff_t>(c.shape()[dim]);
                        strides.push_back(c.strides()[dim] * concrete_range.step);
                        ++dim;
                    }
                    else if constexpr (is_range_slice_v<slice_type>)
                    {
                        strides.push_back(c.strides()[dim] * slice.step);
                        ++dim;
                    }
                    else if constexpr (is_keep_slice_v<slice_type> || is_drop_slice_v<slice_type>)
                    {
                        strides.push_back(c.strides()[dim]);
                        ++dim;
                    }
                    else
                    {
                        // Integer slice - consumes dimension but doesn't add to strides
                        ++dim;
                    }
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
                constexpr std::size_t new_rank = ((is_all_slice_v<std::decay_t<Slices>> || is_range_slice_v<std::decay_t<Slices>> || is_newaxis_v<std::decay_t<Slices>> || is_range_adaptor_v<std::decay_t<Slices>> || is_keep_slice_v<std::decay_t<Slices>> || is_drop_slice_v<std::decay_t<Slices>> ? 1 : 0) + ...);

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
                constexpr std::size_t new_rank = ((is_all_slice_v<std::decay_t<Slices>> || is_range_slice_v<std::decay_t<Slices>> || is_newaxis_v<std::decay_t<Slices>> || is_range_adaptor_v<std::decay_t<Slices>> || is_keep_slice_v<std::decay_t<Slices>> || is_drop_slice_v<std::decay_t<Slices>> ? 1 : 0) + ...);

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

        // ==================== Convenience Helpers ====================

        /**
         * @brief Get a row view from a 2D container
         *
         * Equivalent to view(container, row_index, all())
         *
         * @param c Container to slice
         * @param index Row index (supports negative indices)
         * @return 1D view of the specified row
         *
         * Example:
         * @code{.cpp}
         * xtensor<double, 2> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
         * auto r = row(a, 1);     // Gets [4, 5, 6]
         * auto r_last = row(a, -1);  // Gets last row [7, 8, 9]
         * @endcode
         */
        template <class Container>
        constexpr auto row(Container& c, std::ptrdiff_t index)
        {
            return view(c, index, all());
        }

        /**
         * @brief Get a column view from a 2D container
         *
         * Equivalent to view(container, all(), col_index)
         *
         * @param c Container to slice
         * @param index Column index (supports negative indices)
         * @return 1D view of the specified column
         *
         * Example:
         * @code{.cpp}
         * xtensor<double, 2> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
         * auto col_view = col(a, 1);     // Gets [2, 5, 8]
         * auto col_last = col(a, -1);    // Gets last column [3, 6, 9]
         * @endcode
         */
        template <class Container>
        constexpr auto col(Container& c, std::ptrdiff_t index)
        {
            return view(c, all(), index);
        }

    }  // namespace views2
}  // namespace xt

#endif
