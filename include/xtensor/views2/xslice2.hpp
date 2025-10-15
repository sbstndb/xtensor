/***************************************************************************
 * Copyright (c) 2024, xtensor views2 contributors                          *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_VIEWS2_SLICE_HPP
#define XTENSOR_VIEWS2_SLICE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <map>
#include <tuple>
#include <type_traits>

namespace xt
{
    namespace views2
    {
        // ==================== Slice Types ====================

        /**
         * @brief Tag type representing all elements along a dimension
         */
        struct all_tag
        {
            constexpr all_tag() = default;
        };

        /**
         * @brief Create an all() slice
         */
        inline constexpr all_tag all() noexcept
        {
            return all_tag{};
        }

        /**
         * @brief Tag type representing a newaxis slice (adds a dimension of size 1)
         */
        struct newaxis_tag
        {
            constexpr newaxis_tag() = default;
        };

        /**
         * @brief Create a newaxis() slice
         *
         * newaxis adds a new dimension of size 1 at the specified position.
         *
         * Example:
         * @code{.cpp}
         * xt::xtensor<double, 2> a = {{1, 2, 3}, {4, 5, 6}};  // shape (2, 3)
         * auto v1 = views2::view(a, newaxis(), all(), all());  // shape (1, 2, 3)
         * auto v2 = views2::view(a, all(), newaxis(), all());  // shape (2, 1, 3)
         * auto v3 = views2::view(a, all(), all(), newaxis());  // shape (2, 3, 1)
         * @endcode
         */
        inline constexpr newaxis_tag newaxis() noexcept
        {
            return newaxis_tag{};
        }

        /**
         * @brief Range slice [start:stop:step]
         */
        template <class T = std::ptrdiff_t>
        struct xrange
        {
            T start;
            T stop;
            T step;

            constexpr xrange(T start_, T stop_, T step_ = 1) noexcept
                : start(start_)
                , stop(stop_)
                , step(step_)
            {
            }

            constexpr std::size_t size() const noexcept
            {
                if (step > 0)
                {
                    return static_cast<std::size_t>((stop - start + step - 1) / step);
                }
                else if (step < 0)
                {
                    return static_cast<std::size_t>((start - stop - step - 1) / (-step));
                }
                return 0;
            }

            constexpr T operator()(std::size_t i) const noexcept
            {
                return start + static_cast<T>(i) * step;
            }
        };

        // ==================== Slice Type Traits ====================

        template <class T>
        struct is_all_slice : std::false_type
        {
        };

        template <>
        struct is_all_slice<all_tag> : std::true_type
        {
        };

        template <class T>
        inline constexpr bool is_all_slice_v = is_all_slice<T>::value;

        template <class T>
        struct is_range_slice : std::false_type
        {
        };

        template <class T>
        struct is_range_slice<xrange<T>> : std::true_type
        {
        };

        template <class T>
        inline constexpr bool is_range_slice_v = is_range_slice<T>::value;

        template <class T>
        struct is_integer_slice : std::integral_constant<bool, std::is_integral_v<T>>
        {
        };

        template <class T>
        inline constexpr bool is_integer_slice_v = is_integer_slice<T>::value;

        template <class T>
        struct is_newaxis : std::false_type
        {
        };

        template <>
        struct is_newaxis<newaxis_tag> : std::true_type
        {
        };

        template <class T>
        inline constexpr bool is_newaxis_v = is_newaxis<T>::value;

        // ==================== Placeholder Types ====================

        /**
         * @brief Placeholder type for range() function
         * Used to represent unspecified start/stop/step in ranges
         */
        struct xtuph
        {
        };

        /**
         * @brief Global placeholder constant
         * Usage: range(_, 5) or range(3, _) or range(_, _, 2)
         */
        inline constexpr xtuph _ = xtuph{};

        /**
         * @brief Returns xnone placeholder (same as _)
         */
        inline constexpr xtuph xnone() noexcept
        {
            return xtuph{};
        }

        // ==================== Range Adaptor ====================

        /**
         * @brief Range adaptor supporting placeholders
         * Allows range(_, 5), range(3, _), range(_, _, 2), etc.
         */
        template <class A, class B = A, class C = A>
        struct xrange_adaptor
        {
            A m_start;
            B m_stop;
            C m_step;

            constexpr xrange_adaptor(A start, B stop, C step) noexcept
                : m_start(start)
                , m_stop(stop)
                , m_step(step)
            {
            }

            /**
             * @brief Convert to concrete xrange based on dimension size
             */
            template <class T = std::ptrdiff_t>
            constexpr auto get(std::size_t size) const noexcept
            {
                constexpr bool start_is_placeholder = std::is_same_v<A, xtuph>;
                constexpr bool stop_is_placeholder = std::is_same_v<B, xtuph>;
                constexpr bool step_is_placeholder = std::is_same_v<C, xtuph>;

                T start_val, stop_val, step_val;

                // Handle step
                if constexpr (step_is_placeholder)
                {
                    step_val = 1;
                }
                else
                {
                    step_val = static_cast<T>(m_step);
                }

                // Handle start
                if constexpr (start_is_placeholder)
                {
                    start_val = (step_val > 0) ? 0 : static_cast<T>(size) - 1;
                }
                else
                {
                    start_val = static_cast<T>(m_start);
                    if (start_val < 0)
                    {
                        start_val += static_cast<T>(size);
                    }
                }

                // Handle stop
                if constexpr (stop_is_placeholder)
                {
                    stop_val = (step_val > 0) ? static_cast<T>(size) : T(-1);
                }
                else
                {
                    stop_val = static_cast<T>(m_stop);
                    if (stop_val < 0)
                    {
                        stop_val += static_cast<T>(size);
                    }
                }

                return xrange<T>{start_val, stop_val, step_val};
            }
        };

        // ==================== Enhanced range() with placeholder support ====================

        /**
         * @brief Create a range with placeholder support
         * Examples:
         *   range(_, 5)    -> from start to 5
         *   range(3, _)    -> from 3 to end
         *   range(_, _)    -> entire dimension (same as all())
         *   range(3, _, 2) -> from 3 to end with step 2
         */
        template <class A, class B>
        inline constexpr auto range(A start, B stop) noexcept
        {
            if constexpr (std::is_same_v<A, xtuph> && std::is_same_v<B, xtuph>)
            {
                return xrange_adaptor<xtuph, xtuph, xtuph>(xtuph{}, xtuph{}, xtuph{});
            }
            else if constexpr (std::is_same_v<A, xtuph>)
            {
                return xrange_adaptor<xtuph, std::ptrdiff_t, xtuph>(
                    xtuph{}, static_cast<std::ptrdiff_t>(stop), xtuph{});
            }
            else if constexpr (std::is_same_v<B, xtuph>)
            {
                return xrange_adaptor<std::ptrdiff_t, xtuph, xtuph>(
                    static_cast<std::ptrdiff_t>(start), xtuph{}, xtuph{});
            }
            else
            {
                return xrange_adaptor<std::ptrdiff_t, std::ptrdiff_t, xtuph>(
                    static_cast<std::ptrdiff_t>(start), static_cast<std::ptrdiff_t>(stop), xtuph{});
            }
        }

        /**
         * @brief Create a range with step and placeholder support
         */
        template <class A, class B, class C>
        inline constexpr auto range(A start, B stop, C step) noexcept
        {
            if constexpr (std::is_same_v<A, xtuph> && std::is_same_v<B, xtuph> && std::is_same_v<C, xtuph>)
            {
                return xrange_adaptor<xtuph, xtuph, xtuph>(xtuph{}, xtuph{}, xtuph{});
            }
            else if constexpr (std::is_same_v<A, xtuph> && std::is_same_v<B, xtuph>)
            {
                return xrange_adaptor<xtuph, xtuph, std::ptrdiff_t>(
                    xtuph{}, xtuph{}, static_cast<std::ptrdiff_t>(step));
            }
            else if constexpr (std::is_same_v<A, xtuph> && std::is_same_v<C, xtuph>)
            {
                return xrange_adaptor<xtuph, std::ptrdiff_t, xtuph>(
                    xtuph{}, static_cast<std::ptrdiff_t>(stop), xtuph{});
            }
            else if constexpr (std::is_same_v<B, xtuph> && std::is_same_v<C, xtuph>)
            {
                return xrange_adaptor<std::ptrdiff_t, xtuph, xtuph>(
                    static_cast<std::ptrdiff_t>(start), xtuph{}, xtuph{});
            }
            else if constexpr (std::is_same_v<A, xtuph>)
            {
                return xrange_adaptor<xtuph, std::ptrdiff_t, std::ptrdiff_t>(
                    xtuph{}, static_cast<std::ptrdiff_t>(stop), static_cast<std::ptrdiff_t>(step));
            }
            else if constexpr (std::is_same_v<B, xtuph>)
            {
                return xrange_adaptor<std::ptrdiff_t, xtuph, std::ptrdiff_t>(
                    static_cast<std::ptrdiff_t>(start), xtuph{}, static_cast<std::ptrdiff_t>(step));
            }
            else if constexpr (std::is_same_v<C, xtuph>)
            {
                return xrange_adaptor<std::ptrdiff_t, std::ptrdiff_t, xtuph>(
                    static_cast<std::ptrdiff_t>(start), static_cast<std::ptrdiff_t>(stop), xtuph{});
            }
            else
            {
                return xrange_adaptor<std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t>(
                    static_cast<std::ptrdiff_t>(start),
                    static_cast<std::ptrdiff_t>(stop),
                    static_cast<std::ptrdiff_t>(step));
            }
        }

        // ==================== Keep Slice ====================

        /**
         * @brief Slice that keeps only specified indices
         * Allows non-contiguous indexing
         */
        template <class T = std::ptrdiff_t>
        class xkeep_slice
        {
        public:
            using container_type = std::vector<T>;
            using size_type = T;

            xkeep_slice() = default;

            /**
             * @brief Construct from a container of indices
             */
            template <class C>
            explicit xkeep_slice(C&& cont)
                : m_raw_indices(std::begin(cont), std::end(cont))
            {
            }

            /**
             * @brief Construct from initializer list
             */
            xkeep_slice(std::initializer_list<T> indices)
                : m_raw_indices(indices)
            {
            }

            /**
             * @brief Normalize negative indices based on dimension size
             */
            void normalize(std::size_t shape)
            {
                m_indices.resize(m_raw_indices.size());
                for (std::size_t i = 0; i < m_indices.size(); ++i)
                {
                    m_indices[i] = m_raw_indices[i] < 0
                        ? static_cast<T>(shape) + m_raw_indices[i]
                        : m_raw_indices[i];
                }
            }

            /**
             * @brief Get the i-th index in the slice
             */
            T operator()(std::size_t i) const noexcept
            {
                return m_indices.size() == 1 ? m_indices.front() : m_indices[i];
            }

            /**
             * @brief Get the number of indices
             */
            std::size_t size() const noexcept
            {
                return m_raw_indices.size();
            }

            const container_type& indices() const noexcept
            {
                return m_indices;
            }

            const container_type& raw_indices() const noexcept
            {
                return m_raw_indices;
            }

        private:
            container_type m_raw_indices;  // Original indices (may be negative)
            container_type m_indices;      // Normalized indices (all positive)
        };

        // Helper trait to detect if a type is a container (has begin/end)
        template <class T, class = void>
        struct is_container : std::false_type {};

        template <class T>
        struct is_container<T, std::void_t<
            decltype(std::begin(std::declval<T&>())),
            decltype(std::end(std::declval<T&>()))
        >> : std::true_type {};

        template <class T>
        inline constexpr bool is_container_v = is_container<T>::value;

        /**
         * @brief Create a keep slice from indices
         * Examples:
         *   keep(0, 2, 5)       -> keep indices 0, 2, 5
         *   keep(1, 1, 1)       -> repeat index 1 three times
         *   keep(-1)            -> keep last index
         */
        template <class... Args, std::enable_if_t<(sizeof...(Args) != 1) || !is_container_v<std::decay_t<std::tuple_element_t<0, std::tuple<Args...>>>>, int> = 0>
        inline auto keep(Args... args)
        {
            using T = std::ptrdiff_t;
            return xkeep_slice<T>({static_cast<T>(args)...});
        }

        /**
         * @brief Create a keep slice from a container
         */
        template <class Container, std::enable_if_t<is_container_v<std::decay_t<Container>>, int> = 0>
        inline auto keep(Container&& cont)
        {
            using T = typename std::decay_t<Container>::value_type;
            return xkeep_slice<T>(std::forward<Container>(cont));
        }

        // ==================== Drop Slice ====================

        /**
         * @brief Slice that drops specified indices
         * Returns all indices except the ones specified
         */
        template <class T = std::ptrdiff_t>
        class xdrop_slice
        {
        public:
            using container_type = std::vector<T>;
            using size_type = T;

            xdrop_slice() = default;

            /**
             * @brief Construct from a container of indices to drop
             */
            template <class C>
            explicit xdrop_slice(C&& cont)
                : m_raw_indices(std::begin(cont), std::end(cont))
            {
            }

            /**
             * @brief Construct from initializer list
             */
            xdrop_slice(std::initializer_list<T> indices)
                : m_raw_indices(indices)
            {
            }

            /**
             * @brief Normalize negative indices and compute mapping
             */
            void normalize(std::size_t shape)
            {
                m_size = static_cast<T>(shape - m_raw_indices.size());

                // Normalize negative indices
                m_indices.resize(m_raw_indices.size());
                for (std::size_t i = 0; i < m_indices.size(); ++i)
                {
                    m_indices[i] = m_raw_indices[i] < 0
                        ? static_cast<T>(shape) + m_raw_indices[i]
                        : m_raw_indices[i];
                }

                // Build increment map for efficient index translation
                T cum = 0;
                T prev_cum = cum;
                for (std::size_t i = 0; i < m_indices.size(); ++i)
                {
                    std::size_t ind = i;
                    T d = m_indices[i];

                    // Find consecutive dropped indices
                    while (i + 1 < m_indices.size() && m_indices[i + 1] == m_indices[i] + 1)
                    {
                        ++i;
                    }

                    cum += (static_cast<T>(i) - static_cast<T>(ind)) + 1;
                    m_inc[d - prev_cum] = cum;
                    prev_cum = cum;
                }
            }

            /**
             * @brief Map logical index to actual index (skipping dropped indices)
             */
            T operator()(std::size_t i) const noexcept
            {
                if (m_inc.empty() || static_cast<T>(i) < m_inc.begin()->first)
                {
                    return static_cast<T>(i);
                }
                else
                {
                    auto iter = m_inc.upper_bound(static_cast<T>(i));
                    --iter;
                    return static_cast<T>(i) + iter->second;
                }
            }

            /**
             * @brief Get the size of the resulting dimension
             */
            std::size_t size() const noexcept
            {
                return static_cast<std::size_t>(m_size);
            }

            const container_type& indices() const noexcept
            {
                return m_indices;
            }

            const container_type& raw_indices() const noexcept
            {
                return m_raw_indices;
            }

        private:
            container_type m_raw_indices;        // Original indices to drop
            container_type m_indices;            // Normalized indices to drop
            std::map<T, T> m_inc;                // Increment map for index translation
            T m_size = 0;                        // Resulting size
        };

        /**
         * @brief Create a drop slice from indices
         * Examples:
         *   drop(1, 3)       -> drop indices 1 and 3
         *   drop(0, 2)       -> drop first and third indices
         *   drop(-1)         -> drop last index
         */
        template <class... Args, std::enable_if_t<(sizeof...(Args) != 1) || !is_container_v<std::decay_t<std::tuple_element_t<0, std::tuple<Args...>>>>, int> = 0>
        inline auto drop(Args... args)
        {
            using T = std::ptrdiff_t;
            return xdrop_slice<T>({static_cast<T>(args)...});
        }

        /**
         * @brief Create a drop slice from a container
         */
        template <class Container, std::enable_if_t<is_container_v<std::decay_t<Container>>, int> = 0>
        inline auto drop(Container&& cont)
        {
            using T = typename std::decay_t<Container>::value_type;
            return xdrop_slice<T>(std::forward<Container>(cont));
        }

        // ==================== Type Traits for new slices ====================

        template <class T>
        struct is_keep_slice : std::false_type
        {
        };

        template <class T>
        struct is_keep_slice<xkeep_slice<T>> : std::true_type
        {
        };

        template <class T>
        inline constexpr bool is_keep_slice_v = is_keep_slice<T>::value;

        template <class T>
        struct is_drop_slice : std::false_type
        {
        };

        template <class T>
        struct is_drop_slice<xdrop_slice<T>> : std::true_type
        {
        };

        template <class T>
        inline constexpr bool is_drop_slice_v = is_drop_slice<T>::value;

        template <class T>
        struct is_range_adaptor : std::false_type
        {
        };

        template <class A, class B, class C>
        struct is_range_adaptor<xrange_adaptor<A, B, C>> : std::true_type
        {
        };

        template <class T>
        inline constexpr bool is_range_adaptor_v = is_range_adaptor<T>::value;

    }  // namespace views2
}  // namespace xt

#endif
