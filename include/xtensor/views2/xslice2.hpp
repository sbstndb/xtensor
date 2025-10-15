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

        /**
         * @brief Create a range [start:stop)
         */
        template <class T>
        inline constexpr auto range(T start, T stop) noexcept
        {
            return xrange<T>{start, stop, 1};
        }

        /**
         * @brief Create a range [start:stop:step)
         */
        template <class T>
        inline constexpr auto range(T start, T stop, T step) noexcept
        {
            return xrange<T>{start, stop, step};
        }

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

    }  // namespace views2
}  // namespace xt

#endif
