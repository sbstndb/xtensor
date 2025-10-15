/***************************************************************************
 * Copyright (c) 2024, xtensor views2 contributors                          *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_VIEWS2_NOALIAS_HPP
#define XTENSOR_VIEWS2_NOALIAS_HPP

#include <algorithm>

namespace xt
{
    namespace views2
    {
        // ==================== Assignment Helpers ====================

        namespace detail
        {
            // Helper to check if a type has is_contiguous_view member
            template <class T>
            concept has_is_contiguous_view = requires { T::is_contiguous_view; };

            // Helper to get is_contiguous_view value or false
            template <class T>
            constexpr bool get_is_contiguous_view()
            {
                if constexpr (has_is_contiguous_view<T>)
                {
                    return T::is_contiguous_view;
                }
                else
                {
                    return false;
                }
            }

            // Assign from another view
            template <class Dest, class Src>
            constexpr void assign_impl(Dest& dest, const Src& src)
            {
                // Check if both are contiguous for fast path
                constexpr bool dest_is_contiguous = get_is_contiguous_view<Dest>();
                constexpr bool src_has_is_contiguous = requires { src.is_contiguous(); };

                if constexpr (dest_is_contiguous && src_has_is_contiguous)
                {
                    if (src.is_contiguous())
                    {
                        // Fast memcpy-like path
                        std::copy(src.begin(), src.end(), dest.begin());
                        return;
                    }
                }

                // Element-wise assignment
                auto src_it = src.begin();
                auto dest_it = dest.begin();
                const auto dest_end = dest.end();

                while (dest_it != dest_end)
                {
                    *dest_it = *src_it;
                    ++dest_it;
                    ++src_it;
                }
            }

            // Assign from scalar
            template <class Dest, class Scalar>
                requires std::is_arithmetic_v<Scalar>
            constexpr void assign_impl(Dest& dest, Scalar value)
            {
                std::fill(dest.begin(), dest.end(), value);
            }
        }  // namespace detail

        // ==================== Noalias Proxy ====================

        /**
         * @brief Proxy for no-alias assignment
         */
        template <class View>
        class noalias_proxy
        {
        public:
            constexpr explicit noalias_proxy(View& view) noexcept
                : view_(view)
            {
            }

            // Assignment from another view or expression
            template <class E>
            constexpr View& operator=(const E& expr)
            {
                detail::assign_impl(view_, expr);
                return view_;
            }

            // Assignment from scalar
            template <class Scalar>
                requires std::is_arithmetic_v<Scalar>
            constexpr View& operator=(Scalar value)
            {
                detail::assign_impl(view_, value);
                return view_;
            }

        private:
            View& view_;
        };

        /**
         * @brief Create a noalias proxy for efficient assignment
         */
        template <class View>
        constexpr auto noalias(View&& view) noexcept
        {
            return noalias_proxy<View>{view};
        }

    }  // namespace views2
}  // namespace xt

#endif
