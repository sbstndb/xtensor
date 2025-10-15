/***************************************************************************
 * Copyright (c) 2024, xtensor views2 contributors                          *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_VIEWS2_ARITHMETIC_HPP
#define XTENSOR_VIEWS2_ARITHMETIC_HPP

#include <functional>
#include <cstddef>

namespace xt
{
    namespace views2
    {
        // ==================== Expression Templates ====================

        /**
         * @brief Binary expression template
         */
        template <class Op, class L, class R>
        class binary_expr
        {
        public:
            using value_type = std::decay_t<decltype(Op{}(std::declval<typename L::value_type>(), std::declval<typename R::value_type>()))>;

            constexpr binary_expr(const L& left, const R& right) noexcept
                : left_(left)
                , right_(right)
            {
            }

            constexpr std::size_t size() const noexcept
            {
                return left_.size();
            }

            constexpr bool is_contiguous() const noexcept
            {
                return false;  // Expressions are not contiguous
            }

            // Element access
            template <class... Indices>
            constexpr auto operator()(Indices... indices) const noexcept
            {
                return Op{}(left_(indices...), right_(indices...));
            }

            // Iterator for assignment
            class iterator
            {
            public:
                using value_type = typename binary_expr::value_type;
                using reference = value_type;
                using pointer = value_type*;
                using difference_type = std::ptrdiff_t;
                using iterator_category = std::forward_iterator_tag;

                constexpr iterator(const binary_expr* expr, typename L::iterator lit, typename R::iterator rit) noexcept
                    : expr_(expr)
                    , lit_(lit)
                    , rit_(rit)
                {
                }

                constexpr value_type operator*() const noexcept
                {
                    return Op{}(*lit_, *rit_);
                }

                constexpr iterator& operator++() noexcept
                {
                    ++lit_;
                    ++rit_;
                    return *this;
                }

                constexpr iterator operator++(int) noexcept
                {
                    auto tmp = *this;
                    ++(*this);
                    return tmp;
                }

                constexpr bool operator==(const iterator& other) const noexcept
                {
                    return lit_ == other.lit_;
                }

                constexpr bool operator!=(const iterator& other) const noexcept
                {
                    return !(*this == other);
                }

            private:
                const binary_expr* expr_;
                typename L::iterator lit_;
                typename R::iterator rit_;
            };

            constexpr iterator begin() const noexcept
            {
                return iterator(this, left_.begin(), right_.begin());
            }

            constexpr iterator end() const noexcept
            {
                return iterator(this, left_.end(), right_.end());
            }

        private:
            const L& left_;
            const R& right_;
        };

        /**
         * @brief Scalar-View binary expression
         */
        template <class Op, class Scalar, class View>
            requires std::is_arithmetic_v<Scalar>
        class scalar_expr
        {
        public:
            using value_type = std::decay_t<decltype(Op{}(std::declval<Scalar>(), std::declval<typename View::value_type>()))>;

            constexpr scalar_expr(Scalar scalar, const View& view) noexcept
                : scalar_(scalar)
                , view_(view)
            {
            }

            constexpr std::size_t size() const noexcept
            {
                return view_.size();
            }

            constexpr bool is_contiguous() const noexcept
            {
                return false;
            }

            template <class... Indices>
            constexpr auto operator()(Indices... indices) const noexcept
            {
                return Op{}(scalar_, view_(indices...));
            }

            class iterator
            {
            public:
                using value_type = typename scalar_expr::value_type;
                using reference = value_type;
                using pointer = value_type*;
                using difference_type = std::ptrdiff_t;
                using iterator_category = std::forward_iterator_tag;

                constexpr iterator(Scalar scalar, typename View::iterator it) noexcept
                    : scalar_(scalar)
                    , it_(it)
                {
                }

                constexpr value_type operator*() const noexcept
                {
                    return Op{}(scalar_, *it_);
                }

                constexpr iterator& operator++() noexcept
                {
                    ++it_;
                    return *this;
                }

                constexpr iterator operator++(int) noexcept
                {
                    auto tmp = *this;
                    ++(*this);
                    return tmp;
                }

                constexpr bool operator==(const iterator& other) const noexcept
                {
                    return it_ == other.it_;
                }

                constexpr bool operator!=(const iterator& other) const noexcept
                {
                    return !(*this == other);
                }

            private:
                Scalar scalar_;
                typename View::iterator it_;
            };

            constexpr iterator begin() const noexcept
            {
                return iterator(scalar_, view_.begin());
            }

            constexpr iterator end() const noexcept
            {
                return iterator(scalar_, view_.end());
            }

        private:
            Scalar scalar_;
            const View& view_;
        };

        // ==================== Concepts ====================

        // Concept to identify view-like types (has size(), begin(), value_type)
        template <class T>
        concept is_view_expression = requires(const T& t) {
            typename T::value_type;
            { t.size() } -> std::convertible_to<std::size_t>;
            { t.begin() };
        };

        // ==================== Operators ====================

        // Subtraction: View - View
        template <class L, class R>
            requires is_view_expression<L> && is_view_expression<R>
        constexpr auto operator-(const L& left, const R& right) noexcept
        {
            return binary_expr<std::minus<>, L, R>{left, right};
        }

        // Multiplication: Scalar * View
        template <class Scalar, class View>
            requires std::is_arithmetic_v<Scalar> && is_view_expression<View>
        constexpr auto operator*(Scalar scalar, const View& view) noexcept
        {
            return scalar_expr<std::multiplies<>, Scalar, View>{scalar, view};
        }

        // Multiplication: View * Scalar
        template <class View, class Scalar>
            requires std::is_arithmetic_v<Scalar> && is_view_expression<View>
        constexpr auto operator*(const View& view, Scalar scalar) noexcept
        {
            return scalar_expr<std::multiplies<>, Scalar, View>{scalar, view};
        }

        // Addition: View + View
        template <class L, class R>
            requires is_view_expression<L> && is_view_expression<R>
        constexpr auto operator+(const L& left, const R& right) noexcept
        {
            return binary_expr<std::plus<>, L, R>{left, right};
        }

        // Division: View / Scalar
        template <class View, class Scalar>
            requires std::is_arithmetic_v<Scalar> && is_view_expression<View>
        constexpr auto operator/(const View& view, Scalar scalar) noexcept
        {
            return scalar_expr<std::divides<>, Scalar, View>{scalar, view};
        }

    }  // namespace views2
}  // namespace xt

#endif
