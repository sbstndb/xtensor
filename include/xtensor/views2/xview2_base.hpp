/***************************************************************************
 * Copyright (c) 2024, xtensor views2 contributors                          *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_VIEWS2_BASE_HPP
#define XTENSOR_VIEWS2_BASE_HPP

#include <array>
#include <cstddef>
#include <algorithm>
#include <numeric>

#include "../core/xlayout.hpp"

namespace xt
{
    namespace views2
    {
        // ==================== Iterators ====================

        /**
         * @brief Contiguous iterator (just a pointer)
         */
        template <class T>
        using contiguous_iterator = T*;

        /**
         * @brief Strided iterator for non-contiguous views
         */
        template <class T, std::size_t Rank>
        class strided_iterator
        {
        public:
            using value_type = T;
            using reference = T&;
            using pointer = T*;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::random_access_iterator_tag;

            constexpr strided_iterator(
                T* data,
                const std::array<std::size_t, Rank>& shape,
                const std::array<std::ptrdiff_t, Rank>& strides,
                std::size_t offset = 0
            ) noexcept
                : data_(data)
                , shape_(shape)
                , strides_(strides)
                , base_offset_(offset)
                , index_()
                , linear_index_(0)
            {
                std::fill(index_.begin(), index_.end(), 0);
            }

            constexpr reference operator*() const noexcept
            {
                return data_[compute_offset()];
            }

            constexpr pointer operator->() const noexcept
            {
                return &data_[compute_offset()];
            }

            constexpr strided_iterator& operator++() noexcept
            {
                increment();
                return *this;
            }

            constexpr strided_iterator operator++(int) noexcept
            {
                auto tmp = *this;
                increment();
                return tmp;
            }

            constexpr bool operator==(const strided_iterator& other) const noexcept
            {
                return linear_index_ == other.linear_index_;
            }

            constexpr bool operator!=(const strided_iterator& other) const noexcept
            {
                return !(*this == other);
            }

            constexpr difference_type operator-(const strided_iterator& other) const noexcept
            {
                return static_cast<difference_type>(linear_index_) - static_cast<difference_type>(other.linear_index_);
            }

        private:
            constexpr std::ptrdiff_t compute_offset() const noexcept
            {
                std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(base_offset_);
                for (std::size_t i = 0; i < Rank; ++i)
                {
                    offset += static_cast<std::ptrdiff_t>(index_[i]) * strides_[i];
                }
                return offset;
            }

            constexpr void increment() noexcept
            {
                ++linear_index_;
                // Row-major increment
                for (std::size_t i = Rank; i > 0; --i)
                {
                    if (++index_[i - 1] < shape_[i - 1])
                    {
                        return;
                    }
                    index_[i - 1] = 0;
                }
            }

            T* data_;
            std::array<std::size_t, Rank> shape_;
            std::array<std::ptrdiff_t, Rank> strides_;
            std::size_t base_offset_;
            std::array<std::size_t, Rank> index_;
            std::size_t linear_index_;
        };

        // ==================== Basic View ====================

        /**
         * @brief Basic view wrapping std::mdspan with cached metadata
         *
         * @tparam T Element type
         * @tparam Rank Number of dimensions
         * @tparam IsContiguous Whether the view is contiguous (for optimization)
         */
        template <class T, std::size_t Rank, bool IsContiguous = true>
        class basic_view
        {
        public:
            using value_type = std::remove_const_t<T>;
            using reference = T&;
            using const_reference = const T&;
            using pointer = T*;
            using size_type = std::size_t;
            using shape_type = std::array<size_type, Rank>;
            using strides_type = std::array<std::ptrdiff_t, Rank>;

            using iterator = std::conditional_t<IsContiguous, contiguous_iterator<T>, strided_iterator<T, Rank>>;
            using const_iterator = std::conditional_t<IsContiguous, contiguous_iterator<const T>, strided_iterator<const T, Rank>>;

            static constexpr std::size_t rank = Rank;
            static constexpr bool is_contiguous_view = IsContiguous;

            // ==================== Constructors ====================

            constexpr basic_view(
                T* data,
                const shape_type& shape,
                const strides_type& strides,
                size_type offset = 0,
                layout_type layout = layout_type::row_major
            ) noexcept
                : data_(data)
                , shape_(shape)
                , strides_(strides)
                , offset_(offset)
                , layout_(layout)
                , size_(std::accumulate(shape.begin(), shape.end(), size_type(1), std::multiplies<>()))
            {
            }

            // ==================== Assignment ====================

            // Generic assignment from any view-like type
            template <class E>
            constexpr basic_view& operator=(const E& expr)
            {
                auto src_it = expr.begin();
                auto dest_it = this->begin();
                const auto dest_end = this->end();

                while (dest_it != dest_end)
                {
                    *dest_it = *src_it;
                    ++dest_it;
                    ++src_it;
                }
                return *this;
            }

            // ==================== Shape and Size ====================

            constexpr const shape_type& shape() const noexcept
            {
                return shape_;
            }

            constexpr size_type shape(size_type dim) const noexcept
            {
                return shape_[dim];
            }

            constexpr size_type dimension() const noexcept
            {
                return Rank;
            }

            constexpr size_type size() const noexcept
            {
                return size_;
            }

            constexpr const strides_type& strides() const noexcept
            {
                return strides_;
            }

            constexpr layout_type layout() const noexcept
            {
                return layout_;
            }

            constexpr bool is_contiguous() const noexcept
            {
                return IsContiguous;
            }

            // ==================== Data Access ====================

            constexpr T* data() noexcept
            {
                return data_;
            }

            constexpr const T* data() const noexcept
            {
                return data_;
            }

            constexpr size_type data_offset() const noexcept
            {
                return offset_;
            }

            // ==================== Element Access ====================

            template <class... Indices>
                requires(sizeof...(Indices) == Rank)
            constexpr reference operator()(Indices... indices) const noexcept
            {
                std::array<size_type, Rank> idx{static_cast<size_type>(indices)...};
                std::ptrdiff_t offset = offset_;
                for (size_type i = 0; i < Rank; ++i)
                {
                    offset += static_cast<std::ptrdiff_t>(idx[i]) * strides_[i];
                }
                return data_[offset];
            }

            constexpr reference operator[](size_type i) const noexcept
                requires(Rank == 1)
            {
                return data_[offset_ + static_cast<std::ptrdiff_t>(i) * strides_[0]];
            }

            // ==================== Iterators ====================

            constexpr iterator begin() const noexcept
            {
                if constexpr (IsContiguous)
                {
                    return data_ + offset_;
                }
                else
                {
                    return iterator(data_, shape_, strides_, offset_);
                }
            }

            constexpr iterator end() const noexcept
            {
                if constexpr (IsContiguous)
                {
                    return data_ + offset_ + size_;
                }
                else
                {
                    auto it = iterator(data_, shape_, strides_, offset_);
                    // Move to end position
                    for (size_type i = 0; i < size_; ++i)
                    {
                        ++it;
                    }
                    return it;
                }
            }

            constexpr const_iterator cbegin() const noexcept
            {
                if constexpr (IsContiguous)
                {
                    return data_ + offset_;
                }
                else
                {
                    return const_iterator(data_, shape_, strides_, offset_);
                }
            }

            constexpr const_iterator cend() const noexcept
            {
                if constexpr (IsContiguous)
                {
                    return data_ + offset_ + size_;
                }
                else
                {
                    auto it = const_iterator(data_, shape_, strides_, offset_);
                    for (size_type i = 0; i < size_; ++i)
                    {
                        ++it;
                    }
                    return it;
                }
            }

        private:
            T* data_;
            shape_type shape_;
            strides_type strides_;
            size_type offset_;
            layout_type layout_;
            size_type size_;
        };

        // ==================== Dynamic Rank Support ====================

        /**
         * @brief Strided iterator for dynamic rank views
         */
        template <class T>
        class strided_iterator_dynamic
        {
        public:
            using value_type = T;
            using reference = T&;
            using pointer = T*;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::random_access_iterator_tag;

            strided_iterator_dynamic(
                T* data,
                const std::vector<std::size_t>& shape,
                const std::vector<std::ptrdiff_t>& strides,
                std::size_t offset = 0
            ) noexcept
                : data_(data)
                , shape_(shape)
                , strides_(strides)
                , base_offset_(offset)
                , index_(shape.size(), 0)
                , linear_index_(0)
            {
            }

            reference operator*() const noexcept
            {
                return data_[compute_offset()];
            }

            pointer operator->() const noexcept
            {
                return &data_[compute_offset()];
            }

            strided_iterator_dynamic& operator++() noexcept
            {
                increment();
                return *this;
            }

            strided_iterator_dynamic operator++(int) noexcept
            {
                auto tmp = *this;
                increment();
                return tmp;
            }

            bool operator==(const strided_iterator_dynamic& other) const noexcept
            {
                return linear_index_ == other.linear_index_;
            }

            bool operator!=(const strided_iterator_dynamic& other) const noexcept
            {
                return !(*this == other);
            }

            difference_type operator-(const strided_iterator_dynamic& other) const noexcept
            {
                return static_cast<difference_type>(linear_index_) - static_cast<difference_type>(other.linear_index_);
            }

        private:
            std::ptrdiff_t compute_offset() const noexcept
            {
                std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(base_offset_);
                for (std::size_t i = 0; i < shape_.size(); ++i)
                {
                    offset += static_cast<std::ptrdiff_t>(index_[i]) * strides_[i];
                }
                return offset;
            }

            void increment() noexcept
            {
                ++linear_index_;
                // Row-major increment
                for (std::size_t i = shape_.size(); i > 0; --i)
                {
                    if (++index_[i - 1] < shape_[i - 1])
                    {
                        return;
                    }
                    index_[i - 1] = 0;
                }
            }

            T* data_;
            std::vector<std::size_t> shape_;
            std::vector<std::ptrdiff_t> strides_;
            std::size_t base_offset_;
            std::vector<std::size_t> index_;
            std::size_t linear_index_;
        };

        /**
         * @brief Basic view specialization for dynamic rank (xarray)
         */
        template <class T, bool IsContiguous>
        class basic_view<T, std::size_t(-1), IsContiguous>
        {
        public:
            using value_type = std::remove_const_t<T>;
            using reference = T&;
            using const_reference = const T&;
            using pointer = T*;
            using size_type = std::size_t;
            using shape_type = std::vector<size_type>;
            using strides_type = std::vector<std::ptrdiff_t>;

            using iterator = std::conditional_t<IsContiguous, contiguous_iterator<T>, strided_iterator_dynamic<T>>;
            using const_iterator = std::conditional_t<IsContiguous, contiguous_iterator<const T>, strided_iterator_dynamic<const T>>;

            static constexpr std::size_t rank = std::size_t(-1);  // dynamic
            static constexpr bool is_contiguous_view = IsContiguous;

            // ==================== Constructors ====================

            basic_view(
                T* data,
                const shape_type& shape,
                const strides_type& strides,
                size_type offset = 0,
                layout_type layout = layout_type::row_major
            ) noexcept
                : data_(data)
                , shape_(shape)
                , strides_(strides)
                , offset_(offset)
                , layout_(layout)
                , size_(std::accumulate(shape.begin(), shape.end(), size_type(1), std::multiplies<>()))
            {
            }

            // ==================== Assignment ====================

            // Generic assignment from any view-like type
            template <class E>
            basic_view& operator=(const E& expr)
            {
                auto src_it = expr.begin();
                auto dest_it = this->begin();
                const auto dest_end = this->end();

                while (dest_it != dest_end)
                {
                    *dest_it = *src_it;
                    ++dest_it;
                    ++src_it;
                }
                return *this;
            }

            // ==================== Shape and Size ====================

            const shape_type& shape() const noexcept
            {
                return shape_;
            }

            size_type shape(size_type dim) const noexcept
            {
                return shape_[dim];
            }

            size_type dimension() const noexcept
            {
                return shape_.size();
            }

            size_type size() const noexcept
            {
                return size_;
            }

            const strides_type& strides() const noexcept
            {
                return strides_;
            }

            layout_type layout() const noexcept
            {
                return layout_;
            }

            bool is_contiguous() const noexcept
            {
                return IsContiguous;
            }

            // ==================== Data Access ====================

            T* data() noexcept
            {
                return data_;
            }

            const T* data() const noexcept
            {
                return data_;
            }

            size_type data_offset() const noexcept
            {
                return offset_;
            }

            // ==================== Element Access ====================

            template <class... Indices>
            reference operator()(Indices... indices) const noexcept
            {
                std::array<size_type, sizeof...(Indices)> idx{static_cast<size_type>(indices)...};
                std::ptrdiff_t offset = offset_;
                for (size_type i = 0; i < sizeof...(Indices); ++i)
                {
                    offset += static_cast<std::ptrdiff_t>(idx[i]) * strides_[i];
                }
                return data_[offset];
            }

            reference operator[](size_type i) const noexcept
            {
                return data_[offset_ + static_cast<std::ptrdiff_t>(i) * strides_[0]];
            }

            // ==================== Iterators ====================

            iterator begin() const noexcept
            {
                if constexpr (IsContiguous)
                {
                    return data_ + offset_;
                }
                else
                {
                    return iterator(data_, shape_, strides_, offset_);
                }
            }

            iterator end() const noexcept
            {
                if constexpr (IsContiguous)
                {
                    return data_ + offset_ + size_;
                }
                else
                {
                    auto it = iterator(data_, shape_, strides_, offset_);
                    for (size_type i = 0; i < size_; ++i)
                    {
                        ++it;
                    }
                    return it;
                }
            }

            const_iterator cbegin() const noexcept
            {
                if constexpr (IsContiguous)
                {
                    return data_ + offset_;
                }
                else
                {
                    return const_iterator(data_, shape_, strides_, offset_);
                }
            }

            const_iterator cend() const noexcept
            {
                if constexpr (IsContiguous)
                {
                    return data_ + offset_ + size_;
                }
                else
                {
                    auto it = const_iterator(data_, shape_, strides_, offset_);
                    for (size_type i = 0; i < size_; ++i)
                    {
                        ++it;
                    }
                    return it;
                }
            }

        private:
            T* data_;
            shape_type shape_;
            strides_type strides_;
            size_type offset_;
            layout_type layout_;
            size_type size_;
        };

        // Alias for dynamic rank
        constexpr std::size_t dynamic_rank = std::size_t(-1);

    }  // namespace views2
}  // namespace xt

#endif
