/***************************************************************************
 * Copyright (c) 2024, xtensor views2 contributors                          *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <algorithm>

#include "test_common_macros.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/xviews2.hpp"

namespace xt
{
    using std::size_t;

    TEST(xviews2, simple_view)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        // View with integer slice and range
        auto view1 = views2::view(a, 1, views2::range(std::ptrdiff_t(1), std::ptrdiff_t(4)));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(a(1, 3), view1(2));
        EXPECT_EQ(size_t(1), view1.dimension());
        EXPECT_EQ(size_t(3), view1.shape(0));

        // View with integer and all()
        auto view2 = views2::view(a, 1, views2::all());
        EXPECT_EQ(a(1, 0), view2(0));
        EXPECT_EQ(a(1, 1), view2(1));
        EXPECT_EQ(a(1, 2), view2(2));
        EXPECT_EQ(a(1, 3), view2(3));
        EXPECT_EQ(size_t(1), view2.dimension());
        EXPECT_EQ(size_t(4), view2.shape(0));

        // View with all() and integer
        auto view3 = views2::view(a, views2::all(), 2);
        EXPECT_EQ(a(0, 2), view3(0));
        EXPECT_EQ(a(1, 2), view3(1));
        EXPECT_EQ(a(2, 2), view3(2));
        EXPECT_EQ(size_t(1), view3.dimension());
        EXPECT_EQ(size_t(3), view3.shape(0));
    }

    TEST(xviews2, strided_view)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = views2::strided_view(a, views2::all(), 2);
        EXPECT_EQ(a(0, 2), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(a(2, 2), view1(2));
        EXPECT_EQ(size_t(1), view1.dimension());
        EXPECT_EQ(size_t(3), view1.shape(0));
    }

    TEST(xviews2, shape_and_size)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);

        auto view1 = views2::view(a, 1, views2::all());
        EXPECT_EQ(size_t(1), view1.dimension());
        EXPECT_EQ(size_t(4), view1.size());
        EXPECT_EQ(size_t(4), view1.shape(0));

        auto view2 = views2::view(a, views2::range(std::ptrdiff_t(0), std::ptrdiff_t(2)), views2::all());
        EXPECT_EQ(size_t(2), view2.dimension());
        EXPECT_EQ(size_t(8), view2.size());
        EXPECT_EQ(size_t(2), view2.shape(0));
        EXPECT_EQ(size_t(4), view2.shape(1));
    }

    TEST(xviews2, range_slicing)
    {
        std::array<size_t, 2> shape = {4, 5};
        xtensor<double, 2> a(shape);
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                a(i, j) = static_cast<double>(i * 5 + j);
            }
        }

        // Range on both dimensions
        auto view1 = views2::view(a, views2::range(std::ptrdiff_t(1), std::ptrdiff_t(3)),
                                     views2::range(std::ptrdiff_t(1), std::ptrdiff_t(4)));
        EXPECT_EQ(size_t(2), view1.shape(0));
        EXPECT_EQ(size_t(3), view1.shape(1));
        EXPECT_EQ(a(1, 1), view1(0, 0));
        EXPECT_EQ(a(1, 2), view1(0, 1));
        EXPECT_EQ(a(2, 1), view1(1, 0));
        EXPECT_EQ(a(2, 3), view1(1, 2));
    }

    TEST(xviews2, iterator)
    {
        std::array<size_t, 2> shape = {2, 4};
        xtensor<double, 2> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        // Contiguous view should use pointer iteration
        auto view1 = views2::view(a, 1, views2::all());
        auto iter = view1.begin();
        auto iter_end = view1.end();

        EXPECT_EQ(5, *iter);
        ++iter;
        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xviews2, strided_iterator)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        // Non-contiguous view should use strided_iterator
        auto view1 = views2::strided_view(a, views2::all(), 2);
        auto iter = view1.begin();
        auto iter_end = view1.end();

        EXPECT_EQ(3, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(11, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xviews2, assignment)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        std::array<size_t, 1> shape_b = {4};
        xtensor<double, 1> b(shape_b);
        std::fill(b.begin(), b.end(), 100.0);

        auto view1 = views2::view(a, 1, views2::all());
        views2::noalias(view1) = views2::view(b, views2::all());

        EXPECT_EQ(100.0, a(1, 0));
        EXPECT_EQ(100.0, a(1, 1));
        EXPECT_EQ(100.0, a(1, 2));
        EXPECT_EQ(100.0, a(1, 3));
    }

    TEST(xviews2, arithmetic_operations)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        std::array<size_t, 1> shape_result = {4};
        xtensor<double, 1> result(shape_result);

        auto view1 = views2::view(a, 1, views2::all());
        auto view2 = views2::view(a, 2, views2::all());

        // Subtraction
        views2::noalias(result) = view2 - view1;
        EXPECT_EQ(4.0, result(0));
        EXPECT_EQ(4.0, result(1));
        EXPECT_EQ(4.0, result(2));
        EXPECT_EQ(4.0, result(3));

        // Scalar multiplication
        views2::noalias(result) = 2.0 * view1;
        EXPECT_EQ(10.0, result(0));
        EXPECT_EQ(12.0, result(1));
        EXPECT_EQ(14.0, result(2));
        EXPECT_EQ(16.0, result(3));
    }

    TEST(xviews2, data_offset)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = views2::view(a, 1, views2::all());
        EXPECT_EQ(a.data() + 4, view1.data() + view1.data_offset());
        EXPECT_EQ(a(1, 0), *(view1.data() + view1.data_offset()));

        auto view2 = views2::view(a, 2, views2::range(std::ptrdiff_t(1), std::ptrdiff_t(3)));
        EXPECT_EQ(a.data() + 9, view2.data() + view2.data_offset());
        EXPECT_EQ(a(2, 1), *(view2.data() + view2.data_offset()));
    }

    TEST(xviews2, strides)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2, layout_type::row_major> a(shape);

        auto view1 = views2::view(a, 1, views2::all());
        EXPECT_EQ(std::ptrdiff_t(1), view1.strides()[0]);

        auto view2 = views2::view(a, views2::all(), 2);
        EXPECT_EQ(std::ptrdiff_t(4), view2.strides()[0]);
    }

    TEST(xviews2, three_dimensional)
    {
        std::array<size_t, 3> shape = {2, 3, 4};
        xtensor<double, 3> a(shape);
        std::vector<double> data(24);
        std::iota(data.begin(), data.end(), 1.0);
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = views2::view(a, 1, views2::all(), views2::all());
        EXPECT_EQ(size_t(2), view1.dimension());
        EXPECT_EQ(a(1, 0, 0), view1(0, 0));
        EXPECT_EQ(a(1, 0, 1), view1(0, 1));
        EXPECT_EQ(a(1, 1, 0), view1(1, 0));
        EXPECT_EQ(a(1, 2, 3), view1(2, 3));

        auto view2 = views2::view(a, views2::all(), 1, views2::all());
        EXPECT_EQ(size_t(2), view2.dimension());
        EXPECT_EQ(a(0, 1, 0), view2(0, 0));
        EXPECT_EQ(a(1, 1, 3), view2(1, 3));
    }

    TEST(xviews2, contiguity)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);

        // These should be contiguous
        auto view1 = views2::view(a, 1, views2::all());
        EXPECT_TRUE(view1.is_contiguous());

        auto view2 = views2::view(a, 2, views2::range(std::ptrdiff_t(1), std::ptrdiff_t(4)));
        EXPECT_TRUE(view2.is_contiguous());

        // Strided views are always non-contiguous
        auto view3 = views2::strided_view(a, views2::all(), 1);
        EXPECT_FALSE(view3.is_contiguous());
    }

    TEST(xviews2, copy_semantic)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        // Copy constructor
        auto view1 = views2::view(a, 1, views2::all());
        auto view2(view1);
        EXPECT_EQ(view1(0), view2(0));
        EXPECT_EQ(view1(1), view2(1));
        EXPECT_EQ(view1.size(), view2.size());
    }

    TEST(xviews2, range_with_step)
    {
        std::array<size_t, 1> shape = {10};
        xtensor<double, 1> a(shape);
        std::iota(a.begin(), a.end(), 0.0);

        // Step of 2
        auto view1 = views2::view(a, views2::range(std::ptrdiff_t(0), std::ptrdiff_t(10), std::ptrdiff_t(2)));
        EXPECT_EQ(size_t(5), view1.size());
        EXPECT_EQ(0.0, view1(0));
        EXPECT_EQ(2.0, view1(1));
        EXPECT_EQ(4.0, view1(2));
        EXPECT_EQ(6.0, view1(3));
        EXPECT_EQ(8.0, view1(4));

        // Step of 3
        auto view2 = views2::view(a, views2::range(std::ptrdiff_t(1), std::ptrdiff_t(10), std::ptrdiff_t(3)));
        EXPECT_EQ(size_t(3), view2.size());
        EXPECT_EQ(1.0, view2(0));
        EXPECT_EQ(4.0, view2(1));
        EXPECT_EQ(7.0, view2(2));
    }

    TEST(xviews2, stencil_pattern)
    {
        std::array<size_t, 3> shape = {5, 5, 5};
        xtensor<double, 3> a(shape);
        xtensor<double, 3> b(shape);
        std::fill(a.begin(), a.end(), 1.0);
        std::fill(b.begin(), b.end(), 0.0);

        auto core = views2::range(std::ptrdiff_t(1), std::ptrdiff_t(4));

        // Stencil computation
        views2::noalias(views2::view(b, core, core, core))
            = 0.5 * (views2::view(a, views2::range(std::ptrdiff_t(2), std::ptrdiff_t(5)), core, core)
                     - views2::view(a, views2::range(std::ptrdiff_t(0), std::ptrdiff_t(3)), core, core));

        // Check that interior was computed
        EXPECT_EQ(0.0, b(1, 1, 1));
        EXPECT_EQ(0.0, b(2, 2, 2));
        EXPECT_EQ(0.0, b(3, 3, 3));

        // Check that boundaries are still zero
        EXPECT_EQ(0.0, b(0, 0, 0));
        EXPECT_EQ(0.0, b(4, 4, 4));
    }

    TEST(xviews2, newaxis_basic)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        // Add newaxis at beginning
        auto view1 = views2::view(a, views2::newaxis(), views2::all(), views2::all());
        EXPECT_EQ(size_t(3), view1.dimension());
        EXPECT_EQ(size_t(1), view1.shape(0));
        EXPECT_EQ(size_t(3), view1.shape(1));
        EXPECT_EQ(size_t(4), view1.shape(2));
        EXPECT_EQ(a(0, 0), view1(0, 0, 0));
        EXPECT_EQ(a(1, 2), view1(0, 1, 2));
        EXPECT_EQ(a(2, 3), view1(0, 2, 3));

        // Add newaxis in middle
        auto view2 = views2::view(a, views2::all(), views2::newaxis(), views2::all());
        EXPECT_EQ(size_t(3), view2.dimension());
        EXPECT_EQ(size_t(3), view2.shape(0));
        EXPECT_EQ(size_t(1), view2.shape(1));
        EXPECT_EQ(size_t(4), view2.shape(2));
        EXPECT_EQ(a(0, 0), view2(0, 0, 0));
        EXPECT_EQ(a(1, 2), view2(1, 0, 2));
        EXPECT_EQ(a(2, 3), view2(2, 0, 3));

        // Add newaxis at end
        auto view3 = views2::view(a, views2::all(), views2::all(), views2::newaxis());
        EXPECT_EQ(size_t(3), view3.dimension());
        EXPECT_EQ(size_t(3), view3.shape(0));
        EXPECT_EQ(size_t(4), view3.shape(1));
        EXPECT_EQ(size_t(1), view3.shape(2));
        EXPECT_EQ(a(0, 0), view3(0, 0, 0));
        EXPECT_EQ(a(1, 2), view3(1, 2, 0));
        EXPECT_EQ(a(2, 3), view3(2, 3, 0));
    }

    TEST(xviews2, newaxis_with_integer_slice)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        // newaxis with integer slice
        auto view1 = views2::view(a, views2::newaxis(), 1, views2::all());
        EXPECT_EQ(size_t(2), view1.dimension());
        EXPECT_EQ(size_t(1), view1.shape(0));
        EXPECT_EQ(size_t(4), view1.shape(1));
        EXPECT_EQ(a(1, 0), view1(0, 0));
        EXPECT_EQ(a(1, 1), view1(0, 1));
        EXPECT_EQ(a(1, 2), view1(0, 2));
        EXPECT_EQ(a(1, 3), view1(0, 3));
    }

    TEST(xviews2, newaxis_with_range)
    {
        std::array<size_t, 2> shape = {4, 5};
        xtensor<double, 2> a(shape);
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                a(i, j) = static_cast<double>(i * 5 + j);
            }
        }

        // newaxis with range
        auto view1 = views2::view(a, views2::range(std::ptrdiff_t(1), std::ptrdiff_t(3)),
                                     views2::newaxis(),
                                     views2::range(std::ptrdiff_t(1), std::ptrdiff_t(4)));
        EXPECT_EQ(size_t(3), view1.dimension());
        EXPECT_EQ(size_t(2), view1.shape(0));
        EXPECT_EQ(size_t(1), view1.shape(1));
        EXPECT_EQ(size_t(3), view1.shape(2));
        EXPECT_EQ(a(1, 1), view1(0, 0, 0));
        EXPECT_EQ(a(1, 2), view1(0, 0, 1));
        EXPECT_EQ(a(2, 1), view1(1, 0, 0));
        EXPECT_EQ(a(2, 3), view1(1, 0, 2));
    }

    TEST(xviews2, multiple_newaxis)
    {
        std::array<size_t, 1> shape = {5};
        xtensor<double, 1> a(shape);
        std::iota(a.begin(), a.end(), 0.0);

        // Multiple newaxis
        auto view1 = views2::view(a, views2::newaxis(), views2::all(), views2::newaxis());
        EXPECT_EQ(size_t(3), view1.dimension());
        EXPECT_EQ(size_t(1), view1.shape(0));
        EXPECT_EQ(size_t(5), view1.shape(1));
        EXPECT_EQ(size_t(1), view1.shape(2));
        EXPECT_EQ(a(0), view1(0, 0, 0));
        EXPECT_EQ(a(1), view1(0, 1, 0));
        EXPECT_EQ(a(4), view1(0, 4, 0));
    }

    TEST(xviews2, newaxis_stride_check)
    {
        std::array<size_t, 2> shape = {3, 4};
        xtensor<double, 2> a(shape);

        // Add newaxis at different positions and check strides
        auto view1 = views2::view(a, views2::newaxis(), views2::all(), views2::all());
        EXPECT_EQ(std::ptrdiff_t(0), view1.strides()[0]);  // newaxis dimension has stride 0

        auto view2 = views2::view(a, views2::all(), views2::newaxis(), views2::all());
        EXPECT_EQ(std::ptrdiff_t(0), view2.strides()[1]);  // newaxis dimension has stride 0

        auto view3 = views2::view(a, views2::all(), views2::all(), views2::newaxis());
        EXPECT_EQ(std::ptrdiff_t(0), view3.strides()[2]);  // newaxis dimension has stride 0
    }
}
