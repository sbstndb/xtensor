/***************************************************************************
 * Copyright (c) 2024, xtensor views2 contributors                          *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_XVIEWS2_HPP
#define XTENSOR_XVIEWS2_HPP

/**
 * @brief Main entry point for xtensor views2
 *
 * This is an experimental high-performance view implementation
 * based on C++23 std::mdspan with aggressive caching and
 * compile-time optimizations.
 *
 * Features:
 * - Zero-overhead contiguous views (just pointers)
 * - Cached metadata (no recomputation of layout/strides)
 * - Compile-time contiguity detection
 * - Expression templates for arithmetic operations
 *
 * Usage:
 *   #include <xtensor/xviews2.hpp>
 *   using namespace xt::views2;
 *
 *   xt::xtensor<double, 2> data = ...;
 *   auto v = view(data, all(), 5);  // Contiguous view
 *   auto s = strided_view(data, range(0, 10), all());  // Strided view
 */

#include "views2/xslice2.hpp"
#include "views2/xview2_base.hpp"
#include "views2/xview2.hpp"
#include "views2/xnoalias2.hpp"
#include "views2/xarithmetic2.hpp"

#endif
