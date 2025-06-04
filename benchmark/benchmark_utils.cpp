
#include "benchmark_utils.hpp"

void CustomArguments(
                benchmark::internal::Benchmark* b,
                int start ,
                int end,
                int threshold1,
                int threshold2
                ) {

  // Phase linéaire (incréments de 1)
  for (int i = start; i < threshold1 && i <= end; ++i) {
    b->Arg(i);
  }
  // Phase linéaire (incréments de 16)
  for (int i = threshold1; i <= threshold2 && i <= end; i+=16) {
    b->Arg(i);
  }
  // Phase exponentielle (puissances de 2)
  for (int i = threshold2 * 2; i <= end; i *= 2) {
    b->Arg(i);
  }
}


/**
const int min = 1 ;
const int max = 100000 ;
const int threshold1 = 128 ;
const int threshold2 = 8096 ;

const int min_2d = 1 ;
const int max_2d = 8096 ;
const int threshold1_2d = 32 ;
const int threshold2_2d = 256 ;
**/

