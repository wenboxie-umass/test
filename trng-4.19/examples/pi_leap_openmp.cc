// Copyright (c) 2000-2014, Heiko Bauke
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.  
// 
//   * Redistributions in binary form must reproduce the above
//     copyright notice, this list of conditions and the following
//     disclaimer in the documentation and/or other materials provided
//     with the distribution.  
// 
//   * Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived
//     from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#include <trng/config.hpp>
#if defined TRNG_HAVE_OPENMP

#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>

int main() {
  const long samples=1000000l;          // total number of points in square
  long in=0l;                           // no points in circle
  // distribute workload over all processes and make a global reduction 
#pragma omp parallel reduction(+:in)
  {
    trng::yarn2 rx, ry;                 // random number engines for x- and y-coordinates
    int size=omp_get_num_threads();     // get total number of processes
    int rank=omp_get_thread_num();      // get rank of current process
    // split PRN sequences by leapfrog method
    rx.split(2, 0);                     // choose sub-stream no. 0 out of 2 streams
    ry.split(2, 1);                     // choose sub-stream no. 1 out of 2 streams
    rx.split(size, rank);               // choose sub-stream no. rank out of size streams
    ry.split(size, rank);               // choose sub-stream no. rank out of size streams
    trng::uniform01_dist<> u;           // random number distribution
    // throw random points into square 
    for (long i=rank; i<samples; i+=size) {
      double x=u(rx), y=u(ry);          // choose random x- and y-coordinates
      if (x*x+y*y<=1.0)                 // is point in circle?
	++in;                           // increase thread-local counter
    }
  }
  // print result
  std::cout << "pi = " << 4.0*in/samples << std::endl;
  return EXIT_SUCCESS;
}

#else

#include <cstdlib>
#include <iostream>

int main() {
  std::cerr << "Sorry, OpenMP is not supported by your compiler.\n";
  return EXIT_FAILURE;
}

#endif
