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

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <trng/config.hpp>
#include <trng/yarn2.hpp>
#include <trng/uniform_int_dist.hpp>
#if defined TRNG_HAVE_BOOST
  #include <boost/bind.hpp>
#else

  // helper class 
  template<typename PRN_dist_t, typename PRN_engine_t>
  class binder_cl {
    PRN_dist_t &dist;
    PRN_engine_t &engine;
  public:
    binder_cl(PRN_dist_t &dist, PRN_engine_t &engine) : dist(dist), engine(engine) {
    }
    typename PRN_dist_t::result_type operator()() {
      return dist(engine);
    }
  };

  // convenience function
  template<typename PRN_dist_t, typename PRN_engine_t>
  inline 
  binder_cl<PRN_dist_t, PRN_engine_t> make_binder(PRN_dist_t &dist, PRN_engine_t &engine) {
    return binder_cl<PRN_dist_t, PRN_engine_t>(dist, engine);
  }

#endif


// print an iterator range to stdout
template<typename iter>
void print_range(iter i1, iter i2) {
  while (i1!=i2) std::cout << (*(i1++)) << '\t';
  std::cout << "\n\n";
}

int main() {
  trng::yarn2 R;
  trng::uniform_int_dist U(0, 100);
  std::vector<long> v(10);
  
  std::cout << "random number generation by call operator\n";
  for (std::vector<long>::size_type i=0; i<v.size(); ++i)
    v[i]=U(R);
  print_range(v.begin(), v.end());
  std::vector<long> w(12);
#if defined TRNG_HAVE_BOOST
  std::cout << "random number generation by std::generate\n";
  std::generate(w.begin(), w.end(), boost::bind(U, boost::ref(R)));
  print_range(w.begin(), w.end());
  std::cout << "random number generation by std::generate\n";
  std::generate(w.begin(), w.end(), boost::bind(U, boost::ref(R)));
  print_range(w.begin(), w.end());
#else
  std::cout << "random number generation by std::generate\n";
  std::generate(w.begin(), w.end(), make_binder(U, R));
  print_range(w.begin(), w.end());
  std::cout << "random number generation by std::generate\n";
  std::generate(w.begin(), w.end(), make_binder(U, R));
  print_range(w.begin(), w.end());
#endif
  std::cout << "same sequence as above, but in a random shuffled order\n";
  std::random_shuffle(w.begin(), w.end(), R);
  print_range(w.begin(), w.end());
  return EXIT_SUCCESS;
}
