all:
	g++-5.1 -O3 -Wall -Itrng-4.19 -Ltrng-4.19/src/.libs HLM_KMP_1D_Para_Yao.cpp -o exe_parallel -ltrng4 -fopenmp -std=c++11
	./exe_parallel