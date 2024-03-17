#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <cstring>
#include <random>
#include <immintrin.h>

#define N (1<<30)
#define TRIALS 5

long count_bytes_vectorized(uint8_t *data, long n, uint8_t target) {
	long count = 0;
	// broadcast 8-bit integer target to all 32 elements of target_vectorized
	__m256i target_vectorized = _mm256_set1_epi8(target);

	for (long i = 0; i < n - 32; i += 32) {
		// load 256-bits of integer from memory to data_vectorized
		__m256i data_vectorized = _mm256_loadu_si256((__m256i_u *)(data + i));
		// compared packed 8-bit integers in data_vectorized and target_vectorized, and store the result in cmp_vectorized
		__m256i cmp_vectorized = _mm256_cmpeq_epi8(data_vectorized, target_vectorized);
		// create mask from the highest bit of each 8-bit element in cmp_vectorized
		int mask = _mm256_movemask_epi8(cmp_vectorized);
		// count number of bits set to 1 in a 32-bit integer
		count += _mm_popcnt_u32(mask);
	}

	return count;
}

long count_bytes(uint8_t *data, long n, uint8_t target) {
	long count = 0;

	for (long i = 0; i < n; i++) {
		if (data[i] == target) {
			count += 1;
		}
	}

	return count;
}

int main(int argc, char *argv[]) {
	unsigned int seed = 1;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
				seed = atoi(argv[i + 1]);
				i++;
		}
	}

	uint8_t data[N];
	uint8_t target = 17;
	std::mt19937_64 rng(seed);
	std::uniform_int_distribution<uint8_t> dist(0, UINT8_MAX);
	for (int i = 0; i < N; i++) {
			data[i] = dist(rng);
	}
	auto start_time = std::chrono::steady_clock::now();
	long count;

	for(size_t trial = 0; trial < TRIALS; trial++) {
		count = count_bytes_vectorized(data, N, target);
	}

	auto end_time = std::chrono::steady_clock::now();

	std::chrono::duration<double> diff = end_time - start_time;
	double seconds = diff.count() / TRIALS;

	std::cout << "Time per trial: " << seconds << " seconds, got " << count << " as the count.\n";
}
