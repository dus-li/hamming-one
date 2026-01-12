// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <fstream>
#include <iostream>

#include "sequences.cuh"

using seq::Sequences;

static __host__ Sequences read_sequences(int argc, const char *argv[])
{
	try {
		if (argc == 1)
			return Sequences::from_stream(std::cin);

		std::ifstream file(argv[1]);
		return Sequences::from_stream(file);
	} catch (const std::exception &e) {
		throw;
	}
}

int main(int argc, const char *argv[])
{
	try {
		Sequences seqs = read_sequences(argc, argv);

		thrust::host_vector<seq::ipair> pairs = seqs.hamming_one();
		std::for_each(pairs.begin(), pairs.end(), [](seq::ipair &p) {
			std::cout << p[0] << " and " << p[1] << std::endl;
		});
	} catch (const std::exception &e) {
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
