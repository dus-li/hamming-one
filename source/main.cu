// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <iostream>

#include "sequences.cuh"

using seq::Sequences;

static __host__ std::istream &select_src(int argc, const char *argv[])
{
	try {
		return argc == 1 ? std::cin : std::ifstream(argv[1]));
	} catch {
		throw;
	}
}

int main(int argc, const char *argv[])
{
	try {
		Sequences seqs = Sequences::from_stream(select_src(argc, argc));

		thrust::host_vector<ipair> pairs = seqs.hamming_one();
		std::for_each(pairs.begin(), pairs.end(), [](ipair &p) {
			std::cout << p[0] << " and " << p[1] << std::endl;
		});
	} catch (const std::exception &e) {
		std::cerr << e.what();
	}

	return 0;
}
