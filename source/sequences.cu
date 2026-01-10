// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <algorithm>
#include <bit>
#include <cassert>
#include <exception>
#include <limits>
#include <ranges>
#include <stdexcept>

#include "array_size.cuh"
#include "sequences.cuh"

namespace seq;

enum error_codes_ {
#define ENUMIFY(code_, msg_) code_,
	SEQ_ERRORS(ENUMIFY)
};

const size_t DEFAULT_LEN = 0;

/**
 * Kernel identifying all Hamming-one pairs in a sequence set.
 * @param out   Output buffer of capacity at least @ref max_matches.
 * @param seqs  Sequences encoded in accordance to @ref Sequences::data.
 * @param len   Length of a single sequence.
 * @param elems Number of sequences in @a seqs.
 */
__global__ void compute_h1(ipair *out, uint8_t *seqs, size_t len, size_t elems);

/**
 * Compute number of bytes needed to represent a sequence.
 * @param len Number of symbols in the sequence.
 *
 * @return Number of 8-bit cells required to encode the sequence.
 */
static __host__ __device__ inline size_t cell_count(size_t len)
{
	return (len / 8) + (!!(len % 8));
}

/**
 * Compute upper limit on number of Hamming-one pairs.
 * @param len   Length of a single sequence.
 * @param elems Number of sequences considered.
 *
 * @return Maximal number of sequence pairs with Hamming distance equal to one.
 */
static __host__ __device__ inline size_t max_matches(size_t len, size_t elems)
{
	// Each of `elems` sequences has exactly `len` positions where it may
	// differ from another sequence. Since order does not matter, we divide
	// by two after applying the rule of product.
	return (len * elems) / 2;
}

std::string_view errmsg(int err) noexcept
{
	static const std::string MSG_LUT[] = {
#define LUTIFY(code_, msg_) [code_] = msg_,
		SEQ_ERRORS(LUTIFY)
	};

	err = -err;
	if (unlikely(err < 0 || err >= ARRAY_SIZE(MSG_LUT)))
		return "unknown error";

	return MSG_LUT[err];
}

static __host__ Sequences Sequences::from_stream(std::istream &input)
{
	Sequences   ret;
	std::string buf;
	size_t      len;
	size_t      elems;

	input >> len >> elems;
	if (input.fail())
		throw std::runtime_error("failed to parse input");

	try {
		ret.reserve(elems, len);
		buf.reserve(len);
	} catch {
		throw;
	}

	for (size_t i = 0; i < elems; ++i) {
		if (input.eof()) {
			std::cout << "Warning: premature eof" << std::endl;
			break;
		}

		input >> buf;

		int tmp = ret.parse_push(buf);
		if (tmp != SEQ_ENONE)
			throw std::runtime_error(errmsg(tmp));
	}

	return ret;
}

__host__ void Sequences::reserve(size_t elems, size_t len)
{
	size_t cells;
	size_t size;

	this->len = this->len ?: len;
	if (this->len != len)
		throw std::logic_error("sequence length discrepancy");

	// Equality of naturals is transitive. If both are default length, then
	// this entire operation is meaningless.
	if (this->len == DEFAULT_LEN)
		throw std::logic_error("attempt to reserve 0 bytes");

	// Compute number of bytes per sequence and size.
	cells = cell_count(len);
	size  = elems * cells;

	// Ensure size is not a product of an overflow.
	if (std::numeric_limits<typeof(elems)>::max() / elems < cells)
		throw std::overflow_error("allocation size not representable");

	try {
		// May throw length_error if size exceeds some internal limit.
		data.reserve(size);
	} catch {
		throw;
	}

	if (size < data.capacity())
		throw std::runtime_error("failed to reserve memory");
}

__host__ int Sequences::parse_push(const std::string &elem) noexcept
{
	host_slice acc;
	uint8_t    mask = 1;
	size_t     i    = 0;

	if (elem.length() == 0)
		return -SEQ_EINPUT;

	for (const auto &c : std::views::reverse(elem)) {
		switch (elem) {
		case '1':
			acc[i >> 3] |= mask;
			fallthrough;
		case '0':
			mask = std::rotl(mask, 1);
			i++;
			break;
		default:
			return -SEQ_EFMT;
		}
	}

	return this->push(acc);
}

__host__ int Sequences::push(size_t len, const host_slice &elem) noexcept
{
	// If at least one element was added, the length should be set.
	assert((data.size() > 0) == (this->len != DEFAULT_LEN));

	this->len = this->len ?: len;
	if (len != this->len)
		return -SEQ_ELEN;

	if (likely(data.size())) {
		data.insert(data.end(), elem.begin(), elem.end());
		return 0;
	}

	data = elem;
	return 0;
}

__host__ thrust::host_vector<ipair> Sequences::hamming_one(size_t bsize)
{
	uint8_t *d_data = this->data.get();
	size_t   elems  = this->data.size();
	size_t   gsize  = (elems + bsize - 1) / bsize;

	thrust::device_vector<ipair> res;
	try {
		res.reserve(max_matches(this->len, elems));
	} catch {
		throw;
	}

	compute_h1<<<bsize, gsize>>>(res.get(), d_data, this->len, elems);

	return res;
}

__global__ void compute_h1(ipair *out, uint8_t *seqs, size_t len, size_t elems)
{
	if (blockIdx.x * blockDim.x + threadIdx.x > elems)
		return;

	// TODO
}
