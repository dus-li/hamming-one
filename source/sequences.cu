// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <algorithm>
#include <bit>
#include <cassert>
#include <exception>
#include <limits>
#include <ranges>
#include <stdexcept>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "array_size.cuh"
#include "sequences.cuh"

namespace seq {

enum error_codes_ {
#define ENUMIFY(code_, msg_) code_,
	SEQ_ERRORS(ENUMIFY) SEQ_ERRORS_COUNT
};

const size_t DEFAULT_LEN = 0;

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

/**
 * A functor implementing sequence comparison with a single masked element.
 *
 * Defined over a data set this functor can be used to sort a collection of
 * indices. By masking a single position in the processed words, we effectively
 * cluster potential Hamming-one candidates close to each other.
 *
 * @var Comparator::data
 *   @brief A collection of sequences, as described in @ref Sequences::data.
 *
 *   The comparison is indirect. We are not comparing elements of the data set,
 *   but instead indices into it. This allows us to leave the original input
 *   intact.
 *
 * @var Comparator::stride
 *   @brief Width, in bytes, of a single sequence in @ref Comparator::data.
 *
 *   Observe that conversion between sequence length and stride can be done
 *   through usage of @ref cell_count. Since underlying, backing type is uint8_t
 *   this requires no multiplier.
 *
 * @var Comparator::ignore_bit
 *   @brief Position of the bit that ought to be ignored in every sequence.
 */
struct Comparator {
  private:
	const uint8_t *data;
	size_t         stride;
	size_t         ignore_bit;

  public:
	__host__ Comparator(const device_slice &data, size_t len, size_t bit)
	    : data(thrust::raw_pointer_cast(data.data()))
	    , stride(cell_count(len))
	    , ignore_bit(bit)
	{
	}

	__device__ bool operator()(const size_t &a, const size_t &b) const
	{
		size_t  ignore_byte = ignore_bit >> 3;
		uint8_t ignore_mask = ~(1 << (ignore_bit & 7));

		for (size_t i = 0; i < stride; ++i) {
			uint8_t val_a = data[a * stride + i];
			uint8_t val_b = data[b * stride + i];

			if (i == ignore_byte) {
				val_a &= ignore_mask;
				val_b &= ignore_mask;
			}

			if (val_a < val_b)
				return true;

			if (val_a > val_b)
				return false;
		}

		return false;
	}
};

/**
 * Linear scan finding pairs of indices of sequences with Hamming distance one.
 * @param[out] fst     First indices of identified H1 pairs.
 * @param[out] snd     Second indices of identified H1 pairs.
 * @param[out] count   Number of identified H1 pairs.
 * @param      data    As described in @ref Sequences::data.
 * @param      indices A pre-sorted array of indices to @a data.
 * @param      elems   Number of sequences stored in @a data.
 * @param      stride  Width of a single sequence in bytes.
 * @param      bit     Bit that was masked when sorting @a indices.
 *
 * A strong pre-condition for this kernel to yield sensible results is
 * pre-sorting @a indices using a @ref Comparator instance defined over the same
 * data and the same ignored bit. This implementation relies on adjacency of
 * Hamming-one candidates.
 */
__global__ void scan(size_t *fst, size_t *snd, unsigned *count,
    const uint8_t *data, const size_t *indices, size_t elems, size_t stride,
    size_t bit)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= elems - 1)
		return;

	size_t idx1 = indices[i];
	size_t idx2 = indices[i + 1];

	size_t  byte_idx = bit >> 3;
	uint8_t bit_mask = (1 << (bit & 7));

	// Check equivalence byte by byte.
	bool match = true;
	for (size_t j = 0; j < stride; ++j) {
		uint8_t val1 = data[idx1 * stride + j];
		uint8_t val2 = data[idx2 * stride + j];
		uint8_t tmp  = val1 ^ val2;

		if (likely(j != byte_idx)) {
			if (!tmp)
				continue;
		} else {
			// Ensure values are identical in all bits except `b`
			// and that bit `b` differs.
			if (!(tmp & ~bit_mask) && (tmp & bit_mask))
				continue;
		}

		match = false;
		break;
	}

	if (match) {
		unsigned pos = atomicAdd(count, 1);
		fst[pos]     = min(idx1, idx2);
		snd[pos]     = max(idx1, idx2);
	}
}

const char *errmsg(int err) noexcept
{
	// clang-format off
#define CASIFY__(code_, msg_) case code_: return msg_;
	// clang-format on

	switch (-err) {

		SEQ_ERRORS(CASIFY__)
	}

	return "unknown error";
}

__host__ Sequences Sequences::from_stream(std::istream &input)
{
	Sequences   ret;
	std::string buf;
	size_t      len;
	size_t      elems;

	if (!(input >> len >> elems))
		throw std::runtime_error("failed to parse header");

	try {
		ret.reserve(elems, len);
		buf.reserve(len);
	} catch (const std::exception &e) {
		throw;
	}

	for (size_t i = 0; i < elems; ++i) {
		if (!(input >> buf))
			throw std::runtime_error("failed to read element");

		int tmp = ret.parse_push(buf);
		if (tmp != 0)
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
	} catch (const std::exception &e) {
		throw;
	}

	if (data.capacity() < size)
		throw std::runtime_error("failed to reserve memory");
}

__host__ int Sequences::parse_push(const std::string &elem) noexcept
{
	host_slice acc;
	uint8_t    mask = 1;
	size_t     i    = 0;

	if (elem.length() == 0)
		return -SEQ_EINPUT;

	acc.resize(cell_count(elem.length()), 0);

	for (const char &c : std::views::reverse(elem)) {
		switch (c) {
		case '1':
			acc[i >> 3] |= mask;
			[[fallthrough]];
		case '0':
			mask = std::rotl(mask, 1);
			i++;
			break;
		default:
			return -SEQ_EFMT;
		}
	}

	return this->push(i, acc);
}

__host__ int Sequences::push(size_t len, const host_slice &elem) noexcept
{
	this->len = this->len ?: len;
	if (len != this->len)
		return -SEQ_ELEN;

	data.insert(data.end(), elem.begin(), elem.end());
	return 0;
}

__host__ thrust::host_vector<ipair> Sequences::hamming_one(size_t bsize)
{
	size_t stride = cell_count(this->len);
	size_t elems  = this->data.size() / stride;
	size_t maxp   = max_matches(this->len, elems);

	thrust::device_vector<size_t>   d_out_a(maxp);
	thrust::device_vector<size_t>   d_out_b(maxp);
	thrust::device_vector<unsigned> d_count(1, 0);
	thrust::device_vector<size_t>   indices(elems);

	// This loop will execute exactly l times. Its contents are O(n * l).
	// Therefore the complexity of the solution is O(nl^2).
	for (size_t bit = 0; bit < this->len; ++bit) {
		thrust::sequence(indices.begin(), indices.end());

		// Radix sort is O(n * l)
		Comparator comp(this->data, this->len, bit);
		thrust::sort(indices.begin(), indices.end(), comp);

		// Linear scan is O(n * l)
		size_t gsize = (elems + bsize - 1) / bsize;
		scan<<<gsize, bsize>>>(thrust::raw_pointer_cast(d_out_a.data()),
		    thrust::raw_pointer_cast(d_out_b.data()),
		    thrust::raw_pointer_cast(d_count.data()),
		    thrust::raw_pointer_cast(data.data()),
		    thrust::raw_pointer_cast(indices.data()),
		    elems,
		    stride,
		    bit);
	}

	size_t h_count = d_count[0];

	thrust::host_vector<size_t> h_out_a = d_out_a;
	thrust::host_vector<size_t> h_out_b = d_out_b;
	thrust::host_vector<ipair>  ret(h_count);

	for (unsigned int i = 0; i < h_count; ++i) {
		ret[i] = { h_out_a[i], h_out_b[i] };
	}

	return ret;
}

} // namespace seq
