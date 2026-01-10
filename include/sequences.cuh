// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>

#include <thrust/device_vector>
#include <thrust/host_vector>

namespace seq;

// clang-format off
/** X-macro with error codes of noexcept methods of @ref Sequences. */
#define SEQ_ERRORS(X)                               \
       /*    ID      |           MESSAGE         */ \
        X(SEQ_ENONE  ,          "no error"        ) \
        X(SEQ_EINPUT ,      "invalid argument"    ) \
        X(SEQ_EFMT   ,   "invalid string format"  ) \
        X(SEQ_ELEN   ,  "invalid sequence length" )
// clang-format on

using device_slice = thrust::device_vector<uint8_t>;
using host_slice   = thrust::host_vector<uint8_t>;
using ipair        = std::array<size_t, 2>;

/** Sequence length of an uninitialized instance @ref Sequences. */
extern const size_t DEFAULT_LEN;

/**
 * Decode a noexcept @ref Sequences method error code.
 * @param err Error code returned by the method.
 *
 * This function extracts message column from @ref SEQ_ERRORS and exposes it as
 * a string view. The codes should be passed as-returned, with no negating the
 * values. This function handles that internally.
 */
std::string_view errmsg(int err) noexcept;

/**
 * A struct capturing a collection of sequences over binary alphabet.
 *
 * @var Sequences::len
 *   @brief Length of a single sequence, in bits.
 *
 *   Since we are working with a Hamming space, we take a conservative approach
 *   and enforce trivial comparability of all elements by introducing a
 *   requirement where all sequences have equal length.
 *
 * @var Sequences::data
 *   @brief A buffer containing the sequences.
 *
 *   The sequences are encoded as a single vector of bytes, in big endian order.
 *   It can be thought of as a concatenation of sequences, however if length of
 *   a single such sequence is not divisible by eight, padding bits will be
 *   added so that beginning of each sequence is aligned to byte boundary.
 */
struct Sequences {
  private:
	size_t       len;
	device_slice data;

  public:
	/**
	 * Read a collection of sequences from a stream.
	 * @param input Input stream from which to read.
	 *
	 * The expected format can be described as follows:
	 *
	 *   - First line contains length of a single sequence.
	 *   - Second line contains number of sequences.
	 *   - Remaining lines contain binary sequences.
	 *
	 * @return A new object populated with parsed data.
	 */
	static __host__ Sequences from_stream(std::istream &input);

	/**
	 * Try to reserve memory for a sequence set.
	 * @param elems Number of sequences in the set.
	 * @param len   Length of a single sequence.
	 */
	__host__ void reserve(size_t elems, size_t len = DEFAULT_LEN);

	/**
	 * @brief Push a binary sequence parsed from a string.
	 * @param elem String encoding the sequence.
	 *
	 * This function may fail if the string is not in a valid format or if
	 * the binary word it encodes does not match the length of words in
	 * the sequence.
	 *
	 * @return `0` on success.
	 * @return Negative error code decodable by @ref errmsg on error.
	 */
	__host__ int parse_push(const std::string &elem) noexcept;

	/**
	 * @brief Push a binary sequence to the collection.
	 * @param len  Length of the sequence in bits.
	 * @param elem Sequence that is to be added.
	 *
	 * This function may fail if the passed argument does not match the
	 * length of words in the sequence.
	 *
	 * @return `0` on success.
	 * @return Negative error code decodable by @ref errmsg on error.
	 */
	__host__ int push(size_t len, const host_slice &elem) noexcept;

	/**
	 * Compute a collection of index pairs with Hamming distance one.
	 * @param bsize Number of threads per block to use.
	 */
	__host__ thrust::host_vector<ipair> hamming_one(size_t bsize = 256);
};
