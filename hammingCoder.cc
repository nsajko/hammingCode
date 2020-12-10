// Copyright 2019 Neven Sajko. All rights reserved.
//
// https://github.com/nsajko/hammingCode
//
// A Hamming code coder.
//
// A generator matrix approach is used as an optimization for large
// messages.
//
// Bit vectors are used to compactly represent arbitrarily long strings
// of bits.
//
// For simplicity, I ignored the possibility of heap allocation failing.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

using uint8 = unsigned char;
using uint = unsigned int;
using uintmax = std::uintmax_t;
using intmax = std::intmax_t;

constexpr intmax byteBits{8};

// Hamming code algorithms.
enum class HammingCoderAlgor {
	Rows,
	RowsSparse,
	RowsDense,
	Cols,
	ColsSparse,
	VeryNaive,

	// Incorrect but fast, to get an upper limit on possible throughput.
	Dummy,
};

// Macros are used for conditional compilation, but the goal is to replace them with
// constexpr variables as soon as possible, undeffing the macros simultaneously.

[[maybe_unused]] constexpr HammingCoderAlgor hamCoderAlgo{HAM_COD_ALG};
#undef HAM_COD_ALG

#ifdef USE_STOPWATCH
#   undef USE_STOPWATCH
[[maybe_unused]] constexpr bool useStopwatch{true};
#else
[[maybe_unused]] constexpr bool useStopwatch{false};
#endif

// For profiling or benchmarking the coder.
#ifdef PRINT_LESS
#   undef PRINT_LESS
[[maybe_unused]] constexpr bool printLess{true};
#else
[[maybe_unused]] constexpr bool printLess{false};
#endif

// In bytes.
constexpr int bitStorageAlignment{1UL << 4};

// Configurable initial capacity for the input message in bits.
constexpr intmax initialInputMessageCapacity{bitStorageAlignment * byteBits};

// Shorthand for static_cast.
template<typename X, typename Y>
[[nodiscard]] constexpr X
sc(Y v) { return static_cast<X>(v); }

// Returns the k Hamming code parameter corresponding to a given n.
[[nodiscard]] intmax
hammingK(intmax n) {
	auto N{sc<uintmax>(n)};
	return sc<intmax>(N - std::bit_width(N));
}

// Returns the n Hamming code parameter corresponding to a given k.
[[nodiscard]] intmax
hammingN(intmax k) {
	auto K{sc<uintmax>(k)};
	return sc<intmax>(K + std::bit_width(K + std::bit_width(K)));
}

template<typename T>
[[nodiscard]] char
numToASCII(T a) {
	return sc<char>(sc<uintmax>(a) | 0x30UL);
}

// Converts an ASCII char to the number it represents.
[[nodiscard]] intmax
ASCIIToNum(intmax c) {
	return sc<intmax>(sc<uintmax>(c) & 0x0fUL);
}

// Lexes an ASCII string into a number. Does not look at anything after
// the first char outside the ASCII numeral range.
[[nodiscard]] intmax
lexDecimalASCII(const char *s) {
	intmax r{0};
	for (int i{0};; i++) {
		intmax c{s[i]};
		if (c < '0' || '9' < c) {
			break;
		}
		r = 10 * r + ASCIIToNum(c);
	}
	return r;
}

// Returns a character or EOF.
template<typename X>
concept Reader = requires(X r) {
	{r()} -> std::same_as<int>;
};

template<typename T>
concept IsIndex = (std::is_integral_v<T> && std::is_signed_v<T>);

// A bit storage type is defined by the unsigned integer word type T, and the alignment n
// of the bit storage, in bytes.
template<typename T, int n>
concept BitStorage = (std::is_integral_v<T> && std::is_unsigned_v<T> &&
		      std::has_single_bit(sc<uint>(n)) && (sizeof(T) <= n));

// A bit vector type.
template<typename word, int aligSize>
requires BitStorage<word, aligSize>
class BitVector final {
	// Length in bits.
	intmax len{0};

	struct AlignedBits final {
		alignas(aligSize) word a[aligSize / sizeof(word)];

		[[nodiscard]] word &
		operator[](intmax i) {
			static_assert(sizeof(*this) == aligSize);
			return a[i];
		}

		[[nodiscard]] word
		operator[](intmax i) const {
			return a[i];
		}
	};
	static_assert(sizeof(AlignedBits) == aligSize);

	// Backing storage.
	std::vector<AlignedBits> arr;

	using vst = typename std::vector<AlignedBits>::size_type;

	static constexpr intmax wordBits{byteBits * sizeof(word)};
	static constexpr intmax alignedBits{byteBits * sizeof(AlignedBits)};
	static constexpr intmax alignedWords{aligSize / sizeof(word)};

	// Returns ceiling(n / alignedBits).
	[[nodiscard]] static intmax
	ceilDivAligned(intmax n) {
		return (n - 1) / alignedBits + 1;
	}

	// Returns ceiling(n / wordBits).
	[[nodiscard]] static intmax
	ceilDivWord(intmax n) {
		return (n - 1) / wordBits + 1;
	}

	// Returns ceiling(n / byteBits).
	[[nodiscard]] static intmax
	ceilDivByte(intmax n) {
		return (n - 1) / byteBits + 1;
	}

	// Traps if the bit vector's backing storage is misaligned.
	void
	trapIfMisaligned() {
		constexpr int S{1UL << 15};
		std::size_t sz{S};
		void *p{arr.data()};
		if (std::align(aligSize, aligSize, p, sz) != arr.data() ||
		    p != arr.data() || sz != S) {
			std::cerr << "BitVector<" << sizeof(word) << ", " << aligSize <<
				     ">.trapIfMisaligned: misaligned bit storage\n";
			std::cerr.flush();
			__builtin_trap();
		}
	}

	// Returns the i-th word in the bit storage.
	[[nodiscard]] word
	operator[](intmax i) const {
		return arr[sc<vst>(i / alignedWords)][i % alignedWords];
	}

	[[nodiscard]] word &
	operator[](intmax i) {
		return arr[sc<vst>(i / alignedWords)][i % alignedWords];
	}

	public:

	// The enumeration constants themselves are not important, just the types are.
	enum class ConstrTypeAlloc {e};
	enum class ConstrTypeZero {e};
	enum class ConstrTypeVeryNaive {e};
	enum class ConstrTypeDummyCoder {e};

	[[nodiscard]] intmax
	getLen() const {
		return len;
	}

	void
	resize(intmax l) {
		if (len < l) {
			std::cerr << "resize: enlarging bit vectors isn't implemented\n";
			__builtin_trap();
		}
		len = l;
	}

	// The set parameter should be either 0 or 1.
	//
	// If set is 0, effectively nothing is done.
	//
	// If set is 1, the i-th bit in a is set.
	void
	set(intmax set, intmax i) {
		(*this)[i / wordBits] |= sc<word>(sc<word>(set) << (i % wordBits));
	}

	// XORs the bit vector's i-th bit with the bit bit.
	void
	exOrBit(intmax bit, intmax i) {
		(*this)[i / wordBits] ^= sc<word>(sc<word>(bit) << (i % wordBits));
	}

	// Returns the i-th bit.
	[[nodiscard]] intmax
	isSet(intmax i) const {
		return ((*this)[i / wordBits] >> (i % wordBits)) & 1UL;
	}

	// Allocates enough memory for the given capacity in bits, sets len to 0.
	template<typename ConstrType>
	requires std::is_same_v<ConstrType, ConstrTypeAlloc> || std::is_same_v<ConstrType, ConstrTypeZero>
	BitVector([[maybe_unused]] ConstrType unused, intmax capBits) {
		auto s{ceilDivAligned(capBits)};
		arr.reserve(sc<vst>(s));
		if constexpr (std::is_same_v<ConstrType, ConstrTypeZero>) {
			len = capBits;
			arr.resize(sc<vst>(s));
			trapIfMisaligned();
		}
	}

	// Copies a contiguous range of l bits from in starting at index i to a new BitVector.
	// The following relation must hold: out_len_bit <= in.len - in_off_bit.
	BitVector(const BitVector<word, aligSize> &in, intmax in_off_bit, intmax out_len_bit):
	BitVector(BitVector<word, aligSize>::ConstrTypeZero::e, out_len_bit) {
		intmax in_len_bit{in.len}, in_off_wrd{in_off_bit / wordBits}, out_off_wrd{0},
		     out_len_wrd{ceilDivWord(out_len_bit)}, in_len_wrd{ceilDivWord(in.len)};
		if (!(out_len_bit <= in_len_bit - in_off_bit)) {
			std::cerr << "BitVector(BitVector<" << sizeof(word) << ", " << aligSize <<
			             "> &, intmax, intmax): flawed caller\n";
			std::cerr.flush();
			__builtin_trap();
		}
		in_off_bit %= wordBits;
		in_len_bit %= wordBits;
		out_len_bit %= wordBits;
		if (in_off_bit != 0) {
			for (; out_off_wrd < out_len_wrd - 1; out_off_wrd++, in_off_wrd++) {
				auto w0{in[in_off_wrd]}, w1{in[in_off_wrd + 1]};
				auto i{wordBits - in_off_bit};
				(*this)[out_off_wrd] = sc<word>((w0 >> in_off_bit) | sc<word>(w1 << i));
			}

			(*this)[out_off_wrd] = sc<word>(in[in_off_wrd] >> in_off_bit);

			in_off_wrd++;
			if (in_off_wrd < in_len_wrd) {
				auto i{wordBits - in_off_bit};
				(*this)[out_off_wrd] |= sc<word>(in[in_off_wrd] << i);
			}
		} else {
			for (; out_off_wrd < out_len_wrd; out_off_wrd++, in_off_wrd++) {
				(*this)[out_off_wrd] = in[in_off_wrd];
			}

			// Clear highest bits from the last word after the end of the BitVector.
			out_off_wrd--;
		}
		auto i{(wordBits - out_len_bit) % wordBits};
		(*this)[out_off_wrd] = sc<word>(sc<word>((*this)[out_off_wrd] << i) >> i);
	}

	// Fills the BitVector with input from r.
	template<typename X>
	requires Reader<X>
	BitVector(X r):
	BitVector(BitVector<word, aligSize>::ConstrTypeAlloc::e, initialInputMessageCapacity) {
		for (;;) {
			int c{r()};
			if (c == '	' || c == ' ' || c == '\n' || c == '\r') {
				continue;
			}
			if (c != '0' && c != '1') {
				break;
			}

			if (auto arrSize{arr.size()}; sc<vst>(len) == arrSize * alignedBits) {
				arr.resize(arrSize + 1);
			}
			set(ASCIIToNum(c), len);
			len++;
		}
	}

	// A dummy coder, not actually a coder, just has the coder interface. Faster than
	// a true coder.
	BitVector([[maybe_unused]] ConstrTypeDummyCoder unused,
	  const BitVector<word, aligSize> &in, intmax n):
	BitVector(BitVector<word, aligSize>::ConstrTypeZero::e, n) {
		n = hammingN(in.len);

		auto nWords{ceilDivWord(len)};
		for (intmax i{0}; i < nWords; i++) {
			auto dat{in[i / 2] ^ sc<word>(0xdcbfcdafbe972023UL)};
			(*this)[i] = dat;
			i++;
			if (i == nWords) {
				break;
			}
			(*this)[i] = ~dat;
		}
	}

	// Copies the instance's bits one bit per char into a std::vector<char>.
	[[nodiscard]] std::vector<char>
	fatten() const {
		std::vector<char> r(sc<vst>(len));
		for (intmax i{0};; i++) {
			auto w{(*this)[i]};
			for (int j{0};; j++) {
				auto I{i * wordBits + j};
				if (I == len) {
					return r;
				}
				if (j == wordBits) {
					break;
				}
				r[sc<vst>(I)] = (w >> j) & 1UL;
			}
		}
	}

	// Checks equality between BitVectors. Used just for testing.
	[[nodiscard]] bool
	equal(const BitVector<word, aligSize> &v) const {
		if (len != v.len) {
			return false;
		}
		for (intmax l{ceilDivWord(len)}, i{0}; i < l; i++) {
			if ((*this)[i] != v[i]) {
				return false;
			}
		}
		return true;
	}

	// XORs the current instance with op.
	void
	maskedExOr(const BitVector &op, uint8 bit) {
		auto mask{sc<uint8>(sc<uint8>(~0U) * bit)};
		auto out{std::assume_aligned<aligSize>(reinterpret_cast<uint8*>(arr.data()))};
		auto in{std::assume_aligned<aligSize>(reinterpret_cast<const uint8*>(op.arr.data()))};
		for (intmax l{ceilDivByte(len)}, i{0}; i < l; i++) {
			out[i] ^= in[i] & mask;
		}
	}

	// Shows the bit vector on stdout.
	void
	print() const {
		if constexpr (printLess) {
			for (intmax i{0}; i < len; i += wordBits + 1) {
				std::cout.put(
				  numToASCII(((*this)[i / wordBits] >> (i % wordBits)) & 1UL));
			}
			std::cout.put('\n');
			return;
		}
		using chars = std::vector<char>;
		chars buf(sc<chars::size_type>(len));
		for (intmax i{0}; i < len; i++) {
			buf[sc<chars::size_type>(i)] = numToASCII(((*this)[i / wordBits] >> (i % wordBits)) & 1UL);
		}
		std::cout.write(buf.data(), len).put('\n');
	}
};

// A very naive Hamming code coder, doesn't use matrix multiplication.
[[nodiscard]] std::vector<char>
hamCodeNaive(const std::vector<char> &in, intmax n) {
	using vst = std::vector<char>::size_type;
	std::vector<char> r(sc<vst>(n), 0);
	n = hammingN(sc<intmax>(in.size()));

	// Copy the data bits from the input.
	for (intmax I{0}, pow{4}, i{3}; i <= n; i++) {
		if (i == pow) {
			pow <<= 1;
			continue;
		}

		r[sc<vst>(i - 1)] = in[sc<vst>(I)];
		I++;
	}

	// Create parity bits.
	for (intmax pow{1}; pow <= n; pow <<= 1) {
		// TODO: in the loop below it is possible to halve the number of
		// iterations, but I haven't been able to do that without inducing
		// huge slowdowns, instead of speedups. Probably has something to do with
		// autovectorization.
		for (intmax j{pow + 1}; j <= n; j++) {
			if ((j & pow) != 0) {
				r[sc<vst>(pow - 1)] ^= r[sc<vst>(j - 1)];
			}
		}
	}

	return r;
}

// OEIS sequence A000295, Eulerian numbers.
//
// The position of the first set bit in column n of a Hamming code generator matrix.
//
// In:  0, 1, 2, 3, 4, ...
// Out: 0, 0, 1, 4, 11, ...
[[nodiscard]] intmax
A000295(intmax n) {
	return (1 << n) - n - 1;
}

// OEIS sequence A209229, Characteristic function of powers of 2.
[[nodiscard]] intmax
A036987(intmax n) {
	return sc<intmax>(std::has_single_bit(sc<uintmax>(n)));
}

// Returns the number of set bit ranges in col-th power-of-two column of a Hamming code
// generator matrix with given rank k. Call with col = 0, 1, 2, 3, ...
[[nodiscard]] intmax
hamGenMatColRangesNum(intmax col, intmax k) {
	intmax r{1};
	for (intmax c{1 << (col + 1)}, j{A000295(col) - 1 - 1 + c}; j < k; j += c - A036987(r)) {
		r++;
	}
	return r;
}

// A Hamming code coder that iterates through the columns of an imagined
// generator matrix in the outermost loop.
[[nodiscard]] std::vector<char>
hamCodeCols(const std::vector<char> &in, intmax n) {
	// This is very similar to the GenMatColsSparse constructor.

	using vst = std::vector<char>::size_type;

	std::vector<char> r(sc<vst>(n), 0);

	auto inLen{sc<intmax>(in.size())};

	// The fact that in.len can be smaller than hammingK(n) (which can happen with the
	// last chunk of input) complicates the implementation somewhat.
	// in.len can be smaller than hammingK(n), in which case we need to decrease n
	// accordingly.
	//
	// In our program this only happens with the last chunk of input to be coded.
	n = hammingN(inLen);

	// Create parity/check bits.
	for (intmax pow{1}; pow < n; pow <<= 1) {
		for (intmax i{pow + 1}; i <= n; i++) {
			if ((i & pow) != 0) {
				r[sc<vst>(pow - 1)] ^= in[sc<vst>(hammingK(i) - 1)];
			}
		}
	}

	// Copy the data bits.
	for (intmax i{0}; i < inLen; i++) {
		r[sc<vst>(hammingN(i + 1) - 1)] = in[sc<vst>(i)];
	}

	return r;
}

template<typename T>
requires IsIndex<T>
class GenMatColsSparse final {
	// We store just the power-of-two columns of the generator matrix. Each one is
	// represented as a set of ranges of set bits.

	intmax nCols;

	intmax nPow2Cols;
	T *nRanges;
	T **ranges;

	public:

	// Construct a sparse representation for a generator matrix for a Hamming code
	// with given n.
	GenMatColsSparse(intmax n):
	nCols{n}, nPow2Cols{sc<intmax>(std::bit_width(sc<uintmax>(n)))}, nRanges{new T[sc<uintmax>(nPow2Cols)]},
	ranges{new T*[sc<uintmax>(nPow2Cols)]} {
		for (intmax k{hammingK(n)}, i{0}; i < nPow2Cols; i++) {
			nRanges[i] = sc<T>(hamGenMatColRangesNum(i, k));
			ranges[i] = new T[sc<uintmax>(nRanges[i])];

			// Store the index of the first bit of each range of set bits in the column.
			intmax c{1 << (i + 1)}, j{A000295(i)}, r{0};
			ranges[i][r] = sc<T>(j);
			for (j += c - 2, r++; r < nRanges[i]; j += c - A036987(r)) {
				ranges[i][r] = sc<T>(j);
				r++;
			}
		}
	}

	~GenMatColsSparse() {
		for (intmax i{0}; i < nPow2Cols; i++) {
			delete[] ranges[i];
		}
		delete[] ranges;
		delete[] nRanges;
	}

	// Multiplies the row-vector with the matrix, iterating through the columns of
	// the generator matrix in the outermost loop.
	[[nodiscard]] std::vector<char>
	rowMulMat(const std::vector<char> &row) const {
		using vst = std::vector<char>::size_type;

		std::vector<char> out(sc<vst>(nCols), 0);

		intmax K{std::min(sc<intmax>(row.size()), hammingK(nCols))};

		// Iterate through the power-of-two columns of the generator matrix,
		// creating the parity bits.
		for (intmax i{0}; i < nPow2Cols; i++) {
			// Ranges of set bits within column.

			// Length of range of set bits.
			intmax c{1 << i};

			// First range of set bits, has one less set bit than other ranges.
			intmax j{0};
			for (intmax k{ranges[i][j]}, e{std::min(k - 1 + c, K)}; k < e; k++) {
				out[sc<vst>(c - 1)] ^= row[sc<vst>(k)];
			}

			for (j++; j < nRanges[i]; j++) {
				for (intmax k{ranges[i][j]}, e{std::min(k + c, K)}; k < e; k++) {
					out[sc<vst>(c - 1)] ^= row[sc<vst>(k)];
				}
			}
		}

		// Copy the data bits.
		for (intmax i{0}; i < K; i++) {
			out[sc<vst>(hammingN(i + 1) - 1)] = row[sc<vst>(i)];
		}

		return out;
	}
};

// A Hamming code coder that iterates through the rows of an imagined
// generator matrix in the outermost loop.
[[nodiscard]] std::vector<char>
hamCodeRows(const std::vector<char> &in, intmax n) {
	// This is very similar to the GenMatRowsSparse and GenMatRowsDense constructors.

	using vst = std::vector<char>::size_type;

	std::vector<char> r(sc<vst>(n), 0);

	intmax nRows{sc<intmax>(in.size())};

	n = hammingN(nRows);

	// Add relevant rows of the imagined generator matrix to r.
	for (intmax i{0}; i < nRows; i++) {
		// Add to r the row i of the imagined generator matrix multiplied by
		// the bit in[i].

		auto Col{sc<uintmax>(hammingN(i + 1))};

		// Iterate through the set bits of hammingN(i + 1), effectively going
		// through the positions of the set bits in row[i] of the imagined
		// generator matrix.
		intmax j{0};
		for (uintmax col{Col}; col != 0;) {
			auto d{std::countr_zero(col)};
			j += d;
			r[sc<vst>((1UL << j) - 1)] ^= in[sc<vst>(i)];

			j++;
			col >>= d + 1;
		}

		r[sc<vst>(Col - 1)] = in[sc<vst>(i)];
	}

	return r;
}

template<typename T>
requires IsIndex<T>
class GenMatRowsSparse final {
	intmax rows;
	T *cols;
	T **m;

	public:

	// Construct a sparse representation for a generator matrix for a Hamming code
	// with given n.
	GenMatRowsSparse(intmax n):
	rows(hammingK(n)), cols(new T[sc<uintmax>(rows)]), m(new T*[sc<uintmax>(rows)]) {
		for (intmax i{0}; i < rows; i++) {
			// Number of columns in this row, densely represented, stripped of
			// trailing zeros.
			auto Col{sc<uintmax>(hammingN(i + 1))};

			// Number of columns in this row, sparsely represented.
			intmax spc{std::popcount(Col) + 1};

			cols[i] = sc<T>(spc);
			m[i] = new T[sc<uintmax>(spc)];

			intmax c{0}, j{0};
			for (auto col{Col}; col != 0;) {
				auto d{std::countr_zero(col)};
				j += d;
				m[i][c] = sc<T>((1UL << j) - 1);
				c++;

				j++;
				col >>= d + 1;
			}

			// Bit (i, spc - 1) is always set.
			m[i][spc - 1] = sc<T>(Col - 1);
		}
	}

	~GenMatRowsSparse() {
		for (intmax i{0}; i < rows; i++) {
			delete[] m[i];
		}
		delete[] m;
		delete[] cols;
	}

	// Multiplies the row-vector with the matrix, iterating through the rows of
	// the generator matrix in the outermost loop.
	[[nodiscard]] std::vector<char>
	rowMulMat(const std::vector<char> &row) const {
		// There is always a set bit in the final position (last row, last column)
		// of the generator matrix, so the vector sizes correspond to
		// matrix dimensions.
		intmax nCols{sc<intmax>(m[rows - 1][cols[rows - 1] - 1] + 1)};
		std::vector<char> out(sc<uintmax>(nCols), 0);
		intmax nRows{sc<intmax>(row.size())};

		// Add relevant rows of the matrix to out.
		for (intmax i{0}; i < nRows; i++) {
			// Add to out the row m[i] multiplied by the bit row[i].
			for (intmax c{0}; c < cols[i]; c++) {
				out[sc<uintmax>(m[i][c])] ^= row[sc<uintmax>(i)];
			}
		}
		return out;
	}
};

void
printFatBitVector(const std::vector<char> &bits) {
	using vst = std::vector<char>::size_type;
	auto l{bits.size()};
	if constexpr (printLess) {
		// Here we should do something comparable to what's done in
		// BitVector.print, and we don't know how many bits there are in
		// BitVector's word, but assuming 64 seems fine for now.

		// ceiling(l / 64)
		l = (l - 1) / 64 + 1;

		for (intmax i{0}; sc<vst>(i) < l; i += 64 + 1) {
			std::cout.put(numToASCII(bits[sc<vst>(i)]));
		}
		std::cout.put('\n');
		return;
	}
	std::vector<char> ascii(l);
	for (intmax i{0}; sc<decltype(l)>(i) < l; i++) {
		ascii[sc<vst>(i)] = numToASCII(bits[sc<vst>(i)]);
	}
	std::cout.write(ascii.data(), sc<std::streamsize>(ascii.size())).put('\n');
}

template<typename T, int S>
requires BitStorage<T, S>
class GenMatRowsDense final {
	std::vector<BitVector<T, S>> m;
	using vst = typename std::vector<BitVector<T, S>>::size_type;

	public:

	// Makes the generator matrix for the [n, k] Hamming code.
	GenMatRowsDense(intmax n) {
		intmax rows{hammingK(n)};
		m.reserve(sc<vst>(rows));
		for (intmax i{0}; i < rows; i++) {
			m.emplace_back(BitVector<T, S>(BitVector<T, S>::ConstrTypeZero::e, n));
		}
		for (intmax i{0}; i < rows; i++) {
			auto Col{sc<uintmax>(hammingN(i + 1))};

			intmax j{0};
			for (auto col{Col}; col != 0;) {
				auto d{std::countr_zero(col)};
				j += d;
				m[sc<vst>(i)].set(1, sc<intmax>((1UL << j) - 1));

				j++;
				col >>= d + 1;
			}

			m[sc<vst>(i)].set(1, sc<intmax>(Col - 1));
		}
	}

	// Multiplies the row-vector with the matrix, iterating through the rows of
	// the generator matrix in the outermost loop.
	[[nodiscard]] BitVector<T, S>
	rowMulMat(const BitVector<T, S> &row) const {
		BitVector<T, S> out(BitVector<T, S>::ConstrTypeZero::e, m[0].getLen());

		// Add relevant rows of the matrix to out.
		for (intmax len{row.getLen()}, i{0}; i < len; i++) {
			// Add to out the row m[i] multiplied by the bit row[i].
			out.maskedExOr(m[sc<vst>(i)], sc<uint8>(row.isSet(i)));
		}
		return out;
	}

	// Prints the matrix.
	void
	print() const {
		auto rows{sc<intmax>(m.size())};
		for (intmax r{0}; r < rows; r++) {
			m[sc<vst>(r)].print();
		}
		std::cout.put('\n');
	}
};

}  // namespace

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION

// This is the fuzzing code, used for finding defects in the main code.
// See https://llvm.org/docs/LibFuzzer.html

namespace {

// Replaces two BitVector constructors while testing the rest of the code.
template<typename T, int S, typename X>
requires BitStorage<T, S>
std::vector<BitVector<T, S>>
makeBitVectorVectorWithInput(X r, intmax chunkSize) {
	std::vector<BitVector<T, S>> res;
	res.reserve(1UL << 4);
	res.emplace_back(BitVector<T, S>(BitVector<T, S>::ConstrTypeZero::e, chunkSize));
	intmax len{0};
	for (typename decltype(res)::size_type i{0};;) {
		int c{r()};
		if (c == '	' || c == ' ' || c == '\n' || c == '\r') {
			continue;
		}
		if (c != '0' && c != '1') {
			break;
		}

		if (len == chunkSize) {
			len = 0;
			i++;
			res.emplace_back(BitVector<T, S>(
			  BitVector<T, S>::ConstrTypeZero::e, chunkSize));
		}
		res[i].set(ASCIIToNum(c), len);
		len++;
	}
	res[res.size() - 1].resize(len);
	return res;
}

}  // namespace

extern "C"
int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
	if (size < 2 * sizeof(uint8_t)) {
		return 0;
	}

	uint8_t nByte;
	memcpy(&nByte, data, sizeof(nByte));
	data += sizeof(nByte);
	size -= sizeof(nByte);

	intmax n{nByte};
	if (n == 0 || std::has_single_bit(sc<uint>(n))) {
		return 0;
	}
	intmax k{hammingK(n)};

	using bWord = uint8;
	constexpr int align{4};
	using bV = BitVector<bWord, align>;

	class FuzzReader final {
		// The pointer arithmetic here is a bit ugly, but it's OK because the data
		// members are private and the class is very small.
		intmax cap;
		const uint8 *arr;

		public:

		FuzzReader(intmax cap, const uint8 *arr): cap(cap), arr(arr) {}

		int
		operator()() {
			if (cap == 0) {
				return sc<int>(std::char_traits<uint8>::eof());
			}
			cap--;
			arr++;
			return arr[-1];
		}
	} fakeGet(sc<intmax>(size), data);

	bV inMsg(fakeGet);
	auto inMsgTest{makeBitVectorVectorWithInput<bWord, align>(fakeGet, k)};
	GenMatRowsDense<bWord, align> genMat(n);
	GenMatRowsSparse<int> genMatSprs(n);
	GenMatColsSparse<int> genMatSprsCols(n);
	decltype(inMsgTest)::size_type I{0};
	for (intmax blLen{k}, iMsgLen{inMsg.getLen()}, i{0}; i < iMsgLen; i += k, I++) {
		if (iMsgLen - i < blLen) {
			blLen = iMsgLen - i;
		}
		bV iChunk(inMsg, i, blLen);
		if (!iChunk.equal(inMsgTest[I])) {
			__builtin_trap();
		}
		std::vector<char> iChunkFat{iChunk.fatten()}, naiveResult{hamCodeNaive(iChunkFat, n)};
		if (!(naiveResult == genMat.rowMulMat(iChunk).fatten())) {
			__builtin_trap();
		}
		if (!(naiveResult == hamCodeCols(iChunkFat, n))) {
			__builtin_trap();
		}
		if (!(naiveResult == hamCodeRows(iChunkFat, n))) {
			__builtin_trap();
		}
		if (!(naiveResult == genMatSprs.rowMulMat(iChunkFat))) {
			__builtin_trap();
		}
		if (!(naiveResult == genMatSprsCols.rowMulMat(iChunkFat))) {
			__builtin_trap();
		}
	}

	return 0;
}

#else

int
main(int argc, char *argv[]) {
	std::ios::sync_with_stdio(false);

	// Handle program arguments (argv).
	if (argc != 1 + 2) {
		std::cerr << "coder: wrong number of arguments, start the program\n"
		             "with two arguments, both natural numbers\n";
		return 1;
	}
	intmax n{lexDecimalASCII(argv[1])};
	if (n == 0) {
		std::cerr << "coder: wrong input for first argument (n).\n"
		             "n can not be zero, because no code words would exist in that case\n";
		return 1;
	}
	if (std::has_single_bit(sc<uint>(n))) {
		std::cerr << "coder: wrong input for first argument (n).\n"
		             "n can not be a power of two, because a parity bit\n"
		             "would be wasted in that case as the last bit\n";
		return 1;
	}
	intmax k{lexDecimalASCII(argv[2])};
	if (auto correctK{hammingK(n)}; k != correctK) {
		std::cerr << "coder: the given combination of arguments does not describe\n"
		             "a Hamming code. Try either (" << n << ", " << correctK <<
		             ") or (" << hammingN(k) << ", " << k << ").\n";
		return 1;
	}

	std::cerr << "Linear block code [n = " << n << ", k = " << k <<
	             "]\n(n = code word length) (k = number of source bits in each code word)\n"
	             "code rate = R(K) = " << (sc<double>(k) / sc<double>(n)) <<
	             "\n\nEnter a message in bits (possibly separated by whitespace)\n"
	             "to be Hamming coded using the chosen code parameters:\n\n";
	std::cerr.flush();

	using bWord = uintmax;
	using bV = BitVector<bWord, bitStorageAlignment>;

	bV inMsg([]()->int {return std::cin.get();});
	std::cerr << "\nInput source message:\n";
	std::cerr.flush();
	inMsg.print();

	// Make and show the code's generator matrix.
	GenMatRowsDense<bWord, bitStorageAlignment> genMat(n);
	GenMatRowsSparse<int> genMatSprs(n);
	GenMatColsSparse<int> genMatSprsCols(n);
	if constexpr (hamCoderAlgo == HammingCoderAlgor::RowsDense) {
		std::cerr << "\nThe generator matrix for the code:\n\n";
		std::cerr.flush();
		genMat.print();
	}

	std::cout << '\n';

	std::cerr << "To encode the entire source input string into code words, we divide the\n"
	             "input string into parts of k or less bits, where the last part's\n"
	             "last bits are padded with zeros. Each input part is\n"
	             "multiplied with the generator to produce\nthe corresponding code word.\n\n";
	std::cerr.flush();

	std::chrono::time_point<std::chrono::steady_clock> startTime;
	if constexpr (useStopwatch) {
		startTime = std::chrono::steady_clock::now();
	}

	for (intmax blLen{k}, iMsgLen{inMsg.getLen()}, i{0}; i < iMsgLen; i += k) {
		if (iMsgLen - i < blLen) {
			blLen = iMsgLen - i;
		}

		constexpr bool usingFatBitVectors{hamCoderAlgo == HammingCoderAlgor::Cols ||
		                                  hamCoderAlgo == HammingCoderAlgor::ColsSparse ||
		                                  hamCoderAlgo == HammingCoderAlgor::VeryNaive ||
						  hamCoderAlgo == HammingCoderAlgor::Rows ||
						  hamCoderAlgo == HammingCoderAlgor::RowsSparse};

		// Copy blLen bits from inMsg to iChunk and iChunkFat. Whether iChunk or
		// iChunkFat is used is determined at compilation time.
		bV iChunk(inMsg, i, blLen);
		std::vector<char> iChunkFat;
		if constexpr (usingFatBitVectors) {
			iChunkFat = iChunk.fatten();
		}
		if constexpr (!printLess) {
			std::cerr << "Input " << std::setw(4) << blLen << " bits: ";
			std::cerr.flush();
			if constexpr (usingFatBitVectors) {
				printFatBitVector(iChunkFat);
			} else {
				iChunk.print();
			}
		}

		// Compute the output code word.
		if constexpr (!printLess) {
			std::cerr << "Output: ";
			std::cerr.flush();
		}
		if constexpr (hamCoderAlgo == HammingCoderAlgor::VeryNaive) {
			printFatBitVector(hamCodeNaive(iChunkFat, n));
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::Cols) {
			printFatBitVector(hamCodeCols(iChunkFat, n));
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::ColsSparse) {
			printFatBitVector(genMatSprsCols.rowMulMat(iChunkFat));
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::RowsDense) {
			genMat.rowMulMat(iChunk).print();
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::Dummy) {
			bV(bV::ConstrTypeDummyCoder::e, iChunk, n).print();
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::Rows) {
			printFatBitVector(hamCodeRows(iChunkFat, n));
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::RowsSparse) {
			printFatBitVector(genMatSprs.rowMulMat(iChunkFat));
		} else {
			// We should have covered all enumeration constants above, so this
			// shouldn't ever happen.
			__builtin_trap();
		}
	}

	if constexpr (useStopwatch) {
		auto hca{"VeryNaive "};
		if constexpr (hamCoderAlgo == HammingCoderAlgor::Cols) {
			hca = "Cols      ";
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::ColsSparse) {
			hca = "ColsSparse";
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::RowsDense) {
			hca = "RowsDense ";
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::Dummy) {
			hca = "Dummy     ";
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::Rows) {
			hca = "Rows      ";
		} else if constexpr (hamCoderAlgo == HammingCoderAlgor::RowsSparse) {
			hca = "RowsSparse";
		}
		std::ofstream("/tmp/hammingCoderStopwatch", std::ios_base::app) <<
		  hca << ' ' << bitStorageAlignment << ' ' << n << ": " << std::setprecision(15) <<
		  std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).
		  count() << '\n';
	}

	return 0;
}

#endif
