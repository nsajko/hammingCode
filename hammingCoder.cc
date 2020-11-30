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

constexpr intmax byteBits = 8;

#ifdef NAIVE_HAMMING
#   undef NAIVE_HAMMING
constexpr bool useNaiveHamming = true;
#else
constexpr bool useNaiveHamming = false;
#endif

#ifdef USE_STOPWATCH
#   undef USE_STOPWATCH
constexpr bool useStopwatch = true;
#else
constexpr bool useStopwatch = false;
#endif

// In bytes.
constexpr int bitStorageAlignment = 1UL << 4;

// Configurable initial capacity for the input message in bits.
constexpr intmax initialInputMessageCapacity = bitStorageAlignment * byteBits;

// Shorthand for static_cast.
template<typename X, typename Y>
[[nodiscard]] constexpr X
sc(Y v) { return static_cast<X>(v); }

// Returns the k Hamming code parameter corresponding to a given n.
[[nodiscard]] intmax
hammingK(intmax n) {
	auto N = sc<uintmax>(n);
	return sc<intmax>(N - std::bit_width(N));
}

// Returns the n Hamming code parameter corresponding to a given k.
[[nodiscard]] intmax
hammingN(intmax k) {
	auto K = sc<uintmax>(k);
	return sc<intmax>(K + std::bit_width(K + std::bit_width(K)));
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
	intmax r = 0;
	for (int i = 0;; i++) {
		intmax c = s[i];
		if (c < '0' || '9' < c) {
			break;
		}
		r = 10 * r + ASCIIToNum(c);
	}
	return r;
}

// A bit storage type is defined by the unsigned integer word type T, and the alignment n
// of the bit storage, in bytes.
template<typename T, int n>
concept BitStorage = (std::is_integral_v<T> && std::is_unsigned_v<T> &&
		      std::has_single_bit(sc<uint>(n)) && (sizeof(T) <= n));

// A bit vector type.
template<typename word, int aligSize>
requires BitStorage<word, aligSize>
class bitVector final {
	// Length in bits.
	intmax len = 0;

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

	static constexpr intmax wordBits = byteBits * sizeof(word);
	static constexpr intmax alignedBits = byteBits * sizeof(AlignedBits);
	static constexpr intmax alignedWords = aligSize / sizeof(word);

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
		constexpr int S = 1UL << 15;
		std::size_t sz = S;
		void *p = arr.data();
		if (std::align(aligSize, aligSize, p, sz) != arr.data() ||
		    p != arr.data() || sz != S) {
			std::cerr << "bitVector<" << sizeof(word) << ", " << aligSize <<
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

	// Shows the bit i in word w as '0' or '1' on stdout.
	void
	printBit(intmax i, intmax w) const {
		std::cout << sc<bool>(((*this)[w] >> i) & 1UL);
	}

	public:

	// The enumeration constants themselves are not important, just the types are.
	enum class ConstrTypeAlloc {e};
	enum class ConstrTypeZero {e};

	[[nodiscard]] intmax
	getLen() const {
		return len;
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
	bitVector([[maybe_unused]] ConstrType unused, intmax capBits) {
		auto s = ceilDivAligned(capBits);
		arr.reserve(sc<vst>(s));
		if constexpr (std::is_same_v<ConstrType, ConstrTypeZero>) {
			len = capBits;
			arr.resize(sc<vst>(s));
			trapIfMisaligned();
		}
	}

	// Copies a contiguous range of l bits from in starting at index i to a new bitVector.
	// The following relation must hold: out_len_bit <= in.len - in_off_bit.
	bitVector(bitVector<word, aligSize> &in, intmax in_off_bit, intmax out_len_bit):
	bitVector(bitVector<word, aligSize>::ConstrTypeZero::e, out_len_bit) {
		intmax in_len_bit = in.len, in_off_wrd = in_off_bit / wordBits, out_off_wrd = 0,
		     out_len_wrd = ceilDivWord(out_len_bit), in_len_wrd = ceilDivWord(in.len);
		if (!(out_len_bit <= in_len_bit - in_off_bit)) {
			std::cerr << "bitVector(bitVector<" << sizeof(word) << ", " << aligSize <<
			             "> &, intmax, intmax): flawed caller\n";
			std::cerr.flush();
			__builtin_trap();
		}
		in_off_bit %= wordBits;
		in_len_bit %= wordBits;
		out_len_bit %= wordBits;
		if (in_off_bit != 0) {
			for (; out_off_wrd < out_len_wrd - 1; out_off_wrd++, in_off_wrd++) {
				auto w0 = in[in_off_wrd], w1 = in[in_off_wrd + 1];
				auto i = wordBits - in_off_bit;
				(*this)[out_off_wrd] = sc<word>((w0 >> in_off_bit) | sc<word>(w1 << i));
			}

			(*this)[out_off_wrd] = sc<word>(in[in_off_wrd] >> in_off_bit);

			in_off_wrd++;
			if (in_off_wrd < in_len_wrd) {
				auto i = wordBits - in_off_bit;
				(*this)[out_off_wrd] |= sc<word>(in[in_off_wrd] << i);
			}
		} else {
			for (; out_off_wrd < out_len_wrd; out_off_wrd++, in_off_wrd++) {
				(*this)[out_off_wrd] = in[in_off_wrd];
			}

			// Clear highest bits from the last word after the end of the bitVector.
			out_off_wrd--;
		}
		auto i = (wordBits - out_len_bit) % wordBits;
		(*this)[out_off_wrd] = sc<word>(sc<word>((*this)[out_off_wrd] << i) >> i);
	}

	// Fills the bitVector with input from r.
	template<typename X>
	bitVector(X r):
	bitVector(bitVector<word, aligSize>::ConstrTypeAlloc::e, initialInputMessageCapacity) {
		for (;;) {
			int c = r();
			if (c == '	' || c == ' ' || c == '\n' || c == '\r') {
				continue;
			}
			if (c != '0' && c != '1') {
				break;
			}

			if (auto arrSize = arr.size(); sc<vst>(len) == arrSize * alignedBits) {
				arr.resize(arrSize + 1);
			}
			set(ASCIIToNum(c), len);
			len++;
		}
	}

	// A naive Hamming code coder, replaces bitGenMatrix while testing the rest of the code.
	//
	// Notice how similar this is to the bitGenMatrix constructor.
	bitVector(bitVector<word, aligSize> &in, intmax n):
	bitVector(bitVector<word, aligSize>::ConstrTypeZero::e, n) {
		// The fact that in.len can be smaller than hammingK(n) (which can happen with the
		// last block of input) complicates the implementation somewhat.
		// in.len can be smaller than hammingK(n), in which case we need to decrease n
		// accordingly.
		//
		// In our program this only happens with the last block of input to be coded.
		n = hammingN(in.len);

		for (intmax nonPowerOfTwoColumns = 0, pow = 1, j = 0; j < n; j++) {
			if (j + 1 == pow) {
				// j + 1 is a power of two.
				for (intmax i = pow + 1; i <= n; i++) {
					if ((i & pow) != 0) {
						exOrBit(in.isSet(hammingK(i) - 1), j);
					}
				}
				pow <<= 1;
			} else {
				// j + 1 is not a power of two.
				set(in.isSet(hammingK(j + 1) - 1), j);
				nonPowerOfTwoColumns++;
			}
		}
	}

	// Checks equality between bitVectors. Used just for testing.
	[[nodiscard]] bool
	equal(const bitVector<word, aligSize> &v) {
		for (intmax l = ceilDivWord(len), i = 0; i < l; i++) {
			if ((*this)[i] != v[i]) {
				return false;
			}
		}
		return true;
	}

	// XORs the current instance with op.
	void
	exOr(const bitVector &op) {
		auto out = std::assume_aligned<aligSize>(reinterpret_cast<uint8*>(arr.data()));
		auto in  = std::assume_aligned<aligSize>(reinterpret_cast<const uint8*>(op.arr.data()));
		for (intmax l = ceilDivByte(len), i = 0; i < l; i++) {
			out[i] ^= in[i];
		}
	}

	// Shows the bit vector on stdout.
	void
	print() const {
		// w is for "words", i is for bits.
		auto l = len / wordBits;
		for (intmax w = 0; w < l; w++) {
			for (intmax i = 0; i < wordBits; i++) {
				printBit(i, w);
			}
		}
		for (intmax i = 0; i < len % wordBits; i++) {
			printBit(i, l);
		}
		std::cout << '\n';
	}
};

template<typename T, int S>
requires BitStorage<T, S>
class bitGenMatrix final {
	std::vector<bitVector<T, S>> m;
	intmax rows;
	using vst = typename std::vector<bitVector<T, S>>::size_type;

	public:

	// Makes the generator matrix for the [n, k] Hamming code.
	bitGenMatrix(intmax n): rows(hammingK(n)) {
		m.reserve(sc<vst>(rows));
		for (intmax i = 0; i < rows; i++) {
			m.emplace_back(bitVector<T, S>(bitVector<T, S>::ConstrTypeZero::e, n));
		}
		for (intmax nonPowerOfTwoColumns = 0, pow = 1, j = 0; j < n; j++) {
			if (j + 1 == pow) {
				// j + 1 is a power of two.
				for (auto i = pow + 1; i <= n; i++) {
					// Check columns that have the pow bit set.
					if ((i & pow) != 0) {
						m[sc<vst>(hammingK(i) - 1)].set(1, j);
					}
				}
				pow <<= 1;
			} else {
				// j + 1 is not a power of two. Set the bit
				// m[row=nonPowerOfTwoColumns, column=j].
				m[sc<vst>(nonPowerOfTwoColumns)].set(1, j);
				nonPowerOfTwoColumns++;
			}
		}
	}

	// Multiplies the row-vector with the matrix.
	[[nodiscard]] std::unique_ptr<bitVector<T, S>>
	rowMulMat(const bitVector<T, S> &row) const {
		auto out = std::make_unique<bitVector<T, S>>(
		  bitVector<T, S>(bitVector<T, S>::ConstrTypeZero::e, m[0].getLen()));
		for (intmax len = row.getLen(), i = 0; i < len; i++) {
			if (row.isSet(i) != 0) {
				out->exOr(m[sc<vst>(i)]);
			}
		}
		return out;
	}

	// Prints the matrix.
	void
	print() const {
		for (intmax r = 0; r < rows; r++) {
			m[sc<vst>(r)].print();
		}
		std::cout << '\n';
	}
};

}  // namespace

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION

// This is the fuzzing code, used for finding defects in the main code.
// See https://llvm.org/docs/LibFuzzer.html

namespace {

// Replaces two bitVector constructors while testing the rest of the code.
template<typename T, int S, typename X>
requires BitStorage<T, S>
std::vector<bitVector<T, S>>
makeBitVectorVectorWithInput(X r, intmax blockSize) {
	std::vector<bitVector<T, S>> res;
	res.reserve(1UL << 4);
	res.emplace_back(bitVector<T, S>(bitVector<T, S>::ConstrTypeZero::e, blockSize));
	intmax len = 0;
	for (typename decltype(res)::size_type i = 0;;) {
		int c = r();
		if (c == '	' || c == ' ' || c == '\n' || c == '\r') {
			continue;
		}
		if (c != '0' && c != '1') {
			break;
		}

		if (len == blockSize) {
			len = 0;
			i++;
			res.emplace_back(bitVector<T, S>(
			  bitVector<T, S>::ConstrTypeZero::e, blockSize));
		}
		res[i].set(ASCIIToNum(c), len);
		len++;
	}
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

	intmax n = nByte;
	if (n == 0 || std::has_single_bit(sc<uint>(n))) {
		return 0;
	}
	intmax k = hammingK(n);

	using bWord = uint8;
	constexpr int align = 4;
	using bV = bitVector<bWord, align>;

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
	auto inMsgTest = makeBitVectorVectorWithInput<bWord, align>(fakeGet, k);
	bitGenMatrix<bWord, align> genMat(n);
	decltype(inMsgTest)::size_type I = 0;
	for (intmax blLen = k, iMsgLen = inMsg.getLen(), i = 0; i < iMsgLen; i += k, I++) {
		if (iMsgLen - i < blLen) {
			blLen = iMsgLen - i;
		}
		bV block(inMsg, i, blLen);
		if (!block.equal(inMsgTest[I])) {
			__builtin_trap();
		}
		if (!(*genMat.rowMulMat(block)).equal(
		    bV(block, n))) {
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
	intmax n = lexDecimalASCII(argv[1]);
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
	intmax k = lexDecimalASCII(argv[2]);
	if (auto correctK = hammingK(n); k != correctK) {
		std::cerr << "coder: the given combination of arguments does not describe\n"
		             "a Hamming code. Try either (" << n << ", " << correctK <<
		             ") or (" << hammingN(k) << ", " << k << ").\n";
		return 1;
	}

	std::chrono::time_point<std::chrono::steady_clock> startTime;
	if constexpr (useStopwatch) {
		startTime = std::chrono::steady_clock::now();
	}

	std::cerr << "Linear block code [n = " << n << ", k = " << k <<
	             "]\n(n = code word length) (k = number of source bits in each code word)\n"
	             "code rate = R(K) = " << (sc<double>(k) / sc<double>(n)) <<
	             "\n\nEnter a message in bits (possibly separated by whitespace)\n"
	             "to be Hamming coded using the chosen code parameters:\n\n";
	std::cerr.flush();

	using bWord = uintmax;
	using bV = bitVector<bWord, bitStorageAlignment>;

	bV inMsg([]()->int {return std::cin.get();});
	std::cerr << "\nInput source message:\n";
	std::cerr.flush();
	inMsg.print();
	std::cout.flush();

	// Make and show the code's generator matrix.
	bitGenMatrix<bWord, bitStorageAlignment> genMat(n);
	if constexpr (!useNaiveHamming) {
		std::cerr << "\nThe generator matrix for the code:\n\n";
		std::cerr.flush();
		genMat.print();
	}

	std::cout << '\n';
	std::cout.flush();

	std::cerr << "To encode the entire source input string into codewords, we divide the\n"
	             "input string into parts of k or less bits, where the last part's\n"
	             "last bits are padded with zeros. Each input part is\n"
	             "multiplied with the generator to produce\nthe corresponding codeword.\n\n";
	std::cerr.flush();

	for (intmax blLen = k, iMsgLen = inMsg.getLen(), i = 0; i < iMsgLen; i += k) {
		if (iMsgLen - i < blLen) {
			blLen = iMsgLen - i;
		}

		// Copy blLen bits from inMsg to block.
		bV block(inMsg, i, blLen);
		std::cerr << "Input " << std::setw(4) << blLen << " bits: ";
		std::cerr.flush();
		block.print();
		std::cout.flush();

		// Compute the output code word.
		std::cerr << "Output: ";
		std::cerr.flush();
		if constexpr (useNaiveHamming) {
			bV(block, n).print();
		} else {
			genMat.rowMulMat(block)->print();
		}
		std::cout.flush();
	}

	if constexpr (useStopwatch) {
		std::ofstream("/tmp/hammingCoderStopwatch", std::ios_base::app) <<
		  std::boolalpha <<
		  useNaiveHamming << ' ' << bitStorageAlignment << ' ' << n << ": " <<
		  std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).
		  count() << '\n';
	}

	return 0;
}

#endif
