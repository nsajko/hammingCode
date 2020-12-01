#include <bit>
#include <cstdint>
#include <iostream>

namespace {

using uint8 = unsigned char;
using uint = unsigned int;
using uint64 = std::uint64_t;
using uintmax = std::uintmax_t;
using intmax = std::intmax_t;

constexpr intmax byteBits = 8;

// Size of the smallest amount of output we produce, in bytes.
constexpr intmax chunkSize = 1UL << 21;

// Shorthand for static_cast.
template<typename X, typename Y>
[[nodiscard]] constexpr X
sc(Y v) { return static_cast<X>(v); }

// Xoshiro256 PRNGs by Blackman and Vigna
enum { StarStar, PlusPlus };
template<int variant>
requires (variant == StarStar || variant == PlusPlus)
class Xoshiro256 final {
	// The PRNG state, initialized with random bits.
	uint64 s[4] = {0xdcbfcdafbe972023UL, 0x6f496a9923b1364aUL, 0xb3dcd40b7cd14da1UL, 0xc658ab0e170a5d57UL};

	public:

	[[nodiscard]] uint64
	next(void) {
		uint64 result;
		if constexpr (variant == StarStar) {
			result = std::rotl(s[1] * 5, 7) * 9;
		} else {
			result = std::rotl(s[0] + s[3], 23) + s[0];
		}
	
		uint64 t = s[1] << 17;
	
		s[2] ^= s[0];
		s[3] ^= s[1];
		s[1] ^= s[2];
		s[0] ^= s[3];
	
		s[2] ^= t;
	
		s[3] = std::rotl(s[3], 45);
	
		return result;
	}
};

[[nodiscard]] uint64
numToASCII(uint64 a) {
	return a | 0x30UL;
}

}  // namespace

int
main() {
	std::ios::sync_with_stdio(false);

	constexpr intmax m = 8;

	Xoshiro256<PlusPlus> r;

	for (intmax i = 0; i < m; i++) {
		alignas(256) static char bitStorage[chunkSize / sizeof(char) * byteBits];

		for (intmax j = 0; j < sc<intmax>(chunkSize / sizeof(uint64)); j++) {
			auto v = r.next();
			for (int i = 0; i < sc<int>(sizeof(uint64) * byteBits); i++) {
				bitStorage[j * sizeof(uint64) * byteBits + i] = sc<char>(numToASCII((v >> i) & 1UL));
			}
		}

		std::cout.write(bitStorage, sizeof(bitStorage));
	}

	return 0;
}
