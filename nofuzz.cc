#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

using uint8 = unsigned char;
using uint = unsigned int;
using uintmax = std::uintmax_t;
using intmax = std::intmax_t;

// Shorthand for static_cast.
template<typename X, typename Y>
[[nodiscard]] constexpr X
sc(Y v) { return static_cast<X>(v); }

}  // namespace

extern "C" int
LLVMFuzzerTestOneInput(const unsigned char *data, size_t size);

int
main(int argc, char *argv[]) {
	std::ios::sync_with_stdio(false);

	for (int i = 1; i < argc; i++) {
		std::filesystem::path f(argv[i]);

		if (!std::filesystem::exists(f)) {
			std::cerr << "nofuzz: " << f << " does not exist\n\n";
			continue;
		}
		if (!std::filesystem::is_regular_file(f)) {
			std::cerr << "nofuzz: " << f << " is not a regular file\n\n";
			continue;
		}
		auto sz = std::filesystem::file_size(f);
		if (sc<intmax>(sz) < 0) {
			std::cerr << "nofuzz: unexpected error with " << f << "\n\n";
			continue;
		}
		std::cerr << f << ":\n";

		std::vector<unsigned char> buffer;
		buffer.resize(sc<decltype(buffer)::size_type>(sz));
		auto a = buffer.data();
		std::ifstream(f, std::ios_base::binary).read(reinterpret_cast<char*>(a), sc<intmax>(sz));
		LLVMFuzzerTestOneInput(a, sz);
		std::cerr << '\n';
	}

	return 0;
}
