#! /bin/sh
set -u

opt="-o hammingCoder-$1 -std=c++20 -Werror -Wall -Wextra -Wno-unused-function -Wcast-align -Wconversion -Wold-style-cast -g -march=native -O3 -fno-exceptions hammingCoder.cc"

polly_opt=
# polly_opt="-mllvm -polly"
# polly_opt="-mllvm -polly -mllvm -polly-vectorizer=stripmine"

clang_opt="$polly_opt -stdlib=libstdc++ -ferror-limit=4 -Wcast-qual -Wundefined-reinterpret-cast -Wvector-conversion $opt"

gcc_opt="-fmax-errors=4 -flto $opt"

case "$1" in
gcc-musl)
	x86_64-linux-musl-g++ -static $gcc_opt
	;;
clang-callgraph)
	clang++ -std=c++20 -g -march=native -O3 -fno-exceptions -stdlib=libstdc++ -emit-llvm -S -o - hammingCoder.cc | opt -dot-callgraph
	c++filt < '<stdin>.callgraph.dot' | sed 's,>,\\>,g; s,-\\>,->,g; s,<,\\<,g' | dot -Tsvg > callgraph.svg
	rm '<stdin>.callgraph.dot'
	;;
clang)
	clang++ $clang_opt
	;;
gcc)
	g++ $gcc_opt
	;;
fuzz)
	clang++ -fsanitize=fuzzer,address,undefined -fsanitize-trap=undefined -fno-omit-frame-pointer -D FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION $clang_opt
	;;
nofuzz-clang)
	clang++ -fsanitize=address,undefined -fsanitize-trap=undefined -fno-omit-frame-pointer -D FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION $clang_opt nofuzz.cc
	;;
nofuzz-gcc)
	g++ -fsanitize=address,undefined -fsanitize-undefined-trap-on-error -fno-omit-frame-pointer -D FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION $gcc_opt nofuzz.cc
	;;
nofuzz-gdb-clang)
	clang++ -fno-omit-frame-pointer -D FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION $clang_opt nofuzz.cc
	;;
nofuzz-gdb-gcc)
	g++ -fno-omit-frame-pointer -D FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION $gcc_opt nofuzz.cc
	;;
esac
