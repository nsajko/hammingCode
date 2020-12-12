#! /bin/sh
set -u

cxx_opt='-std=c++20 -Werror -Wall -Wextra -Wno-unused-function -Wcast-align -Wconversion -Wold-style-cast -g -march=native -O3 -fno-exceptions'

opt="-D FORCE_COMPUTATION_AND_DISALLOW_REORDERING -D USE_STOPWATCH -D PRINT_LESS -D FORCE_COMPUTATION_AND_DISALLOW_REORDERING -o hammingCoder-$1 $cxx_opt hammingCoder.cc"

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
clang-RowsDense)
	clang++ -D HAM_COD_ALG=HammingCoderAlgor::RowsDense $clang_opt
	;;
clang-Cols)
	clang++ -D HAM_COD_ALG=HammingCoderAlgor::Cols $clang_opt
	;;
clang-ColsSparse)
	clang++ -D HAM_COD_ALG=HammingCoderAlgor::ColsSparse $clang_opt
	;;
clang-VeryNaive)
	clang++ -D HAM_COD_ALG=HammingCoderAlgor::VeryNaive $clang_opt
	;;
clang-Dummy)
	clang++ -D HAM_COD_ALG=HammingCoderAlgor::Dummy $clang_opt
	;;
clang-Rows)
	clang++ -D HAM_COD_ALG=HammingCoderAlgor::Rows $clang_opt
	;;
clang-RowsSparse)
	clang++ -D HAM_COD_ALG=HammingCoderAlgor::RowsSparse $clang_opt
	;;
gcc)
	g++ $gcc_opt
	;;
gcc-RowsDense)
	g++ -D HAM_COD_ALG=HammingCoderAlgor::RowsDense $gcc_opt
	;;
gcc-Cols)
	g++ -D HAM_COD_ALG=HammingCoderAlgor::Cols $gcc_opt
	;;
gcc-ColsSparse)
	g++ -D HAM_COD_ALG=HammingCoderAlgor::ColsSparse $gcc_opt
	;;
gcc-VeryNaive)
	g++ -D HAM_COD_ALG=HammingCoderAlgor::VeryNaive $gcc_opt
	;;
gcc-Dummy)
	g++ -D HAM_COD_ALG=HammingCoderAlgor::Dummy $gcc_opt
	;;
gcc-Rows)
	g++ -D HAM_COD_ALG=HammingCoderAlgor::Rows $gcc_opt
	;;
gcc-RowsSparse)
	g++ -D HAM_COD_ALG=HammingCoderAlgor::RowsSparse $gcc_opt
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
util-random-ascii-bits)
	g++ -o util-random-ascii-bits $cxx_opt -flto util/random-ascii-bits.cc
	;;
*)
	printf "build.sh: unrecognized argument $1 \n" 1>&2
esac
