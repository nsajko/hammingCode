// Copyright 2019 Neven Sajko. All rights reserved.

// A Hamming code coder.
//
// A generator matrix approach is used as an optimization for large
// messages.
//
// Bit vectors are used to compactly represent strings of bits.
//
// For simplicity, I ignored the possibility of heap allocation failing.

// TODO: Some functions may be counterproductively specified as 'inline'.
// TODO: Mark pointer parameters as const where possible.

////// XXX convert all operands of '>>' to unsigned if necessary. Otherwise, maybe use division instead of shifting.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long bitVectorSmall;
typedef struct {
	// Length and backing storage capacity, both in bits.
	long len, cap;

	// Backing storage. The element type is chosen for efficiency,
	// it enables easily doing arithmetic on bits with great
	// parallelism.
	bitVectorSmall *arr;
} bitVector;

enum {
	// Just for clarity, probably not configurable.
	bitsInAByte = 8,

	bitsInABitVectorSmall = bitsInAByte * sizeof(bitVectorSmall),
};

// Is l a power of two? Or, equivalently, does l have a set bit count
// (population count/popcount) of one?
static inline
int
isPowerOfTwo(long l) {
	return (l & (l - 1)) == 0;
}

// Returns ceiling(n / bitsInABitVectorSmall).
static inline
long
ceilDiv(long n) {
	return (n - 1) / bitsInABitVectorSmall + 1;
}

// Makes the generator matrix for the [n, k] Hamming code.
static inline
bitVector *
makeGen(long n, long k) {
	bitVector *r = malloc(sizeof(*r) * k);
	long i, j;
	for (i = 0; i < k; i++) {
		r[i].len = n;
		r[i].arr = calloc(sizeof(r[i].arr[0]), ceilDiv(n));
	}
	long nonPowerOfTwoColumns = 0, pow = 1;
	for (j = 0; j < n; j++) {
		if (j + 1 == pow) {
			// j + 1 is a power of two.
			for (i = pow + 1; i < n; i++) {
				// We check columns that have the pow
				// bit set.
				if (i & pow) {
					// m will hold the index of the
					// row in which we will set the bit.
					long m = 0, l;
					for (l = 1; l < i; l++) {
						if (!isPowerOfTwo(l)) {
							m++;
						}
					}

					// Set the bit
					// r[row=m, column=j].
					printf("n=%ld, k=%ld, m=%ld, j=%ld\n", n, k, m, j); //// XXX DEBUGGING
					r[m].arr[j / bitsInABitVectorSmall] |= 1 << (j % bitsInABitVectorSmall);
				}
			}
			pow <<= 1;
		} else {
			// j + 1 is not a power of two. Set the bit
			// r[row=nonPowerOfTwoColumns, column=j].
			printf("nPOTC=%ld, j=%ld %ld\n", nonPowerOfTwoColumns, j, j / bitsInABitVectorSmall); //// XXX DEBUGGING
			r[nonPowerOfTwoColumns].arr[j / bitsInABitVectorSmall] |= 1 << (j % bitsInABitVectorSmall);
			nonPowerOfTwoColumns++;
		}
	}
	return r;
}

// Performs bitwise exclusive-or between the op and out bit vectors.
static inline
void
xor(bitVector *out, bitVector *op) {
	long i, l = ceilDiv(out->len);
	for (i = 0; i < l; i++) {
		out->arr[i] ^= op->arr[i];
	}
}

// Counts the number of contiguous bits b in the bit vector, starting at
// index i.
static
long
countContiguous(bitVector *bV, long i, unsigned b) {
	long j;
	bitVectorSmall s = 0;
	if (b != 0) {
		s = (bitVectorSmall)~0UL;
	}
	for (j = i; j < bV->len && bV->arr[j / bitsInABitVectorSmall] == s; j += bitsInABitVectorSmall) {}
	for (; j < bV->len &&
			(b == (1 & (bV->arr[j / bitsInABitVectorSmall] >> (j % bitsInABitVectorSmall))))
			; j++) {}
	return j - i;
}

// Multiplies the row-vector with the matrix. Out is expected to be zeroed.
static inline
void
rowMulMat(bitVector *out, bitVector *row, bitVector *mat) {
	// We operate by finding ranges of set bits in the row, prefixed
	// by ranges of unset bits, and then adding up with XOR the
	// corresponding rows from mat.
	long i;
	for (i = 0; i < row->len;) {
		// Skip range of unset bits.
		i += countContiguous(row, i, 0);

		// Find the range of set bits.
		long j = i + countContiguous(row, i, 1);

		// Add up the corresponding range of rows from mat into
		// out.
		long k;
		for (k = i; k < j; k++) {
			xor(out, &(mat[k]));
		}

		i += j;
	}
}

// Frees all k rows belonging to the matrix, then the matrix itself.
static inline
void
freeMat(bitVector *mat, long k) {
	long i;
	for (i = 0; i < k; i++) {
		free(mat[i].arr);
	}
	free(mat);
}

// Moves a contiguous range of bits from in starting at index i to out.
static inline
void
bitVectorMoveInto(bitVector *out, bitVector *in, long i) {
	long w = i / bitsInABitVectorSmall, y = 0, l = ceilDiv(out->len);
	i %= bitsInABitVectorSmall;
	for (; y < l && w + 1 < ceilDiv(in->len); w++, y++) {
		out->arr[y] = in->arr[w] >> i;
		out->arr[y] |= in->arr[w + 1] << (bitsInABitVectorSmall - i);
	}
	out->arr[y] = in->arr[w] >> i;  // TODO: Not sure in correctness here.
}

// Shows the boolean argument as bits '0' or '1' on stdout.
static inline
void
printBool(int b) {
	if (b) {
		putchar('1');
	} else {
		putchar('0');
	}
}

// Shows the bit vector on stdout.
static inline
void
printBitVector(bitVector *bV) {
	// w is for "words", i is for bits.
	long w, i, l = bV->len / bitsInABitVectorSmall;
	for (w = 0; w < l; w++) {
		for (i = 0; i < bitsInABitVectorSmall; i++) {
			printBool((1 << i) & bV->arr[w]);
		}
	}
	for (i = 0; i < bV->len % bitsInABitVectorSmall; i++) {
		printBool((1 << i) & bV->arr[w]);
	}
	printf("\n");
}

// Shows the array of bit vectors/rows on stdout (as a matrix).
static inline
void
printMatrix(bitVector *m, long rows) {
	long r;
	for (r = 0; r < rows; r++) {
		printBitVector(&(m[r]));
	}
	printf("\n");
}

// Converts an ASCII char to the number it represents.
static inline
long
ASCIIToNum(long c) {
	return c - '0';
}

// Lexes an ASCII string into a number. Does not look at anything after
// the first char outside the ASCII numeral range.
static inline
long
lexDecimalASCII(const char *s) {
	int i = 0;
	long r = 0;
	for (;; i++) {
		long c = s[i];
		if (c < '0' || '9' < c) {
			break;
		}
		r = 10*r + ASCIIToNum(c);
	}
	return r;
}

// Floor of the binary logarithm.
static inline
long
floorLog2(long n) {
	long r = 0;
	for (; n >>= 1; r++) {}
	return r;
}

int
main(int argc, char *argv[]) {
	// Handle program arguments (argv).
	if (argc != 1 + 2) {
		fprintf(stderr, "coder: wrong number of arguments\n");
		return 1;
	}
	long n = lexDecimalASCII(argv[1]);
	if (n == 0) {
		fprintf(stderr, "coder: wrong input for first argument (n)\n");
		return 1;
	}
	long k = lexDecimalASCII(argv[2]);
	if (k == 0) {
		fprintf(stderr, "coder: wrong input for second argument (k)\n");
		return 1;
	}
	printf("Linear block code [n = %ld, k = %ld]\ncode rate = R(K) = %g\n", n, k, (double)k / (double)n);
	if (n <= k) {
		fprintf(stderr, "coder: n <= k\n");
		return 1;
	}
	if (n - 1 == k || n - floorLog2(n + 1) < k) {
		fprintf(stderr, "coder: warning: these n,k parameters do not formally define a Hamming code, because the usual theorems do not apply\n");
	}

	// Stdin input of source input message.
	printf("\nEnter a message in bits (separated by whitespace) to be Hamming coded using the chosen code parameters:\n\n");
	enum {
		// In bits.
		initialInputMessageCapacity = 1 << 13,
	};
	bitVector inMsg = {0, initialInputMessageCapacity};
	inMsg.arr = calloc(sizeof(inMsg.arr[0]), inMsg.cap / bitsInABitVectorSmall);
	bitVectorSmall tmpBits = 0;
	for (;;) {
		int c = fgetc(stdin);
		if (c == '	' || c == ' ' || c == '\n') {
			continue;
		}
		if (c != '0' && c != '1') {
			break;
		}

		c -= '0';

		// c is now either zero or one. Set or clear the
		// corresponding bit accordingly.
		tmpBits |= c << (inMsg.len % bitsInABitVectorSmall);

		inMsg.len++;

		if (inMsg.cap - 1 < inMsg.len) {
			inMsg.cap <<= 1;
			inMsg.arr = realloc(inMsg.arr, inMsg.cap / bitsInABitVectorSmall * sizeof(inMsg.arr[0]));
		}

		if (inMsg.len % bitsInABitVectorSmall == 0) {
			// Copy temporary bit storage to the backing array.
			inMsg.arr[inMsg.len / bitsInABitVectorSmall] = tmpBits;
		}
	}
	printBitVector(&inMsg);

	// Make and show the code's generator matrix.
	bitVector *genMat = makeGen(n, k);
	printf("\nThe generator matrix for the code:\n\n");
	printMatrix(genMat, k);

	// Encode all of the source input string into codewords. The
	// possible arbitrary bits of the possible undersized last
	// block are padded with zero bits.
	//
	// The capacity struct field is unused here, so it is OK to set
	// it to zero, although that is not the real capacity.
	bitVector codeWord = {n, 0};
	codeWord.arr = calloc(sizeof(codeWord.arr[0]), ceilDiv(n));
	long i;
	bitVector block = {k, 0};
	block.arr = malloc(sizeof(block.arr[0]) * ceilDiv(k));
	for (i = 0; i < inMsg.len; i += k) {
		// Copy k bits from inMsg to block.
		memset(&(block.arr), 0, sizeof(block.arr[0]) * ceilDiv(k));
		bitVectorMoveInto(&block, &inMsg, i);

		// Compute the output code word.
		rowMulMat(&codeWord, &block, genMat);
		printBitVector(&codeWord);
	}

	// Deallocate memory.
	free(block.arr);
	free(codeWord.arr);
	freeMat(genMat, k);
	free(inMsg.arr);

	// C main function must return an int, and it should be zero in
	// case no error occured.
	return 0;
}
