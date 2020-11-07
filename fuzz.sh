#! /bin/sh
set -u
: "$1"
UBSAN_OPTIONS=print_stacktrace=1 exec ./hammingCoder-fuzz -max_len=100000 -len_control=10000 -timeout=3600 -rss_limit_mb=3000 "$@"
