#! /bin/sh

# Run this as:
#
# chrt -f 99 ./misc/bench.sh DIR

set -u
set -e

dir=$1
mkdir "/tmp/$dir"

rt_per=/proc/sys/kernel/sched_rt_period_us
rt_run=/proc/sys/kernel/sched_rt_runtime_us

# Enable real-time scheduling.
printf '%s' -1 > "$rt_run"

IFS=-$IFS
for code in '7-4' '31-26' '63-57' '95-88' '127-120' '511-502' '1023-1013' '2047-2036'; do
	for alg in hammingCoder-gcc-*; do
		for i in `seq 32`; do
			./misc/util-random-ascii-bits | "./$alg" $code >> "/tmp/${dir}/hammingCoderStopwatch-${alg}-${code}"
			sleep 0.25
		done
	done
done

# Set the default real-time scheduling period and limit.
printf '%s' 1000000 > "$rt_per"
printf '%s' 950000 > "$rt_run"
