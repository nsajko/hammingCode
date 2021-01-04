# Copyright 2020 Neven Sajko. All rights reserved. See LICENSE for a license.

# Name for the total results must be given in argv, followed by at least one input file.
#
# Each file given by file name in argv must contain one floating point number per line.
# We render a number of outputs with Gadfly for each input file and a number of
# additional outputs that are sourced from all the inputs.
#
# Types of output files:
#
# * A plot of the function that maps line numbers to floating point values, based on
#   a single input file.
#
# * A probability density function (PDF) estimation for a single input file. This is like
#   a histogram in that it represents the distribution of the floating point values, but
#   much better.
#
# * For each input file, a function that maps line numbers to floating point values.
#
# * Multiple density estimations, each sourced from a single input file.

module PlotDat

using DataFrames, Gadfly

export main

const Q = Rational{Int}
const F = BigFloat
const V = Vector{F}
const VV = Vector{V}
const VI = Vector{Int}

# Floating point number representation precision.
const fpprec = 8192

Fl(x::T) where {T} = F(x, precision=fpprec)

const numTicks = 30

# Can be adjusted. E.g. I used 20 for Intel Skylake.
const ticksDenominator = 14

const ticksVal = F[Fl(n // ticksDenominator) for n in 0:numTicks]

# Maximal (expected) measured value.
const maxVal = Fl(numTicks // ticksDenominator)

function drawSeq(outName::String, y::V)::Nothing
	# Make indices.
	local numSamples::Int = size(y, 1)
	local x::VI = [i for i in 1:numSamples]

	draw(SVG(string("plots-seq/", outName, ".svg"), 40cm, 20cm), plot(
	  Coord.cartesian(xmin=0, ymin=0, xmax=numSamples, ymax=maxVal),
	  Guide.xticks(ticks=[i for i in 0:(numSamples ÷ 4):numSamples]),
	  Guide.yticks(ticks=ticksVal),
	  Guide.xlabel("index in the sequence"),
	  Guide.ylabel("value (seconds)"),
	  layer(DataFrame(
	  seq_num=x, value=y), x=:seq_num, y=:value, size=[1pt], Geom.point, Theme(highlight_width=0pt))))

	return nothing
end

function drawHist(outName::String, x::V)::Nothing
	local df::DataFrame = DataFrame(value=x)
	draw(SVG(string("plots-hist/", outName, ".svg"), 40cm, 20cm), plot(
	  Coord.cartesian(xmin=0, xmax=maxVal),
	  Guide.xticks(ticks=ticksVal),
	  Guide.xlabel("value (seconds)"),
	  layer(df, x=:value, Geom.density)))

	return nothing
end

function main()::Nothing
	local n::Int = size(ARGS, 1)

	n < 3 && return nothing

	local combinedName::String = ARGS[1]
	local args::Vector{String} = ARGS[2:n]
	n = n - 1

	setprecision(F, fpprec)

	# Read input files.
	local x::VV = [F[] for _ in 1:n]
	for i in 1:n
		open(args[i], lock=false, read=true) do f
			while !eof(f)
				local l::String = readline(f)
				push!(x[i], F(l[findfirst(':', l) + 1:length(l)], precision=fpprec))
			end
		end
	end

	# Draw the sequence plots.
	for i in 1:n
		drawSeq(basename(args[i]), x[i])
	end

	# Draw the histograms and probability density estimations.
	for i in 1:n
		drawHist(basename(args[i]), x[i])
	end

	# Draw the combined sequence plot.
	local numSamples::Int = size(x[1], 1)     # Assuming all files have the same amount of samples!
	local indices::VI = [j for j in 1:numSamples]
	draw(SVG(string("plots-seq/", combinedName, ".svg"), 40cm, 20cm), plot(Theme(key_position=:bottom),
	  Coord.cartesian(xmin=0, ymin=0, xmax=numSamples, ymax=maxVal),
	  Guide.xticks(ticks=[i for i in 0:(numSamples ÷ 4):numSamples]),
	  Guide.yticks(ticks=ticksVal),
	  Guide.xlabel("index in the sequence"),
	  Guide.ylabel("value (seconds)"),
	  [layer(DataFrame(seq_num=indices, value=x[i]), x=:seq_num, y=:value,
	  size=[1pt], Geom.point, Theme(highlight_width=0pt), color=[basename(args[i])]) for i in 1:n]...))

	# Draw the combined histograms.
	draw(SVG(string("plots-hist/", combinedName, ".svg"), 40cm, 20cm), plot(Theme(key_position=:bottom),
	  Coord.cartesian(xmin=0, xmax=maxVal),
	  Guide.xticks(ticks=ticksVal),
	  Guide.xlabel("value (seconds)"),
	  [layer(DataFrame(value=x[i]), x=:value, Geom.density, color=[basename(args[i])]) for i in 1:n]...))

	return nothing
end

end

PlotDat.main()
