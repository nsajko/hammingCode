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
# * A histogram that represents the distribution of the floating point values in
#   a single input file. The histograms come in several different bin sizes. Each
#   histogram is overlayed with a probability density function (PDF) estimation.
#
# * For each input file, a function that maps line numbers to floating point values.
#
# * Multiple probability density function (PDF) estimations, each sourced from
#   a single input file.

module PlotDat

using DataFrames, Gadfly

export main

const F = BigFloat
const V = Vector{F}
const VV = Vector{V}
const VI = Vector{Int}

const fpprec = 8192

function drawSeq(outName::String, y::V)::Nothing
	# Make indices.
	local x::VI = [0 for _ in 1:size(y, 1)]
	for i in 1:size(y, 1)
		x[i] = i
	end

	draw(SVG(string("plots-seq/", outName, ".svg"), 20cm, 20cm), plot(layer(DataFrame(
	  seq_num=x, value=y), x=:seq_num, y=:value, Geom.point)))

	return nothing
end

function drawHist(outName::String, x::V, bc::Int)::Nothing
	local df::DataFrame = DataFrame(value=x)
	draw(SVG(string("plots-hist/", outName, "-bincount:", bc, ".svg"), 20cm, 20cm), plot(
	  layer(df, x=:value, Geom.density, color=[:pdf]),
	  layer(df, x=:value, Geom.histogram(bincount=bc), color=[:histogram])))

	return nothing
end

function main()::Nothing
	local n::Int = size(ARGS, 1)

	n < 3 && return nothing

	local combinedName::String = ARGS[1]
	local args::Vector{String} = ARGS[2:n]
	n = n - 1

	setprecision(BigFloat, fpprec)

	# Read input files.
	local x::VV = [F[] for _ in 1:n]
	for i in 1:n
		open(args[i], lock=false, read=true) do f
			while !eof(f)
				local l::String = readline(f)
				push!(x[i], BigFloat(l[findfirst(':', l) + 1:length(l)], precision=fpprec))
			end
		end
	end

	# Draw the sequence plots.
	for i in 1:n
		drawSeq(basename(args[i]), x[i])
	end

	# Draw the histograms and probability density estimations.
	for i in 1:n
		for bc in 30:10:60
			drawHist(basename(args[i]), x[i], bc)
		end
	end

	# Draw the combined sequence plot.
	local numSamples::Int = size(x[1], 1)     # Assuming all files have the same amount of samples!
	local indices::VI = [j for j in 1:numSamples]
	draw(SVG(string("plots-seq/", combinedName, ".svg"), 20cm, 20cm), plot(
	  [layer(DataFrame(seq_num=indices, value=x[i]), x=:seq_num, y=:value, Geom.point, color=[basename(args[i])]) for i in 1:n]...))

	# Draw the combined histograms.
	draw(SVG(string("plots-hist/", combinedName, ".svg"), 20cm, 20cm), plot(
	  [layer(DataFrame(value=x[i]), x=:value, Geom.density, color=[basename(args[i])]) for i in 1:n]...))

	return nothing
end

end

PlotDat.main()
