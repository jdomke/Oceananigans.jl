push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Oceananigans.Models: ShallowWaterModel
include("src/Benchmarks.jl")
using .Benchmarks

# Benchmark function

function benchmark_shallow_water_model(Arch, FT, N)
    grid = RectilinearGrid(Arch(), FT, size=(N, N), extent=(1, 1), topology=(Periodic, Periodic, Flat), halo=(3, 3))
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1.0)
    set!(model, h=1)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters
#
#
Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Float_types = [Float64]
Ns = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_shallow_water_model; Architectures, Float_types, Ns)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Float_types, :Ns])
benchmarks_pretty_table(df, title="Shallow water model benchmarks")
