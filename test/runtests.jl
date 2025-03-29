using Test
using GridInterpolations
using NearestNeighbors
using Distances

# Load the main module
include("../src/LocalFunctionApproximation.jl")
using .LocalFunctionApproximation

@testset "LocalGIFunctionApproximator" begin
    grid = RectangleGrid(0:0.5:1.0, 0:0.5:1.0)
    gifa = LocalGIFunctionApproximator(grid)

    # Set dummy values
    set_all_interpolating_values(gifa, fill(1.0, n_interpolating_points(gifa)))

    # Evaluate at interior point
    val = compute_value(gifa, [0.25, 0.75])
    @test isapprox(val, 1.0; atol=1e-8)

    # Test multiple points
    vals = compute_value(gifa, [[0.25, 0.75], [0.5, 0.5]])
    @test all(isapprox(v, 1.0; atol=1e-8) for v in vals)

    # Test interpolating point count
    @test n_interpolating_points(gifa) == length(get_all_interpolating_points(gifa))
end

@testset "LocalNNFunctionApproximator - kNN" begin
    pts = [[x, y] for x in 0:1, y in 0:1]
    tree = KDTree(pts)
    nnfa = LocalNNFunctionApproximator(tree, pts, 2)

    set_all_interpolating_values(nnfa, fill(2.0, length(pts)))

    val = compute_value(nnfa, [0.1, 0.9])
    @test isapprox(val, 2.0; atol=1e-8)

    vals = compute_value(nnfa, [[0.1, 0.9], [0.5, 0.5]])
    @test all(isapprox(v, 2.0; atol=1e-8) for v in vals)

    @test n_interpolating_points(nnfa) == length(get_all_interpolating_points(nnfa))
end

@testset "LocalNNFunctionApproximator - radius" begin
    pts = [[x, y] for x in 0:1, y in 0:1]
    tree = KDTree(pts; metric=Euclidean())
    nnfa = LocalNNFunctionApproximator(tree, pts, 1.0)

    set_all_interpolating_values(nnfa, fill(3.0, length(pts)))

    val = compute_value(nnfa, [0.0, 0.0])
    @test isapprox(val, 3.0; atol=1e-8)
end
