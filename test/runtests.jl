using Test
using GridInterpolations
using NearestNeighbors
using Distances

# Load your module from source
include("../src/LocalFunctionApproximation.jl")
using .LocalFunctionApproximation

@testset "LocalGIFunctionApproximator" begin
    grid = RectangleGrid(0:0.5:1.0, 0:0.5:1.0)
    gifa = LocalGIFunctionApproximator(grid)

    set_all_interpolating_values(gifa, fill(1.0, n_interpolating_points(gifa)))

    val = compute_value(gifa, [0.25, 0.75])
    @test isapprox(val, 1.0; atol=1e-8)

    vals = compute_value(gifa, [[0.25, 0.75], [0.5, 0.5]])
    @test all(isapprox(v, 1.0; atol=1e-8) for v in vals)

    @test n_interpolating_points(gifa) == length(get_all_interpolating_points(gifa))
end

@testset "LocalNNFunctionApproximator - kNN" begin
    raw_pts = vec([[Float64(x), Float64(y)] for x in 0:1, y in 0:1])
    pts_matrix = hcat(raw_pts...)
    tree = KDTree(pts_matrix)

    nnfa = LocalNNFunctionApproximator(tree, raw_pts, 2)
    set_all_interpolating_values(nnfa, fill(2.0, length(raw_pts)))

    val = compute_value(nnfa, [0.1, 0.9])
    @test isapprox(val, 2.0; atol=1e-8)

    vals = compute_value(nnfa, [[0.1, 0.9], [0.5, 0.5]])
    @test all(isapprox(v, 2.0; atol=1e-8) for v in vals)

    @test n_interpolating_points(nnfa) == length(get_all_interpolating_points(nnfa))
end

@testset "LocalNNFunctionApproximator - radius" begin
    raw_pts = vec([[Float64(x), Float64(y)] for x in 0:1, y in 0:1])
    pts_matrix = hcat(raw_pts...)
    tree = KDTree(pts_matrix, Euclidean())

    nnfa = LocalNNFunctionApproximator(tree, raw_pts, 1.0)
    set_all_interpolating_values(nnfa, fill(3.0, length(raw_pts)))

    val = compute_value(nnfa, [0.0, 0.0])
    @test isapprox(val, 3.0; atol=1e-8)
end

@testset "LocalGIFunctionApproximator - extra coverage" begin
    grid = RectangleGrid(0:0.5:1.0, 0:0.5:1.0)
    gifa = LocalGIFunctionApproximator(grid)

    # get_all_interpolating_values
    vals = get_all_interpolating_values(gifa)
    @test length(vals) == n_interpolating_points(gifa)

    # get_interpolating_nbrs_idxs_wts
    idxs, wts = get_interpolating_nbrs_idxs_wts(gifa, [0.25, 0.75])
    @test length(idxs) == length(wts)

    # finite_horizon_extension
    extended_gifa = finite_horizon_extension(gifa, 1:3)
    @test n_interpolating_points(extended_gifa) > n_interpolating_points(gifa)
end
