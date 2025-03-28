using Test
using StaticArrays
using NearestNeighbors
using GridInterpolations
include("../src/LocalFunctionApproximation.jl")

using .LocalFunctionApproximation

@testset "LocalNNFunctionApproximator" begin
    points = [SVector(0.,0.), SVector(0.,1.), SVector(1.,1.), SVector(1.,0.)]
    vals = [1., 1., -1., -1]
    nntree = KDTree(points)
    k = 2
    r = 0.5 * sqrt(2)

    knnfa = LocalNNFunctionApproximator(nntree, points, k)
    set_all_interpolating_values(knnfa, vals)

    @test n_interpolating_points(knnfa) == 4
    @test get_all_interpolating_points(knnfa) == points
    @test get_all_interpolating_values(knnfa) == vals

    for i = 1:10
        pt = [rand()/2, 0.5]
        val = compute_value(knnfa, pt)
        @test val ≈ 1.0 atol=1e-5
    end

    pts = [[0.25, 0.5], [0.1, 0.9]]
    results = compute_value(knnfa, pts)
    @test length(results) == 2

    rnnfa = LocalNNFunctionApproximator(nntree, points, r)
    set_all_interpolating_values(rnnfa, vals)

    for i = 1:10
        pt = [1.0 - rand()/2, 0.5]
        val = compute_value(rnnfa, pt)
        @test val ≈ -1.0 atol=1e-5
    end

    idxs, wts = get_interpolating_nbrs_idxs_wts(knnfa, [0.0, 0.0])
    @test all(0.0 .<= wts .<= 1.0)
    @test isapprox(sum(wts), 1.0; atol=1e-10)
end


@testset "LocalGIFunctionApproximator" begin
    grid = RectangleGrid(LinRange(0, 1, 5))
    gifa = LocalGIFunctionApproximator(grid)
    values = sin.(vertices(grid))  # ✅ FIXED THIS LINE

    @test n_interpolating_points(gifa) == length(grid)
    @test get_all_interpolating_points(gifa) == collect(vertices(grid))

    set_all_interpolating_values(gifa, values)
    @test get_all_interpolating_values(gifa) ≈ values

    test_pt = [0.5]
    val = compute_value(gifa, test_pt)
    @test isapprox(val, sin(0.5); atol=1e-3)

    test_pts = [[0.1], [0.2], [0.3]]
    results = compute_value(gifa, test_pts)
    expected = sin.([0.1, 0.2, 0.3])
    @test results ≈ expected atol=1e-3

    idxs, wts = get_interpolating_nbrs_idxs_wts(gifa, [0.5])
    @test typeof(idxs) <: AbstractVector{Int}
    @test typeof(wts) <: AbstractVector{Float64}

    extended = finite_horizon_extension(gifa, 1:3)
    @test isa(extended, LocalGIFunctionApproximator)
    @test size(extended.grid.cutPoints) == (2,)
end
