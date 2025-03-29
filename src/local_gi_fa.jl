mutable struct LocalGIFunctionApproximator{G<:AbstractGrid} <: LocalFunctionApproximator
    grid::G
    gvalues::Vector{Float64}
end

"""
    LocalGIFunctionApproximator(grid::G)

Create a `LocalGIFunctionApproximator` with the given `grid`, 
initializing function values to zeros.
"""
function LocalGIFunctionApproximator(grid::G) where {G<:AbstractGrid}
    return LocalGIFunctionApproximator(grid, zeros(length(vertices(grid))))
end


################ INTERFACE FUNCTIONS ################

"""
    n_interpolating_points(gifa::LocalGIFunctionApproximator)

Return the number of interpolating points used by the grid-based approximator.
"""
function n_interpolating_points(gifa::LocalGIFunctionApproximator)
    return length(gifa.grid)
end

"""
    get_all_interpolating_points(gifa::LocalGIFunctionApproximator)

Return the full set of grid points used for interpolation.
"""
function get_all_interpolating_points(gifa::LocalGIFunctionApproximator)
    return vertices(gifa.grid)
end

"""
    get_all_interpolating_values(gifa::LocalGIFunctionApproximator)

Return the function values associated with each grid point.
"""
function get_all_interpolating_values(gifa::LocalGIFunctionApproximator)
    return gifa.gvalues
end

"""
    get_interpolating_nbrs_idxs_wts(gifa, v)

Return a tuple `(indices, weights)` of the grid points and their 
respective interpolation weights used to estimate the function at point `v`.
"""
function get_interpolating_nbrs_idxs_wts(gifa::LocalGIFunctionApproximator, v::AbstractVector{Float64})
    return interpolants(gifa.grid, v)
end

"""
    compute_value(gifa, v)

Compute the interpolated function value at a single query point `v`.
"""
function compute_value(gifa::LocalGIFunctionApproximator, v::AbstractVector{Float64})
    return interpolate(gifa.grid, gifa.gvalues, v)
end

"""
    compute_value(gifa, v_list)

Compute the interpolated function values for a list of query points.
"""
function compute_value(gifa::LocalGIFunctionApproximator, v_list::AbstractVector{V}) where V <: AbstractVector{Float64}
    @assert length(v_list) > 0 "Query list must be non-empty"
    return [compute_value(gifa, pt) for pt in v_list]
end

"""
    set_all_interpolating_values(gifa, gvalues)

Set all function values for the grid to the provided vector `gvalues`.
"""
function set_all_interpolating_values(gifa::LocalGIFunctionApproximator, gvalues::AbstractVector{Float64})
    gifa.gvalues = copy(gvalues)
end

"""
    finite_horizon_extension(gifa, hor)

Extend the grid along a new time dimension to support finite-horizon value functions.
"""
function finite_horizon_extension(gifa::LocalGIFunctionApproximator, hor::StepRange{Int64,Int64})
    cut_points = Tuple(gifa.grid)  # This extracts all axis cut points safely
    extended_grid = RectangleGrid(cut_points..., hor)
    return LocalGIFunctionApproximator(extended_grid)
end
