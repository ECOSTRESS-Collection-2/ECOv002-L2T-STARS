using Base.Threads
using Dates
import ArchGDAL
using Rasters
using LinearAlgebra
using Statistics
using SparseArrays
using BlockDiagonals
using ProgressMeter
using Distances
import GaussianRandomFields: CovarianceFunction, Matern, apply
using Logging

BLAS.set_num_threads(1)

struct CustomLogger <: AbstractLogger
    stream::IO
    min_level::LogLevel
end

Logging.min_enabled_level(logger::CustomLogger) = logger.min_level
Logging.shouldlog(logger::CustomLogger, level, _module, group, id) = level >= logger.min_level

function Logging.handle_message(logger::CustomLogger, level, message, _module, group, id, file, line; kwargs...)
    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println(logger.stream, "[$t $(uppercase(string(level)))] $message")
end

global_logger(CustomLogger(stdout, Logging.Info))

struct BBox
    xmin::Float64
    ymin::Float64
    xmax::Float64
    ymax::Float64
end

function BBox(raster::Raster)::BBox
    ((xmin, xmax), (ymin, ymax)) = bounds(raster[:, :, 1])[1:2]
    return BBox(xmin, ymin, xmax, ymax)
end

buffer(bbox::BBox, distance::Float64)::BBox =
    BBox(bbox.xmin - distance, bbox.ymin - distance, bbox.xmax + distance, bbox.ymax + distance)

function extract_tile_dims(reference_tif::AbstractString)::Tuple{X, Y}
    ref = Raster(reference_tif)
    return (dims(ref, X), dims(ref, Y))
end

@inline function read_tif_float64(filename::AbstractString)::Matrix{Float64}
    arr = Array(Raster(filename))
    return Float64.(coalesce.(arr, NaN))
end

@inline function read_prior_raster_float64(filename::AbstractString)::Raster
    r = Raster(filename)
    return rebuild(r; data=Float64.(coalesce.(Array(r), NaN)), missingval=NaN)
end

struct DataFusionState
    mean::Raster
    SD::Raster
    mean_bias::Raster
    SD_bias::Raster
end

function kalman_filter_step(H::SparseMatrixCSC{Float64, Int}, y::AbstractVector{Float64}, err_vars::AbstractVector{Float64}, x_pred::Vector{Float64}, P_pred::Matrix{Float64})::Tuple{Vector{Float64}, Matrix{Float64}}
    ym = .!isnan.(y)
    sum(ym) == 0 && return (x_pred, P_pred)

    R = @views Diagonal(err_vars[ym])
    y_obs = @views y[ym]
    Ht = H[ym, :]

    res_pred = y_obs - Ht * x_pred
    HPpT = P_pred * Ht'
    S = Ht * HPpT + R

    LAPACK.potrf!('U', S)
    LAPACK.potri!('U', S)
    K = BLAS.symm('R', 'U', S, HPpT)

    x_new = x_pred + K * res_pred
    P_new = P_pred - K * (Ht * P_pred)

    return (x_new, P_new)
end

function unif_weighted_obs_operator(sensor::Matrix{Float64}, target::Matrix{Float64}, sensor_res::Vector{Float64}, target_res::Vector{Float64})::SparseMatrixCSC{Float64, Int}
    d1 = pairwise(Euclidean(), sensor[:, 1]' .+ sensor_res[1] / 2, target[:, 1]' .+ target_res[1] / 2, dims=2) .<= sensor_res[1] / 2
    d2 = pairwise(Euclidean(), sensor[:, 2]' .+ sensor_res[2] / 2, target[:, 2]' .+ target_res[2] / 2, dims=2) .<= sensor_res[2] / 2
    H = d1 .* d2
    return sparse(broadcast(/, H, sum(H, dims=2)))
end

function matern_cor(X::AbstractArray{Float64}; reg::Float64=1e-10, ν::Float64=1.5, σ::Float64=200.0)::Matrix{Float64}
    cc = CovarianceFunction(2, Matern(σ, ν))
    return Diagonal(reg * ones(size(X, 2))) + apply(cc, X, X)
end

function cell_size(raster::Raster)::Tuple{Float64, Float64}
    rows, cols = size(raster, 1), size(raster, 2)
    bbox = bounds(raster[:, :, 1])
    xmin, xmax = bbox[1]
    ymin, ymax = bbox[2]
    return (xmax - xmin) / rows, -(ymax - ymin) / cols
end

get_x_matrix(image::Raster) =
    reshape(repeat(vec(image.dims[1].val[:]), inner=1, outer=size(image, 2)), size(image, 1), size(image, 2))

get_y_matrix(image::Raster) =
    transpose(reshape(repeat(vec(image.dims[2].val[:]), inner=1, outer=size(image, 1)), size(image, 2), size(image, 1)))

function STARS_fusion_step(
        measurements::Vector{Matrix{Float64}},
        measurement_error_vars::Vector{Matrix{Float64}},
        measurement_coords::Vector{Matrix{Float64}},
        resolutions::Vector{Vector{Float64}},
        target_coords::Matrix{Float64},
        target_resolution::Vector{Float64},
        prior_mean::Matrix{Float64},
        prior_sd::Matrix{Float64},
        target_time_index::Int,
        cov_pars::Vector{Float64},
        offset_ar::Vector{Float64},
        offset_var::Vector{Float64})::Tuple{Matrix{Float64}, Matrix{Float64}}

    ni = length(measurements)
    nnobs = [size(measurements[i], 1) for i in 1:ni]
    nf = size(target_coords, 1)
    nsteps = target_time_index

    H = spzeros(Float64, 0, nf)
    ys = Array{Float64}(undef, 0, nsteps)
    err_vars = Array{Float64}(undef, 0, nsteps)

    for i in 1:ni
        H1 = unif_weighted_obs_operator(measurement_coords[i], target_coords, resolutions[i], target_resolution)
        H = vcat(H, H1)
        ys = vcat(ys, measurements[i][:, 1:nsteps])
        err_vars = vcat(err_vars, measurement_error_vars[i][:, 1:nsteps])
    end

    Q = cov_pars[1] .* matern_cor(transpose(target_coords), reg=cov_pars[3], ν=cov_pars[4], σ=cov_pars[2])

    add_bias = offset_ar .> 0
    if any(add_bias)
        nfs = sum(nnobs[add_bias])
        ar_vals = vcat(fill.(offset_ar[add_bias], nnobs[add_bias])...)
        ar_vars = vcat(fill.(offset_var[add_bias], nnobs[add_bias])...)
        Hb = spzeros(Float64, sum(nnobs), nfs)
        Hb[vcat(fill.(add_bias, nnobs)...), :] = sparse(I, nfs, nfs)
        H = hcat(Hb, H)

        F = Diagonal(vcat(ar_vals, ones(nf)))
        Q = Matrix(BlockDiagonal([diagm(ar_vars), Q]))
    else
        F = Diagonal(ones(nf))
    end

    x_filt = convert(Vector{Float64}, prior_mean[:])
    P_filt = convert(Matrix{Float64}, Diagonal(prior_sd[:] .^ 2))

    size(P_filt) == size(Q) || error("Dimension of prior_mean/prior_sd doesn't match model dimension.")

    for t in 1:nsteps
        x_pred = F * x_filt
        P_pred = F * P_filt * F' + Q
        x_filt, P_filt = kalman_filter_step(H, ys[:, t], err_vars[:, t], x_pred, P_pred)
    end

    return reshape(x_filt, :, 1), reshape(sqrt.(diag(P_filt)), :, 1)
end

function coarse_fine_data_fusion(
        coarse_images::Raster,
        fine_images::Raster,
        cov_pars::Raster,
        prior::Union{DataFusionState, Nothing},
        target_date::Date;
        coarse_err_var::Float64 = 1e-6,
        fine_err_var::Float64 = 1e-6,
        offset_ar::Vector{Float64} = [1.0, 0.0],
        offset_var::Vector{Float64} = [1e-5, 1e-5],
        default_mean::Float64 = 0.3,
        default_SD::Float64 = 0.01,
        default_bias_mean::Float64 = 0.0,
        default_bias_SD::Float64 = 1e-5,
        buffer_distance::Float64 = 100.0)::DataFusionState

    coarse_dims = coarse_images.dims[1:2]
    fine_dims = fine_images.dims[1:2]
    coarse_rows, coarse_cols, _ = size(coarse_images)
    fine_rows, fine_cols, _ = size(fine_images)

    if prior === nothing
        prior_fused_mean = Raster(fill(default_mean, fine_rows, fine_cols), dims=fine_dims, missingval=NaN)
        prior_fused_sd = Raster(fill(default_SD, fine_rows, fine_cols), dims=fine_dims, missingval=NaN)
        prior_bias_mean = Raster(fill(default_bias_mean, coarse_rows, coarse_cols), dims=coarse_dims, missingval=NaN)
        prior_bias_sd = Raster(fill(default_bias_SD, coarse_rows, coarse_cols), dims=coarse_dims, missingval=NaN)
    else
        prior_fused_mean = length(prior.mean.dims) == 3 ? prior.mean[:, :, end] : prior.mean
        prior_fused_sd = length(prior.SD.dims) == 3 ? prior.SD[:, :, end] : prior.SD
        prior_bias_mean = length(prior.mean_bias.dims) == 3 ? prior.mean_bias[:, :, end] : prior.mean_bias
        prior_bias_sd = length(prior.SD_bias.dims) == 3 ? prior.SD_bias[:, :, end] : prior.SD_bias
    end

    target_time_index = findfirst(==(target_date), Array(fine_images.dims[3].val))
    target_time_index === nothing && error("target date $(target_date) not found in fine images")

    bias_dims = (coarse_dims[1], coarse_dims[2], Ti([target_date]))
    target_dims = (fine_dims[1], fine_dims[2], Ti([target_date]))

    fine_cell_width, fine_cell_height = cell_size(fine_images)
    coarse_cell_width, coarse_cell_height = cell_size(coarse_images)

    resolutions = [[coarse_cell_width, -coarse_cell_height], [fine_cell_width, -fine_cell_height]]
    target_resolution = [fine_cell_width, -fine_cell_height]

    fused_target_images = Raster(fill(NaN, fine_rows, fine_cols, 1), dims=target_dims, missingval=NaN)
    fused_UQ_target_images = Raster(fill(NaN, fine_rows, fine_cols, 1), dims=target_dims, missingval=NaN)
    fused_coarse_bias_images = Raster(fill(NaN, coarse_rows, coarse_cols, 1), dims=bias_dims, missingval=NaN)
    fused_UQ_coarse_bias_images = Raster(fill(NaN, coarse_rows, coarse_cols, 1), dims=bias_dims, missingval=NaN)

    m1 = hcat(repeat(1:coarse_cols, inner=[1], outer=[coarse_rows]),
              repeat(1:coarse_rows, inner=[coarse_cols], outer=[1]))
    n = size(m1, 1)

    fused_vectors_list = Vector{Matrix{Float64}}(undef, n)
    fused_UQ_vectors_list = Vector{Matrix{Float64}}(undef, n)

    @info "running Kalman filter fusion across $(n) moving windows using $(nthreads()) threads"
    p = Progress(n; desc="Kalman fusion: ")
    update!(p, 0)
    jj = Threads.Atomic{Int}(0)
    l = Threads.SpinLock()

    Threads.@threads for k in 1:n
        Threads.atomic_add!(jj, 1)
        Threads.lock(l)
        update!(p, jj[])
        Threads.unlock(l)

        coarse_col, coarse_row = m1[k, :]
        target_bbox = buffer(BBox(coarse_images[coarse_row:coarse_row, coarse_col:coarse_col, :]), fine_cell_width)
        window_bbox = buffer(target_bbox, buffer_distance)

        coarse_window = view(coarse_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        bias_mean_window = view(prior_bias_mean, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        bias_sd_window = view(prior_bias_sd, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))

        coarse_window_rows, coarse_window_cols, coarse_window_timesteps = size(coarse_window)
        coarse_window_values = Matrix{Float64}(reshape(read(coarse_window), coarse_window_rows * coarse_window_cols, coarse_window_timesteps))
        bias_mean_window_values = Matrix{Float64}(reshape(read(bias_mean_window), coarse_window_rows * coarse_window_cols, 1))
        bias_sd_window_values = Matrix{Float64}(reshape(read(bias_sd_window), coarse_window_rows * coarse_window_cols, 1))

        fine_window = view(fine_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        prior_mean_window = view(prior_fused_mean, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        prior_sd_window = view(prior_fused_sd, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))

        fine_window_rows, fine_window_cols, fine_window_timesteps = size(fine_window)
        fine_window_values = Matrix{Float64}(reshape(read(fine_window), fine_window_rows * fine_window_cols, fine_window_timesteps))
        prior_mean_window_values = Matrix{Float64}(reshape(read(prior_mean_window), fine_window_rows * fine_window_cols, 1))
        prior_sd_window_values = Matrix{Float64}(reshape(read(prior_sd_window), fine_window_rows * fine_window_cols, 1))

        coarse_window_coords = hcat(vec(get_x_matrix(coarse_window)), vec(get_y_matrix(coarse_window)))
        fine_window_coords = hcat(vec(get_x_matrix(fine_window)), vec(get_y_matrix(fine_window)))

        cov_pars_window = view(cov_pars, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        cov_values = vec(read(cov_pars_window))

        coarse_err_vars = fill(coarse_err_var, size(coarse_window_values))
        fine_err_vars = fill(fine_err_var, size(fine_window_values))

        fused_vectors_list[k], fused_UQ_vectors_list[k] = STARS_fusion_step(
            [coarse_window_values, fine_window_values],
            [coarse_err_vars, fine_err_vars],
            [coarse_window_coords, fine_window_coords],
            resolutions,
            fine_window_coords,
            target_resolution,
            vcat(bias_mean_window_values, prior_mean_window_values),
            vcat(bias_sd_window_values, prior_sd_window_values),
            target_time_index,
            cov_values,
            offset_ar,
            offset_var
        )
    end

    @info "stitching $(n) fused windows into output fine ($(fine_rows)x$(fine_cols)) and coarse ($(coarse_rows)x$(coarse_cols)) grids"
    p_stitch = Progress(n; desc="Grid stitching: ")
    for k in 1:n
        next!(p_stitch)

        coarse_col, coarse_row = m1[k, :]
        target_bbox = buffer(BBox(coarse_images[coarse_row:coarse_row, coarse_col:coarse_col, :]), fine_cell_width)
        window_bbox = buffer(target_bbox, buffer_distance)

        coarse_window = view(coarse_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        fine_window = view(fine_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))

        coarse_window_rows, coarse_window_cols, _ = size(coarse_window)
        fine_window_rows, fine_window_cols, _ = size(fine_window)
        nf_win = fine_window_rows * fine_window_cols

        x_sel = X(Rasters.Between(target_bbox.xmin, target_bbox.xmax))
        y_sel = Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax))

        fused_target_images_subset = view(fused_target_images, x_sel, y_sel)
        fused_fine_raster = Raster(reshape(fused_vectors_list[k][(end - nf_win + 1):end, :], fine_window_rows, fine_window_cols, 1), dims=(fine_window.dims[1:2]..., Band(1:1)), missingval=NaN)
        resampled_subset = view(fused_fine_raster, x_sel, y_sel)
        inds = isnan.(fused_target_images_subset)
        fused_target_images_subset[inds] = resampled_subset[inds]

        fused_UQ_target_images_subset = view(fused_UQ_target_images, x_sel, y_sel)
        fused_UQ_fine_raster = Raster(reshape(fused_UQ_vectors_list[k][(end - nf_win + 1):end, :], fine_window_rows, fine_window_cols, 1), dims=(fine_window.dims[1:2]..., Band(1:1)), missingval=NaN)
        resampled_subset = view(fused_UQ_fine_raster, x_sel, y_sel)
        inds = isnan.(fused_UQ_target_images_subset)
        fused_UQ_target_images_subset[inds] = resampled_subset[inds]

        fused_bias_images_subset = view(fused_coarse_bias_images, x_sel, y_sel)
        fused_bias_raster = Raster(reshape(fused_vectors_list[k][1:(end - nf_win), :], coarse_window_rows, coarse_window_cols, 1), dims=(coarse_window.dims[1:2]..., Band(1:1)), missingval=NaN)
        if size(fused_bias_raster)[1:2] == (1, 1) && size(fused_bias_images_subset)[1:2] == (1, 1)
            fused_bias_images_subset[:, :, :] = fused_bias_raster[:, :, :]
        else
            resampled_subset = view(fused_bias_raster, x_sel, y_sel)
            inds = isnan.(fused_bias_images_subset)
            fused_bias_images_subset[inds] = resampled_subset[inds]
        end

        fused_UQ_bias_images_subset = view(fused_UQ_coarse_bias_images, x_sel, y_sel)
        fused_UQ_bias_raster = Raster(reshape(fused_UQ_vectors_list[k][1:(end - nf_win), :], coarse_window_rows, coarse_window_cols, 1), dims=(coarse_window.dims[1:2]..., Band(1:1)), missingval=NaN)
        if size(fused_UQ_bias_raster)[1:2] == (1, 1) && size(fused_UQ_bias_images_subset)[1:2] == (1, 1)
            fused_UQ_bias_images_subset[:, :, :] = fused_UQ_bias_raster[:, :, :]
        else
            resampled_subset = view(fused_UQ_bias_raster, x_sel, y_sel)
            inds = isnan.(fused_UQ_bias_images_subset)
            fused_UQ_bias_images_subset[inds] = resampled_subset[inds]
        end
    end

    return DataFusionState(
        fused_target_images,
        fused_UQ_target_images,
        fused_coarse_bias_images,
        fused_UQ_coarse_bias_images
    )
end

function fast_var_est(coarse_images::Raster; n_eff_agg::Float64, min_num_obs::Int=8, default_var::Float64=1e-4)
    num_obs = sum(.!(isnan.(coarse_images)), dims=3)
    result = mapslices(x -> var(x[.!(isnan.(x))]), diff(coarse_images, dims=3), dims=3) .* n_eff_agg
    result[num_obs .< min_num_obs] .= default_var
    return result
end

function compute_n_eff(agg_scale::Int, range::Real; smoothness::Float64=1.5)::Float64
    subsample_inds = hcat(repeat(1:agg_scale, inner=[1], outer=[agg_scale]),
                          repeat(1:agg_scale, inner=[agg_scale], outer=[1]))'
    return 1.0 / (sum(matern_cor(Float64.(subsample_inds), reg=1e-16, ν=smoothness, σ=Float64(range))) / agg_scale^4)
end

@inline function downsampled_tif_path(dir::AbstractString, d::Date, tile::AbstractString, product::AbstractString, cell_size::Int64)::String
    date_str = Dates.format(d, dateformat"yyyy-mm-dd")
    return joinpath(dir, string(year(d)), date_str, tile, "STARS_$(product)_$(tile)_$(cell_size)m.tif")
end

@info "processing STARS data fusion"
tile = ARGS[1]
@info "tile: $(tile)"
coarse_cell_size = parse(Int64, ARGS[2])
@info "coarse cell size: $(coarse_cell_size)"
fine_cell_size = parse(Int64, ARGS[3])
@info "fine cell size: $(fine_cell_size)"
VIIRS_start_date = Date(ARGS[4])
@info "VIIRS start date: $(VIIRS_start_date)"
VIIRS_end_date = Date(ARGS[5])
@info "VIIRS end date: $(VIIRS_end_date)"
HLS_start_date = Date(ARGS[6])
@info "HLS start date: $(HLS_start_date)"
HLS_end_date = Date(ARGS[7])
@info "HLS end date: $(HLS_end_date)"
downsampled_directory = ARGS[8]
@info "downsampled inputs directory: $(downsampled_directory)"
product_name = ARGS[9]
@info "Computing $(product_name) product"
posterior_filename = ARGS[10]
@info "posterior filename: $(posterior_filename)"
posterior_UQ_filename = ARGS[11]
@info "posterior UQ filename: $(posterior_UQ_filename)"
posterior_bias_filename = ARGS[12]
@info "posterior bias filename: $(posterior_bias_filename)"
posterior_bias_UQ_filename = ARGS[13]
@info "posterior bias UQ filename: $(posterior_bias_UQ_filename)"

if length(ARGS) >= 17
    prior_filename = ARGS[14]
    @info "prior filename: $(prior_filename)"
    prior_mean = read_prior_raster_float64(prior_filename)
    prior_UQ_filename = ARGS[15]
    @info "prior UQ filename: $(prior_UQ_filename)"
    prior_sd = read_prior_raster_float64(prior_UQ_filename)
    prior_bias_filename = ARGS[16]
    @info "prior bias filename: $(prior_bias_filename)"
    prior_bias_mean = read_prior_raster_float64(prior_bias_filename)
    prior_bias_UQ_filename = ARGS[17]
    @info "prior bias UQ filename: $(prior_bias_UQ_filename)"
    prior_bias_sd = read_prior_raster_float64(prior_bias_UQ_filename)
    prior = DataFusionState(prior_mean, prior_sd, prior_bias_mean, prior_bias_sd)
else
    prior = nothing
end

covariance_dates = collect(VIIRS_start_date:Day(1):VIIRS_end_date)
dates = collect(HLS_start_date:Day(1):HLS_end_date)

coarse_paths = [downsampled_tif_path(downsampled_directory, d, tile, product_name, coarse_cell_size) for d in covariance_dates]
fine_paths = [downsampled_tif_path(downsampled_directory, d, tile, product_name, fine_cell_size) for d in dates]

ref_coarse_tif = isnothing(prior) ? coarse_paths[findfirst(ispath, coarse_paths)] : prior_bias_filename
ref_fine_tif = isnothing(prior) ? fine_paths[findfirst(ispath, fine_paths)] : prior_filename

x_coarse, y_coarse = extract_tile_dims(ref_coarse_tif)
x_coarse_size = length(x_coarse)
y_coarse_size = length(y_coarse)
@info "coarse x size: $(x_coarse_size)"
@info "coarse y size: $(y_coarse_size)"

x_fine, y_fine = extract_tile_dims(ref_fine_tif)
x_fine_size = length(x_fine)
y_fine_size = length(y_fine)
@info "fine x size: $(x_fine_size)"
@info "fine y size: $(y_fine_size)"

covariance_data = fill(NaN, x_coarse_size, y_coarse_size, length(covariance_dates))
cov_index_by_date = Dict{Date, Int}()
sizehint!(cov_index_by_date, length(covariance_dates))

for (i, d) in enumerate(covariance_dates)
    cov_index_by_date[d] = i
    date_str = Dates.format(d, dateformat"yyyy-mm-dd")
    filename = downsampled_tif_path(downsampled_directory, d, tile, product_name, coarse_cell_size)
    if !ispath(filename)
        @info "coarse image is not available on $(date_str)"
    else
        @info "ingesting coarse image on $(date_str): $(filename)"
        @views covariance_data[:, :, i] .= read_tif_float64(filename)
    end
end

@info "building coarse image raster for covariance calculation"
covariance_images = Raster(covariance_data, dims=(x_coarse, y_coarse, Ti(covariance_dates)), missingval=NaN)

n_eff = compute_n_eff(Int(round(coarse_cell_size / fine_cell_size)), 2, smoothness=1.5)
sp_var = fast_var_est(covariance_images, n_eff_agg=n_eff)
cov_pars_raster = Raster(fill(NaN, x_coarse_size, y_coarse_size, 4), dims=(x_coarse, y_coarse, Band(1:4)), missingval=NaN)
cov_pars_raster[:, :, 1] = sp_var
cov_pars_raster[:, :, 2] .= 200.0
cov_pars_raster[:, :, 3] .= 1e-10
cov_pars_raster[:, :, 4] .= 1.5

if all(haskey(cov_index_by_date, d) for d in dates)
    idx = [cov_index_by_date[d] for d in dates]
    coarse_data = @view covariance_data[:, :, idx]
else
    coarse_data = fill(NaN, x_coarse_size, y_coarse_size, length(dates))
    for (i, d) in enumerate(dates)
        if haskey(cov_index_by_date, d)
            @views coarse_data[:, :, i] .= covariance_data[:, :, cov_index_by_date[d]]
        else
            date_str = Dates.format(d, dateformat"yyyy-mm-dd")
            filename = downsampled_tif_path(downsampled_directory, d, tile, product_name, coarse_cell_size)
            if !ispath(filename)
                @info "coarse image is not available on $(date_str)"
            else
                @info "ingesting coarse image on $(date_str): $(filename)"
                @views coarse_data[:, :, i] .= read_tif_float64(filename)
            end
        end
    end
end

coarse_images = Raster(coarse_data, dims=(x_coarse, y_coarse, Ti(dates)), missingval=NaN)

fine_data = fill(NaN, x_fine_size, y_fine_size, length(dates))
for (i, d) in enumerate(dates)
    date_str = Dates.format(d, dateformat"yyyy-mm-dd")
    filename = downsampled_tif_path(downsampled_directory, d, tile, product_name, fine_cell_size)
    if !ispath(filename)
        @info "fine image is not available on $(date_str)"
    else
        @info "ingesting fine image on $(date_str): $(filename)"
        @views fine_data[:, :, i] .= read_tif_float64(filename)
    end
end

fine_images = Raster(fine_data, dims=(x_fine, y_fine, Ti(dates)), missingval=NaN)

@info "running data fusion"
fusion_results = coarse_fine_data_fusion(
    coarse_images,
    fine_images,
    cov_pars_raster,
    prior,
    dates[end]
)

@info "writing fused mean: $(posterior_filename)"
write(posterior_filename, Raster(fusion_results.mean, dims=(x_fine, y_fine, Band(1:1)), missingval=NaN); force=true, driver="GTiff")
@info "writing fused SD: $(posterior_UQ_filename)"
write(posterior_UQ_filename, Raster(fusion_results.SD, dims=(x_fine, y_fine, Band(1:1)), missingval=NaN); force=true, driver="GTiff")
@info "writing bias mean: $(posterior_bias_filename)"
write(posterior_bias_filename, Raster(fusion_results.mean_bias, dims=(x_coarse, y_coarse, Band(1:1)), missingval=NaN); force=true, driver="GTiff")
@info "writing bias SD: $(posterior_bias_UQ_filename)"
write(posterior_bias_UQ_filename, Raster(fusion_results.SD_bias, dims=(x_coarse, y_coarse, Band(1:1)), missingval=NaN); force=true, driver="GTiff")