using LsqFit, LinearAlgebra, StaticArrays, Random
Random.seed!(123)

# Pre-allocate reusable structures outside the loop
struct FitCache
    x::Vector{Float64}
    y::Vector{Float64}
    weights::Vector{Float64}
    p0::Vector{Float64}
    fit::LsqFit.LsqFitResult
end

# Optimized linear model (avoid allocations)
#@inline function linear_model(x::AbstractVector, p::AbstractVector)
#    @inbounds @simd for i in eachindex(x)
#        x[i] * p[1] + p[2]
#    end
#    return x  # Return modified x (in-place)
#end
# Linear model + fit helper
linear_model(x,p) = p[1] .* x .+ p[2]  # p[1] = slope (m), p[2] = intercept (b)

quadratic_model(x,p) = p[1].*x.*x .+ p[2].*x .+ p[3]


# General polynomial model generator
function polynomial_model(degree::Int)
    """
    Returns a polynomial model function of given degree.
    Example: degree=2 -> p[1]*x^2 + p[2]*x + p[3]
    """
    return function(x, p)
        result = zeros(length(x))
        for i in 1:length(p)
            result .+= p[i] .* x.^(degree - i + 1)
        end
        return result
    end
end

# Alternative: More efficient polynomial evaluation using Horner's method
function polynomial_model_horner(degree::Int)
    return function(x, p)
        result = fill(p[1], length(x))
        for i in 2:length(p)
            result = result .* x .+ p[i]
        end
        return result
    end
end

# Main fitting function
function perform_fit(x::Vector{Float64}, y::Vector{Float64}; 
                     σ::Union{Vector{Float64}, Vector{Nothing}}, degree::Int=1)
    """
    Fit a polynomial of specified degree to data.
    
    Arguments:
    - x, y: data points
    - degree: polynomial degree (1=linear, 2=quadratic, 3=cubic, etc.)
    - σ: optional error bars
    
    Returns:
    - params: fitted parameters (coefficients)
    - errors: standard errors of parameters
    - residuals: weighted/unweighted residuals
    - r_squared: R² value
    """
    
    # Create polynomial model of specified degree
    model = polynomial_model(degree)
    
    # Initial parameters (all ones, degree+1 coefficients)
    p0 = ones(Float64, degree + 1)
    
    if typeof(σ) == Vector{Nothing}
        # Unweighted fit
        fit = curve_fit(model, x, y, p0)
        
        # Calculate R-squared
        y_pred = model(x, fit.param)
        ss_res = sum((y .- y_pred).^2)
        ss_tot = sum((y .- mean(y)).^2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # For unweighted fit, residuals are just differences
        residuals = (y .- y_pred).^2
        
    else
        # Weighted fit
        weights = 1 ./ σ.^2
        fit = curve_fit(model, x, y, Diagonal(weights), p0)
        
        # Weighted R-squared
        y_pred = model(x, fit.param)
        ss_res = sum(weights .* (y .- y_pred).^2)
        ss_tot = sum(weights .* (y .- mean(y)).^2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Weighted residuals
        residuals = weights .* (y .- y_pred).^2
    end

    
    # Get parameter errors
    errs = stderror(fit)
    
    # Return parameters, errors, residuals, and R²
    return fit.param, errs, residuals, r_squared
end

# Convenience function for specific fits
function perform_linear_fit(x::Vector{Float64}, y::Vector{Float64}, σ::Union{Vector{Float64}, Vector{Nothing}})
    """Wrapper for linear fit (degree=1)"""
    params, errs, residuals, r2 = perform_fit(x, y; σ=σ, degree=1)
    return params[1], params[2], (errs[1], errs[2]), residuals, r2
end

function perform_quadratic_fit(x::Vector{Float64}, y::Vector{Float64}, σ::Union{Vector{Float64}, Vector{Nothing}})
    """Wrapper for quadratic fit (degree=2)"""
    params, errs, residuals, r2 = perform_fit(x, y; σ=σ, degree=2)
    return params[1], params[2], params[3], (errs[1], errs[2], errs[3]), residuals, r2
end

function perform_cubic_fit(x::Vector{Float64}, y::Vector{Float64}, σ::Union{Vector{Float64}, Vector{Nothing}})
    """Wrapper for cubic fit (degree=3)"""
    params, errs, residuals, r2 = perform_fit(x, y; σ=σ, degree=3)
    return params[1], params[2], params[3], params[4], 
           (errs[1], errs[2], errs[3], errs[4]), residuals, r2
end

# Helper function to evaluate polynomial at new x values
function evaluate_polynomial(x::Vector{Float64}, params::Vector{Float64})
    """Evaluate polynomial with given coefficients at points x"""
    result = zeros(length(x))
    for i in 1:length(params)
        result .+= params[i] .* x.^(length(params) - i)
    end
    return result
end

# Fast fit without plotting
@inline function perform_fit_fast(x::Vector{Float64}, y::Vector{Float64}, σ::Union{Vector{Float64}, Vector{Nothing}})
    n = length(x)
    # Precompute sums for simple linear regression (much faster than curve_fit)
    if nothing in σ 
        # Unweighted fit using analytical solution (O(n))
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = 0.0
        sum_x2 = 0.0
        
        @inbounds @simd for i in 1:n
            sum_xy += x[i] * y[i]
            sum_x2 += x[i] * x[i]
        end
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator != 0
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
        else
            slope, intercept = 1.0, 0.0
        end
        
        # Calculate residuals and R²
        ss_res = 0.0
        ss_tot = 0.0
        mean_y = sum_y / n
        
        @inbounds @simd for i in 1:n
            y_pred = slope * x[i] + intercept
            resid = y[i] - y_pred
            ss_res += resid * resid
            dev = y[i] - mean_y
            ss_tot += dev * dev
        end
        r2 = 1 - (ss_res / ss_tot)
        
        # Estimate standard errors
        #if n > 2
        #    s2 = ss_res / (n - 2)
        #    slope_err = sqrt(s2 * n / denominator)
        #    intercept_err = sqrt(s2 * sum_x2 / denominator)
        #else
        #    slope_err, intercept_err = 0.0, 0.0
        #end
        
        return r2# slope, intercept, (slope_err, intercept_err), ss_res, r2
    else
        # Weighted fit using analytical solution
        @assert length(σ) == n
        sum_w = 0.0
        sum_wx = 0.0
        sum_wy = 0.0
        sum_wxy = 0.0
        sum_wx2 = 0.0
        
        @inbounds @simd for i in 1:n
            w = 1.0 / (σ[i] * σ[i])
            sum_w += w
            sum_wx += w * x[i]
            sum_wy += w * y[i]
            sum_wxy += w * x[i] * y[i]
            sum_wx2 += w * x[i] * x[i]
        end
        
        denominator = sum_w * sum_wx2 - sum_wx * sum_wx
        if denominator != 0
            slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator
            intercept = (sum_wy - slope * sum_wx) / sum_w
        else
            slope, intercept = 1.0, 0.0
        end
        
        # Calculate weighted residuals and R²
        ss_res = 0.0
        ss_tot = 0.0
        sum_w = 0.0
        sum_wy = 0.0
        
        @inbounds @simd for i in 1:n
            w = 1.0 / (σ[i] * σ[i])
            sum_w += w
            sum_wy += w * y[i]
        end
        weighted_mean_y = sum_wy / sum_w
        
        @inbounds @simd for i in 1:n
            w = 1.0 / (σ[i] * σ[i])
            y_pred = slope * x[i] + intercept
            resid = y[i] - y_pred
            ss_res += w * resid * resid
            dev = y[i] - weighted_mean_y
            ss_tot += w * dev * dev
        end
        r2 = 1 - (ss_res / ss_tot)
        
        # Estimate standard errors for weighted fit
        #if n > 2
        #    s2 = ss_res / (n - 2)
        #    slope_err = sqrt(s2 * sum_w / denominator)
        #    intercept_err = sqrt(s2 * sum_wx2 / denominator)
        #else
        #    slope_err, intercept_err = 0.0, 0.0
        #end
        
        return r2 # slope, intercept, (slope_err, intercept_err), ss_res, r2
    end
end

# New non-plotting function (just fit)
@inline function calculate_pit_fit(id::Int, x::Vector{Float64}, y::Vector{Float64}, 
                                   dy::Union{Vector{Float64}, Nothing}, 
                                   do_fit::Bool)
    if do_fit
        slope, intercept, (dm, db), ss_res, r2 = perform_fit_fast(x, y, dy)
        return slope, dm, ss_res, r2
    else
        return 1.0, 1.0, 0.0, 0.0
    end
end

# Batch processing function for maximum speed
@inline function batch_fit_all(x_data::Vector{Vector{Float64}}, 
                               y_data::Vector{Vector{Float64}}, 
                               σ_data::Union{Vector{Vector{Float64}}, Nothing},
                               do_fit::Bool=true)
    n_pits = length(x_data)
    results = Vector{NTuple{4,Float64}}(undef, n_pits)
    
    @inbounds @simd for i in 1:n_pits
        if do_fit
            σ = σ_data === nothing ? nothing : σ_data[i]
            slope, dm, ss_res, r2 = perform_fit_fast(x_data[i], y_data[i], σ)
            results[i] = (slope, dm, ss_res, r2)
        else
            results[i] = (1.0, 1.0, 0.0, 1.0)
        end
    end
    
    return results
end

# Pre-allocate arrays for even faster repeated fits
mutable struct FitWorker
    x::Vector{Float64}
    y::Vector{Float64}
    σ::Union{Vector{Float64},Vector{Nothing}}
    cache_x::Vector{Float64}
    cache_y::Vector{Float64}
end

function FitWorker(n::Int)
    return FitWorker(
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n), #[nothing for i in 1:n],#Union{Vector{Float64},Vector{Nothing}},#(undef, n),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n)
    )
end

# Usage example for 400,000 fits
function run_many_fits(n_fits, x, y, dy, n_points)
    
    # Pre-allocate all data
    #all_y = [1.0 for _ in 1:n_fits]
    #all_σ = [0.0 for _ in 1:n_fits]
    
    # Batch fit (fastest)
    #@time results = batch_fit_all(all_x, all_y, all_σ, true)
    
    # Or use worker for sequential fits
    worker = FitWorker(n_points)
    results2 = Vector{Float64}(undef,n_fits)#Vector{NTuple{4,Float64}}(undef, n_fits)
    
    for i in 1:n_fits
        # Copy data into pre-allocated arrays
        perm = randperm(n_points)
        copy!(worker.x, x)
        copy!(worker.y, y[perm])
        copy!(worker.σ, dy[perm])
        
        r2 = perform_fit_fast(worker.x, worker.y, worker.σ) #slope, dm, ss_res, 
        results2[i] = r2#(slope, dm, ss_res, r2)
    end
    
    return results2
end

function run_many_interval_fits(n_points, x, y, dy, ntot_points)
    
    # Pre-allocate all data
    #all_y = [1.0 for _ in 1:n_fits]
    #all_σ = [0.0 for _ in 1:n_fits]
    
    # Batch fit (fastest)
    #@time results = batch_fit_all(all_x, all_y, all_σ, true)
    
    # Or use worker for sequential fits
    worker = FitWorker(n_points)
    results2 = Vector{Float64}(undef,ntot_points - n_points+1)#Vector{NTuple{4,Float64}}(undef, n_fits)
    
    for i in 1:(ntot_points-n_points+1)
        copy!(worker.x, x[i:i+n_points-1])
        copy!(worker.y, y[i:i+n_points-1])
        copy!(worker.σ, dy[i:i+n_points-1])
        
        r2 = perform_fit_fast(worker.x, worker.y, worker.σ) #slope, dm, ss_res, 
        results2[i] = r2#(slope, dm, ss_res, r2)
    end
    
    return results2
end



# If you still need the plotting function separately
function plot_pit_fits2(fig, id::Int, x::Vector{Float64}, y::Vector{Float64}, 
                       dy::Union{Vector{Float64}, Vector{Nothing}}, IDs_pit::Vector{String},
                       x_lims::Vector{Float64}; x_label::String, y_label::String,
                       do_fit::Bool, subplot::Int64)
    
    color = cgrad(:rainbow)[id/length(IDs_pit)]
    
    args = (fig[subplot], x, y)
    kwargs = Dict(
        :label => (subplot == 1 || length(IDs_pit)==1 || IDs_pit[1]=="" ? nothing : IDs_pit[id][1:end-5]),
        :xlabel => x_label,
        :ylabel => y_label,
        :color => color,
        :markerstrokecolor => color,
        :alpha => 0.7
    )
    if dy !== nothing
        kwargs[:yerr] = dy
        scatter!(args...; kwargs...) 
    else
        kwargs[:marker] = :circle
        plot!(args...; kwargs...) 
    end

    if do_fit 
        if length(IDs_pit) > 0 && IDs_pit[1] == "black"
            color = :black
        end
        m, b, (dm, _), ss_res, r2 = perform_fit_fast(x, y, dy)
        xfit = range(x_lims[1], x_lims[2], length=100)
        plot!(fig[subplot], xfit, linear_model(xfit, (m,b)),
              linestyle=:dash, linewidth=2, color=color,
              label="slope=$(value_with_error(m,dm,2))")
        return m, dm, ss_res, r2
    else
        return 1.0, 1.0, 0.0, 1.0
    end
end



using Distributions

function chi2_sf(chi2_stat::Float64, df::Int)
    """
    Survival function (1 - CDF) of chi-square distribution
    Equivalent to scipy.stats.chi2.sf(chi2_stat, df)
    """
    dist = Chisq(df)
    println("&&&&&&&&&&&&&&&&&& ",chi2_stat,"  ",dist)
    return ccdf(dist, chi2_stat)  # Complementary CDF = 1 - cdf
end

# For reduced chi2
function fit_goodness_pvalue(chi2_stat::Float64, df::Int)
    """
    Calculate p-value from reduced chi-square
    """
    return chi2_sf(chi2_stat, df)
end













using Statistics, LinearAlgebra, Distributions#, SpecialFunctions

function weighted_linear_fit(x, y, y_err; x_err=nothing)
    """
    Perform weighted linear regression with uncertainties.
    
    Returns:
    - slope, intercept
    - slope_err, intercept_err
    - chi2, reduced_chi2
    - p_value (from F-test)
    - r2_weighted
    """
    n = length(x)
    
    # If x_err provided, combine uncertainties in quadrature
    if x_err !== nothing
        # For weighted regression with x errors, use effective variance method
        # First do unweighted fit to estimate slope
        X = [ones(n) x]
        coeffs = X \ y
        slope_init = coeffs[2]
        
        # Combine errors: σ_eff² = σ_y² + (slope_init * σ_x)²
        eff_err = sqrt.(y_err.^2 .+ (slope_init * x_err).^2)
        weights = 1 ./ eff_err.^2
    else
        weights = 1 ./ y_err.^2
    end
    
    # Weighted sums
    W = sum(weights)
    Wx = sum(weights .* x)
    Wy = sum(weights .* y)
    Wx2 = sum(weights .* x.^2)
    Wxy = sum(weights .* x .* y)
    
    # Denominator
    Δ = W * Wx2 - Wx^2
    
    if Δ == 0
        error("Cannot fit: denominator is zero")
    end
    
    # Weighted slope and intercept
    slope = (W * Wxy - Wx * Wy) / Δ
    intercept = (Wy * Wx2 - Wx * Wxy) / Δ
    
    # Uncertainties
    slope_err = sqrt(W / Δ)
    intercept_err = sqrt(Wx2 / Δ)
    
    # Calculate chi-squared
    y_pred = intercept .+ slope * x
    residuals = y .- y_pred
    
    if x_err !== nothing
        # Include x errors in chi2
        eff_err = sqrt.(y_err.^2 .+ (slope * x_err).^2)
        chi2 = sum((residuals ./ eff_err).^2)
    else
        chi2 = sum((residuals ./ y_err).^2)
    end
    
    dof = n - 2
    reduced_chi2 = chi2 / dof
    
    # Weighted R²
    y_mean_weighted = Wy / W
    ss_tot = sum(weights .* (y .- y_mean_weighted).^2)
    ss_res = sum(weights .* residuals.^2)
    r2_weighted = 1 - ss_res / ss_tot
    
    # F-test for slope significance
    if dof > 0 && ss_res > 0
        # Weighted SSR
        ss_reg = sum(weights .* (y_pred .- y_mean_weighted).^2)
        f_stat = (ss_reg / 1) / (ss_res / dof)
        p_value = 1 - cdf(FDist(1, dof), f_stat)
    else
        f_stat = 0.0
        p_value = 1.0
    end
    
    return (slope, intercept, slope_err, intercept_err, chi2, reduced_chi2, p_value, r2_weighted, f_stat)
end

function find_largest_flat_interval_weighted(x, y, y_err; x_err=nothing, min_points=3, max_points=length(x), alpha=0.99)
    """
    Find the LARGEST consecutive interval where we CANNOT reject H₀: slope = 0
    using weighted least squares with uncertainties.
    
    Returns the interval with the most points that has p-value ≥ alpha.
    """
    n = length(x)
    
    # Precompute prefix sums for fast weighted statistics
    # We need: W, Wx, Wy, Wx2, Wxy
    prefix_W = [0.0]
    prefix_Wx = [0.0]
    prefix_Wy = [0.0]
    prefix_Wx2 = [0.0]
    prefix_Wxy = [0.0]
    
    # Compute weights
    if x_err !== nothing
        # Need initial slope estimate for effective variance
        # Use unweighted fit first
        X = [ones(n) x]
        coeffs = X \ y
        slope_init = coeffs[2]
        weights = 1 ./ (y_err.^2 .+ (slope_init * x_err).^2)
    else
        weights = 1 ./ y_err.^2
    end
    
    # Build prefix sums
    for i in 1:n
        wi = weights[i]
        xi = x[i]
        yi = y[i]
        
        push!(prefix_W, prefix_W[end] + wi)
        push!(prefix_Wx, prefix_Wx[end] + wi * xi)
        push!(prefix_Wy, prefix_Wy[end] + wi * yi)
        push!(prefix_Wx2, prefix_Wx2[end] + wi * xi^2)
        push!(prefix_Wxy, prefix_Wxy[end] + wi * xi * yi)
    end
    
    best_start = 1
    best_endpoint = min_points
    best_ncount = 0
    best_p_value = -Inf
    best_slope = 0.0
    best_slope_err = 0.0
    best_intercept = 0.0
    best_intercept_err = 0.0
    best_chi2 = 0.0
    best_reduced_chi2 = 0.0
    best_r2 = 0.0
    best_f_stat = 0.0
    
    max_endpoint = min(n, max_points)
    
    #println("Searching for largest flat interval with uncertainties (α = $alpha)...")
    #println("Testing all intervals with at least $min_points points...")
    #println("Weighted regression using $(x_err === nothing ? "y-errors only" : "x and y errors")")
    
    for start in 1:n-min_points+1
        actual_max_endpoint = min(n, start + max_points - 1)
        
        for endpoint in start+min_points-1:actual_max_endpoint
            ncount = endpoint - start + 1
            
            # Get range statistics from prefix sums
            W = prefix_W[endpoint+1] - prefix_W[start]
            Wx = prefix_Wx[endpoint+1] - prefix_Wx[start]
            Wy = prefix_Wy[endpoint+1] - prefix_Wy[start]
            Wx2 = prefix_Wx2[endpoint+1] - prefix_Wx2[start]
            Wxy = prefix_Wxy[endpoint+1] - prefix_Wxy[start]
            
            # Skip if not enough weight
            if W == 0
                continue
            end
            
            # Weighted fit for this interval
            Δ = W * Wx2 - Wx^2
            
            if Δ == 0
                continue
            end
            
            slope = (W * Wxy - Wx * Wy) / Δ
            intercept = (Wy * Wx2 - Wx * Wxy) / Δ
            
            # Uncertainties
            slope_err = sqrt(W / Δ)
            intercept_err = sqrt(Wx2 / Δ)
            
            # Get data for this interval
            x_interval = x[start:endpoint]
            y_interval = y[start:endpoint]
            y_err_interval = y_err[start:endpoint]
            
            # Calculate residuals and chi2
            y_pred = intercept .+ slope * x_interval
            residuals = y_interval .- y_pred
            
            if x_err !== nothing
                x_err_interval = x_err[start:endpoint]
                eff_err = sqrt.(y_err_interval.^2 .+ (slope * x_err_interval).^2)
                chi2 = sum((residuals ./ eff_err).^2)
            else
                chi2 = sum((residuals ./ y_err_interval).^2)
            end
            
            dof = ncount - 2
            reduced_chi2 = chi2 / dof
            
            # Weighted R²
            y_mean_weighted = Wy / W
            ss_tot = sum((1 ./ y_err_interval.^2) .* (y_interval .- y_mean_weighted).^2)
            ss_res = sum((1 ./ y_err_interval.^2) .* residuals.^2)
            r2_weighted = 1 - ss_res / ss_tot
            
            # F-test for slope significance
            if dof > 0 && ss_res > 0
                ss_reg = sum((1 ./ y_err_interval.^2) .* (y_pred .- y_mean_weighted).^2)
                f_stat = (ss_reg / 1) / (ss_res / dof)
                p_value = 1 - cdf(FDist(1, dof), f_stat)
            else
                f_stat = 0.0
                p_value = 1.0
            end
            
            # Check if this interval is flat (cannot reject H₀)
            if p_value >= alpha
                # We want the LARGEST interval first, then highest p-value
                if ncount > best_ncount || (ncount == best_ncount && p_value > best_p_value)
                    best_start = start
                    best_endpoint = endpoint
                    best_ncount = ncount
                    best_p_value = p_value
                    best_slope = slope
                    best_slope_err = slope_err
                    best_intercept = intercept
                    best_intercept_err = intercept_err
                    best_chi2 = chi2
                    best_reduced_chi2 = reduced_chi2
                    best_r2 = r2_weighted
                    best_f_stat = f_stat
                end
            end
        end
    end
    
    return (best_start, best_endpoint, best_ncount, best_slope, best_slope_err, 
            best_intercept, best_intercept_err, best_p_value, best_r2, 
            best_chi2, best_reduced_chi2, best_f_stat)
end

using StatsBase, Distributions
function find_largest_flat_interval_bootstrap(x, y, y_err; 
                                             x_err=nothing, 
                                             min_points=3, 
                                             max_points=length(x), 
                                             alpha=0.99,
                                             n_bootstrap=500,
                                             bootstrap_method=:residual)
    """
    Find the LARGEST consecutive interval where we CANNOT reject H₀: slope = 0
    using weighted least squares with uncertainties and bootstrap p-values.
    
    Returns the interval with the most points that has p-value ≥ alpha.
    
    Bootstrap methods:
    - :residual: Residual bootstrap (recommended, robust)
    - :parametric: Parametric bootstrap (assumes normality)
    - :wild: Wild bootstrap (handles heteroscedasticity)
    """
    n = length(x)
    
    # Precompute prefix sums for fast weighted statistics
    prefix_W = [0.0]
    prefix_Wx = [0.0]
    prefix_Wy = [0.0]
    prefix_Wx2 = [0.0]
    prefix_Wxy = [0.0]
    
    # Compute weights
    if x_err !== nothing
        # Need initial slope estimate for effective variance
        X = [ones(n) x]
        coeffs = X \ y
        slope_init = coeffs[2]
        weights = 1 ./ (y_err.^2 .+ (slope_init * x_err).^2)
    else
        weights = 1 ./ y_err.^2
    end
    
    # Build prefix sums
    for i in 1:n
        wi = weights[i]
        xi = x[i]
        yi = y[i]
        
        push!(prefix_W, prefix_W[end] + wi)
        push!(prefix_Wx, prefix_Wx[end] + wi * xi)
        push!(prefix_Wy, prefix_Wy[end] + wi * yi)
        push!(prefix_Wx2, prefix_Wx2[end] + wi * xi^2)
        push!(prefix_Wxy, prefix_Wxy[end] + wi * xi * yi)
    end
    
    best_start = 1
    best_endpoint = min_points
    best_ncount = 0
    best_p_value = -Inf
    best_slope = 0.0
    best_slope_err = 0.0
    best_intercept = 0.0
    best_intercept_err = 0.0
    best_chi2 = 0.0
    best_reduced_chi2 = 0.0
    best_r2 = 0.0
    best_f_stat = 0.0
    best_bootstrap_ci = (0.0, 0.0)
    
    max_endpoint = min(n, max_points)
    
    for start in 1:n-min_points+1
        actual_max_endpoint = min(n, start + max_points - 1)
        
        for endpoint in start+min_points-1:actual_max_endpoint
            ncount = endpoint - start + 1
            
            # Get range statistics from prefix sums
            W = prefix_W[endpoint+1] - prefix_W[start]
            Wx = prefix_Wx[endpoint+1] - prefix_Wx[start]
            Wy = prefix_Wy[endpoint+1] - prefix_Wy[start]
            Wx2 = prefix_Wx2[endpoint+1] - prefix_Wx2[start]
            Wxy = prefix_Wxy[endpoint+1] - prefix_Wxy[start]
            
            # Skip if not enough weight
            if W == 0
                continue
            end
            
            # Weighted fit for this interval
            Δ = W * Wx2 - Wx^2
            
            if Δ == 0
                continue
            end
            
            slope = (W * Wxy - Wx * Wy) / Δ
            intercept = (Wy * Wx2 - Wx * Wxy) / Δ
            
            # Uncertainties
            slope_err = sqrt(W / Δ)
            intercept_err = sqrt(Wx2 / Δ)
            
            # Get data for this interval
            x_interval = x[start:endpoint]
            y_interval = y[start:endpoint]
            y_err_interval = y_err[start:endpoint]
            
            # Calculate residuals and chi2
            y_pred = intercept .+ slope * x_interval
            residuals = y_interval .- y_pred
            
            if x_err !== nothing
                x_err_interval = x_err[start:endpoint]
                eff_err = sqrt.(y_err_interval.^2 .+ (slope * x_err_interval).^2)
                chi2 = sum((residuals ./ eff_err).^2)
            else
                chi2 = sum((residuals ./ y_err_interval).^2)
            end
            
            dof = ncount - 2
            reduced_chi2 = chi2 / dof
            
            # Weighted R²
            y_mean_weighted = Wy / W
            ss_tot = sum((1 ./ y_err_interval.^2) .* (y_interval .- y_mean_weighted).^2)
            ss_res = sum((1 ./ y_err_interval.^2) .* residuals.^2)
            r2_weighted = 1 - ss_res / ss_tot
            
            # F-test for slope significance (for reference) #*********** sacar
            if dof > 0 && ss_res > 0
                ss_reg = sum((1 ./ y_err_interval.^2) .* (y_pred .- y_mean_weighted).^2)
                f_stat = (ss_reg / 1) / (ss_res / dof)
                f_p_value = 1 - cdf(FDist(1, dof), f_stat)
            else
                f_stat = 0.0
                f_p_value = 1.0
            end
            
            # ================================================================
            # BOOTSTRAP P-VALUE CALCULATION
            # ================================================================
            
            # Skip bootstrap if not enough points
            if ncount < 4
                p_value = 1.0  # With few points, we can't reject H₀
                bootstrap_ci = (0.0, 0.0)
            else
                # Prepare data for bootstrap
                X_local = [ones(ncount) x_interval]
                
                # Perform bootstrap
                boot_slopes = zeros(n_bootstrap)
                
                for b in 1:n_bootstrap
                    if bootstrap_method == :residual
                        # Residual bootstrap (recommended)
                        # Resample residuals with replacement
                        idx = sample(1:ncount, ncount, replace=true)
                        
                        # Generate new y under H₀ (slope = 0)
                        # Keep the same x, remove the slope effect, add resampled residuals
                        y_centered = y_interval .- slope * x_interval
                        y_boot = y_centered[idx] .+ y_err_interval[idx] .* randn(ncount)
                        
                    elseif bootstrap_method == :parametric
                        # Parametric bootstrap (assumes normal errors)
                        sigma = sqrt(chi2 / dof)
                        y_boot = y_interval .- slope * x_interval .+ randn(ncount) * sigma
                        
                    elseif bootstrap_method == :wild
                        # Wild bootstrap (for heteroscedasticity)
                        # Rademacher distribution: ±1 with equal probability
                        signs = 2 * rand(Bernoulli(0.5), ncount) .- 1
                        y_boot = y_interval .- slope * x_interval .+ signs .* residuals
                        
                    else
                        error("Unknown bootstrap method: $bootstrap_method")
                    end
                    
                    # Fit bootstrap sample with weights
                    # Use weighted least squares
                    weights_boot = weights[start:endpoint]
                    W_boot = sum(weights_boot)
                    Wx_boot = sum(weights_boot .* x_interval)
                    Wy_boot = sum(weights_boot .* y_boot)
                    Wx2_boot = sum(weights_boot .* x_interval.^2)
                    Wxy_boot = sum(weights_boot .* x_interval .* y_boot)
                    
                    Δ_boot = W_boot * Wx2_boot - Wx_boot^2
                    
                    if Δ_boot != 0
                        boot_slopes[b] = (W_boot * Wxy_boot - Wx_boot * Wy_boot) / Δ_boot
                    else
                        boot_slopes[b] = 0.0
                    end
                end
                
                # Calculate bootstrap p-value (two-sided)
                # p-value = proportion of |boot_slopes| ≥ |slope|
                p_value = mean(abs.(boot_slopes) .>= abs(slope))
                
                # Calculate bootstrap confidence intervals
                alpha_ci = 0.05  # 95% CI
                ci_lower = quantile(boot_slopes, alpha_ci/2)
                ci_upper = quantile(boot_slopes, 1 - alpha_ci/2)
                bootstrap_ci = (ci_lower, ci_upper)
            end
            
            # Check if this interval is flat (cannot reject H₀)
            if p_value >= alpha
                # We want the LARGEST interval first, then highest p-value
                if ncount > best_ncount || (ncount == best_ncount && p_value > best_p_value)
                    best_start = start
                    best_endpoint = endpoint
                    best_ncount = ncount
                    best_p_value = p_value
                    best_slope = slope
                    best_slope_err = slope_err
                    best_intercept = intercept
                    best_intercept_err = intercept_err
                    best_chi2 = chi2
                    best_reduced_chi2 = reduced_chi2
                    best_r2 = r2_weighted
                    best_f_stat = f_stat
                    best_bootstrap_ci = bootstrap_ci
                end
            end
        end
    end
    
    return (best_start, best_endpoint, best_ncount, best_slope, best_slope_err, 
            best_intercept, best_intercept_err, best_p_value, best_r2, 
            best_chi2, best_reduced_chi2, best_bootstrap_ci) #best_f_stat, 
end

# ============================================================================
# HELPER: Analyze function with bootstrap
# ============================================================================

function analyze_largest_flat_interval_bootstrap(x, y, y_err; 
                                                           x_err=nothing,
                                                           min_points=3, 
                                                           max_points=length(x), 
                                                           alpha=0.99,
                                                           n_bootstrap=500,
                                                           bootstrap_method=:residual)
    """
    Analyze the largest flat interval with bootstrap p-values.
    """
    println("="^80)
    println("LARGEST FLAT INTERVAL ANALYSIS (BOOTSTRAP)")
    println("="^80)
    println("Significance level: α = $alpha")
    println("Null hypothesis: slope = 0 (flat line)")
    println("Bootstrap method: $bootstrap_method")
    println("Number of bootstrap samples: $n_bootstrap")
    println("-"^80)
    
    # Find the largest flat interval
    result = find_largest_flat_interval_bootstrap(
        x, y, y_err,
        x_err=x_err,
        min_points=min_points,
        max_points=max_points,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        bootstrap_method=bootstrap_method
    )
    
    start_idx, endpoint_idx, ncount, slope, slope_err, intercept, intercept_err, 
    p_value, r2, chi2, reduced_chi2, bootstrap_ci = result
    
    if ncount == 0
        println("❌ No flat intervals found at α = $alpha")
        println("Try increasing alpha (e.g., 0.95) or reducing min_points")
        return nothing
    end
    
    println("\n✅ FOUND FLAT INTERVAL!")
    println("-"^80)
    println("Interval indices: $start_idx to $endpoint_idx")
    println("Number of points: $ncount")
    println("\nFit parameters:")
    println("  Slope     = $(round(slope, digits=6)) ± $(round(slope_err, digits=6))")
    println("  Intercept = $(round(intercept, digits=6)) ± $(round(intercept_err, digits=6))")
    println("  R²        = $(round(r2, digits=6))")
    println("\nGoodness of fit:")
    println("  χ²        = $(round(chi2, digits=6))")
    println("  χ²/dof    = $(round(reduced_chi2, digits=6))")
    #println("  F-stat    = $(round(f_stat, digits=6))")
    
    println("\n📊 BOOTSTRAP RESULTS:")
    println("-"^80)
    println("  Bootstrap p-value: $(round(p_value, digits=6))")
    println("  95% Bootstrap CI: [$(round(bootstrap_ci[1], digits=6)), $(round(bootstrap_ci[2], digits=6))]")
    
    println("\n📊 STATISTICAL INTERPRETATION:")
    println("-"^80)
    if p_value >= alpha
        println("✓ p-value ($(round(p_value, digits=4))) ≥ α ($alpha)")
        println("✓ CANNOT reject H₀: slope = 0")
        println("✓ This interval is STATISTICALLY FLAT")
    else
        println("✗ p-value ($(round(p_value, digits=4))) < α ($alpha)")
        println("✗ REJECT H₀: slope ≠ 0")
        println("✗ This interval has a SIGNIFICANT TREND")
    end
    
    # Check if zero is in the bootstrap CI
    if bootstrap_ci[1] <= 0 <= bootstrap_ci[2]
        println("\n✓ 0 is within the 95% bootstrap confidence interval")
        println("  → Slope is consistent with zero")
    else
        println("\n✗ 0 is NOT within the 95% bootstrap confidence interval")
        println("  → Slope is significantly different from zero")
    end
    
    # Number of standard deviations
    slope_sigma = abs(slope / slope_err)
    println("\n📐 UNCERTAINTY ANALYSIS:")
    println("-"^80)
    println("Slope is $(round(slope_sigma, digits=2)) σ away from zero")
    if slope_sigma < 1
        println("✓ Slope is consistent with zero within 1σ")
    elseif slope_sigma < 2
        println("→ Slope is within 2σ of zero")
    else
        println("⚠ Slope is more than 2σ away from zero")
    end
    
    println("="^80)
    
    return (start_idx, endpoint_idx, ncount, slope, slope_err, intercept, intercept_err, 
            p_value, r2, chi2, reduced_chi2, bootstrap_ci)
    
end


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# Create test data with a flat region
#x = 1:100
#y = 10 .+ 0.001 * x .+ 0.1 * randn(100)  # Almost flat
#y_err = 0.1 * ones(100)

# Add a truly flat region
#y[30:70] .= 10.5 .+ 0.01 * randn(41)

# Analyze with bootstrap
#result = analyze_largest_flat_interval_weighted_bootstrap(
#    x, y, y_err,
#    min_points=5,
#    max_points=50,
#    alpha=0.95,  # 95% confidence
#    n_bootstrap=200,
#    bootstrap_method=:residual
#)

# Compare with F-test version
#println("\n" * "="^80)
#println("COMPARISON: F-test vs Bootstrap")
#println("="^80)

# Original F-test version
#function find_largest_flat_interval_ftest(x, y, y_err; x_err=nothing, min_points=3, max_points=length(x), alpha=0.99)
    # ... (your original function here) ...
    # For demonstration, we'll just use the bootstrap result
#    nothing
#end

# The bootstrap version returns additional information
#if result !== nothing
#    println("\nBootstrap p-value: $(round(result[8], digits=4))")
#    println("F-statistic (for reference): $(round(result[11], digits=4))")
#    println("95% Bootstrap CI: [$(round(result[13][1], digits=4)), $(round(result[13][2], digits=4))]")
#end


using StatsBase

function find_all_flat_intervals_bootstrap(x, y, y_err; min_points=3, max_points=length(x), 
                                           alpha=0.99, n_bootstrap=1000)
    n = length(x)
    flat_intervals = []
    
    max_endpoint = min(n, max_points)
    
    for start in 1:n-min_points+1
        actual_max_endpoint = min(n, start + max_points - 1)
        
        for endpoint in start+min_points-1:actual_max_endpoint
            ncount = endpoint - start + 1
            x_sub = x[start:endpoint]
            y_sub = y[start:endpoint]
            y_err_sub = y_err[start:endpoint]
            
            # Calculate observed slope
            X = [ones(ncount) x_sub]
            coeffs = X \ y_sub
            slope_obs = coeffs[2]
            
            # Bootstrap to get p-value
            boot_slopes = zeros(n_bootstrap)
            
            for b in 1:n_bootstrap
                # Generate bootstrap sample
                idx = sample(1:ncount, ncount, replace=true)
                y_boot = y_sub[idx] .+ y_err_sub[idx] .* randn(ncount)
                
                # Fit bootstrap sample
                coeffs_boot = X \ y_boot
                boot_slopes[b] = coeffs_boot[2]
            end
            
            # Two-sided bootstrap p-value
            # H0: slope = 0
            p_value = 2 * min(
                mean(boot_slopes .<= 0),
                mean(boot_slopes .>= 0)
            )
            
            if p_value >= alpha
                push!(flat_intervals, (start, endpoint, ncount, slope_obs, 
                                       std(boot_slopes), p_value))
            end
        end
    end
    
    sort!(flat_intervals, by = x -> (-x[3], -x[6]))
    return flat_intervals
end

function find_largest_flatest_interval_weighted(x, y, y_err; x_err=nothing, min_points=3, max_points=length(x), alpha=0.99)
    """
    Find the LARGEST consecutive interval where slope is closest to 0 and we rejected H₀
    using weighted least squares with uncertainties.
    
    Returns the interval with the most points that has slope closest to zero.
    """
    n = length(x)
    
    # Precompute prefix sums for fast weighted statistics
    # We need: W, Wx, Wy, Wx2, Wxy
    prefix_W = [0.0]
    prefix_Wx = [0.0]
    prefix_Wy = [0.0]
    prefix_Wx2 = [0.0]
    prefix_Wxy = [0.0]
    
    # Compute weights
    if x_err !== nothing
        # Need initial slope estimate for effective variance
        # Use unweighted fit first
        X = [ones(n) x]
        coeffs = X \ y
        slope_init = coeffs[2]
        weights = 1 ./ (y_err.^2 .+ (slope_init * x_err).^2)
    else
        weights = 1 ./ y_err.^2
    end
    
    # Build prefix sums
    for i in 1:n
        wi = weights[i]
        xi = x[i]
        yi = y[i]
        
        push!(prefix_W, prefix_W[end] + wi)
        push!(prefix_Wx, prefix_Wx[end] + wi * xi)
        push!(prefix_Wy, prefix_Wy[end] + wi * yi)
        push!(prefix_Wx2, prefix_Wx2[end] + wi * xi^2)
        push!(prefix_Wxy, prefix_Wxy[end] + wi * xi * yi)
    end
    
    best_start = 1
    best_endpoint = min_points
    best_ncount = 0
    best_p_value = -Inf
    best_slope = 0.0
    best_slope_err = 0.0
    best_intercept = 0.0
    best_intercept_err = 0.0
    best_chi2 = 0.0
    best_reduced_chi2 = 0.0
    best_r2 = 0.0
    best_f_stat = 0.0
    
    max_endpoint = min(n, max_points)
    
    #println("Searching for largest flat interval with uncertainties (α = $alpha)...")
    #println("Testing all intervals with at least $min_points points...")
    #println("Weighted regression using $(x_err === nothing ? "y-errors only" : "x and y errors")")
    
    for start in 1:n-min_points+1
        actual_max_endpoint = min(n, start + max_points - 1)
        
        for endpoint in start+min_points-1:actual_max_endpoint
            ncount = endpoint - start + 1
            
            # Get range statistics from prefix sums
            W = prefix_W[endpoint+1] - prefix_W[start]
            Wx = prefix_Wx[endpoint+1] - prefix_Wx[start]
            Wy = prefix_Wy[endpoint+1] - prefix_Wy[start]
            Wx2 = prefix_Wx2[endpoint+1] - prefix_Wx2[start]
            Wxy = prefix_Wxy[endpoint+1] - prefix_Wxy[start]
            
            # Skip if not enough weight
            if W == 0
                continue
            end
            
            # Weighted fit for this interval
            Δ = W * Wx2 - Wx^2
            
            if Δ == 0
                continue
            end
            
            slope = (W * Wxy - Wx * Wy) / Δ
            intercept = (Wy * Wx2 - Wx * Wxy) / Δ
            
            # Uncertainties
            slope_err = sqrt(W / Δ)
            intercept_err = sqrt(Wx2 / Δ)
            
            # Get data for this interval
            x_interval = x[start:endpoint]
            y_interval = y[start:endpoint]
            y_err_interval = y_err[start:endpoint]
            
            # Calculate residuals and chi2
            y_pred = intercept .+ slope * x_interval
            residuals = y_interval .- y_pred
            
            if x_err !== nothing
                x_err_interval = x_err[start:endpoint]
                eff_err = sqrt.(y_err_interval.^2 .+ (slope * x_err_interval).^2)
                chi2 = sum((residuals ./ eff_err).^2)
            else
                chi2 = sum((residuals ./ y_err_interval).^2)
            end
            
            dof = ncount - 2
            reduced_chi2 = chi2 / dof
            
            # Weighted R²
            y_mean_weighted = Wy / W
            ss_tot = sum((1 ./ y_err_interval.^2) .* (y_interval .- y_mean_weighted).^2)
            ss_res = sum((1 ./ y_err_interval.^2) .* residuals.^2)
            r2_weighted = 1 - ss_res / ss_tot
            
            # F-test for slope significance
            if dof > 0 && ss_res > 0
                ss_reg = sum((1 ./ y_err_interval.^2) .* (y_pred .- y_mean_weighted).^2)
                f_stat = (ss_reg / 1) / (ss_res / dof)
                p_value = 1 - cdf(FDist(1, dof), f_stat)
            else
                f_stat = 0.0
                p_value = 1.0
            end
            
            # Check if this interval is flat (cannot reject H₀)
            #if p_value >= alpha
                # We want the LARGEST interval first, then highest p-value
            if ncount > best_ncount || (ncount == best_ncount && slope < best_slope)
                best_start = start
                best_endpoint = endpoint
                best_ncount = ncount
                best_p_value = p_value
                best_slope = slope
                best_slope_err = slope_err
                best_intercept = intercept
                best_intercept_err = intercept_err
                best_chi2 = chi2
                best_reduced_chi2 = reduced_chi2
                best_r2 = r2_weighted
                best_f_stat = f_stat
            end
            #end
        end
    end
    
    return (best_start, best_endpoint, best_ncount, best_slope, best_slope_err, 
            best_intercept, best_intercept_err, best_p_value, best_r2, 
            best_chi2, best_reduced_chi2, best_f_stat)
end

using StatsBase, Distributions

#https://stats.stackexchange.com/questions/260708/non-parametric-test-for-regression-slope0

function find_largest_flattest_interval_bootstrap(x, y, y_err; 
                                                  x_err=nothing, 
                                                  min_points=3, 
                                                  max_points=length(x), 
                                                  n_bootstrap=500,
                                                  bootstrap_method=:residual,
                                                  prioritize_size=true)
    """
    Find the LARGEST consecutive interval where slope is closest to 0.
    Uses weighted least squares with uncertainties and bootstrap p-values.
    
    Returns the interval with the most points that has the slope closest to zero.
    
    Parameters:
    - prioritize_size: If true, maximizes interval size first, then minimizes slope.
                       If false, minimizes slope first, then maximizes size.
    """
    n = length(x)
    
    # Precompute prefix sums for fast weighted statistics
    prefix_W = [0.0]
    prefix_Wx = [0.0]
    prefix_Wy = [0.0]
    prefix_Wx2 = [0.0]
    prefix_Wxy = [0.0]
    
    # Compute weights
    if x_err !== nothing
        # Need initial slope estimate for effective variance
        X = [ones(n) x]
        coeffs = X \ y
        slope_init = coeffs[2]
        weights = 1 ./ (y_err.^2 .+ (slope_init * x_err).^2)
    else
        weights = 1 ./ y_err.^2
    end
    
    # Build prefix sums
    for i in 1:n
        wi = weights[i]
        xi = x[i]
        yi = y[i]
        
        push!(prefix_W, prefix_W[end] + wi)
        push!(prefix_Wx, prefix_Wx[end] + wi * xi)
        push!(prefix_Wy, prefix_Wy[end] + wi * yi)
        push!(prefix_Wx2, prefix_Wx2[end] + wi * xi^2)
        push!(prefix_Wxy, prefix_Wxy[end] + wi * xi * yi)
    end
    
    best_start = 1
    best_endpoint = min_points
    best_ncount = 0
    best_slope = Inf
    best_slope_err = 0.0
    best_p_value = 0.0
    best_bootstrap_p = 0.0
    best_intercept = 0.0
    best_intercept_err = 0.0
    best_chi2 = 0.0
    best_reduced_chi2 = 0.0
    best_r2 = 0.0
    best_f_stat = 0.0
    best_bootstrap_ci = (0.0, 0.0)
    
    max_endpoint = min(n, max_points)
    
    for start in 1:n-min_points+1
        actual_max_endpoint = min(n, start + max_points - 1)
        
        for endpoint in start+min_points-1:actual_max_endpoint
            ncount = endpoint - start + 1
            
            # Get range statistics from prefix sums
            W = prefix_W[endpoint+1] - prefix_W[start]
            Wx = prefix_Wx[endpoint+1] - prefix_Wx[start]
            Wy = prefix_Wy[endpoint+1] - prefix_Wy[start]
            Wx2 = prefix_Wx2[endpoint+1] - prefix_Wx2[start]
            Wxy = prefix_Wxy[endpoint+1] - prefix_Wxy[start]
            
            # Skip if not enough weight
            if W == 0
                continue
            end
            
            # Weighted fit for this interval
            Δ = W * Wx2 - Wx^2
            
            if Δ == 0
                continue
            end
            
            slope = (W * Wxy - Wx * Wy) / Δ
            intercept = (Wy * Wx2 - Wx * Wxy) / Δ
            
            # Uncertainties
            slope_err = sqrt(W / Δ)
            intercept_err = sqrt(Wx2 / Δ)
            
            # Get data for this interval
            x_interval = x[start:endpoint]
            y_interval = y[start:endpoint]
            y_err_interval = y_err[start:endpoint]
            
            # Calculate residuals and chi2
            y_pred = intercept .+ slope * x_interval
            residuals = y_interval .- y_pred
            
            if x_err !== nothing
                x_err_interval = x_err[start:endpoint]
                eff_err = sqrt.(y_err_interval.^2 .+ (slope * x_err_interval).^2)
                chi2 = sum((residuals ./ eff_err).^2)
            else
                chi2 = sum((residuals ./ y_err_interval).^2)
            end
            
            dof = ncount - 2
            reduced_chi2 = chi2 / dof
            
            # Weighted R²
            y_mean_weighted = Wy / W
            ss_tot = sum((1 ./ y_err_interval.^2) .* (y_interval .- y_mean_weighted).^2)
            ss_res = sum((1 ./ y_err_interval.^2) .* residuals.^2)
            r2_weighted = 1 - ss_res / ss_tot
            
            # F-test for slope significance (for reference)
            if dof > 0 && ss_res > 0
                ss_reg = sum((1 ./ y_err_interval.^2) .* (y_pred .- y_mean_weighted).^2)
                f_stat = (ss_reg / 1) / (ss_res / dof)
                f_p_value = 1 - cdf(FDist(1, dof), f_stat)
            else
                f_stat = 0.0
                f_p_value = 1.0
            end
            
            # ================================================================
            # BOOTSTRAP P-VALUE CALCULATION: H0: m=0
            # ================================================================
            
            # Skip bootstrap if not enough points
            if ncount < 4
                bootstrap_p = 1.0  # With few points, we can't reject H₀
                bootstrap_ci = (0.0, 0.0)
            else
                # Prepare data for bootstrap
                X_local = [ones(ncount) x_interval]
                
                # Perform bootstrap
                boot_slopes = zeros(n_bootstrap)
                
                for b in 1:n_bootstrap
                    if bootstrap_method == :residual
                        # Residual bootstrap (recommended) https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Resampling_residuals ; https://slcodia-stat-142.share.connect.posit.cloud/bootstrapmodel.html#semiparametric-bootstrap-regression
                        idx = sample(1:ncount, ncount, replace=true) # reshuffle values (with replacement: might be sampled more than once)
                        # Generate new y under H₀ (slope = 0)
                        # Use the same x but with resampled residuals
                        y_centered = y_interval .- slope * x_interval # residuals
                        y_boot = y_centered[idx] .+ y_err_interval[idx] .* randn(ncount) # Creás datos sintéticos con los residuos
                        
                    elseif bootstrap_method == :parametric # assumes normal errors
                        # Parametric bootstrap
                        sigma = sqrt(chi2 / dof)
                        y_boot = y_interval .- slope * x_interval .+ randn(ncount) * sigma
                        
                    elseif bootstrap_method == :wild
                        # Wild bootstrap (for heteroscedasticity)
                        # Uses Rademacher distribution: ±1 with equal probability
                        signs = 2 * rand(Bernoulli(0.5), ncount) .- 1
                        y_boot = y_interval .- slope * x_interval .+ signs .* residuals
                        
                    else
                        error("Unknown bootstrap method: $bootstrap_method")
                    end
                    
                    # Fit bootstrap sample with weights
                    weights_boot = weights[start:endpoint]
                    W_boot = sum(weights_boot)
                    Wx_boot = sum(weights_boot .* x_interval)
                    Wy_boot = sum(weights_boot .* y_boot)
                    Wx2_boot = sum(weights_boot .* x_interval.^2)
                    Wxy_boot = sum(weights_boot .* x_interval .* y_boot)
                    
                    Δ_boot = W_boot * Wx2_boot - Wx_boot^2
                    
                    if Δ_boot != 0
                        boot_slopes[b] = (W_boot * Wxy_boot - Wx_boot * Wy_boot) / Δ_boot
                    else
                        boot_slopes[b] = 0.0
                    end
                end
                
                # Calculate bootstrap p-value (two-sided)
                bootstrap_p = mean(abs.(boot_slopes) .>= abs(slope))
                
                # Calculate bootstrap confidence intervals
                alpha_ci = 0.05  # 95% CI
                ci_lower = quantile(boot_slopes, alpha_ci/2)
                ci_upper = quantile(boot_slopes, 1 - alpha_ci/2)
                bootstrap_ci = (ci_lower, ci_upper)
            end
            
            # ================================================================
            # SELECTION CRITERIA
            # ================================================================
            
            # Check if this interval is better (larger size and flatter slope)
            is_better = false
            
            if prioritize_size
                # First prioritize size, then slope closeness to zero
                if ncount > best_ncount
                    is_better = true
                elseif ncount == best_ncount && abs(slope) < abs(best_slope)
                    is_better = true
                elseif ncount == best_ncount && abs(slope) == abs(best_slope) && bootstrap_p > best_bootstrap_p
                    # Tie-breaker: higher p-value (more evidence of flatness)
                    is_better = true
                end
            else
                # First prioritize slope closeness to zero, then size
                if abs(slope) < abs(best_slope)
                    is_better = true
                elseif abs(slope) == abs(best_slope) && ncount > best_ncount
                    is_better = true
                elseif abs(slope) == abs(best_slope) && ncount == best_ncount && bootstrap_p > best_bootstrap_p
                    # Tie-breaker: higher p-value
                    is_better = true
                end
            end
            
            if is_better
                best_start = start
                best_endpoint = endpoint
                best_ncount = ncount
                best_slope = slope
                best_slope_err = slope_err
                best_p_value = f_p_value
                best_bootstrap_p = bootstrap_p
                best_intercept = intercept
                best_intercept_err = intercept_err
                best_chi2 = chi2
                best_reduced_chi2 = reduced_chi2
                best_r2 = r2_weighted
                best_f_stat = f_stat
                best_bootstrap_ci = bootstrap_ci
            end
        end
    end
    
    return (best_start, best_endpoint, best_ncount, best_slope, best_slope_err, 
            best_intercept, best_intercept_err, best_bootstrap_p, best_r2, 
            best_chi2, best_reduced_chi2, #best_f_stat, 
            best_bootstrap_ci,
            #best_p_value
            )  # Also return F-test p-value for comparison
end

function analyze_largest_flat_interval_weighted(x, y, y_err; x_err=nothing, min_points=3, max_points=length(x), alpha=0.99)
    """
    Analyze the largest flat interval with uncertainty information.
    """
    println("="^80)
    println("LARGEST FLAT INTERVAL ANALYSIS WITH UNCERTAINTIES")
    println("="^80)
    println("Significance level: α = $alpha")
    println("Null hypothesis: slope = 0 (flat line)")
    println("We accept intervals where p-value ≥ $alpha (cannot reject H₀)")
    println("Uncertainties: $(x_err === nothing ? "y-errors only" : "x and y errors")")
    println("-"^80)
    
    # Find the largest flat interval
    result = find_largest_flat_interval_weighted(
        x, y, y_err, x_err=x_err, min_points=min_points, 
        max_points=max_points, alpha=alpha
    )
    
    start_idx, endpoint_idx, ncount, slope, slope_err, intercept, intercept_err, 
    p_value, r2, chi2, reduced_chi2, f_stat = result
    
    if ncount == 0
        println("❌ No flat intervals found at α = $alpha")
        println("Try increasing alpha (e.g., 0.99) or reducing min_points")
        return nothing
    end
    
    println("\n✅ FOUND FLAT INTERVAL!")
    println("-"^80)
    println("Interval indices: $start_idx to $endpoint_idx")
    println("Number of points: $ncount")
    println("\nFit parameters:")
    println("  Slope     = $(round(slope, digits=6)) ± $(round(slope_err, digits=6))")
    println("  Intercept = $(round(intercept, digits=6)) ± $(round(intercept_err, digits=6))")
    println("  R²        = $(round(r2, digits=6))")
    println("\nGoodness of fit:")
    println("  χ²        = $(round(chi2, digits=6))")
    println("  χ²/dof    = $(round(reduced_chi2, digits=6))")
    println("  F-stat    = $(round(f_stat, digits=6))")
    println("  P-value   = $(round(p_value, digits=6))")
    
    println("\n📊 STATISTICAL INTERPRETATION:")
    println("-"^80)
    if p_value >= alpha
        println("✓ p-value ($(round(p_value, digits=4))) ≥ α ($alpha)")
        println("✓ CANNOT reject H₀: slope = 0")
        println("✓ This interval is STATISTICALLY FLAT")
        println("  (There is no significant evidence of a trend)")
    else
        println("✗ p-value ($(round(p_value, digits=4))) < α ($alpha)")
        println("✗ REJECT H₀: slope ≠ 0")
        println("✗ This interval has a SIGNIFICANT TREND")
    end
    
    # Check if slope is consistent with zero within uncertainties
    slope_sigma = abs(slope / slope_err)
    println("\n📐 UNCERTAINTY ANALYSIS:")
    println("-"^80)
    println("Slope is $(round(slope_sigma, digits=2)) σ away from zero")
    if slope_sigma < 1
        println("✓ Slope is consistent with zero within 1σ")
    elseif slope_sigma < 2
        println("→ Slope is within 2σ of zero")
    else
        println("⚠ Slope is more than 2σ away from zero")
    end
    
    # Check reduced chi²
    println("\n📊 GOODNESS OF FIT:")
    println("-"^80)
    if reduced_chi2 < 1
        println("✓ χ²/dof = $(round(reduced_chi2, digits=3)) < 1 (fit might be overestimated)")
    elseif reduced_chi2 < 2
        println("✓ χ²/dof = $(round(reduced_chi2, digits=3)) (good fit)")
    else
        println("⚠ χ²/dof = $(round(reduced_chi2, digits=3)) > 2 (poor fit or underestimated errors)")
    end
    
    println("="^80)
    
    return (start_idx, endpoint_idx, ncount, slope, slope_err, intercept, intercept_err, 
            p_value, r2, chi2, reduced_chi2, f_stat)
end

# Function to find all flat intervals with uncertainties
function find_all_flat_intervals_weighted(x, y, y_err; x_err=nothing, min_points=3, max_points=length(x), alpha=0.99)
    """
    Find ALL intervals where we cannot reject H₀: slope = 0 with uncertainties.
    Returns sorted by interval size (largest first), then by p-value.
    """
    n = length(x)
    flat_intervals = []
    
    # Compute weights
    if x_err !== nothing
        X = [ones(n) x]
        coeffs = X \ y
        slope_init = coeffs[2]
        weights = 1 ./ (y_err.^2 .+ (slope_init * x_err).^2)
    else
        weights = 1 ./ y_err.^2
    end
    
    # Prefix sums
    prefix_W = [0.0]
    prefix_Wx = [0.0]
    prefix_Wy = [0.0]
    prefix_Wx2 = [0.0]
    prefix_Wxy = [0.0]
    
    for i in 1:n
        wi = weights[i]
        xi = x[i]
        yi = y[i]
        
        push!(prefix_W, prefix_W[end] + wi)
        push!(prefix_Wx, prefix_Wx[end] + wi * xi)
        push!(prefix_Wy, prefix_Wy[end] + wi * yi)
        push!(prefix_Wx2, prefix_Wx2[end] + wi * xi^2)
        push!(prefix_Wxy, prefix_Wxy[end] + wi * xi * yi)
    end
    
    max_endpoint = min(n, max_points)
    
    for start in 1:n-min_points+1
        actual_max_endpoint = min(n, start + max_points - 1)
        
        for endpoint in start+min_points-1:actual_max_endpoint
            ncount = endpoint - start + 1
            
            W = prefix_W[endpoint+1] - prefix_W[start]
            Wx = prefix_Wx[endpoint+1] - prefix_Wx[start]
            Wy = prefix_Wy[endpoint+1] - prefix_Wy[start]
            Wx2 = prefix_Wx2[endpoint+1] - prefix_Wx2[start]
            Wxy = prefix_Wxy[endpoint+1] - prefix_Wxy[start]
            
            if W == 0
                continue
            end
            
            Δ = W * Wx2 - Wx^2
            
            if Δ == 0
                continue
            end
            
            slope = (W * Wxy - Wx * Wy) / Δ
            slope_err = sqrt(W / Δ)
            
            # Calculate residuals and chi2
            x_interval = x[start:endpoint]
            y_interval = y[start:endpoint]
            y_err_interval = y_err[start:endpoint]
            
            y_pred = (Wy * Wx2 - Wx * Wxy) / Δ .+ slope * x_interval
            residuals = y_interval .- y_pred
            
            if x_err !== nothing
                x_err_interval = x_err[start:endpoint]
                eff_err = sqrt.(y_err_interval.^2 .+ (slope * x_err_interval).^2)
                chi2 = sum((residuals ./ eff_err).^2)
            else
                chi2 = sum((residuals ./ y_err_interval).^2)
            end
            
            dof = ncount - 2
            
            if dof > 0 && chi2 > 0
                # For F-test using chi2
                # Calculate chi2_null (fit with slope=0)
                y_mean_weighted = Wy / W
                chi2_null = sum((1 ./ y_err_interval.^2) .* (y_interval .- y_mean_weighted).^2)
                
                # F-statistic from chi2 comparison
                f_stat = ((chi2_null - chi2) / 1) / (chi2 / dof)
                p_value = 1 - cdf(FDist(1, dof), f_stat)
            else
                f_stat = 0.0
                p_value = 1.0
            end
            
            if p_value >= alpha
                push!(flat_intervals, (start, endpoint, ncount, slope, slope_err, p_value, f_stat, chi2))
            end
        end
    end
    
    # Sort by: 1) size (descending), 2) p-value (descending)
    sort!(flat_intervals, by = x -> (-x[3], -x[6]))
    
    return flat_intervals
end




"""
# Demonstration
println("\n" * "="^80)
println("DEMONSTRATION: Weighted analysis with uncertainties")
println("="^80)

# Create test data with uncertainties
x = 1:150

# True values with flat regions
y_true = zeros(150)
y_true[20:50] .= 5.0   # Flat region 1 (31 points)
y_true[80:130] .= 8.0  # Flat region 2 (51 points) - LARGER
y_true[1:19] .= 3.0 .+ 0.05 * (1:19)
y_true[51:79] .= 5.5 .+ 0.03 * (1:29)
y_true[131:150] .= 8.5 .+ 0.02 * (1:20)

# Add noise with varying uncertainties
y_err = 0.1 .+ 0.05 * rand(150)  # Different errors for each point
y = y_true .+ y_err .* randn(150)

# Also create x uncertainties (optional)
x_err = 0.05 * ones(150)  # Small x errors

# Analyze with y-errors only
println("\n" * "="^80)
println("ANALYSIS WITH Y-ERRORS ONLY")
println("="^80)

# Analyze with both x and y errors
println("\n" * "="^80)
println("ANALYSIS WITH X AND Y ERRORS")
println("="^80)
result_xy = analyze_largest_flat_interval_weighted(
    x, y, y_err, x_err=x_err,
    min_points=5, max_points=100, alpha=0.99
)
"""
# Visualization
using Plots

function get_largest_flat_interval(x,y,y_err)
    line=true
    alpha = 0.95
    result_yonly = analyze_largest_flat_interval_bootstrap(
        x, y, y_err, 
        min_points=5, max_points=length(x), alpha=alpha
    )
    tries = 1
    while result_yonly == nothing && tries < 5
        alpha -= 0.05
        result_yonly = analyze_largest_flat_interval_bootstrap(
            x, y, y_err, 
            min_points=5, max_points=length(x), alpha=alpha
        )
        tries+=1
    end
    while result_yonly == nothing && tries <10
        result_yonly = find_largest_flattest_interval_bootstrap(
            x, y, y_err, x_err=nothing, min_points=5, 
            max_points=length(x), alpha=alpha
        )
        tries+=1
        line = false
    end

    if result_yonly !== nothing
        start_idx, endpoint_idx, ncount, slope, slope_err, intercept, intercept_err, 
        p_value, r2, chi2, reduced_chi2, f_stat = result_yonly

        y_true = slope.*x .+ intercept
        
        # Plot with error bars
        p1 = scatter(x, y, yerr=y_err, label="Data with errors", 
                     markershape=:circle, markersize=3, color=:blue, alpha=0.7, legend=:best)
        
        # Highlight the selected interval
        vline!([x[start_idx], x[endpoint_idx]], #[y[start_idx], y[endpoint_idx]], 
              label="Largest flat interval (n=$ncount)", linewidth=2, color=:red)
        
        # Add fitted line with uncertainty band
        x_fit = x[start_idx:endpoint_idx]
        y_fit = slope * x_fit .+ intercept
        
        # Uncertainty band for the fit
        x_range = range(x[start_idx], x[endpoint_idx], length=100)
        y_range = slope * x_range .+ intercept
        y_upper = y_range .+ sqrt.(intercept_err^2 .+ (x_range .* slope_err).^2) #1.96 *  es confidence interval gaussiano para el 95%
        y_lower = y_range .- sqrt.(intercept_err^2 .+ (x_range .* slope_err).^2) #1.96 * 
        
        plot!(x_range, y_range, label="Weighted fit", linewidth=3, color=:green)
        #plot!(x_range, y_upper, label="Uncertainties", linewidth=1, color=:green, #$(round(alpha*100;digits=2))% confidence band
        #      linestyle=:dash, alpha=0.5)
        #plot!(x_range, y_lower, label="", linewidth=1, color=:green, 
        #      linestyle=:dash, alpha=0.5)
        #plot!(x_range, y_upper, fill_range=y_lower, color=:green, alpha=0.4, label="") 
        
        title!("Mean speed per interval, with flattest interval fitted\nSlope = $(round(slope, digits=4)) ± $(round(slope_err, digits=4)), p = $(round(p_value, digits=6))")
        xlabel!("t (hs)")
        ylabel!("mean speed in [t-1,t]")
        
        # Plot 2: Show χ² distribution
        p2 = plot(1:length(x), (y .- y_true), 
                  label="Residuals", linewidth=2, color=:blue)
        hline!([mean((y .- y_true))], label="Mean residual", 
               linewidth=2, color=:red, linestyle=:dash)
        
        title!("Residuals (data - true value)")
        xlabel!("Index")
        ylabel!("Absolute residual")
        
        fig = plot(p1, p2, layout=(1,2), size=(1400, 500))
        display(fig)
        
        return p_value, alpha, start_idx, endpoint_idx, intercept, intercept_err, line, p1
    else
        return 1,1, 1, length(x), 0.0, 0.0, false, nothing
    end
    
end
#get_largest_flat_interval(x,y,y_err)

# Compare with unweighted analysis
#println("\n" * "="^80)
#println("COMPARISON: Weighted vs Unweighted Analysis")
#println("="^80)

# Quick unweighted analysis for comparison
function quick_unweighted_fit(x, y, min_points=5, max_points=100, alpha=0.99)
    n = length(x)
    best_n = 0
    best_p = -Inf
    best_s = 0
    best_start = 1
    best_end = min_points
    
    for start in 1:n-min_points+1
        for endpoint in start+min_points-1:min(n, start+max_points-1)
            ncount = endpoint - start + 1
            x_sub = x[start:endpoint]
            y_sub = y[start:endpoint]
            
            X = [ones(ncount) x_sub]
            coeffs = X \ y_sub
            slope = coeffs[2]
            
            y_pred = coeffs[1] .+ slope * x_sub
            residuals = y_sub .- y_pred
            
            if ncount > 2
                ss_reg = sum((y_pred .- mean(y_sub)).^2)
                ss_res = sum(residuals.^2)
                f_stat = (ss_reg / 1) / (ss_res / (ncount - 2))
                p_value = 1 - cdf(FDist(1, ncount - 2), f_stat)
                
                if p_value >= alpha && ncount > best_n
                    best_n = ncount
                    best_p = p_value
                    best_s = slope
                    best_start = start
                    best_end = endpoint
                end
            end
        end
    end
    
    return (best_start, best_end, best_n, best_s, best_p)
end

using GLM
function hinge_fit(x, y, σ, τ)

    h = max.(0.0, x .- τ)

    df = DataFrame(
        y = y,
        x = x,
        h = h
    )

    if σ === nothing
        model = lm(@formula(y ~ x + h), df)

        weights = ones(length(y))
    else
        weights = 1.0 ./ σ.^2
        model = lm(@formula(y ~ x + h), df, wts=weights)
    end

    β = coef(model)

    """
    Σ = vcov(model)

    se = sqrt.(diag(Σ))

    dof = dof_residual(model)

    tβ2 = β[3] / se[3]

    pβ2 = 2 * (1 - cdf(TDist(dof), abs(tβ2)))

    """
    
    intercept = β[1]
    slope1    = β[2]
    slope2    = β[2] + β[3]

    ypred = predict(model)

    χ² = sum(weights .* (y .- ypred).^2)

    dof = length(y) - length(β)

    χ²red = χ² / dof

    return (
        τ = τ,
        intercept = intercept,
        slope1 = slope1,
        slope2 = slope2,
        #Δslope = β[3],
        #se_Δslope = se[3],
        #t = tβ2,
        #p = pβ2,
        χ² = χ²,
        χ²red = χ²red,
        model = model
    )

end
function linear_fit(x,y,σ=nothing)

    df = DataFrame(x=x,y=y)

    if isnothing(σ)
        weights = ones(length(y))
        model = lm(@formula(y~x),df)
    else
        weights = 1 ./ σ.^2
        model = lm(@formula(y~x),df,wts=weights)
    end

    ypred = predict(model)

    χ² = sum(weights .* (y .- ypred).^2)

    return (
        model=model,
        χ²=χ²,
        ypred=ypred
    )

end
function exhaustive_hinge(x, y, σ=nothing; minpoints=5)

    best = nothing
    bestχ² = Inf

    for k in minpoints:(length(x)-minpoints)

        fit = hinge_fit(x, y, σ, x[k])
        linear = linear_fit(x,y,σ)

        if fit.χ² < bestχ²
            bestχ² = fit.χ²
            best = fit
        end

    end

    return best

end

function bootstrap_breakpoint_test(x, y, σ; B=5000, minpoints=5, tolerance=0.05)

    # Observed statistic
    linear = linear_fit(x,y,σ)
    segmented = exhaustive_hinge(x, y, σ; minpoints=minpoints)
    Δχ²_obs = linear.χ² - segmented.χ²

    # Bootstrap
    Δχ²_boot = zeros(B)

    for b in 1:B
        yboot = similar(y)
        for i in eachindex(y)
            yboot[i] = linear.ypred[i] + randn()*σ[i]
        end

        linear_boot = linear_fit(x,yboot,σ)
        segmented_boot = exhaustive_hinge(x, yboot, σ; minpoints=minpoints)
        Δχ²_boot[b] = linear_boot.χ² - segmented_boot.χ²
    end

    # p-value
    p = mean(Δχ²_boot .>= Δχ²_obs)

    if p>tolerance
        τ = 0.0
        intercept = coef(linear.model)[1]
        slope1 = coef(linear.model)[2]
        slope2 = 0.0
    else
        τ = segmented.τ
        intercept = segmented.intercept 
        slope1 = segmented.slope1 
        slope2 = segmented.slope2 
        if p == 0.0
            p = 1/B
        end
    end

    return (p=p, segmented=(τ, intercept, slope1, slope2))
end





function two_lines_broken_model(x, p)
    return p[1] .+ x.* p[2] .+ p[3] .* max.(0,x.-p[4])
end
function hinge_fit_mal(x,y,σ) #**no funciona xq el modelo es discontinuo

    model = two_lines_broken_model
    # Initial parameters (all ones, degree+1 coefficients)
    p0 = ones(Float64, 4)

    if typeof(σ) == Vector{Nothing}
        # Unweighted fit
        fit = curve_fit(model, x, y, p0)
        
        # Calculate R-squared
        y_pred = model(x, fit.param)
        ss_res = sum((y .- y_pred).^2)
        ss_tot = sum((y .- mean(y)).^2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # For unweighted fit, residuals are just differences
        residuals = (y .- y_pred).^2
        
    else
        # Weighted fit
        weights = 1 ./ σ.^2
        fit = curve_fit(model, x, y, Diagonal(weights), p0)
        
        # Weighted R-squared
        y_pred = model(x, fit.param)
        ss_res = sum(weights .* (y .- y_pred).^2)
        ss_tot = sum(weights .* (y .- mean(y)).^2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Weighted residuals
        residuals = weights .* (y .- y_pred).^2
    end

    
    # Get parameter errors
    errs = stderror(fit)
    
    # Return parameters, errors, residuals, and R²
    return fit.param, errs, residuals, r_squared

end
    
