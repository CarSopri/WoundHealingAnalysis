using Images
using ImageMorphology
using ImageSegmentation
using Statistics
using Logging

code_folder = @__DIR__
include(joinpath(code_folder,"fast_fitting.jl"))

# Results calculations
function get_row_widths_and_walls(im::Matrix{RGB{N0f8}}, color_wound::RGB{Float64})::Tuple{Vector{Float64}, Vector{Union{Int, Nothing}}, Vector{Union{Int, Nothing}}}
    h, _ = size(im)
    widths = zeros(Float64, h)
    left_walls = Vector{Union{Int, Nothing}}(undef, h)
    right_walls = Vector{Union{Int, Nothing}}(undef, h)

    for row in 1:h
        widths[row] = count(x -> x == color_wound, im[row, :])
        left_walls[row] = findlast(i -> im[row, i] == RGB(1.0, 0.0, 0.0) && im[row, i+1] == color_wound, 1:size(im, 2)-1)
        right_walls[row] = findfirst(i -> im[row, i] == color_wound && im[row, i+1] == RGB(0.0, 1.0, 0.0), 1:size(im, 2)-1)
    end
    return widths, left_walls, right_walls
end

# L is segment length to compute roughness
function compute_roughness(walls::Vector{Int}, L::Int)::Tuple{Float64, Float64}
    n = length(walls)
    if L == 0
        return std(walls; corrected=false), 0.0
    elseif L > n
        return -1.0, -1.0
    else
        roughness_all = [std(walls[(i-1)*L+1:i*L]; corrected=false)
                                for i in 1:div(n, L) ]
        
        return mean(roughness_all), std(roughness_all)
    end
end

function summarize_wall_data(walls::Vector{Union{Int, Nothing}}, segment_length::Int)::Tuple{Float64, Float64, Float64, Float64} #img_width::Int, 
    valid_wall_data = map(x -> x::Int, filter(!isnothing, walls)) # Get rid of nothing values (where wound is closed)
    if isempty(valid_wall_data)
        return NaN, 0.0, 0.0, 0.0 #img_width, 0.0, 0.0, 0.0 #*************creo que no necesito esto porque en realidad con valid wound ya me saco de encima este caso
    else
        μ = mean(valid_wall_data)
        σ = std(valid_wall_data)
        roughness = compute_roughness(valid_wall_data, segment_length)
        return μ, σ, roughness[1], roughness[2]
    end
end

function store_results!(
    dicts::Tuple{Dict,Dict,Dict,Dict,Dict},
    ID::String,
    width_stats::Tuple{Float64,Float64},
    left_stats::Tuple{Float64,Float64},
    right_stats::Tuple{Float64,Float64},
    left_roughness_stats::Tuple{Float64,Float64},
    right_roughness_stats::Tuple{Float64,Float64},
    valid_wound::Bool
)
    widths_distribution, left_wall, right_wall, left_roughness, right_roughness = dicts
    if valid_wound
        widths_distribution[ID] = [width_stats...]
        left_wall[ID] = [left_stats...]
        right_wall[ID] = [right_stats...]
        left_roughness[ID] = [left_roughness_stats...]
        right_roughness[ID] = [right_roughness_stats...]
    else
        widths_distribution[ID] = [-1.0, -1.0]
        left_wall[ID] = [-1.0, -1.0]
        right_wall[ID] = [-1.0, -1.0]
        left_roughness[ID] = [-1.0, -1.0]
        right_roughness[ID] = [-1.0, -1.0]
    end
end

# Plotting functions
function init_histograms()
    hist_horizontal = plot(legend=:outertopright, grid=true)
    hist_vertical = plot(legend=:outertopright, grid=true)
    return hist_horizontal, hist_vertical
end 

function add_histogram_traces!(
    hist_horizontal,
    hist_vertical,
    time::Int,
    tf::Int,
    ID::String,
    widths::Vector{Float64},
    left_wall_ID::Vector{Int},
    right_wall_ID::Vector{Int},
    mean_width::Float64,
    std_width::Float64,
    mean_left::Float64,
    std_left::Float64,
    mean_right::Float64,
    std_right::Float64
)
    color = cgrad(:rainbow)[time / tf]
    base_y = 230 + time / tf * 50

    if time in (1,5,tf)
        histogram!(hist_horizontal, widths, label=nothing, color=color, fillalpha=0.5, line=(:2, :black))
        histogram!(hist_vertical, left_wall_ID, label=nothing, color=color, fillalpha=0.5, line=(:2, :black))
        histogram!(hist_vertical, right_wall_ID, label=nothing, color=color, fillalpha=0.5, line=(:2, :black))
    end
    plot!(hist_horizontal, [mean_width, mean_width], [0, 300], color=color, linestyle=:dash, lw=2, label=ID)
    plot!(hist_horizontal, [mean_width - std_width, mean_width + std_width], [base_y, base_y], color=color, linestyle=:dash, lw=2, label=nothing)

    plot!(hist_vertical, [mean_left, mean_left], [0, 300], color=color, linestyle=:dash, lw=2, label=ID)
    plot!(hist_vertical, [mean_left - std_left, mean_left + std_left], [base_y, base_y], color=color, linestyle=:dash, lw=2, label=nothing)

    plot!(hist_vertical, [mean_right, mean_right], [0, 300], color=color, linestyle=:dash, lw=2, label=nothing)
    plot!(hist_vertical, [mean_right - std_right, mean_right + std_right], [base_y, base_y], color=color, linestyle=:dash, lw=2, label=nothing)
end

function finalize_and_display_plots!(
    hist_horizontal,
    hist_vertical,
    general_ID::AbstractString,
    main_folder::String
)
    title!(hist_horizontal, "Row width distribution")
    xlabel!(hist_horizontal, "px")

    title!(hist_vertical, "Mean wall position")
    xlabel!(hist_vertical, "px")

    display(hist_horizontal)
    display(hist_vertical)

    if general_ID == "B4_03_1_3"
        save(main_folder * "histogram.png", hist_horizontal)
        save(main_folder * "histogram_wall_positions.png", hist_vertical)
    end
end







# Initialize per-pit storage
"""
    init_pit_containers(tf)

Returns three arrays of length 6: means, stds, IDs, each of which is
a Vector{Vector{Vector{Float64}}} (for means/stds) or Vector{Vector{String}}.
"""
function init_pit_containers(tf::Int, n_of_classes::Int)
    means = [ [ Vector{Float64}[] for t in 1:tf ] for _ in 1:n_of_classes]
    stds  = [ [ Vector{Float64}[] for t in 1:tf ] for _ in 1:n_of_classes]
    IDs   = [ String[]    for _ in 1:n_of_classes]
    return means, stds, IDs
end

#***************** esta función hay que reescribirla toda para poder usarla con otras nomenclaturas !!!
function accumulate_data!(means::Vector{Vector{Vector{Vector{Float64}}}}, stds::Vector{Vector{Vector{Vector{Float64}}}}, 
                          IDs::Vector{Vector{String}}, left_wall::Dict, right_wall::Dict, pits::Vector{String}, tf::Int,n_rois::Int, invert_classification::Bool)
    
    for pit in pits
        if invert_classification
            pit_num = findfirst(x->x==first(pit),"ABCD")
            mi, si, IDlist = means[pit_num], stds[pit_num], IDs[pit_num] # this vectors are empty the first time you call them
            for t in 1:tf
                for n_photo in 1:n_rois
                    ID = "$(pit)ROI$(n_photo)_T"*lpad(t,3,'0')
                    if haskey(left_wall, ID)
                        if t == 1
                            push!(IDlist, ID)
                        else
                            if !haskey(left_wall, "$(pit)ROI$(n_photo)_T"*lpad(1,3,'0'))
                                continue
                            end
                        end
                        push!(mi[t], [left_wall[ID][1],right_wall[ID][1]])
                        push!(si[t], [left_wall[ID][2],right_wall[ID][2]])
                    else
                        if haskey(left_wall, "$(pit)ROI$(n_photo)_T"*lpad(1,3,'0'))
                            push!(mi[t], [-1,-1])
                            push!(si[t], [-1,-1])
                        end
                    end
                end
            end
        else
            pit_num = parse(Int, last(pit))
            if pit_num == 6 #********************** sólo queratinocitos de junio porque no me salió juntar condiciones
                pit_num = 1
            end
            mi, si, IDlist = means[pit_num], stds[pit_num], IDs[pit_num] # this vectors are empty the first time you call them
    
            #println("D1ROI4_T001" ∈ IDlist,"D1ROI4_T001" ∈ keys(left_wall), "D1ROI4_T021" ∈ IDlist,"D1ROI4_T021" ∈ keys(left_wall))
            for t in 1:tf
                for n_photo in 1:n_rois
                    ID = "$(pit)ROI$(n_photo)_T"*lpad(t,3,'0')
                    if haskey(left_wall, ID)
                        if t == 1
                            push!(IDlist, ID) #**no me está poniendo los que tienen mal la primera segmentación ?
                        else
                            if !haskey(left_wall, "$(pit)ROI$(n_photo)_T"*lpad(1,3,'0'))
                                continue
                            end
                        end
                        #if ID == "D1ROI4_T021"
                        #    println(ID,[left_wall[ID][1],right_wall[ID][1]])
                        #end          
                        #if pit[2] == '1'
                        #    println("t ",t," nroi ",n_photo," id ",ID)
                        #end
                        push!(mi[t], [left_wall[ID][1],right_wall[ID][1]])
                        push!(si[t], [left_wall[ID][2],right_wall[ID][2]])
                    else
                        if haskey(left_wall, "$(pit)ROI$(n_photo)_T"*lpad(1,3,'0'))
                            push!(mi[t], [-1,-1])
                            push!(si[t], [-1,-1])
                        end
                    end
                end
            end
        end
    end
end

# Plot initial‐width summary
function plot_initial_widths(IDs::Vector{Vector{String}}, left_wall::Dict, right_wall::Dict,
                             pits::Vector{String}, tf::Int64, savepath::String, n_rois::Int)
    fig = plot(legend=:outertopright, grid=true)
    kf=0
    for ID in keys(left_wall)
        if ID[1:2] in pits
            kf+=1
        end
    end
    kf/=tf
    k = 1
    for pit in pits
        pit_num = parse(Int, last(pit))
        for n_photo in 1:n_rois
            ID = "$(pit)_03_1_$(n_photo)_T"*lpad(1,3,'0') #***corregir !!!
            if haskey(left_wall, ID)
                dl, dr = left_wall[ID], right_wall[ID]
                scatter!(fig, [k], [dl[1]],
                         yerr=[dl[2]], label=ID[1:end-5],
                         color=cgrad(:rainbow)[k/kf])
                scatter!(fig, [k], [dr[1]],
                         yerr=[dr[2]], label=nothing,
                         color=cgrad(:rainbow)[k/kf])
                k+=1
            end
        end
    end
    plot!(size=(600+200*ceil(Int64,(k)/25),500), legendcolumns=ceil(Int64,(k)/25),  xlabel = "ID image",ylabel = "Initial wall position (px)" , margin=0.5cm) #400+100*ceil(Int64,(k/tf)/25)
    display(fig)
    savefig(fig, savepath)
end

function plot_initial_widths(IDs::Vector{Vector{String}}, widths::Dict,
                             pits::Vector{String}, tf::Int64, savepath::String, n_rois::Int)
    fig = plot(legend=:outertopright, grid=true)
    kf = 0
    for ID in keys(widths)
        if ID[1:2] in pits
            kf+=1
        end
    end
    kf/=tf
    k = 1
    for pit in pits
        pit_num = parse(Int, last(pit))
        for n_photo in 1:n_rois
            ID = "$(pit)_03_1_$(n_photo)_T"*lpad(1,3,'0')
            if haskey(widths, ID)
                width = widths[ID]
                scatter!(fig, [k], [width[1]],
                         yerr=[width[2]], label=ID[1:end-5],
                         color=cgrad(:rainbow)[k/kf])
                k+=1
            end
        end
    end
   
    
    plot!(size=(600+200*ceil(Int64,(k)/25),500), legendcolumns=ceil(Int64,(k)/25),  xlabel = "ID image",ylabel = "Initial width (px)" , margin=0.5cm) #400+100*ceil(Int64,(k/tf)/25)
    display(fig)
    savefig(fig, savepath)
end

function plot_pit_fits(fig, id::Int, x::Vector{Float64}, y::Vector{Float64}, 
                       dy::Union{Vector{Float64}, Vector{Nothing}}, IDs_pit::Vector{String},
                       x_lims::Vector{Float64}; x_label::String, y_label::String,
                       do_fit::Bool, subplot::Int64)
    
    color = cgrad(:rainbow)[id/length(IDs_pit)]
    
    args = (fig[subplot], x, y)
    kwargs = Dict(
        :label => (subplot == 1 || length(IDs_pit)==1 || IDs_pit[1]=="" ? "" : IDs_pit[id][1:end-5]),
        :xlabel => x_label,
        :ylabel => y_label,
        :color => color,
        :markerstrokecolor => color,
        :alpha => 0.7
    )
    if typeof(dy) != Vector{Nothing}
        kwargs[:yerr] = dy
        scatter!(args...; kwargs...) 
    else
        kwargs[:marker] = :circle
        plot!(args...; kwargs...) 
    end

    # perform and plot fit in log-log
    if do_fit 
        if IDs_pit[1] == "black"
            color = :black
        end
        m,b,(dm,_), ss_res, r2 = perform_linear_fit(x, y, dy)
        xfit = range(x_lims[1], x_lims[2], length=100)
        plot!(fig[subplot], xfit, linear_model(xfit, (m,b)),
              linestyle=:dash, linewidth=2, color=color,
              label="slope=$(value_with_error(m,dm,2))",legend_marker=false,marker_legend=false, # Make marker invisible in legend only
        markersize_legend=0,        # Makes marker size 0 in legend
        markerstrokewidth_legend=0, # Removes marker stroke in legend
        legend_margin=-0.5mm,             # Reduce margins
        legend_title_margin=-0.5mm,
        legend_entry_spacing=0.2mm)

        return m,dm, ss_res, r2
    else
        return 1.0,1.0,[1.0 for point in 1:length(y)], 0.0
    end
    
end

function plot_pit_fits2(fig, id::Int, x::Vector{Float64}, y::Vector{Float64}, 
                       dy::Union{Vector{Float64}, Vector{Nothing}}, IDs_pit::Vector{String},
                       x_lims::Vector{Float64}; x_label::String, y_label::String,
                       do_fit::Bool)
    
    color = cgrad(:rainbow)[id/length(IDs_pit)]
    
    args = (fig, x, y)
    kwargs = Dict(
        :label => (length(IDs_pit)==1 || IDs_pit[1]=="" ? "" : IDs_pit[id][1:end-5]),
        :xlabel => x_label,
        :ylabel => y_label,
        :color => color,
        :markerstrokecolor => color,
        :alpha => 0.7
    )
    if typeof(dy) != Vector{Nothing}
        kwargs[:yerr] = dy
        scatter!(args...; kwargs...) 
    else
        kwargs[:marker] = :circle
        plot!(args...; kwargs...) 
    end

    # perform and plot fit in log-log
    if do_fit 
        if IDs_pit[1] == "black"
            color = :black
        end
        m,b,(dm,_), ss_res, r2 = perform_linear_fit(x, y, dy)
        xfit = range(x_lims[1], x_lims[2], length=100)
        plot!(fig, xfit, linear_model(xfit, (m,b)),
              linestyle=:dash, linewidth=2, color=color,
              label="slope=$(value_with_error(m,dm,2))",legend_marker=false,marker_legend=false, # Make marker invisible in legend only
        markersize_legend=0,        # Makes marker size 0 in legend
        markerstrokewidth_legend=0, # Removes marker stroke in legend
        legend_margin=-0.5mm,             # Reduce margins
        legend_title_margin=-0.5mm,
        legend_entry_spacing=0.2mm)

        return m,dm, ss_res, r2
    else
        return 1.0,1.0,[1.0 for point in 1:length(y)], 0.0
    end
    
end

function plot_quadratic_fits(fig, id::Int, x::Vector{Float64}, y::Vector{Float64}, 
                       dy::Union{Vector{Float64}, Vector{Nothing}}, IDs_pit::Vector{String},
                       x_lims::Vector{Float64}; x_label::String, y_label::String,
                       do_fit::Bool, subplot::Int64)
    
    color = cgrad(:rainbow)[id/length(IDs_pit)]
    
    args = (fig[subplot], x, y)
    kwargs = Dict(
        :label => (subplot == 1 || length(IDs_pit)==1 || IDs_pit[1]=="" ? "" : IDs_pit[id][1:end-5]),
        :xlabel => x_label,
        :ylabel => y_label,
        :color => color,
        :markerstrokecolor => color,
        :alpha => 0.7
    )
    if typeof(dy) != Vector{Nothing}
        kwargs[:yerr] = dy
        scatter!(args...; kwargs...) 
    else
        kwargs[:marker] = :circle
        plot!(args...; kwargs...) 
    end

    # perform and plot fit in log-log
    if do_fit 
        if IDs_pit[1] == "black"
            color = :black
        end
        a,m,b,(da,dm,_), ss_res, r2 = perform_quadratic_fit(x, y, dy)
        plot!(fig[subplot], x, evaluate_polynomial(x, [a,m,b]),
              linestyle=:dash, linewidth=2, color=color,
              label="acc=$(value_with_error(a,da,2)), v0=$(value_with_error(m,dm,2))",# Make marker invisible in legend only
        markersize_legend=0,        # Makes marker size 0 in legend
        markerstrokewidth_legend=0) # Removes marker stroke in legend)

        return a,da,m,dm, ss_res, r2
    else
        return 1.0,1.0,1.0,1.0,[1.0 for point in 1:length(y)], 0.0
    end
    
end

function get_x_and_y_and_ylog(means_pit, stds_pit, times, id::Int64, tf::Int64, left_or_right::Int64)
    if left_or_right == 3 # get widths
        y_data = [means_pit[t][id][2]-means_pit[t][id][1] for t in 2:tf]
        y0 = means_pit[1][id][2]-means_pit[1][id][1]
        dy0 = stds_pit[1][id][2]^2+stds_pit[1][id][1]^2
        dy = [stds_pit[t][id][2]^2+stds_pit[t][id][1]^2 for t in 2:tf]
        
        good_data = [means_pit[t][id][1] for t in 1:tf] .!= -1.0
        y_data = y_data[good_data[2:end]]
        times_data = times[good_data]
        dy = dy[good_data[2:end]]
        
        y_ref_y0 = abs.(y_data .- y0)
        dylog = log10(exp(1.0)).*sqrt.(dy.^2 .+dy0.^2)./y_ref_y0 
        dy = sqrt.(dy)
        dy0 = sqrt(y0)
        xlog = log10.(times_data[2:end])
        ylog = log10.(y_ref_y0)
    else 
        #println(size(means_pit)," id: ",id)
        #println(length(means_pit[1])," len tf",length(means_pit[22])," tf: ",tf)
        #for t in 2:tf
        #println(1,[length(means_pit[t]) for t in 2:tf])
        #end
        y_data = [means_pit[t][id][left_or_right] for t in 2:tf]
        y0 = means_pit[1][id][left_or_right]
        dy0 = stds_pit[1][id][left_or_right]
        dy = [stds_pit[t][id][left_or_right] for t in 2:tf] #if y_data[t-1] != -1.0]

        good_data = [means_pit[t][id][left_or_right] for t in 2:tf] .<= (1.1*means_pit[1][id][left_or_right])
        y_data = y_data[good_data]
        times_data = vcat(times[1],times[2:end][good_data])
        dy = dy[good_data]

        times_data = times_data[vcat(y0,y_data) .!= -1.0]
        dy = dy[y_data .!= -1.0]
        y_data = y_data[y_data .!= -1.0]    
        

        
        y_ref_y0 = abs.(y_data .- y0)
        dylog = log10(exp(1.0)).*sqrt.(dy.^2 .+dy0.^2)./y_ref_y0 
        xlog = log10.(times_data[2:end])
        ylog = log10.(y_ref_y0)
    end

    if y0!=-1
        return times_data, vcat(y0, y_data), vcat(dy0,dy), xlog, ylog, dylog
    else
        return times_data, y_data, dy, xlog, ylog, dylog
    end
end


#**se puede hacer que esta función dependa de la clase ? creo q no se pueden asociar funciones a clases en julia mmmm
#**probablemente sería más prolijo que esto se tome del struct en vez de ser una función con ifs !!! en params for analyze o alguasi poner cómo separar los datos !
#****es horrible esto !!! 
function set_title(pit_number, data_struct="k")
    if data_struct == "f"
    # Fibroblasts:
        return "Pits $(pit_number), " * (pit_number ≤ 3 ? "10⁵" : "3×10⁵") * " cells, " * (pit_number in [1,5] ? "100" : pit_number in [2,6] ? "200" : "0") * " μg/ml"
    elseif data_struct == "k"
    # Keratinocytes
    #***ver qué hacer si quiero la versión invertida ??
        #if data_struct.invert == true
        #    pit_l = "ABCD"
        #    return "Pits $(pit_l[pit_number]), 1-3: 3x10⁵, 4-5: 4×10⁵ cells, " * (pit_number == 1 ? "control" : pit_number == 2 ? "hUC" : pit_number == 3 ? "100 μg/ml hAM" : pit_number == 4 ? "200 μg/ml hAM" : "0")
        #else
        return "Pits $(pit_number), A: control, B:hUC, C: 100 μg/ml hAM, D: 200 μg/ml hAM; " * (pit_number ∈ [1,2,3] ? "3x10⁵ cells" : pit_number ∈ [4,5,6] ? "4x10⁵ cells" : "0")
        #end
    end
end

function get_pit_charac(pit, invert)
    if invert
        pit_l = "ABCD"
        return findfirst(x->x==pit[1],pit_l)
    else
        return parse(Int64,pit[2])
    end
end




# **se podrá invertir el gráfico y poner w(L) para varios t?
#**para eso necesito poder invocar fácilmente w(L), y no estar recalculándolo con todo lo demás
function plot_roughness(means, which_pits, stds, IDs, exclude, times, L, with_uncertainties, save_folder)
    tf::Int64 = length(times)
    for pit_number in which_pits
        means_pit = means[pit_number]
        stds_pit = stds[pit_number]
        IDs_pit = IDs[pit_number]
        
        title = set_title(pit_number)
        
        fig = plot(layout=grid(1,2, widths=(4/8,4/8)), size=(1000,400),  margin=5mm, legend=:outertopright, grid=true)
    
        xlog_lims = [log10(times[2]),log10(times[end]+1)]
        x_lims = [minimum(times),maximum(times)]

        println(IDs_pit, length(means_pit[tf]))
        for (i,ID) in enumerate(IDs_pit)

            if ID[1:end-5] ∉ exclude
                times_data, y, dy, xlog, ylog, dylog = get_x_and_y_and_ylog(means_pit, stds_pit, times, i, tf, 1) # acá en roughness dy está lleno de 0
                if !with_uncertainties
                    dy = [nothing for n in times_data]
                end
                if length(times_data)<2
                    println(length(times_data)," ",length(y)," ",ID)
                end
                plot_pit_fits(fig, i, times_data, y, dy,
                                  IDs_pit, x_lims, x_label="t (hs)", y_label="left w (px)", 
                                  do_fit=false, subplot=1)
                times_data, y, dy, xlog, ylog, dylog = get_x_and_y_and_ylog(means_pit, stds_pit, times, i, tf, 2)
                if !with_uncertainties
                    dy = [nothing for n in times_data]
                end
                plot_pit_fits(fig, i, times_data, y, dy,
                                  IDs_pit, x_lims, x_label="t (hs)", y_label="right w (px)", 
                                  do_fit=false, subplot=2)

            else
                println(ID)
            end
    
        end
        plot!(fig, plot_title="Roughness L=$(L) "*title, legend=:outertopright)
        display(fig)
        savefig(fig,save_folder*"roughness_pit_vs_t"*string(pit_number))
    end
end



function plot_roughness_vs_L(left_roughnesses_L, right_roughnesses_L, which_pits, IDs, time, which_Ls, with_uncertainties::Bool, log_plot::Bool, save_folder::String)
    for pit_number in which_pits
        IDs_pit = IDs[pit_number]
        
       title = set_title(pit_number)
        
        fig = plot(layout=grid(1,2, widths=(4/8,4/8)), size=(1000,400),  margin=5mm, legend=:outertopright, grid=true)
    
        x_lims = [float(minimum(which_Ls)),float(maximum(which_Ls))]
        xlog_lims = [log10(minimum(which_Ls)),log10(maximum(which_Ls))]
        
        for (i,ID) in enumerate(IDs_pit)
            left_pit = [left_roughnesses_L[ID[1:end-5]*"_L"*lpad(l,4,"0")][1] for l in which_Ls] #**mmm ese 1:end-5 es malísimo
            right_pit = [right_roughnesses_L[ID[1:end-5]*"_L"*lpad(l,4,"0")][1] for l in which_Ls]

            L_data = which_Ls[left_pit.!=-1.0]
            left_pit = left_pit[left_pit.!=-1.0]
            right_pit = right_pit[right_pit.!=-1.0]

            if with_uncertainties
                dy_l = [left_roughnesses_L[ID[1:end-5]*"_L"*lpad(l,4,"0")][2] for l in L_data]
                dy_r = [right_roughnesses_L[ID[1:end-5]*"_L"*lpad(l,4,"0")][2] for l in L_data]
            else
                dy_l = [nothing for xl in L_data]
                dy_r = [nothing for xl in L_data]
            end
            if log_plot
                if with_uncertainties
                    dlogy_l = log10(exp(1.0)).*dy_l./left_pit 
                    dlogy_r = log10(exp(1.0)).*dy_r./right_pit 
                else
                    dlogy_l = [nothing for xl in L_data]
                    dlogy_r = [nothing for xl in L_data]
                end
                plot_pit_fits(fig, i, log10.(L_data), log10.(left_pit), dlogy_l,
                                  IDs_pit, xlog_lims, x_label="log10(L)", y_label="left w (log10)", 
                                  do_fit=false, subplot=1)
                plot_pit_fits(fig, i, log10.(L_data), log10.(right_pit), dlogy_r,
                                  IDs_pit, xlog_lims, x_label="log10(L)", y_label="right w (log10)", 
                                  do_fit=false, subplot=2) 
            else
                plot_pit_fits(fig, i, float.(L_data), left_pit, dy_l,
                                  IDs_pit, x_lims, x_label="L (px)", y_label="left w (px)", 
                                  do_fit=false, subplot=1)
                plot_pit_fits(fig, i, float.(L_data), right_pit, dy_r,
                                  IDs_pit, x_lims, x_label="L (px)", y_label="right w (px)", 
                                  do_fit=false, subplot=2) 
            end
    
        end
        plot!(fig, plot_title="Roughness t=$(time) "*title, legend=:outertopright,)
        display(fig)
        savefig(fig,save_folder*"roughness_pit_vs_L"*string(pit_number))
        
    end
end








using LsqFit
using LinearAlgebra
using Printf
using Measures
using Plots

function value_with_error(val::Float64, err::Float64, precision::Int=2)
    # Number of digits in the error
    err_decimals = precision - floor(Int, log10(err) + 1)

    # Output error with the desired number of significant digits
    err_out = round(err, digits=err_decimals)

    # Removes leading zeros for fractional errors
    if err_out < 1
        err_out = round(Int, err_out * 10^err_decimals)
        err_format = 0
    else
        err_format = max(err_decimals, 0)
    end

    # Format the value to have the same significant digits as the error
    val_out = round(val, digits=err_decimals)
    val_format = max(err_decimals, 0)

    return @sprintf("%.*f(%.*f)", val_format, val_out, err_format, err_out)
end

# Utility for formatting “value ± error”
"""
function value_with_error(val::Float64, err::Float64; precision::Int=2)
    err_digits = precision - floor(Int, log10(err) + 1)
    err_out    = round(err, digits=err_digits)
    val_out    = round(val, digits=err_digits)
    return @sprintf("%.*f(%.*f)", max(err_digits,0), val_out, max(err_digits,0), err_out)
end
"""

# Linear model + fit helper
#linear_model(x,p) = p[1] .* x .+ p[2]  # p[1] = slope (m), p[2] = intercept (b)

"""
    perform_fit(x, y; σ = nothing)

Fit y ~ m*x + b (possibly weighted by 1/σ^2 if σ provided).
Returns (m, b, (Δm, Δb)).
"""
function perform_fit(x::Vector, y::Vector; σ::Union{Vector{Float64}, Vector{Nothing}}, model=linear_model, p0 = [1.0, 1.0])

    if typeof(σ) == Vector{Nothing} 
        fit = LsqFit.curve_fit(model, x, y, p0)

        # Calculate R-squared
        y_pred = model(x, fit.param)
        res = (y .- y_pred).^2
        ss_res = sum(res)
        ss_tot = sum((y .- mean(y)).^2)
        r_squared = 1 - (ss_res / ss_tot)

    else
        weights = 1 ./ σ.^2
        fit = LsqFit.curve_fit(model, x, y, Diagonal(weights), p0)
        
        # Weighted R-squared for when σ is provided
        y_pred = model(x, fit.param)
        res = weights .* (y .- y_pred).^2
        ss_res = sum(res)
        ss_tot = sum(weights .* (y .- mean(y)).^2)
        r_squared = 1 - (ss_res / ss_tot)
        
    end
    errs = stderror(fit)

    
    
    return fit.param[1], fit_param[2], (errs[1],errs[2]), res, r_squared
end





function main_walls(pits, which_pits, times, tf, left_wall, right_wall, widths_distribution, main_folder, plot_widths_t1::Bool, plot_log_wall_position, plot_linear_wall_position, plot_widths_from_walls, n_rois::Int, invert::Bool, n_of_classes::Int)
    # Set up containers
    means, stds, IDs = init_pit_containers(tf, n_of_classes)
    
    # Initial widths plot
    #**acá hay dos cosas, el plot initial y el general means y stds
    accumulate_data!(means, stds, IDs, left_wall, right_wall, pits, tf, n_rois, invert)
    if plot_widths_t1
        plot_initial_widths(IDs, left_wall, right_wall, pits, tf, main_folder*"fit_initial_position_wall.png", n_rois)
    end
    #plot_initial_widths!(means, stds, IDs, left_wall, right_wall, pits, tf, main_folder*"fit_initial_position_wall.png", n_rois)
    
    # Dynamic fits
    for pit_number in which_pits
        means_pit = means[pit_number]
        stds_pit = stds[pit_number]
        IDs_pit = IDs[pit_number]
        
        title = set_title(pit_number)
        
        log_fig = plot(layout=grid(1,2, widths=(4/8,4/8)), size=(1000,400), 
                    margin=5mm, legend=:outertopright, grid=true)
        widths_fig = plot(layout=grid(1,2, widths=(4/8,4/8)), size=(1400,400), 
                    margin=5mm, legend=:outertopright, grid=true, legendcolumns=2)
        plot!(widths_fig,subplot=1,legendcolumns=1)
        linear_fig = plot(layout=grid(1,2, widths=(4/8,4/8)), size=(1400,400), 
                    margin=5mm, legend=:outertopright, grid=true, legendcolumns=2)
        plot!(linear_fig,subplot=1,legendcolumns=1)
        
        xlog_lims = [log10(times_data[2]),log10(times_data[end]+1)]
        x_lims = [minimum(times_data),maximum(times_data)]
        for (i,ID) in enumerate(IDs_pit)
           
            # Left wall
            times_data, y, dy, xlog, ylog, dylog = get_x_and_y_and_ylog(means_pit, stds_pit, times, i, tf, 1)
            if plot_log_wall_position
                    # Log plot [nothing for xl in xlog]
                plot_pit_fits(log_fig, i, xlog, ylog, dylog,
                              IDs_pit, xlog_lims,
                              x_label="log10(t)", y_label="log10(<h>-<h0>) left wall", 
                              do_fit=true, subplot=1)
            end
            if plot_linear_wall_position
                    # Linear plot
                plot_pit_fits(linear_fig, i, times_data, y, dy,
                              IDs_pit, x_lims,
                              x_label="t (hs)", y_label="left <h> (px)", 
                              do_fit=true, subplot=1)
            end
            # Right wall
            times_data, y, dy, xlog, ylog, dylog = get_x_and_y_and_ylog(means_pit, stds_pit, times, i, tf, 2)
            if plot_log_wall_position
                    # log plot
                plot_pit_fits(log_fig, i, xlog, ylog, dylog,
                              IDs_pit, xlog_lims,
                              x_label="log10(t)", y_label="log10(<h>-<h0>) right wall", 
                              do_fit=true, subplot=2)
            end
            if plot_linear_wall_position
                    # linear plot
                plot_pit_fits(linear_fig, i, times_data, y, dy,
                              IDs_pit, x_lims,
                              x_label="t (hs)", y_label="right <h> (px)", 
                              do_fit=true, subplot=2)
            end

            if plot_widths_from_walls
                # Widths plot
                times_data, y, dy, xlog, ylog, dylog = get_x_and_y_and_ylog(means_pit, stds_pit, times, i, tf, 3)
                    # Linear scale
                plot_pit_fits(widths_fig, i, times_data, y, dy,
                              IDs_pit, x_lims,
                              x_label="t (hs)", y_label="d=<hr>-<hl> (px)", 
                              do_fit=true, subplot=1)
                    # Log scale
                plot_pit_fits(widths_fig, i, xlog, ylog, dylog,
                              IDs_pit, xlog_lims,
                              x_label="log10(t)", y_label="log10(d0-d)", 
                              do_fit=true, subplot=2)
            end

        end
        if plot_log_wall_position
            plot!(log_fig, plot_title="Mean wall position "*title, legend=:outertopright,)
            display(log_fig)
        end
        if plot_linear_wall_position
            plot!(linear_fig, plot_title="Mean wall position + fit "*title, legend=:outertopright,)
            display(linear_fig)
        end
        if plot_widths_from_walls
            plot!(widths_fig, plot_title="Width (from walls) "*title, legend=:outertopright,)
            display(widths_fig)
        end
        
        if pit_number == 2
            #savefig(log_fig, main_folder*"fit_pit$(pit_number)_vertical_log_h.png")
            #savefig(widths_fig, main_folder*"fit_pit$(pit_number)_vertical_d.png")
            #savefig(linear_fig, main_folder*"fit_pit$(pit_number)_vertical_h.png")
        end
    end
end



#main_walls(pits, 1:n_of_classes, times_data, length(times_data), left_walls, right_walls, widths_distribution, main_folder, true, false, false, n_rois, invert, n_of_classes)



#**le tuve que sacar los errores porque da cualquier cosa shoro







using LsqFit
using LinearAlgebra
using Printf

function weighted_mean_and_std(m,w) # Inverse-variance weighting
    numerator = [m/w[i]^2 for (i,m) in enumerate(m)]
    denominator = sum([1/σ^2 for σ in w])
    return sum(numerator)/denominator, sqrt(1/denominator)
end


function init_report_file(path::String, time_batches::Vector{Vector{Int}})
    if isfile(path)
        df = CSV.read(path, DataFrame, delim='\t')
        for (n_batch,time_batch) in enumerate(time_batches)
            for time in time_batch 
                col_name = "T"*lpad(time, 3, "0")
                if !hasproperty(df, col_name)
                    println("Creating new column: ", col_name)
                    df[!, col_name] .= 0.0
                end
            end
        end
    else
        columns = (; ID=AbstractString[], (Symbol("T", lpad(i, 3, "0")) => Float64[] for i in 1:time_batches[end][end])...)
        df = DataFrame(columns)
    end
    return df
end

function filter_cols(IDs,col_num::Nothing, invert) #**no hace nada, está por las dudas que no querramos filtrar nada! si pasamos which_pits = nothing
    return IDs
end

function filter_cols(IDs,col_num::Vector{Int64}, invert) #**si queremos filtrar alguna condición para que no aparezca
    if invert
        idx = 2
        inverse_idx = 1
    else
        idx = 1
        inverse_idx = 2
    end
    IDs_small_rho = deepcopy(IDs)
    for (c,col) in enumerate(IDs_small_rho)
        for roi in col
            if invert
                if parse(Int,roi[idx]) ∉ col_num
                    filter!(x -> x != roi, IDs[c])
                end
            else
                if findfirst(roi[idx], "ABCD") ∉ col_num #***voy a tener que corregir esto ! para que sea más general
                    filter!(x -> x != roi, IDs[c])
                end
            end
        end
    end
    return IDs
end

function filter_cols(IDs, which_pits::Vector{Vector{Int}}, invert=false) #**si queremos filtrar versátilmente ciertos pits (se queda sólo con las columnas y filas que pedís en la combinación de ambas)
    """
    Filter rows based on BOTH number (second character) AND letter (first character)
    col_num: numbers to keep (e.g., [1,2,3])
    letters_to_keep: letters to keep (e.g., "AB" or ["A","B"])
    invert: if true, keep elements that DON'T match criteria
    """

    col_num::Vector{Int64} = which_pits[2]
    letters_to_keep::Union{String,Vector} = which_pits[1]
    
    # Convert letters to set for efficient lookup
    if letters_to_keep isa String
        letter_set = Set(collect(letters_to_keep))
    else
        letters = "ABCD" #********************** corregir!!
        letter_set = [string(letters[i]) for i in letters_to_keep]
        #letter_set = Set(collect(selected_letters))#Set(letters_to_keep)
    end
    num_set = Set(col_num)
    
    result = deepcopy(IDs)
    for (c, col) in enumerate(result)
        to_keep = Bool[]
        
        for roi in col
            # Extract first character (letter) and second character (number)
            letter = string(roi[1])
            number = parse(Int, roi[2])
            
            # Check both conditions
            letter_ok = letter ∈ letter_set
            number_ok = number ∈ num_set
            both_ok = letter_ok && number_ok

            keep = both_ok
            """
            if invert
                # Keep if NOT meeting both criteria
                keep = !both_ok
            else
                # Keep only if meeting both criteria
                keep = both_ok
            end
            """
            
            push!(to_keep, keep)
        end
        
        # Apply filter
        result[c] = result[c][to_keep]
    end
    return result
end

function filter_cols(IDs,col_num::Tuple{Vector{Int64}, Vector{Int64}}, invert) #**le pasás exactamente el índice que querés
    if invert
        idx = 2
        inverse_idx = 1
    else
        idx = 1
        inverse_idx = 2
    end
    IDs_small_rho = deepcopy(IDs)
    for (c,col) in enumerate(IDs_small_rho)
        for roi in col
            if invert
                if parse(Int,roi[idx]) ∉ col_num[inverse_idx]
                    filter!(x -> x != roi, IDs[c])
                elseif parse(Int,roi[end-5]) != col_num[idx][findfirst(x->x==parse(Int,roi[idx]),col_num[inverse_idx])] #if parse(Int,roi[end-5])  col_num[2]
                    filter!(x -> x != roi, IDs[c])
                end
            else
                if findfirst(roi[idx], "ABCD") ∉ col_num[inverse_idx] #***acomodar esto para que sea más general
                    filter!(x -> x != roi, IDs[c])
                elseif roi[end-4] != col_num[idx][findfirst(x->x==findfirst(roi[idx],"ABCD"),col_num[inverse_idx])] #if parse(Int,roi[end-5])  col_num[2]
                    filter!(x -> x != roi, IDs[c])
                end
            end
        end
    end
    return IDs
end


function best_two_line_fit(x, y, dy) 
    n = length(x)
    best_i = 0
    best_ssr = Inf
    best_fit1 = nothing
    best_fit2 = nothing
    errs1 = nothing
    errs2 = nothing

    for i in 3:(n-3)  # avoid very small segments
        # Fit first segment
        model(x, p) = p[1] .* x .+ p[2]
        fit1 = LsqFit.curve_fit(linear_model, x[1:i], y[1:i], Diagonal(1 ./ dy[1:i].^2), [1.0, 0.0]) #curve_fit(model, x[1:i], y[1:i], [1.0, 0.0])
        fit2 = LsqFit.curve_fit(linear_model, x[i+1:end], y[i+1:end], Diagonal(1 ./ dy[i+1:end].^2), [1.0, 0.0]) #curve_fit(model, x[i+1:end], y[i+1:end], [1.0, 0.0])

        ssr = sum(abs2, y[1:i] .- model(x[1:i], fit1.param)) +
              sum(abs2, y[i+1:end] .- model(x[i+1:end], fit2.param))

        if ssr < best_ssr
            best_ssr = ssr
            best_i = i
            best_fit1 = fit1
            best_fit2 = fit2
            errs1 = stderror(fit1)
            errs2 = stderror(fit2)
        end

    end

    #@info ("slope1=$(value_with_error(best_fit1.param[1],errs1[1],2));    breakpoint: t=$(10^(x[best_i])) hs; points:$(best_i);    slope2=$(value_with_error(best_fit2.param[1],errs2[1],2))")

    return (
        split_index = best_i,
        x_split = x[best_i],
        fit1 = (best_fit1.param,errs1),
        fit2 = (best_fit2.param,errs2),
        best_ssr = best_ssr
    )
end


# ============================================================================
# HELPER: Find smallest range with point selection
# ============================================================================

function find_smallest_overlap_range_with_selection(vectors::Vector{Vector{T}}, 
                                                     intervals::Vector{Tuple{T,T}}, 
                                                     tolerance::T) where T<:Real
    """
    Find the smallest range that contains at least one point from each vector,
    and select the specific points (one per vector) that minimize the range,
    prioritizing points closest to the first vector.
    """
    n_vectors = length(vectors)
    n_intervals = length(intervals)
    
    # If we have only one interval, just return it with the closest points
    if n_intervals == 1
        min_val, max_val = intervals[1]
        # Find the point in each vector closest to the center of the interval
        center = (min_val + max_val) / 2
        selected_points = [find_closest_to_value(v, center) for v in vectors]
        return (min_val, max_val, nothing, selected_points)
    end
    
    # Collect all candidate points from all vectors
    # We'll try to find the combination that gives the smallest range
    best_range = Inf
    best_min = nothing
    best_max = nothing
    best_selection = nothing
    
    # For each vector, get points that fall within the pairwise intervals
    # First, find the overall range of all intervals
    global_min = minimum([interval[1] for interval in intervals])
    global_max = maximum([interval[2] for interval in intervals])
    
    # For each vector, get points within the global range
    candidate_points = []
    for v in vectors
        points_in_range = filter(p -> p >= global_min - tolerance && p <= global_max + tolerance, v)
        if isempty(points_in_range)
            # If no points in range, use the closest point
            center = (global_min + global_max) / 2
            push!(candidate_points, [find_closest_to_value(v, center)])
        else
            push!(candidate_points, points_in_range)
        end
    end
    
    # Now we need to find the combination of points (one from each vector)
    # that gives the smallest range, prioritizing points closest to the first vector
    
    # Start with the first vector's points
    first_points = candidate_points[1]
    
    for first_point in first_points
        # For each point in the first vector, find the best matching points
        # from other vectors that minimize the range
        current_selection = [first_point]
        current_min = first_point
        current_max = first_point
        
        # For each subsequent vector, find the point closest to the current range
        for i in 2:n_vectors
            best_point = find_best_point_for_range(candidate_points[i], current_min, current_max, first_point)
            push!(current_selection, best_point)
            current_min = min(current_min, best_point)
            current_max = max(current_max, best_point)
        end
        
        # Check if this selection gives a better range
        current_range = current_max - current_min
        if current_range < best_range
            best_range = current_range
            best_min = current_min
            best_max = current_max
            best_selection = copy(current_selection)
        end
    end
    
    if best_min !== nothing
        return (best_min - tolerance, best_max + tolerance, nothing, best_selection)
    else
        return (nothing, nothing, nothing, nothing)
    end
end

# ============================================================================
# HELPER: Find best point for a range
# ============================================================================

function find_best_point_for_range(points::Vector{T}, 
                                   current_min::T, 
                                   current_max::T, 
                                   reference_point::T) where T<:Real
    """
    Find the point that minimizes the range expansion while staying close to the reference.
    """
    if isempty(points)
        # Shouldn't happen, but just in case
        return reference_point
    end
    
    if length(points) == 1
        return points[1]
    end
    
    best_point = points[1]
    best_score = Inf
    
    for p in points
        # Calculate how much this point expands the range
        new_min = min(current_min, p)
        new_max = max(current_max, p)
        range_expansion = (new_max - new_min) - (current_max - current_min)
        
        # Calculate distance to reference point (first vector's point)
        distance_to_ref = abs(p - reference_point)
        
        # Combined score: prioritize minimizing range expansion,
        # then minimizing distance to reference
        score = range_expansion * 1000 + distance_to_ref
        
        if score < best_score
            best_score = score
            best_point = p
        end
    end
    
    return best_point
end

# ============================================================================
# HELPER: Find closest point to a value
# ============================================================================

function find_closest_to_value(v::Vector{T}, target::T) where T<:Real
    """
    Find the point in vector v closest to the target value.
    """
    if isempty(v)
        return target
    end
    
    closest = v[1]
    min_dist = abs(v[1] - target)
    
    for p in v
        dist = abs(p - target)
        if dist < min_dist
            min_dist = dist
            closest = p
        end
    end
    
    return closest
end

# ============================================================================
# HELPER: Select closest points to first vector
# ============================================================================

function select_closest_points(vectors::Vector{Vector{T}}, 
                               overlapping_points::Vector{Vector{Int}}, 
                               first_idx::Int) where T<:Real
    """
    Select the points (one per vector) that are closest to the first vector's selected point.
    """
    n_vectors = length(vectors)
    selected_points = []
    
    # Get the first vector's selected point (median of its overlapping points)
    first_points = vectors[first_idx][overlapping_points[first_idx]]
    if isempty(first_points)
        # If no points, use the median of the whole vector
        first_reference = median(vectors[first_idx])
    else
        first_reference = median(first_points)
    end
    
    # For each vector, select the point closest to the first reference
    for i in 1:n_vectors
        if !isempty(overlapping_points[i])
            # Get points in this vector's overlap range
            points = vectors[i][overlapping_points[i]]
            # Select the one closest to the first reference
            selected = find_closest_to_value(points, first_reference)
            push!(selected_points, selected)
        else
            # If no overlapping points, use the closest point in the whole vector
            selected = find_closest_to_value(vectors[i], first_reference)
            push!(selected_points, selected)
        end
    end
    
    return selected_points
end  
function find_common_range(vectors::Vector{Vector{T}}; std_multiplier=1.0, tolerance=1e-10) where T<:Real
    """
    Find the range of data that coincides between any number of vectors.
    
    Returns:
    - overlap_min, overlap_max: the common range boundaries
    - overlapping_points: indices of points within the common range for each vector
    - selected_points: the actual selected points (one per vector) forming the smallest range
    - status: "exact_overlap", "std_range", "partial_overlap", "pairwise_overlap", or "warning"
    - pair_info: information about pairwise overlaps (for debugging)
    """
    n_vectors = length(vectors)
    
    # Check if we have at least 2 vectors
    if n_vectors < 2
        @warn "Need at least 2 vectors. Returning the only vector's range."
        sorted_v = sort(vectors[1])
        return (minimum(sorted_v), maximum(sorted_v), 
                [findall(v .>= minimum(sorted_v) .&& v .<= maximum(sorted_v)) for v in vectors], 
                [minimum(sorted_v)], "single_vector", nothing) #**me agregó algo ahí
    end
    
    # Sort each vector for easier processing
    sorted_vectors = [sort(v) for v in vectors]
    
    # Find global min and max
    global_min = minimum([minimum(v) for v in sorted_vectors])
    global_max = maximum([maximum(v) for v in sorted_vectors])
    
    # Find exact overlap range across ALL vectors
    overlap_min = maximum([minimum(v) for v in sorted_vectors])
    overlap_max = minimum([maximum(v) for v in sorted_vectors])
    
    if overlap_min <= overlap_max + tolerance
        overlapping_points = [
            findall(v .>= overlap_min - tolerance .&& v .<= overlap_max + tolerance)
            for v in vectors
        ]
        # Select the points closest to the first vector's mean
        selected_points = select_closest_points(vectors, overlapping_points, 1) #**agregó esto
        return (overlap_min, overlap_max, overlapping_points, selected_points, "exact_overlap", nothing)
    end

    # Check pairwise overlaps with the first vector
    first_vector = vectors[1]
    min_first = minimum(first_vector)
    max_first = maximum(first_vector)
    
    pair_overlaps = []
    all_pair_overlap = true
    overlapping_points = [collect(1:length(vectors[1]))]  # Start with all indices of first vector #**cambió
    
    for i in 2:n_vectors
        v = vectors[i]
        min_v = minimum(v)
        max_v = maximum(v)
        
        # Find overlap between first vector and this vector
        pair_min = max(min_first, min_v)
        pair_max = min(max_first, max_v)
        
        if pair_min <= pair_max + tolerance
            # There is an overlap
            push!(pair_overlaps, (i, pair_min, pair_max, true, nothing))
            push!(overlapping_points, findall(v .>= pair_min - tolerance .&& v .<= pair_max + tolerance))
        else
            # No overlap - find the two closest points
            all_pair_overlap = false
            
            # Find closest points between first vector and this vector
            closest_pair = find_closest_points(first_vector, v)
            
            # Get the closest point from each vector
            closest_first, closest_v = closest_pair
            
            # Create a tiny range around these points
            mid_point = (closest_first + closest_v) / 2
            half_range = abs(closest_first - closest_v) / 2
            
            pair_min = mid_point - half_range * 1.1  # Slightly expand
            pair_max = mid_point + half_range * 1.1

            @warn "No overlapping range found between vectors. Using closest points."
            
            push!(pair_overlaps, (i, pair_min, pair_max, false, (closest_first, closest_v)))
            push!(overlapping_points, findall(v .>= pair_min - tolerance .&& v .<= pair_max + tolerance))
        end
    end
    
    # Return partial overlap - SMALLEST RANGE with at least one point from each
    if all_pair_overlap
        # Get all pairwise overlap intervals
        intervals = [(pair[2], pair[3]) for pair in pair_overlaps]
        
        # Find the smallest range that contains at least one point from each vector
        overlap_min, overlap_max, overlapping_indices, selected_points = find_smallest_overlap_range_with_selection(
            vectors, intervals, tolerance
        )
        
        if overlap_min !== nothing && overlap_max !== nothing
            overlapping_points = [
                findall(v .>= overlap_min - tolerance .&& v .<= overlap_max + tolerance)
                for v in vectors
            ]
            return (overlap_min, overlap_max, overlapping_points, selected_points, "pairwise_overlap", pair_overlaps)
        end
    end
    
    # If we get here, use the pairwise overlaps we already found
    # Find the smallest range from the pairwise overlaps
    if length(pair_overlaps) > 0
        # Get all pairwise overlap intervals
        intervals = [(pair[2], pair[3]) for pair in pair_overlaps]
        
        # Find the smallest range
        overlap_min, overlap_max, overlapping_indices, selected_points = find_smallest_overlap_range_with_selection(
            vectors, intervals, tolerance
        )
        
        if overlap_min !== nothing && overlap_max !== nothing
            overlapping_points = [
                findall(v .>= overlap_min - tolerance .&& v .<= overlap_max + tolerance)
                for v in vectors
            ]
            return (overlap_min, overlap_max, overlapping_points, selected_points, "pairwise_overlap_fallback", pair_overlaps)
        end
    end
    
    # Ultimate fallback: return global range
    @warn "No overlapping range found between vectors. Using global range."
    overlapping_points = [
        findall(v .>= global_min .&& v .<= global_max)
        for v in vectors
    ]
    # Select points closest to first vector's mean
    selected_points = select_closest_points(vectors, overlapping_points, 1)
    
    return (global_min, global_max, overlapping_points, selected_points, "warning", nothing)
end


function find_common_range_old(vectors::Vector{Vector{T}}; std_multiplier=1.0, tolerance=1e-10) where T<:Real
    """
    Find the range of data that coincides between any number of vectors.
    
    Returns:
    - overlap_min, overlap_max: the common range boundaries
    - overlapping_points: indices of points within the common range for each vector
    - status: "exact_overlap", "std_range", "partial_overlap", "pairwise_overlap", or "warning"
    - pair_info: information about pairwise overlaps (for debugging)
    """
    n_vectors = length(vectors)
    
    # Check if we have at least 2 vectors
    if n_vectors < 2
        @warn "Need at least 2 vectors. Returning the only vector's range."
        sorted_v = sort(vectors[1])
        return (minimum(sorted_v), maximum(sorted_v), 
                [findall(v .>= minimum(sorted_v) .&& v .<= maximum(sorted_v)) for v in vectors], 
                "single_vector", nothing)
    end
    
    # Sort each vector for easier processing
    sorted_vectors = [sort(v) for v in vectors]
    
    # Find global min and max
    global_min = minimum([minimum(v) for v in sorted_vectors])
    global_max = maximum([maximum(v) for v in sorted_vectors])
    
    # Method 1: Find exact overlap range across ALL vectors
    overlap_min = maximum([minimum(v) for v in sorted_vectors])
    overlap_max = minimum([maximum(v) for v in sorted_vectors])
    
    if overlap_min <= overlap_max + tolerance
        overlapping_points = [
            findall(v .>= overlap_min - tolerance .&& v .<= overlap_max + tolerance)
            for v in vectors
        ]
        return (overlap_min, overlap_max, overlapping_points, "exact_overlap", nothing)
    end

    """
    # Method 2: Use standard deviation expansion
    stds = [std(v) for v in vectors]
    max_std = maximum(stds)
    
    if max_std > 0
        all_data = vcat(vectors...)
        global_mean = mean(all_data)
        
        overlap_min = global_mean - std_multiplier * max_std
        overlap_max = global_mean + std_multiplier * max_std
        
        overlapping_points = [
            findall(v .>= overlap_min .&& v .<= overlap_max)
            for v in vectors
        ]
        
        if all(length(points) > 0 for points in overlapping_points)
            return (overlap_min, overlap_max, overlapping_points, "std_range", nothing)
        end
    end
    """
    
    # Method 3: Check pairwise overlaps with the first vector
    first_vector = vectors[1]
    min_first = minimum(first_vector)
    max_first = maximum(first_vector)
    
    pair_overlaps = []
    all_pair_overlap = true
    overlapping_points = [vectors[1]]
    
    for i in 2:n_vectors
        v = vectors[i]
        min_v = minimum(v)
        max_v = maximum(v)
        
        # Find overlap between first vector and this vector
        pair_min = max(min_first, min_v)
        pair_max = min(max_first, max_v)
        
        if pair_min <= pair_max + tolerance
            # There is an overlap
            push!(pair_overlaps, (i, pair_min, pair_max, true, nothing))
            push!(overlapping_points, findall(v .>= pair_min - tolerance .&& v .<= pair_max + tolerance))
        else
            # No overlap - find the two closest points
            all_pair_overlap = false
            
            # Find closest points between first vector and this vector
            closest_pair = find_closest_points(first_vector, v)
            
            # Get the closest point from each vector
            closest_first, closest_v = closest_pair
            
            # Create a tiny range around these points
            mid_point = (closest_first + closest_v) / 2
            half_range = abs(closest_first - closest_v) / 2
            
            pair_min = mid_point - half_range * 1.1  # Slightly expand
            pair_max = mid_point + half_range * 1.1

            # Method 5: Fallback - return global range with warning
            @warn "No overlapping range found between vectors. Using closest points."
            
            push!(pair_overlaps, (i, pair_min, pair_max, false, (closest_first, closest_v)))

            push!(overlapping_points, findall(v .>= pair_min - tolerance .&& v .<= pair_max + tolerance))
            
        end
    end
    
    #return (overlap_min, overlap_max, overlapping_points, "pairwise_overlap", pair_overlaps)

    

    
    # Method 4: Return partial overlap - SMALLEST RANGE with at least one point from each
    if all_pair_overlap
        # Get all pairwise overlap intervals
        intervals = [(pair[2], pair[3]) for pair in pair_overlaps]
        
        # Find the smallest range that contains at least one point from each vector
        overlap_min, overlap_max, overlapping_indices = find_smallest_overlap_range(
            vectors, intervals, tolerance
        )
        
        if overlap_min !== nothing && overlap_max !== nothing
            overlapping_points = [
                findall(v .>= overlap_min - tolerance .&& v .<= overlap_max + tolerance)
                for v in vectors
            ]
            return (overlap_min, overlap_max, overlapping_points, "pairwise_overlap", pair_overlaps)
        end
    end
    
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function find_closest_points(v1::Vector{T}, v2::Vector{T}) where T<:Real
    """
    Find the closest pair of points between two vectors.
    Returns the pair (point_from_v1, point_from_v2).
    """
    # Sort vectors for efficient searching
    sorted_v1 = sort(v1)
    sorted_v2 = sort(v2)
    
    closest_dist = Inf
    closest_pair = (sorted_v1[1], sorted_v2[1])
    
    # Use two-pointer technique for O(n + m) complexity
    i = 1
    j = 1
    
    while i <= length(sorted_v1) && j <= length(sorted_v2)
        diff = sorted_v1[i] - sorted_v2[j]
        dist = abs(diff)
        
        if dist < closest_dist
            closest_dist = dist
            closest_pair = (sorted_v1[i], sorted_v2[j])
        end
        
        # Move the pointer of the smaller element
        if diff < 0
            i += 1
        else
            j += 1
        end
    end
    
    return closest_pair
end

# Alternative brute-force version (use for small vectors)
function find_closest_points_bruteforce(v1::Vector{T}, v2::Vector{T}) where T<:Real
    """
    Find the closest pair of points between two vectors using brute force.
    O(n*m) complexity - only use for small vectors.
    """
    closest_dist = Inf
    closest_pair = (v1[1], v2[1])
    
    for a in v1
        for b in v2
            dist = abs(a - b)
            if dist < closest_dist
                closest_dist = dist
                closest_pair = (a, b)
            end
        end
    end
    
    return closest_pair
end

# ============================================================================
# PRINTING AND DISPLAY FUNCTIONS
# ============================================================================

function print_overlap_results(result, vector_names=nothing)
    """
    Pretty print the overlap results with selected points.
    """
    if length(result) == 6
        overlap_min, overlap_max, overlapping_points, selected_points, status, pair_info = result
    else
        overlap_min, overlap_max, overlapping_points, status = result
        selected_points = nothing
        pair_info = nothing
    end
    
    n_vectors = length(overlapping_points)
    
    if vector_names === nothing
        vector_names = ["Vector $i" for i in 1:n_vectors]
    end
    
    println("="^70)
    println("OVERLAP ANALYSIS RESULTS")
    println("="^70)
    println("Status: $status")
    println("Overlap range: [$overlap_min, $overlap_max]")
    println("Range width: $(overlap_max - overlap_min)")
    
    println("\nPoints within range:")
    total_points = 0
    for (i, name) in enumerate(vector_names)
        points = overlapping_points[i]
        println("  $name: $(length(points)) points at indices $points")
        total_points += length(points)
    end
    println("\nTotal points in overlap: $total_points")
    
    if selected_points !== nothing
        println("\nSELECTED POINTS (one per vector, smallest range):")
        for (i, name) in enumerate(vector_names)
            if i <= length(selected_points)
                println("  $name: $(selected_points[i])")
            end
        end
        println("\nSelected range: [$(minimum(selected_points)), $(maximum(selected_points))]")
        println("Selected range width: $(maximum(selected_points) - minimum(selected_points))")
    end
    
    if status == "pairwise_overlap" && pair_info !== nothing
        println("\nPairwise overlaps with first vector:")
        for pair in pair_info
            idx, pmin, pmax, has_overlap, closest = pair
            if has_overlap
                println("  Vector $idx: overlap [$pmin, $pmax]")
            else
                println("  Vector $idx: NO overlap. Closest points: $(closest[1]) and $(closest[2])")
            end
        end
    end
    
    if status == "warning"
        println("\n⚠️  WARNING: Using global range as fallback!")
    end
    
    println("="^70)
end

function print_overlap_results_old(result, vector_names=nothing)
    """
    Pretty print the overlap results.
    """
    if length(result) == 5
        overlap_min, overlap_max, overlapping_points, status, pair_info = result
    else
        overlap_min, overlap_max, overlapping_points, status = result
        pair_info = nothing
    end
    
    n_vectors = length(overlapping_points)
    
    if vector_names === nothing
        vector_names = ["Vector $i" for i in 1:n_vectors]
    end
    
    println("="^70)
    println("OVERLAP ANALYSIS RESULTS")
    println("="^70)
    println("Status: $status")
    println("Overlap range: [$overlap_min, $overlap_max]")
    println("Range width: $(overlap_max - overlap_min)")
    
    println("\nPoints within range:")
    total_points = 0
    for (i, name) in enumerate(vector_names)
        points = overlapping_points[i]
        println("  $name: $(length(points)) points at indices $points")
        total_points += length(points)
    end
    println("\nTotal points in overlap: $total_points")
    
    if status == "pairwise_overlap" && pair_info !== nothing
        println("\nPairwise overlaps with first vector:")
        for pair in pair_info
            idx, pmin, pmax, has_overlap, closest = pair
            if has_overlap
                println("  Vector $idx: overlap [$pmin, $pmax]")
            else
                println("  Vector $idx: NO overlap. Closest points: $(closest[1]) and $(closest[2])")
            end
        end
    end
    
    if status == "warning"
        println("\n⚠️  WARNING: Using global range as fallback!")
        println("   Consider increasing std_multiplier or checking your data.")
    end
    
    println("="^70)
end

# ============================================================================
# ADDITIONAL UTILITY: Find overlapping points as actual values
# ============================================================================

function get_overlapping_values(result)
    """
    Extract the actual overlapping values from the result.
    """
    _, _, overlapping_points, status, _ = result
    return [v[idx] for (v, idx) in zip(vectors, overlapping_points)]
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

function demonstrate_common_range()
    println("\n" * "="^70)
    println("DEMONSTRATION: Common Range Finder for Any Number of Vectors")
    println("="^70)
    
    # Example 1: Vectors with good overlap
    println("\n1. VECTORS WITH GOOD OVERLAP")
    println("-"^70)
    A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    B = [2.5, 3.0, 3.5, 4.5, 5.5]
    C = [2.8, 3.2, 3.8, 4.2, 5.0]
    D = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    E = [2.9, 3.4, 4.0, 4.6]
    
    result1 = find_common_range([A, B, C, D, E])
    print_overlap_results(result1, ["A", "B", "C", "D", "E"])
    
    # Example 2: Vectors with no common overlap
    println("\n2. VECTORS WITH NO COMMON OVERLAP")
    println("-"^70)
    A2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    B2 = [10.0, 11.0, 12.0, 13.0]
    C2 = [20.0, 21.0, 22.0]
    D2 = [30.0, 31.0, 32.0, 33.0]
    
    result2 = find_common_range([A2, B2, C2, D2], std_multiplier=2.0)
    print_overlap_results(result2, ["A2", "B2", "C2", "D2"])
    
    # Example 3: Some vectors overlap with first, some don't
    println("\n3. PARTIAL OVERLAP WITH FIRST VECTOR")
    println("-"^70)
    A3 = [1.0, 2.0, 3.0, 4.0, 5.0]
    B3 = [2.5, 3.0, 3.5, 4.0]  # Overlaps with A
    C3 = [0.5, 0.8, 1.2]        # Overlaps with A (at low end)
    D3 = [10.0, 11.0, 12.0]     # No overlap with A
    
    result3 = find_common_range([A3, B3, C3, D3])
    println(result3)
    print_overlap_results(result3, ["A3", "B3", "C3", "D3"])
    
    # Example 4: Using with one vector only
    println("\n4. SINGLE VECTOR")
    println("-"^70)
    A4 = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    result4 = find_common_range([A4])
    print_overlap_results(result4, ["A4"])
    
    return nothing
end

# Run demonstration
#demonstrate_common_range()

# ============================================================================
# ADVANCED: Find pairwise overlaps between all vectors
# ============================================================================

function find_all_pairwise_overlaps(vectors::Vector{Vector{T}}; tolerance=1e-10) where T<:Real
    """
    Find overlaps between all pairs of vectors.
    Returns a matrix of overlap information.
    """
    n = length(vectors)
    overlap_matrix = Matrix{Union{Nothing,Tuple{T,T,Vector{Int},Vector{Int}}}}(undef, n, n)
    
    for i in 1:n
        for j in i+1:n
            vi = vectors[i]
            vj = vectors[j]
            
            min_i = minimum(vi)
            max_i = maximum(vi)
            min_j = minimum(vj)
            max_j = maximum(vj)
            
            overlap_min = max(min_i, min_j)
            overlap_max = min(max_i, max_j)
            
            if overlap_min <= overlap_max + tolerance
                # Find overlapping points
                idx_i = findall(vi .>= overlap_min - tolerance .&& vi .<= overlap_max + tolerance)
                idx_j = findall(vj .>= overlap_min - tolerance .&& vj .<= overlap_max + tolerance)
                overlap_matrix[i, j] = (overlap_min, overlap_max, idx_i, idx_j)
                overlap_matrix[j, i] = (overlap_min, overlap_max, idx_j, idx_i)
            else
                # No overlap - find closest points
                closest_i, closest_j = find_closest_points(vi, vj)
                overlap_matrix[i, j] = (closest_i, closest_j, nothing, nothing)
                overlap_matrix[j, i] = (closest_j, closest_i, nothing, nothing)
            end
        end
    end
    
    return overlap_matrix
end

function print_pairwise_overlaps(vectors::Vector{Vector{T}}, vector_names=nothing) where T<:Real
    """
    Print pairwise overlap information.
    """
    n = length(vectors)
    
    if vector_names === nothing
        vector_names = ["Vector $i" for i in 1:n]
    end
    
    overlap_matrix = find_all_pairwise_overlaps(vectors)
    
    println("="^70)
    println("PAIRWISE OVERLAP ANALYSIS")
    println("="^70)
    
    for i in 1:n
        for j in i+1:n
            info = overlap_matrix[i, j]
            if info !== nothing
                if length(info) == 4 && info[1] !== nothing
                    min_val, max_val, idx_i, idx_j = info
                    println("$i-$j ($(vector_names[i])-$(vector_names[j])):")
                    println("  Overlap: [$min_val, $max_val]")
                    println("  $(length(idx_i)) points in $(vector_names[i]), $(length(idx_j)) points in $(vector_names[j])")
                else
                    closest_i, closest_j, _, _ = info
                    println("$i-$j ($(vector_names[i])-$(vector_names[j])):")
                    println("  NO OVERLAP - Closest points: $(closest_i) and $(closest_j)")
                end
            end
        end
    end
    println("="^70)
end

function find_smallest_overlap_range(vectors::Vector{Vector{T}}, 
                                     intervals::Vector{Tuple{T,T}}, 
                                     tolerance::T) where T<:Real
    """
    Find the smallest range that contains at least one point from every vector.
    
    Strategy:
    1. Collect all points from all vectors
    2. Sort them
    3. Use a sliding window to find the smallest window that includes at least
       one point from each vector
    """
    n_vectors = length(vectors)
    
    # Create a list of (value, vector_index) pairs
    all_points = []
    for (idx, v) in enumerate(vectors)
        for point in v
            push!(all_points, (point, idx))
        end
    end
    
    # Sort by value
    sort!(all_points, by = x -> x[1])
    
    # Use sliding window to find smallest range covering all vectors
    best_min = nothing
    best_max = nothing
    best_width = Inf
    
    # Count how many vectors we've seen in current window
    count = Dict{Int,Int}()
    unique_vectors = 0
    left = 1
    
    for right in 1:length(all_points)
        # Add the right point
        val, vec_idx = all_points[right]
        count[vec_idx] = get(count, vec_idx, 0) + 1
        if count[vec_idx] == 1
            unique_vectors += 1
        end
        
        # While we have all vectors in the window, try to shrink from the left
        while unique_vectors == n_vectors
            # Check if this window is better than the current best
            window_min = all_points[left][1]
            window_max = all_points[right][1]
            window_width = window_max - window_min
            
            if window_width < best_width
                best_width = window_width
                best_min = window_min
                best_max = window_max
            end
            
            # Remove the left point
            left_val, left_idx = all_points[left]
            count[left_idx] -= 1
            if count[left_idx] == 0
                unique_vectors -= 1
            end
            left += 1
            
        end
    end
    
    if best_min !== nothing
        # Add a small tolerance to ensure we capture all points
        return (best_min - tolerance, best_max + tolerance, nothing)
    else
        return (nothing, nothing, nothing)
    end
end

# ============================================================================
# ALTERNATIVE: More precise version using pairwise overlaps
# ============================================================================

function find_smallest_overlap_range_precise(vectors::Vector{Vector{T}}, 
                                             intervals::Vector{Tuple{T,T}}, 
                                             tolerance::T) where T<:Real
    """
    More precise version that uses the pairwise overlap intervals directly.
    
    This finds the smallest range that overlaps with every pairwise interval.
    """
    n_vectors = length(vectors)
    n_intervals = length(intervals)
    
    # If we have only one interval, just return it
    if n_intervals == 1
        return intervals[1]
    end
    
    # Collect all interval endpoints
    endpoints = []
    for (i, (min_val, max_val)) in enumerate(intervals)
        push!(endpoints, (min_val, "start", i))
        push!(endpoints, (max_val, "end", i))
    end
    sort!(endpoints, by = x -> x[1])
    
    # Use sweep line to find the smallest range that intersects all intervals
    best_min = nothing
    best_max = nothing
    best_width = Inf
    
    active_intervals = Set{Int}()
    start_idx = 1

    
    for (idx, (point, type, interval_id)) in enumerate(endpoints)
        if type == "start"
            push!(active_intervals, interval_id)
            
            # If we have all intervals, try to find a good range
            if length(active_intervals) == n_intervals
                # Find the latest start among active intervals
                latest_start = maximum([intervals[i][1] for i in active_intervals])
                # Current point is the earliest end
                earliest_end = point
                width = earliest_end - latest_start
                if width < best_width
                    best_width = width
                    best_min = latest_start
                    best_max = earliest_end
                        
                end
            end
        else  # type == "end"
            delete!(active_intervals, interval_id)
        end
            
    end
    
    if best_min !== nothing
        return (best_min - tolerance, best_max + tolerance)
    else
        return (nothing, nothing)
    end
end

# ============================================================================
# EVEN SIMPLER: Find the common range with at least one point from each
# ============================================================================

function find_smallest_common_range(vectors::Vector{Vector{T}}; tolerance=1e-10) where T<:Real
    """
    Simplified version that finds the smallest range containing at least one point
    from each vector.
    
    This uses the "intersection of all pairwise overlaps" approach.
    """
    n = length(vectors)
    
    if n < 2
        return (minimum(vectors[1]), maximum(vectors[1]))
    end
    
    # Get all pairwise overlap intervals with the first vector
    first_vector = vectors[1]
    intervals = []

        
    for i in 2:n
        v = vectors[i]
        # Find overlap between first vector and this vector
        min_val = max(minimum(first_vector), minimum(v))
        max_val = min(maximum(first_vector), maximum(v))
        
        if min_val <= max_val + tolerance
            push!(intervals, (min_val, max_val))
        else
            # No overlap - find closest points
            closest_first, closest_v = find_closest_points(first_vector, v)
            mid = (closest_first + closest_v) / 2
            half = abs(closest_first - closest_v) / 2 * 1.1
            push!(intervals, (mid - half, mid + half))
        end
    end
    
    # Find the intersection of all intervals
    # But we want the SMALLEST range that intersects all intervals,
    # not the intersection itself (which might be empty)
    
    # Method: Collect all candidate points (one from each interval)
    candidates = []
    for (min_val, max_val) in intervals
        # For each interval, add its min and max as candidates
        push!(candidates, min_val)
        push!(candidates, max_val)
    end
    
    # For each candidate point, check if it can serve as an anchor
    # that creates a range containing at least one point from each vector
    best_range = Inf
    best_min = NaN
    best_max = NaN
    
    for anchor in candidates
        # Find the farthest point from anchor that we need to include
        farthest = 0.0
        for (idx, (min_val, max_val)) in enumerate(intervals)
            if anchor < min_val
                farthest = max(farthest, min_val)
            elseif anchor > max_val
                farthest = max(farthest, abs(anchor - max_val))
            end
        end
        
        range_width = abs(anchor - farthest)
        if range_width < best_range
            best_range = range_width
            best_min = min(anchor, farthest)
            best_max = max(anchor, farthest)
        end
    end
    
    # Add tolerance
    return (best_min - tolerance, best_max + tolerance)
end



    


#*¨******esta función es una vergüenzaaaaaaaaaaaaaaaaa

function main(pits::Vector{String}, which_pits::Vector{Int64}, times::Vector{Float64},
                tf::Int64, left_wall::Dict, right_wall::Dict, widths_distribution::Dict, 
                main_folder::String, plot_widths_t1::Bool, plot_area::Bool, plot_all::Bool, plot_sqrt::Bool, plot_mean_widths::Bool, 
                nrois::Int, invert::Bool, n_of_classes::Int, cut_ids=nothing, make_files=false)
    # Set up containers
    means, stds, IDs_old = init_pit_containers(tf, n_of_classes)
    
    accumulate_data!(means, stds, IDs_old, widths_distribution, widths_distribution, pits, tf, nrois, invert)

    IDs = filter_cols(deepcopy(IDs_old),cut_ids, invert)
    println(cut_ids,IDs,IDs_old)
    
    if plot_widths_t1 #&& pits[1] == 1
        # pits # ["A1", "D1","A2","D2","A3","B3","C3","D3"]  ["A1", "B1","C1","D1","A2","B2","C2","D2","A3","B3","C3","D3"]
        plot_initial_widths(IDs, widths_distribution, ["A1", "B1","C1","D1","A2","B2","C2","D2","A3","B3","C3","D3","A4", "B4","C4","D4","A5","B5","C5","D5","A6","B6","C6","D6"], tf, main_folder*"fit_initial_position_wall.png",n_rois) ########################################## 
    end
    df_areas = CSV.File(main_folder*"/scratch_report.txt") |> DataFrame

    xlog_lims::Vector{Float64} = [log10(times[2]),log10(times[end]+1)]
    x_lims::Vector{Float64} = [minimum(times),maximum(times)]

    
    if isfile(main_folder*"mean_slope_report.txt")
        mean_slopes_file = CSV.read(main_folder*"mean_slope_report.txt", DataFrame, delim='\t')
    else
        columns = (; pit_num=Int[], slope=Float64[], std_slope=Float64[], r2_slope=Float64[])
        mean_slopes_file = DataFrame(columns)
    end

    #** O sea, de acá tendría que pasarle para cada ID en IDs (no sólo IDs_pit, o sea tengo que comparar todos los pits)
    #**ahhh pero entonces tengo que adaptar la función para que tome un número variable de vectores para comparar !!
    @info([[widths_distribution[id][1] for id in IDs_pits if id in keys(widths_distribution)] for IDs_pits in IDs if !isempty(IDs_pits)])
    result = find_common_range([[widths_distribution[id][1] for id in IDs_pits if id in keys(widths_distribution)] for IDs_pits in IDs if !isempty(IDs_pits)])
    
    print_overlap_results(result)
    overlap_min, overlap_max, overlapping_points, selected_points, status, pair_info = result
    @info (overlapping_points, selected_points)

    similar_di_fig = plot(size=(2000,400), margin=10mm, legend=:outertopright, grid=true, legend_columns=2)

        
    just_widths_fig1 = plot(legend=false, grid=true,title = "Control")#
    just_widths_fig2 = plot(legend=false, grid=true, title="600hUC")#
        
    # Dynamic fits
    for pit_number in which_pits
        means_pit::Vector{Vector{Vector{Float64}}} = means[pit_number]
        stds_pit::Vector{Vector{Vector{Float64}}} = stds[pit_number]
        IDs_pit::Vector{String} = IDs[pit_number]

        if isempty(IDs_pit)
            continue
        end

        
        title = set_title(pit_number)
        
        if plot_area

            k = length(IDs_pit)
            
            area_fig = plot(layout=grid(1,2, widths=(4/8,4/8)), size=(1400,400), 
                        margin=10mm, legend=:outertopright, grid=true, legendcolumns=2)#


            alternative_poly_plot = plot(layout=grid(1,2, widths=(4/8,4/8)), size=(1400,400), 
                        margin=10mm, legend=:outertopright, grid=true, legendcolumns=2)#

            res_fig = plot(size=(2000,400), margin=10mm, legend=:outertopright, grid=true, legend_columns=2)
            r2_hist_fig = Vector{Any}(undef,length(IDs_pit))#plot(size=(2000,400), margin=10mm, legend=:outertopright, grid=true)
            #fig_r2_vs_t = plot(size=(2000,400), margin=10mm, legend=:outertopright, grid=true)

            mean_slope = zeros(Float64, length(IDs_pit))
            std_mean_slope = zeros(Float64, length(IDs_pit))
            r2_slope = zeros(Float64, length(IDs_pit))

            initial_w = zeros(Float64, length(IDs_pit))

            if make_files            
                width_file = init_report_file(main_folder*"width_report.txt", [[i for i in 1:tf]])
                slope_file = init_report_file(main_folder*"slope_report.txt", [[1,2]])
            end
            
        end
        if plot_all
            all_fig = plot(layout=grid(1,2, widths=(2/8,6/8)), size=(2200,600), 
                    margin=10mm, legend=:outertopright, grid=true, legendcolumns=3)
            plot!(all_fig,subplot=1,legend=false)#, legendcolumns=2)

            all_fig2 = plot(layout=grid(1,2, widths=(3/8,5/8)), size=(2000,400), 
                    margin=10mm, legend=:outertopright, grid=true, legendcolumns=3)
            plot!(all_fig2,subplot=1,legend=false)#, legendcolumns=2)

            d_vs_t_plot = plot(grid=true)


        end
        if plot_sqrt
            sqrt_fig = plot(layout=grid(1,2, widths=(4/8,4/8)), size=(1400,400), 
                    margin=10mm, legend=:outertopright, grid=true, legendcolumns=2)
        end
        if plot_mean_widths
            mean_fig = plot(layout=grid(1,2, widths=(4/8,4/8)), size=(1400,400), 
                    margin=10mm, legend=:outertopright, grid=true, legendcolumns=2)
        end

        
        
        for (i,ID) in enumerate(IDs_pit)

            index_id = findfirst(IDs_old[pit_number] .== ID)

            times_data, y, dy, xlog, ylog, dylog = get_x_and_y_and_ylog(means_pit, stds_pit, times, index_id, tf, 2)
            
            if plot_area
                #@info (ID,y)
                initial_w[i] = y[1]
                # Area fig
                
                    # areas
                target_ID = ID[1:end-5]
                row_idx = findfirst(df_areas.ID .== target_ID)
                area_init = df_areas[row_idx,2]                                                       
                areas = [df_areas[row_idx,j+1]/area_init for j in 1:tf]

                #@info("Af/Ai es di x cte: ", [areas[k] for (k,t) in enumerate(times) if t in times_data]./y)

                # acá ya me estoy salteando los puntos donde el área es más que el área inicial
                # peeero no estoy sacando, como sí en y y times_data, los valores donde puse -1 en el ancho !
                plot_pit_fits(area_fig, i, times[areas.<1.01], areas[areas.<1.01], [nothing for td in times[areas.<1.01]],
                              ["" for _ in IDs_pit], xlog_lims, x_label="t (hs)", y_label="Area (% of A(t0))", 
                              do_fit=false, subplot=1)
                
                    # widths
                
                m,dm, res, r2 = plot_pit_fits(area_fig, i, times_data, y, dy,               
                              IDs_pit, x_lims,
                              x_label="t (hs)", y_label="Wound width (px)", 
                              do_fit=true, subplot=2) 

                if pit_number == 1
                    plot_pit_fits2(just_widths_fig1, i, times_data, y, dy,               
                                  ["" for _ in IDs_pit], x_lims,
                                  x_label="t (hs)", y_label="Wound width (px)", 
                                  do_fit=false) 
                elseif pit_number == 4
                    plot_pit_fits2(just_widths_fig2, i, times_data, y, dy,               
                                  ["" for _ in IDs_pit], x_lims,
                                  x_label="t (hs)", y_label="Wound width (px)", 
                                  do_fit=false) 
                end

                    
                if y[1] in selected_points#i in overlapping_points[pit_number]
                    scatter!(similar_di_fig, times_data,y,yerr=dy, label=ID,xlabel="t (hs)",ylabel="d (px)", color=cgrad(:rainbow)[pit_number/length(IDs)], markerstrokecolor=cgrad(:rainbow)[pit_number/length(IDs)], markersize=2, alpha=0.4)
                    plot!(similar_di_fig,times_data,y, label="",color=cgrad(:rainbow)[pit_number/length(IDs)], markerstrokecolor=cgrad(:rainbow)[pit_number/length(IDs)])
                end

                scatter!(alternative_poly_plot, times_data.*times_data, y, yerr=dy, label="", xlabel="t² (hs²)", ylabel="d (px)",color=cgrad(:rainbow)[i/length(IDs_pit)], markerstrokecolor=cgrad(:rainbow)[i/length(IDs_pit)], subplot=1)
                scatter!(alternative_poly_plot, sqrt.(times_data), y, yerr=dy, label="", xlabel="sqrt(t) sqrt(hs)", ylabel="d (px)", color=cgrad(:rainbow)[i/length(IDs_pit)], markerstrokecolor=cgrad(:rainbow)[i/length(IDs_pit)], subplot=2)

                # Histogramas con R2 de los fits con los datos reshuffled al azar
                #n_fits = 100000
                #r2_dist = run_many_fits(n_fits, times_data, y, dy, length(y))
                #r2_hist_fig[i] = histogram(r2_dist,bins=30,title=ID, color = cgrad(:rainbow)[i/length(IDs_pit)], legend=false, alpha=0.7)
                #scatter!([r2], [1], markershape=:star5, markersize=12, color=:black, markercolor=:black, label="Selected point", markerstrokewidth=0)
                #xlims!(-0.05,1.05)

                # Cómo cambia r2 ajustando de a distintos intervalos
                #@info (ID,"  ",length(y))
                #r2_vs_t = run_many_interval_fits(5, times_data, y, dy, length(y))
                #scatter!(fig_r2_vs_t, times_data[1:end-5+1], r2_vs_t,label=ID, color = cgrad(:rainbow)[i/length(IDs_pit)], alpha=0.7)

                # Ajuste cuadrático, la idea era comparar r2 con lineal pero tenemos incertezas muy altas
                #***bueno en ningún caso ajustar cuadrática al principio da algo razonable. las barras de error son muy grandes :(
                #quad_term, _, _, (err_quad,_,_), _, r2_quad = perform_quadratic_fit(times_data[1:minimum([8,length(y)])],y[1:minimum([8,length(y)])],dy[1:minimum([8,length(y)])]) #***pensar cómo marcar ese quiebre
                #@info ("maybe quadratic at first fits better? ",abs(quad_term)>err_quad," : ",quad_term,"+-",err_quad,"    r2: ",r2_quad)

                # Best split fit teniendo en cuenta maximizar r2
                #@info ("breaking fit in two parts: compare to fit of all data: ",m)
                #begin2, x_split, fit1, fit2, best_ssr = best_two_line_fit(times_data,y,dy)
                
                #@info (ID,": p-value from chi2 fit: ",chi2_sf(sum(res), length(y)-2)) #** bueno parece que un p-value de >0.99 indica barras de error demasiado grandes, o sea chi2 es sospechosamente (?) pequeño. pasa que si las saco, me pasa exactamente lo contrario y tampoco me sirve. claramente necesito más puntos. 
                #@info ("R2 at 99% : ",sort(r2_dist)[floor(Int64,0.99*n_fits)], "   ,  where my R2 lies : ",(argmin(abs.(sort(r2_dist) .- r2)) )/n_fits*100)



                mean_slope[i] = m
                std_mean_slope[i] = dm

                
                scatter!(res_fig, times_data, res, label=ID, color = cgrad(:rainbow)[i/length(IDs_pit)])
                r2_slope[i] = r2

                if make_files
                    push!(slope_file,vcat([ID],[m],[dm]))
                    
                    if length(y) == tf
                        push!(width_file, vcat([ID],y))#abs.(y.-y[1])))
                    else
                        push!(width_file, vcat([ID],y,[0.0 for t in 1:tf-length(y)]))
                    end
                end
            end
            if plot_all



                # All fig fit 
                     # linear plot
                #**********************en vez de dejar el linear plot, graficar derivada del log? a ver si se ve una buena tendencia de la pendiente! (no sé cómo tomarle incertezas a eso)
                #plot_pit_fits(all_fig, i, times_data, abs.(y.-y[1]), vcat(0.0001,sqrt.(dy[2:end].^2 .+dy[1]^2)),
                #              IDs_pit, x_lims,
                #              x_label="t (hs)", y_label="d=d0-d(t) (px)", 
                #              do_fit=true, subplot=1)
                    # derivative plot

                plot_pit_fits(all_fig, i, times_data[2:end-1], [(ylog[t+1]-ylog[t])/(xlog[t+1]-xlog[t]) for t in 1:length(ylog)-1], [nothing for b in 1:length(dylog)-1],
                              IDs_pit, x_lims,
                              x_label="t (hs)", y_label="dlog", 
                              do_fit=false, subplot=1)
                ylims!(all_fig,subplot=1,(-2.5,2.5))
                    # log plot


                #***cómo elijo el mejor quiebre en realidad? debería salir del log-log? quisiera que fuera el mismo para todos 
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
                begin2, x_split, fit1, fit2, best_ssr = best_two_line_fit(xlog,ylog,dylog)

                m1,dm1, res1, r2_1 = plot_pit_fits(all_fig, i, xlog[1:begin2], ylog[1:begin2], [nothing for b in 1:begin2],#dylog[1:begin2],
                          ["" for i in IDs_pit], xlog_lims,
                          x_label="log10(t)", y_label="log10(d)", 
                          do_fit=true, subplot=2)
                m2,dm2, res2, r2_2 = plot_pit_fits(all_fig, i, xlog[begin2+1:end], ylog[begin2+1:end], [nothing for b in (begin2+1):length(dylog)],#dylog[begin2+1:end],
                          IDs_pit, xlog_lims,
                          x_label="log10(t)", y_label="log10(d0-d)", 
                          do_fit=true, subplot=2)

                #mean_slope1[i] = m1
                #std_mean_slope1[i] = dm1

                #mean_slope2[i] = m2
                #std_mean_slope2[i] = dm2


                plot_pit_fits(all_fig2, i, times_data[2:end], [abs(y[t]-y[t-1]) for t in 2:length(y)],  [nothing for b in (begin2+1):length(dylog)],#[sqrt(dy[t]*dy[t]+dy[t-1]*dy[t-1]) for t in 2:length(dy)],
                              ["" for i in IDs_pit], x_lims,
                              x_label="t (hs)", y_label="dy", 
                              do_fit=false, subplot=1)
                
                
                

                """
                if length(times_data) > 8                                   
                    begin2 = 6
                    plot_pit_fits(all_fig, i, xlog[1:begin2], ylog[1:begin2], dylog[1:begin2],
                              ["" for i in IDs_pit], xlog_lims,
                              x_label="log10(t)", y_label="log10(d)", 
                              do_fit=true, subplot=2)
                    plot_pit_fits(all_fig, i, xlog[begin2+1:end], ylog[begin2+1:end], dylog[begin2+1:end],
                          IDs_pit, xlog_lims,
                          x_label="log10(t)", y_label="log10(d0-d)", 
                          do_fit=true, subplot=2)
                else
                    begin2 = length(times_data)-1
                    plot_pit_fits(all_fig, i, xlog[1:begin2], ylog[1:begin2], dylog[1:begin2],
                              IDs_pit, xlog_lims,
                              x_label="log10(t)", y_label="log10(d)", 
                              do_fit=true, subplot=2)
                end
                """
                
                              
            end

            if plot_sqrt

                # All fig fit 
                     # linear plot
                plot_pit_fits(sqrt_fig, i, times_data, abs.(y.-y[1]), vcat(0.0001,sqrt.(dy[2:end].^2 .+dy[1]^2)),
                              IDs_pit, x_lims,
                              x_label="t (hs)", y_label="d=d0-d(t) (px)", 
                              do_fit=true, subplot=1)
                    # sqrt plot
                plot_pit_fits(sqrt_fig, i, sqrt.(times_data)[2:end], abs.(y.-y[1])[2:end], sqrt.(dy[2:end].^2 .+dy[1]^2),
                          ["" for i in IDs_pit], sqrt.(x_lims),
                          x_label="sqrt(t) (sqrt(hs))", y_label="d (px)", 
                          do_fit=true, subplot=2)
                              
            end

        end

        if plot_area
            plot!(area_fig, plot_title="All (area + widths from histograms) $(pit_number)")
            ylims!(area_fig,subplot=2,(-10.0,Inf))
            display(area_fig)
            savefig(area_fig, main_folder*"areas_and_widths_$(pit_number).png")


            plot!(res_fig, plot_title="Residuals from linear fit "*title)
            display(res_fig)

            plot!(alternative_poly_plot, plot_title="d vs t² and d vs sqrt(t) $(pit_number)")
            display(alternative_poly_plot)
            savefig(alternative_poly_plot, main_folder*"d_vs_t2_and_sqrt_t_$(pit_number).png")
                

            #plot!(fig_r2_vs_t, plot_title="R2 vs t for 5 point-linear fitting "*title)
            #display(fig_r2_vs_t)

            #fig = plot(r2_hist_fig..., layout=grid(ceil(Int64,length(IDs_pit)/3) ,3), size=(1200, 600))
            #for i in length(IDs_pit)
            #    scatter!(fig[i], [r2_slope[i]], [1], markershape=:star5, markersize=12, color=:black, markercolor=:black, label="Selected point", markerstrokewidth=0)
            #    xlims!(fig[i],(-0.05,1.05))
            #end
            #display(fig)
            #savefig(fig, main_folder*"r2_dist_histograms.png")
            

            @info ("Mean slope: ",value_with_error(weighted_mean_and_std(mean_slope,std_mean_slope)[1],weighted_mean_and_std(mean_slope,std_mean_slope)[2],1))
            #**en realidad quiero hacer un boxplot como xime we así que debería usar todas las que están guardadas ya
            
            @info ("Mean slope: ",weighted_mean_and_std(mean_slope,std_mean_slope)[1]," +- ",weighted_mean_and_std(mean_slope,std_mean_slope)[2])
            w_thr = 350
            if true ∈ (initial_w.>w_thr)
                @info ("Mean slope wide initial (>$(w_thr)px): ",value_with_error(weighted_mean_and_std(mean_slope[initial_w.>w_thr],std_mean_slope[initial_w.>w_thr])[1],weighted_mean_and_std(mean_slope[initial_w.>w_thr],std_mean_slope[initial_w.>w_thr])[2],1))
                @info ("Mean slope wide initial (>$(w_thr)px): ",weighted_mean_and_std(mean_slope[initial_w.>w_thr],std_mean_slope[initial_w.>w_thr])[1]," +- ",weighted_mean_and_std(mean_slope[initial_w.>w_thr],std_mean_slope[initial_w.>w_thr])[2])
            end
            if true ∈ (initial_w.<w_thr)
                @info ("Mean slope narrow initial (<$(w_thr)px): ",value_with_error(weighted_mean_and_std(mean_slope[initial_w.<w_thr],std_mean_slope[initial_w.<w_thr])[1],weighted_mean_and_std(mean_slope[initial_w.<w_thr],std_mean_slope[initial_w.<w_thr])[2],1))
                @info ("Mean slope narrow initial (<$(w_thr)px): ",weighted_mean_and_std(mean_slope[initial_w.<w_thr],std_mean_slope[initial_w.<w_thr])[1], " +- ",weighted_mean_and_std(mean_slope[initial_w.<w_thr],std_mean_slope[initial_w.<w_thr])[2],1)
            end

            if make_files
                CSV.write(main_folder*"width_report.txt", width_file, delim='\t')
                CSV.write(main_folder*"slope_report.txt", slope_file, delim='\t')
                push!(mean_slopes_file, vcat([pit_number],[weighted_mean_and_std(mean_slope,std_mean_slope)[1]],[weighted_mean_and_std(mean_slope,std_mean_slope)[2]],Statistics.mean(r2_slope))) #**no sé si está bien promedir R2 we
                CSV.write(main_folder*"mean_slope_report.txt",mean_slopes_file, delim="\t")
            end
        end
        if plot_mean_widths
            ######### acá también separar del promedio los puntos que claramente están mal, tipo dan un ancho mayor al doble del ancho inicial !!
            
            cut_indices = [[means_pit[t][id][1] for id in 1:length(IDs_pit)] .!= -1.0 for t in 1:tf] # esto me da QUÉ índices tienen puntos bien calculados para cada id para cada tiempo (1 es true, 0 dato malo)
            y_to_mean = [[means_pit[t][i][1] for (i,c) in enumerate(cut_indices[t]) if c==1] for t in 1:tf] # tx10, idx4, m+stdx2
            y_to_std = [[stds_pit[t][i][1] for (i,c) in enumerate(cut_indices[t]) if c==1] for t in 1:tf]
    
            times_data = [times[t] for t in 1:tf if length(y_to_mean[t])>(length(IDs_pit)/2)] 
            y0 = weighted_mean_and_std(y_to_mean[1],y_to_std[1])[1]
            y_data = [weighted_mean_and_std(y_to_mean[t],y_to_std[t])[1] for t in 2:tf if length(y_to_mean[t])>(length(IDs_pit)/2)]
            @info ("###### ", y_to_mean[end-1],"  ",y_to_std[end-1],"   ",weighted_mean_and_std(y_to_mean[end-1],y_to_std[end-1]))
            dy0 = weighted_mean_and_std(y_to_mean[1],y_to_std[1])[2]
            dy = [weighted_mean_and_std(y_to_mean[t],y_to_std[t])[2] for t in 2:tf if length(y_to_mean[t])>(length(IDs_pit)/2)]
            
            y_ref_y0 = abs.(y_data .- y0)
            xlog = log10.(times_data[2:end])
            ylog = log10.(y_ref_y0)
            dylog = log10(exp(1.0)).*sqrt.(dy.^2 .+dy0.^2)./y_ref_y0
    
            y = vcat(y0,y_data)
            dy = vcat(dy0,dy)
        
            # Widths plot 
                # Linear scale
            p, params = bootstrap_breakpoint_test(times_data,abs.(y.-y0), sqrt.(dy.*dy.+dy0*dy0))
            _,begin2_best = findmin(x->abs(x-params[1]),times_data)

            if begin2_best>3
                plot_pit_fits(mean_fig, 1, times_data[1:begin2_best], abs.(y.-y0)[1:begin2_best], sqrt.(dy.*dy.+dy0*dy0)[1:begin2_best],
                              [""], x_lims,
                              x_label="t (hs)", y_label="", 
                              do_fit=true, subplot=1)
            end
            if begin2_best<length(y)-3
                plot_pit_fits(mean_fig, 1, times_data[begin2_best+1:end], abs.(y.-y0)[begin2_best+1:end], sqrt.(dy.*dy.+dy0*dy0)[begin2_best+1:end],
                          ["black"], x_lims,
                          x_label="t (hs)", y_label="|d-d0| (px)", 
                          do_fit=true, subplot=1)
            end
                

            vline!(mean_fig,[times_data[begin2_best]],color=:black,linestyle=:dash,label="p-value: $p", subplot=1)
                # Log scale
            # best_two_line_fit(x, y)
    #split_index = best_i,
    #x_split = x[best_i],
    #fit1 = best_fit1.param,errs1
    #fit2 = best_fit2.param,errs2
    #best_ssr = best_ssr

            begin2, x_split, fit1, fit2, best_ssr = best_two_line_fit(xlog,ylog,dylog)


            
            #p, params = bootstrap_breakpoint_test(xlog,ylog, dylog)
            _,begin2 = findmin(x->abs(x-params[1]),times_data)


            if begin2>3
                plot_pit_fits(mean_fig, 1, xlog[1:begin2], ylog[1:begin2], dylog[1:begin2],
                          [""], xlog_lims,
                          x_label="log10(t)", y_label="log10(d0-d)", 
                          do_fit=true, subplot=2)
            end
            if begin2_best<length(y)-3
                plot_pit_fits(mean_fig, 1, xlog[begin2+1:end], ylog[begin2+1:end], dylog[begin2+1:end],
                          ["black"], xlog_lims,
                          x_label="log10(t)", y_label="log10(d0-d)", 
                          do_fit=true, subplot=2)
            end
            
            #begin2 = length(times_data)-1
            #plot_pit_fits(mean_fig, 1, xlog[1:begin2], ylog[1:begin2], dylog[1:begin2],
            #              [""], xlog_lims,
            #              x_label="log10(t)", y_label="log10(d0-d)", 
            #              do_fit=true, subplot=2)
            #if length(times_data) > 7
            #    begin2 = 6
            #    plot_pit_fits(mean_fig, 1, xlog[begin2:end], ylog[begin2:end], dylog[begin2:end],
            #                  ["black"], xlog_lims,
            #                  x_label="log10(t)", y_label="log10(d0-d)", 
            #                  do_fit=true, subplot=2)
            #end
            plot!(mean_fig, plot_title="Mean distance traveled while closing wound: control", legend=:outertopright,) #***poner condition de config 
            display(mean_fig)
            savefig(mean_fig, main_folder*"mean_widths_$(pit_number).png")
        end
            
        if plot_all
            plot!(all_fig, plot_title="Log-scale fits "*title)
            display(all_fig)
            savefig(all_fig, main_folder*"log-scale_fits_$(pit_number).png")


            cut_indices = [[means_pit[t][id][1] for id in 1:length(IDs_pit)] .!= -1.0 for t in 1:tf] # esto me da QUÉ índices tienen puntos bien calculados para cada id para cada tiempo (1 es true, 0 dato malo)
            y_to_mean = [[(means_pit[t][i][1]-means_pit[t-1][i][1])/(times[t]-times[t-1]) for (i,c) in enumerate(cut_indices[t]) if c==1 && cut_indices[t-1][i]==1 && (times[t]-times[t-1])>=(times[2]-times[1]-0.01)] for t in 2:tf] # tx10, idx4, m+stdx2
            y_to_std = [[sqrt(stds_pit[t][i][1]^2+stds_pit[t-1][i][1]^2)/(times[t]-times[t-1]) for (i,c) in enumerate(cut_indices[t]) if c==1 && cut_indices[t-1][i]==1 && (times[t]-times[t-1])>=(times[2]-times[1]-0.01)] for t in 2:tf]
    
            times_data = [times[t] for t in 2:tf if !isempty(y_to_mean[t-1])] 
            y_data = [weighted_mean_and_std(y_to_mean[t],y_to_std[t])[1] for t in 1:tf-1 if !isempty(y_to_mean[t])]
            dy = [weighted_mean_and_std(y_to_mean[t],y_to_std[t])[2] for t in 1:tf-1 if !isempty(y_to_mean[t])]
            
            # Widths plot 
                # Linear scale
            plot_pit_fits(all_fig2, 1, times_data, y_data, dy,
                          ["" for _ in IDs_pit], x_lims,
                          x_label="t (hs)", y_label="<v>=dx/dt (px/h)", 
                          do_fit=false, subplot=2)            

            #scatter!(allfig2, times_data[2:end], weighted_mean_and_std(mean_slope,std_mean_slope)[1] )
                #** y plotear arriba el promedio con las barras de error también **

            plot!(all_fig2, plot_title ="Distance traveled in interval + mean speeds "*title)
            display(all_fig2)
            savefig(all_fig2, main_folder*"distance_and_v_per_interval_$(pit_number).png")



                #$$$$$$$$$$$$ acá es donde tengo que aplicar el fit partido mejor
            
            p_value, alpha, constant_v_start, constant_v_end, v0, dv0, is_line, constant_v_plot = get_largest_flat_interval(times_data,abs.(y_data),dy)
            
            
            v0= -v0
            
                
            begin2 = floor(Int,tf/2)
            
            if constant_v_start<tf/2
                if constant_v_end<times[end-4]
                    begin2 = constant_v_end
                else
                    #**acá ver si conviene para un lado o para el otro
                    begin2 = constant_v_start
                end
            else
                begin2=constant_v_start
            end
            if !is_line
                begin2 = floor(Int,tf/2)
            end

            begin2 = begin2_best
                
            time_begin2 = times_data[begin2]
            #$$$$$$$$$$$$$$$$$$$$$$$
            #time_begin2 = params.τ

            
        

            #************************* ojo con el begin2 que no va a ser el mismo t para todos los casos, con una cantidad distinta de puntos! debería encontrar el t más cercano para cada lado 
            
            mean_slope1 = zeros(Float64, length(IDs_pit))
            std_mean_slope1 = zeros(Float64, length(IDs_pit))
            mean_r21 = zeros(Float64, length(IDs_pit))

            mean_slope2 = zeros(Float64, length(IDs_pit))
            std_mean_slope2 = zeros(Float64, length(IDs_pit))
            mean_r22 = zeros(Float64, length(IDs_pit))

            v_half_time_all = zeros(Float64, length(IDs_pit))
            dv_half_time_all = zeros(Float64,length(IDs_pit))

            mean_slopes_10hs = zeros(Float64, length(IDs_pit))
            std_slopes_10hs = zeros(Float64, length(IDs_pit))

            n_cols_in_legend = (begin2<2 || begin2>tf-2 || is_line==false) ? 2 : 3
            
            fig_split_fit = plot(constant_v_plot, constant_v_plot, layout=grid(1,2, widths=(3/8,5/8)), size=(2500,600),
                    margin=10mm, grid=true)
            empty!(fig_split_fit.subplots[2])

            dummyfig = plot(constant_v_plot, constant_v_plot, layout=grid(1,2, widths=(3/8,5/8)), size=(2500,600),
                    margin=10mm, grid=true)
                
            if isfile(main_folder*"careful_fit_slope_report.txt")
                careful_fit_slope_file = CSV.read(main_folder*"careful_fit_slope_report.txt", DataFrame, delim='\t')
            else
                columns = (; pit_num=String[], slope=Float64[], std_slope=Float64[], r2_slope=Float64[],slope2=Float64[],std_slope2=Float64[],r2_slope2=Float64[],p_value=Float64[], alpha=Float64[], break_point=Float64[],is_line=Bool[])
                careful_fit_slope_file = DataFrame(columns)
            end
            if isfile(main_folder*"careful_fit_mean_slope_report.txt")
                careful_fit_mean_slope_file = CSV.read(main_folder*"careful_fit_mean_slope_report.txt", DataFrame, delim='\t')
            else
                columns = (; pit_num=Int[], slope=Float64[], std_slope=Float64[], r2_slope=Float64[],slope2=Float64[],std_slope2=Float64[],r2_slope2=Float64[],p_value=Float64[], alpha=Float64[],break_point=Float64[],is_line=Bool[])
                careful_fit_mean_slope_file = DataFrame(columns)
            end
            if isfile(main_folder*"careful_fit_v0_report.txt")
                careful_fit_v0_file = CSV.read(main_folder*"careful_fit_v0_report.txt", DataFrame, delim='\t')
            else
                columns = (; pit_num=Int[], slope=Float64[], std_slope=Float64[],is_line=Bool[])
                careful_fit_v0_file = DataFrame(columns)
            end
            if isfile(main_folder*"v_half_time_report.txt")
                v_half_time_file = CSV.read(main_folder*"v_half_time_report.txt", DataFrame, delim='\t')
            else
                columns = (; pit_num=String[], slope=Float64[], std_slope=Float64[])
                v_half_time_file = DataFrame(columns)
            end
            if isfile(main_folder*"mean_v_half_time_report.txt")
                mean_v_half_time_file = CSV.read(main_folder*"mean_v_half_time_report.txt", DataFrame, delim='\t')
            else
                columns = (; pit_num=Int[], slope=Float64[], std_slope=Float64[])
                mean_v_half_time_file = DataFrame(columns)
            end
            if isfile(main_folder*"v_fit_10hs_report.txt")
                v_fit_10hs_file = CSV.read(main_folder*"v_fit_10hs_report.txt", DataFrame, delim='\t')
            else
                columns = (; pit_num=String[], slope=Float64[], std_slope=Float64[])
                v_fit_10hs_file = DataFrame(columns)
            end
            if isfile(main_folder*"mean_v_fit_10hs_report.txt")
                mean_v_fit_10hs_file = CSV.read(main_folder*"mean_v_fit_10hs_report.txt", DataFrame, delim='\t')
            else
                columns = (; pit_num=Int[], slope=Float64[], std_slope=Float64[])
                mean_v_fit_10hs_file = DataFrame(columns)
            end
            if isfile(main_folder*"all_vs_report.txt")
                all_vs_file = CSV.read(main_folder*"all_vs_report.txt", DataFrame, delim='\t')
            else
                columns = (; pit_num=Int[], naive_fit=Float64[], dnaive_fit=Float64[],careful_fit1=Float64[], dcareful_fit1=Float64[], careful_fit2=Float64[], dcareful_fit2=Float64[],careful_fit_v0=Float64[], dcareful_fit_v0=Float64[],v_half_time=Float64[], dv_half_time=Float64[],v_fit_10hs=Float64[], dv_fit_10hs=Float64[],is_line=Bool[])
                all_vs_file = DataFrame(columns)
            end

            for (i,ID) in enumerate(IDs_pit)

                index_id = findfirst(IDs_old[pit_number] .== ID)
    
                times_data, y, dy, xlog, ylog, dylog = get_x_and_y_and_ylog(means_pit, stds_pit, times, index_id, tf, 2)

                _,begin2 = findmin(x->abs(x-time_begin2),times_data) #**podría encontrar los dos más cercanos y tomarlos como índices para uno y otro lado para hacer el ajuste
                if times_data[begin2]<time_begin2
                    shift_begin = 1
                elseif times_data[begin2]>time_begin2
                    shift_begin = 1
                    begin2 -=1
                else
                    shift_begin = 0
                end

                _,half_time_index = findmin(x -> abs(x-tf/2), times_data)
                v_half_time = (y[half_time_index]-y[1])/(times_data[half_time_index]-times_data[1])
                dv_half_time = sqrt(dy[half_time_index]+dy[1])^2/(times_data[half_time_index]-times_data[1])

                v_half_time_all[i] = v_half_time
                dv_half_time_all[i] = dv_half_time

                _, idx_t10hs = findmin(x->abs(x-10),times_data)
                m10hs,dm10hs,_,_ = plot_pit_fits(dummyfig, i, times_data[1:idx_t10hs], y[1:idx_t10hs], dy[1:idx_t10hs],#dylog[1:begin2],
                              ["" for i in IDs_pit], x_lims,
                              x_label="t (hs)", y_label="d (px)", 
                              do_fit=true, subplot=1)

                mean_slopes_10hs[i] = m10hs
                std_slopes_10hs[i] = dm10hs

                #if !is_line
                    #begin2 = 1
                        
                    #a,da,m2,dm2, res2, r2_2 = plot_quadratic_fits(fig_split_fit, i, times_data[begin2:end], y[begin2:end], dy[begin2:end],#dylog[1:begin2],
                    #          ["" for i in IDs_pit], x_lims,
                    #          x_label="t (hs)", y_label="d (px)", 
                    #          do_fit=true, subplot=2)

                    #m2 = a*(tf+times[1])*0.5+m2
                    #dm2=sqrt(da*da*(tf+times[1])^2*0.25+dm2*dm2) #*****nidea

                    #m1,dm1,r2_1=NaN,NaN,NaN

                    #scatter!(fig_split_fit, times_data[begin2:end], res2, label=ID, color = cgrad(:rainbow)[i/length(IDs_pit)], subplot=1)

                #$$$$$$$$$$$$$$
                if begin2>3
                    m1,dm1, res1, r2_1 = plot_pit_fits(fig_split_fit, i, times_data[1:begin2], y[1:begin2], dy[1:begin2],#dylog[1:begin2],
                              ["" for i in IDs_pit], x_lims,
                              x_label="t (hs)", y_label="d (px)", 
                              do_fit=true, subplot=2)
                    #scatter!(fig_split_fit, times_data[1:begin2], res1, label=ID, color = cgrad(:rainbow)[i/length(IDs_pit)], subplot=1)                        
                else
                    m1,dm1,r2_1=NaN,NaN,NaN
                    begin2 = 1
                end
                if begin2<length(y)-3
                    m2,dm2, res2, r2_2 = plot_pit_fits(fig_split_fit, i, times_data[begin2+shift_begin:end], y[begin2+shift_begin:end], dy[(begin2+shift_begin):end],#dylog[begin2+1:end],
                          IDs_pit, x_lims,
                          x_label="t (hx)", y_label="d0-d (px)", 
                          do_fit=true, subplot=2)
                    #scatter!(fig_split_fit, times_data[begin2:end], res2, label=ID, color = cgrad(:rainbow)[i/length(IDs_pit)], subplot=1)    
                else
                    m2,dm2,r2_2=NaN,NaN,NaN
                end
                
                
                mean_slope1[i] = m1
                std_mean_slope1[i] = dm1
                mean_r21[i] = r2_1

                mean_slope2[i] = m2
                std_mean_slope2[i] = dm2
                mean_r22[i] = r2_2

                if make_files
                    push!(careful_fit_slope_file,vcat([ID],[m1],[dm1],[r2_1],[m2],[dm2],[r2_2],[p_value],[alpha],[times_data[begin2]],[is_line]))
                    push!(v_half_time_file,vcat([ID],[v_half_time_all[i]],[dv_half_time_all[i]]))
                    push!(v_fit_10hs_file,vcat([ID],[m10hs],[dm10hs]))
                end

            end

            if !is_line
                v0 = Statistics.mean(mean_slope2)
                dv0 = Statistics.mean(std_mean_slope2)
            end
            if make_files
                CSV.write(main_folder*"careful_fit_slope_report.txt",careful_fit_slope_file, delim="\t")
                CSV.write(main_folder*"v_half_time_report.txt",v_half_time_file, delim="\t")
                CSV.write(main_folder*"v_fit_10hs_report.txt",v_fit_10hs_file, delim="\t")
                
                push!(careful_fit_mean_slope_file, vcat([pit_number],[weighted_mean_and_std(filter(!isnan, mean_slope1),filter(!isnan, std_mean_slope1))[1]],[weighted_mean_and_std(filter(!isnan, mean_slope1),filter(!isnan, std_mean_slope1))[2]],Statistics.mean(filter(!isnan, mean_r21)),
                                [weighted_mean_and_std(filter(!isnan, mean_slope2),filter(!isnan, std_mean_slope2))[1]],[weighted_mean_and_std(filter(!isnan, mean_slope2),filter(!isnan, std_mean_slope2))[2]],Statistics.mean(filter(!isnan, mean_r22)),[p_value],[alpha],[time_begin2],[is_line]))
                CSV.write(main_folder*"careful_fit_mean_slope_report.txt",careful_fit_mean_slope_file, delim="\t")
                
                push!(careful_fit_v0_file, vcat([pit_number], [v0], [dv0], [is_line]))
                CSV.write(main_folder*"careful_fit_v0_report.txt",careful_fit_v0_file, delim="\t") 
                
                push!(mean_v_half_time_file, vcat([pit_number],[weighted_mean_and_std(v_half_time_all,dv_half_time_all)[1]],[weighted_mean_and_std(v_half_time_all,dv_half_time_all)[2]]))
                CSV.write(main_folder*"mean_v_half_time_report.txt",mean_v_half_time_file, delim="\t")

                push!(mean_v_fit_10hs_file, vcat([pit_number],[weighted_mean_and_std(mean_slopes_10hs,std_slopes_10hs)[1]],[weighted_mean_and_std(mean_slopes_10hs,std_slopes_10hs)[2]]))
                CSV.write(main_folder*"mean_v_fit_10hs_report.txt",mean_v_fit_10hs_file, delim="\t")
                
                #**acá ojo que estoy usando resultados de plot_area !!! va a tirar error si no corro los dos juntos
                push!(all_vs_file, vcat([pit_number],[weighted_mean_and_std(mean_slope,std_mean_slope)[1]],[weighted_mean_and_std(mean_slope,std_mean_slope)[2]],
                                [weighted_mean_and_std(filter(!isnan, mean_slope1),filter(!isnan, std_mean_slope1))[1]],[weighted_mean_and_std(filter(!isnan, mean_slope1),filter(!isnan, std_mean_slope1))[2]],
                                [weighted_mean_and_std(filter(!isnan, mean_slope2),filter(!isnan, std_mean_slope2))[1]],[weighted_mean_and_std(filter(!isnan, mean_slope2),filter(!isnan, std_mean_slope2))[2]],
                                [v0], [dv0],
                                [weighted_mean_and_std(v_half_time_all,dv_half_time_all)[1]],[weighted_mean_and_std(v_half_time_all,dv_half_time_all)[2]],
                                [weighted_mean_and_std(mean_slopes_10hs,std_slopes_10hs)[1]],[weighted_mean_and_std(mean_slopes_10hs,std_slopes_10hs)[2]],
                                [is_line]))
                CSV.write(main_folder*"all_vs_report.txt",all_vs_file, delim="\t")

            end
            #vline!(fig_split_fit, [time_begin2], label="fit break", color="black", linestyle=:dash, subplot=1)
            vline!(fig_split_fit, [time_begin2], label="", color="black", linestyle=:dash, legend=:outertopright, legendcolumns=n_cols_in_legend, subplot=2)
            plot!(fig_split_fit, plot_title ="Split fit d(t) $(pit_number)")
            ylims!(fig_split_fit,subplot=2,(-10.0,Inf))
            display(fig_split_fit)
            savefig(fig_split_fit, main_folder*"careful_fit_split_$(pit_number).png")

            
        end
        if plot_sqrt
            plot!(sqrt_fig, plot_title="Distance traveled "*title)
            display(sqrt_fig)

            if pit_number == 2
                #savefig(log_fig, main_folder*"fit_pit$(pit_number)_vertical_log_h.png")
                #savefig(widths_fig, main_folder*"fit_pit$(pit_number)_vertical_d.png")
                #savefig(linear_fig, main_folder*"fit_pit$(pit_number)_vertical_h.png")
                savefig(sqrt_fig, main_folder*"linear_and_sqrt_fit_$(pit_number).png")
            end
        end
        
        
        
    end
    plot!(similar_di_fig, plot_title="d(t) with similar d(0)", legend=:outertopright)
    display(similar_di_fig)
    savefig(similar_di_fig, main_folder*"similar_di_plots.png")

    just_widths_fig = plot(just_widths_fig1, just_widths_fig2, layout=grid(1,2, widths=(4/8,4/8)), size=(1200,400), 
                    margin=10mm, legend=false, grid=true, plot_title="Widths vs time")
    ylims!(just_widths_fig,subplot=2,(-10.0,Inf))
    display(just_widths_fig)
    savefig(just_widths_fig, main_folder*"just_widths_1_and_4.png")


end





















# ============================================================================
# CONFIGURATION AND SETUP HELPERS
# ============================================================================

function setup_containers(tf::Int, n_of_classes::Int)
    """Initialize containers for means, stds, and IDs."""
    means = [Vector{Vector{Vector{Float64}}}(undef, tf) for _ in 1:n_of_classes]
    stds = [Vector{Vector{Vector{Float64}}}(undef, tf) for _ in 1:n_of_classes]
    IDs_old = [Vector{String}(undef, tf) for _ in 1:n_of_classes]
    return means, stds, IDs_old
end

function setup_plot_containers(IDs_pit::Vector{String})
    """Initialize containers for plot data."""
    return (
        mean_slope = zeros(Float64, length(IDs_pit)),
        std_mean_slope = zeros(Float64, length(IDs_pit)),
        r2_slope = zeros(Float64, length(IDs_pit)),
        initial_w = zeros(Float64, length(IDs_pit))
    )
end

function setup_all_plot_containers(IDs_pit::Vector{String})
    """Initialize containers for all-plots."""
    n = length(IDs_pit)
    return (
        mean_slope1 = zeros(Float64, n),
        std_mean_slope1 = zeros(Float64, n),
        mean_slope2 = zeros(Float64, n),
        std_mean_slope2 = zeros(Float64, n)
    )
end

function get_x_limits(times::Vector{Float64})
    """Get linear and log x-axis limits."""
    xlog_lims = [log10(times[2]), log10(times[end] + 1)]
    x_lims = [minimum(times), maximum(times)]
    return xlog_lims, x_lims
end

function load_mean_slopes_file(main_folder::String)
    """Load or create mean slopes report file."""
    filepath = main_folder * "mean_slope_report.txt"
    if isfile(filepath)
        return CSV.read(filepath, DataFrame, delim='\t')
    else
        columns = (; pit_num=Int[], slope=Float64[], std_slope=Float64[], r2_slope=Float64[])
        return DataFrame(columns)
    end
end

# ============================================================================
# PLOT CREATION HELPERS
# ============================================================================

function create_area_plots()
    """Create figure containers for area analysis."""
    area_fig = plot(
        layout = grid(1, 2, widths=(4/8, 4/8)),
        size = (1400, 400),
        margin = 10mm,
        legend = :outertopright,
        grid = true,
        legendcolumns = 2
    )
    res_fig = plot(
        size = (2000, 400),
        margin = 10mm,
        legend = :outertopright,
        grid = true
    )
    r2_hist_fig = Vector{Any}(undef, 0)  # Will be filled later
    return area_fig, res_fig, r2_hist_fig
end

function create_all_plots()
    """Create figure containers for all-plots."""
    all_fig = plot(
        layout = grid(1, 2, widths=(3/8, 5/8)),
        size = (2000, 400),
        margin = 10mm,
        legend = :outertopright,
        grid = true,
        legendcolumns = 3
    )
    plot!(all_fig, subplot=1, legend=false)
    
    all_fig2 = plot(
        layout = grid(1, 2, widths=(3/8, 5/8)),
        size = (2000, 400),
        margin = 10mm,
        legend = :outertopright,
        grid = true,
        legendcolumns = 3
    )
    plot!(all_fig2, subplot=1, legend=false)
    
    return all_fig, all_fig2
end

function create_sqrt_plot()
    """Create figure container for sqrt analysis."""
    return plot(
        layout = grid(1, 2, widths=(4/8, 4/8)),
        size = (1400, 400),
        margin = 10mm,
        legend = :outertopright,
        grid = true,
        legendcolumns = 2
    )
end

function create_mean_width_plot()
    """Create figure container for mean width analysis."""
    return plot(
        layout = grid(1, 2, widths=(4/8, 4/8)),
        size = (1400, 400),
        margin = 10mm,
        legend = :outertopright,
        grid = true,
        legendcolumns = 2
    )
end

# ============================================================================
# DATA PROCESSING HELPERS
# ============================================================================

function get_pit_data(means_pit, stds_pit, times, ID, index_id, tf)
    """Extract x, y, and log data for a given pit and ID."""
    times_data, y, dy, xlog, ylog, dylog = get_x_and_y_and_ylog(
        means_pit, stds_pit, times, index_id, tf, 2
    )
    return times_data, y, dy, xlog, ylog, dylog
end

function get_area_data(df_areas, ID, tf)
    """Extract area data for a given ID."""
    target_ID = ID[1:end-5]
    row_idx = findfirst(df_areas.ID .== target_ID)
    area_init = df_areas[row_idx, 2]
    areas = [df_areas[row_idx, j+1] / area_init for j in 1:tf]
    return areas
end

function filter_areas(times, areas)
    """Filter areas where area < 2 (not exploded)."""
    mask = areas .< 2
    return times[mask], areas[mask]
end

# ============================================================================
# FIT AND STATISTICS HELPERS
# ============================================================================

function perform_single_fit(times_data, y, dy)
    """Perform a single linear fit and return statistics."""
    m, dm, res, r2 = perform_linear_fit(times_data, y, dy)
    return m, dm, res, r2
end

function compute_r2_distribution(n_fits, times_data, y, dy, n_points)
    """Compute R² distribution via bootstrap."""
    return run_many_fits(n_fits, times_data, y, dy, n_points)
end

function check_quadratic_fit(times_data, y, dy, max_points=8)
    """Check if a quadratic fit is better at the beginning."""
    n = min(max_points, length(y))
    quad_term, _, _, (err_quad, _, _), _, r2_quad = perform_quadratic_fit(
        times_data[1:n], y[1:n], dy[1:n]
    )
    is_significant = abs(quad_term) > err_quad
    return quad_term, err_quad, r2_quad, is_significant
end

function compute_weighted_stats(values, errors)
    """Compute weighted mean and std of values with errors."""
    return weighted_mean_and_std(values, errors)
end

# ============================================================================
# REPORTING HELPERS
# ============================================================================

function initialize_report_files(main_folder::String, tf::Int)
    """Initialize report files for width and slope."""
    width_file = init_report_file(
        main_folder * "width_report.txt", 
        [[i for i in 1:tf]]
    )
    slope_file = init_report_file(
        main_folder * "slope_report.txt", 
        [[1, 2]]
    )
    return width_file, slope_file
end

function save_width_data!(width_file, ID, y, tf)
    """Save width data to report file."""
    if length(y) == tf
        push!(width_file, vcat([ID], y))
    else
        push!(width_file, vcat([ID], y, [0.0 for _ in 1:tf-length(y)]))
    end
end

function save_slope_data!(slope_file, ID, m, dm)
    """Save slope data to report file."""
    push!(slope_file, vcat([ID], [m], [dm]))
end

function log_slope_statistics(mean_slope, std_mean_slope, initial_w, w_thr=350)
    """Log slope statistics with optional width-based grouping."""
    # All slopes
    wm, ws = weighted_mean_and_std(mean_slope, std_mean_slope)
    @info "Mean slope: $(value_with_error(wm, ws, 1))"
    @info "Mean slope: $wm ± $ws"
    
    # Wide initial widths
    wide_mask = initial_w .> w_thr
    if any(wide_mask)
        wm_wide, ws_wide = weighted_mean_and_std(
            mean_slope[wide_mask], 
            std_mean_slope[wide_mask]
        )
        @info "Mean slope wide initial (>$(w_thr)px): $(value_with_error(wm_wide, ws_wide, 1))"
        @info "Mean slope wide initial (>$(w_thr)px): $wm_wide ± $ws_wide"
    end
    
    # Narrow initial widths
    narrow_mask = initial_w .< w_thr
    if any(narrow_mask)
        wm_narrow, ws_narrow = weighted_mean_and_std(
            mean_slope[narrow_mask], 
            std_mean_slope[narrow_mask]
        )
        @info "Mean slope narrow initial (<$(w_thr)px): $(value_with_error(wm_narrow, ws_narrow, 1))"
        @info "Mean slope narrow initial (<$(w_thr)px): $wm_narrow ± $ws_narrow"
    end
end

function update_mean_slopes_file!(mean_slopes_file, pit_number, mean_slope, std_mean_slope, r2_slope)
    """Update the mean slopes report file."""
    wm, ws = weighted_mean_and_std(mean_slope, std_mean_slope)
    push!(mean_slopes_file, vcat(
        [pit_number], 
        [wm], 
        [ws], 
        [Statistics.mean(r2_slope)]
    ))
end

# ============================================================================
# VELOCITY DATA PROCESSING
# ============================================================================

function compute_velocity_data(means_pit, stds_pit, IDs_pit, times, tf)
    """Compute velocity (dx/dt) data from pit measurements."""
    # Determine which indices have valid points for each time
    cut_indices = [
        [means_pit[t][id][1] != -1.0 for id in 1:length(IDs_pit)]
        for t in 1:tf
    ]
    
    # Compute velocities and their errors
    y_to_mean = [
        [
            abs(means_pit[t][i][1] - means_pit[t-1][i][1]) / (times[t] - times[t-1])
            for (i, c) in enumerate(cut_indices[t])
            if c && cut_indices[t-1][i]
        ]
        for t in 2:tf
    ]
    
    y_to_std = [
        [
            sqrt(stds_pit[t][i][1]^2 + stds_pit[t-1][i][1]^2) / (times[t] - times[t-1])
            for (i, c) in enumerate(cut_indices[t])
            if c && cut_indices[t-1][i]
        ]
        for t in 2:tf
    ]
    
    # Aggregate velocity data
    times_data = [times[t] for t in 2:tf if !isempty(y_to_mean[t-1])]
    y_data = [weighted_mean_and_std(y_to_mean[t], y_to_std[t])[1] for t in 1:tf-1 if !isempty(y_to_mean[t])]
    dy = [weighted_mean_and_std(y_to_mean[t], y_to_std[t])[2] for t in 1:tf-1 if !isempty(y_to_mean[t])]
    
    return times_data, y_data, dy
end

function compute_mean_width_data(means_pit, stds_pit, IDs_pit, times, tf)
    """Compute mean width data across all IDs for each time point."""
    # Determine which indices have valid points for each time
    cut_indices = [
        [means_pit[t][id][1] != -1.0 for id in 1:length(IDs_pit)]
        for t in 1:tf
    ]
    
    # Collect width data
    y_to_mean = [
        [means_pit[t][i][1] for (i, c) in enumerate(cut_indices[t]) if c]
        for t in 1:tf
    ]
    y_to_std = [
        [stds_pit[t][i][1] for (i, c) in enumerate(cut_indices[t]) if c]
        for t in 1:tf
    ]
    
    # Filter times with enough data
    min_points = length(IDs_pit) / 2
    times_data = [times[t] for t in 1:tf if length(y_to_mean[t]) > min_points]
    
    # Compute weighted means
    y0, dy0 = weighted_mean_and_std(y_to_mean[1], y_to_std[1])
    y_data = [
        weighted_mean_and_std(y_to_mean[t], y_to_std[t])[1]
        for t in 2:tf
        if length(y_to_mean[t]) > min_points
    ]
    dy = [
        weighted_mean_and_std(y_to_mean[t], y_to_std[t])[2]
        for t in 2:tf
        if length(y_to_mean[t]) > min_points
    ]
    
    # Compute log-space quantities
    y_ref_y0 = abs.(y_data .- y0)
    xlog = log10.(times_data[2:end])
    ylog = log10.(y_ref_y0)
    dylog = log10(exp(1.0)) .* sqrt.(dy.^2 .+ dy0^2) ./ y_ref_y0
    
    return times_data, vcat(y0, y_data), vcat(dy0, dy), xlog, ylog, dylog
end

# ============================================================================
# MAIN PIT PROCESSING
# ============================================================================

function process_area_analysis!(
    area_fig, res_fig, r2_hist_fig,
    means_pit, stds_pit, IDs_pit, times, tf,
    df_areas, xlog_lims, x_lims,
    containers, report_files, main_folder, pit_number,
    make_files
)
    """Process area analysis for a single pit."""
    n_ids = length(IDs_pit)
    
    for (i, ID) in enumerate(IDs_pit)
        # Get data
        index_id = findfirst(IDs_old[pit_number] .== ID)
        times_data, y, dy, xlog, ylog, dylog = get_pit_data(
            means_pit, stds_pit, times, ID, index_id, tf
        )
        
        containers.initial_w[i] = y[1]
        
        # Plot areas
        areas = get_area_data(df_areas, ID, tf)
        times_area, areas_filtered = filter_areas(times, areas)
        
        plot_pit_fits(
            area_fig, i,
            times_area, areas_filtered,
            [nothing for _ in times_area],
            IDs_pit, xlog_lims,
            x_label="t (hs)", y_label="Area (% of A(t0))",
            do_fit=false, subplot=1
        )
        
        # Plot widths with fit
        m, dm, res, r2 = plot_pit_fits(
            area_fig, i,
            times_data, y, dy,
            IDs_pit, x_lims,
            x_label="t (hs)", y_label="Wound width (px)",
            do_fit=true, subplot=2
        )
        
        # R² distribution
        n_fits = 100000
        r2_dist = compute_r2_distribution(n_fits, times_data, y, dy, length(y))
        r2_hist_fig[i] = histogram(
            r2_dist, bins=30, title=ID,
            color=cgrad(:rainbow)[i/n_ids], legend=false, alpha=0.7
        )
        scatter!([r2], [1], markershape=:star5, markersize=12,
                 color=:black, markercolor=:black, label="Selected point",
                 markerstrokewidth=0)
        xlims!(-0.05, 1.05)
        
        # Check quadratic fit
        quad_term, err_quad, r2_quad, is_significant = check_quadratic_fit(
            times_data, y, dy
        )
        
        # Store results
        containers.mean_slope[i] = m
        containers.std_mean_slope[i] = dm
        containers.r2_slope[i] = r2
        
        # Residuals
        scatter!(res_fig, times_data, res, label=ID,
                 color=cgrad(:rainbow)[i/n_ids])
        
        # Save to files
        if make_files
            save_slope_data!(report_files.slope_file, ID, m, dm)
            save_width_data!(report_files.width_file, ID, y, tf)
        end
    end
    
    return containers
end

function process_all_analysis!(
    all_fig, all_fig2,
    means_pit, stds_pit, IDs_pit, times, tf,
    xlog_lims, x_lims,
    containers
)
    """Process all-plots analysis for a single pit."""
    n_ids = length(IDs_pit)
    
    for (i, ID) in enumerate(IDs_pit)
        index_id = findfirst(IDs_old[pit_number] .== ID)
        times_data, y, dy, xlog, ylog, dylog = get_pit_data(
            means_pit, stds_pit, times, ID, index_id, tf
        )
        
        # Log-log derivative plot
        plot_pit_fits(
            all_fig, i,
            times_data[2:end-1],
            [(ylog[t+1] - ylog[t]) / (xlog[t+1] - xlog[t]) for t in 1:length(ylog)-1],
            [nothing for _ in 1:length(dylog)-1],
            IDs_pit, x_lims,
            x_label="t (hs)", y_label="dlog",
            do_fit=false, subplot=1
        )
        ylims!(all_fig, subplot=1, (-2.5, 2.5))
        
        # Two-line fit in log-log space
        begin2, x_split, fit1, fit2, best_ssr = best_two_line_fit(
            xlog, ylog, dylog
        )
        
        # First segment fit
        m1, dm1, res1, r2_1 = plot_pit_fits(
            all_fig, i,
            xlog[1:begin2], ylog[1:begin2],
            [nothing for _ in 1:begin2],
            ["" for _ in IDs_pit], xlog_lims,
            x_label="log10(t)", y_label="log10(d)",
            do_fit=true, subplot=2
        )
        
        # Second segment fit
        m2, dm2, res2, r2_2 = plot_pit_fits(
            all_fig, i,
            xlog[begin2+1:end], ylog[begin2+1:end],
            [nothing for _ in (begin2+1):length(dylog)],
            IDs_pit, xlog_lims,
            x_label="log10(t)", y_label="log10(d0-d)",
            do_fit=true, subplot=2
        )
        
        containers.mean_slope1[i] = m1
        containers.std_mean_slope1[i] = dm1
        containers.mean_slope2[i] = m2
        containers.std_mean_slope2[i] = dm2
        
        # Distance traveled in time
        plot_pit_fits(
            all_fig2, i,
            times_data[2:end],
            [abs(y[t] - y[t-1]) for t in 2:length(y)],
            [nothing for _ in 2:length(y)],
            IDs_pit, x_lims,
            x_label="t (hs)", y_label="dy",
            do_fit=false, subplot=1
        )
    end
    
    return containers
end

function process_sqrt_analysis!(
    sqrt_fig,
    means_pit, stds_pit, IDs_pit, times, tf,
    x_lims
)
    """Process sqrt-plot analysis for a single pit."""
    for (i, ID) in enumerate(IDs_pit)
        index_id = findfirst(IDs_old[pit_number] .== ID)
        times_data, y, dy, _, _, _ = get_pit_data(
            means_pit, stds_pit, times, ID, index_id, tf
        )
        
        # Linear plot: d vs t
        d = abs.(y .- y[1])
        d_err = vcat(0.0001, sqrt.(dy[2:end].^2 .+ dy[1]^2))
        
        plot_pit_fits(
            sqrt_fig, i,
            times_data, d, d_err,
            IDs_pit, x_lims,
            x_label="t (hs)", y_label="d=d0-d(t) (px)",
            do_fit=true, subplot=1
        )
        
        # Sqrt plot: d vs sqrt(t)
        plot_pit_fits(
            sqrt_fig, i,
            sqrt.(times_data)[2:end],
            d[2:end],
            sqrt.(d_err[2:end].^2 .+ d_err[1]^2),
            ["" for _ in IDs_pit],
            sqrt.(x_lims),
            x_label="sqrt(t) (sqrt(hs))", y_label="d (px)",
            do_fit=true, subplot=2
        )
    end
end

function process_mean_width_analysis!(
    mean_fig,
    means_pit, stds_pit, IDs_pit, times, tf,
    xlog_lims, x_lims
)
    """Process mean width analysis for a single pit."""
    # Compute mean width data
    times_data, y, dy, xlog, ylog, dylog = compute_mean_width_data(
        means_pit, stds_pit, IDs_pit, times, tf
    )
    
    # Linear plot
    plot_pit_fits(
        mean_fig, 1,
        times_data, y, dy,
        [""], x_lims,
        x_label="t (hs)", y_label="d (px)",
        do_fit=false, subplot=1
    )
    
    # Log-log two-line fit
    begin2, x_split, fit1, fit2, best_ssr = best_two_line_fit(
        xlog, ylog, dylog
    )
    
    plot_pit_fits(
        mean_fig, 1,
        xlog[1:begin2], ylog[1:begin2], dylog[1:begin2],
        [""], xlog_lims,
        x_label="log10(t)", y_label="log10(d0-d)",
        do_fit=true, subplot=2
    )
    
    plot_pit_fits(
        mean_fig, 1,
        xlog[begin2+1:end], ylog[begin2+1:end], dylog[begin2+1:end],
        ["black"], xlog_lims,
        x_label="log10(t)", y_label="log10(d0-d)",
        do_fit=true, subplot=2
    )
    
    plot!(mean_fig, plot_title="Mean width from "*title, 
          legend=:outertopright)
end

# ============================================================================
# MAIN FUNCTION (REFACTORED)
# ============================================================================

function main2(
    pits::Vector{String},
    which_pits::Vector{Int64},
    times::Vector{Float64},
    tf::Int64,
    left_wall::Dict,
    right_wall::Dict,
    widths_distribution::Dict,
    main_folder::String,
    plot_widths_t1::Bool,
    plot_area::Bool,
    plot_all::Bool,
    plot_sqrt::Bool,
    plot_mean_widths::Bool,
    nrois::Int,
    invert::Bool,
    n_of_classes::Int,
    cut_ids=nothing,
    make_files=false
)
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    # Setup containers
    means, stds, IDs_old = setup_containers(tf, n_of_classes)
    accumulate_data!(means, stds, IDs_old, widths_distribution, widths_distribution,
                     pits, tf, nrois, invert)
    
    # Filter IDs
    IDs = filter_cols(deepcopy(IDs_old), cut_ids, invert)
    println(cut_ids, IDs, IDs_old)
    
    # Load data
    df_areas = CSV.File(main_folder * "/scratch_report.txt") |> DataFrame
    xlog_lims, x_lims = get_x_limits(times)
    mean_slopes_file = load_mean_slopes_file(main_folder)
    
    # Initial widths plot
    if plot_widths_t1
        plot_initial_widths(
            IDs, widths_distribution,
            ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2", 
             "A3", "B3", "C3", "D3", "A4", "B4", "C4", "D4",
             "A5", "B5", "C5", "D5", "A6", "B6", "C6", "D6"],
            tf,
            main_folder * "fit_initial_position_wall.png",
            nrois
        )
    end
    
    # ========================================================================
    # MAIN LOOP OVER PITS
    # ========================================================================
    
    for pit_number in which_pits
        means_pit = means[pit_number]
        stds_pit = stds[pit_number]
        IDs_pit = IDs[pit_number]
        title = set_title(pit_number)
        
        # Initialize plot-specific containers
        if plot_area
            area_fig, res_fig, r2_hist_fig = create_area_plots()
            containers = setup_plot_containers(IDs_pit)
            report_files = initialize_report_files(main_folder, tf)
        end
        
        if plot_all
            all_fig, all_fig2 = create_all_plots()
            all_containers = setup_all_plot_containers(IDs_pit)
        end
        
        if plot_sqrt
            sqrt_fig = create_sqrt_plot()
        end
        
        if plot_mean_widths
            mean_fig = create_mean_width_plot()
        end
        
        # --------------------------------------------------------------------
        # AREA ANALYSIS
        # --------------------------------------------------------------------
        
        if plot_area
            containers = process_area_analysis!(
                area_fig, res_fig, r2_hist_fig,
                means_pit, stds_pit, IDs_pit, times, tf,
                df_areas, xlog_lims, x_lims,
                containers, report_files, main_folder, pit_number,
                make_files
            )
            
            # Finalize area plots
            plot!(area_fig, plot_title="All (area + widths from histograms) "*title)
            ylims!(area_fig, subplot=2, (-10.0, Inf))
            display(area_fig)
            savefig(area_fig, main_folder * "areas_and_widths_$(pit_number).png")
            
            plot!(res_fig, plot_title="Residuals from linear fit "*title)
            display(res_fig)
            
            # Log statistics
            log_slope_statistics(
                containers.mean_slope,
                containers.std_mean_slope,
                containers.initial_w
            )
            
            # Save reports
            if make_files
                CSV.write(main_folder * "width_report.txt", report_files.width_file, delim='\t')
                CSV.write(main_folder * "slope_report.txt", report_files.slope_file, delim='\t')
                update_mean_slopes_file!(
                    mean_slopes_file, pit_number,
                    containers.mean_slope,
                    containers.std_mean_slope,
                    containers.r2_slope
                )
                CSV.write(main_folder * "mean_slope_report.txt", mean_slopes_file, delim="\t")
            end
        end
        
        # --------------------------------------------------------------------
        # ALL ANALYSIS
        # --------------------------------------------------------------------
        
        if plot_all
            all_containers = process_all_analysis!(
                all_fig, all_fig2,
                means_pit, stds_pit, IDs_pit, times, tf,
                xlog_lims, x_lims,
                all_containers
            )
            
            plot!(all_fig, plot_title="Distance traveled "*title)
            display(all_fig)
            
            # Compute velocity data
            times_data, y_data, dy = compute_velocity_data(
                means_pit, stds_pit, IDs_pit, times, tf
            )
            
            # Plot velocity
            plot_pit_fits(
                all_fig2, 1,
                times_data, y_data, dy,
                [""], x_lims,
                x_label="t (hs)", y_label="<v>=dx/dt (px/h)",
                do_fit=false, subplot=2
            )
            
            plot!(all_fig2, plot_title="Distance traveled in interval "*title)
            display(all_fig2)
            
            # Find largest flat interval
            get_largest_flat_interval(times_data, y_data, dy)
        end
        
        # --------------------------------------------------------------------
        # SQRT ANALYSIS
        # --------------------------------------------------------------------
        
        if plot_sqrt
            process_sqrt_analysis!(
                sqrt_fig,
                means_pit, stds_pit, IDs_pit, times, tf,
                x_lims
            )
            
            plot!(sqrt_fig, plot_title="Distance traveled "*title)
            display(sqrt_fig)
            
            if pit_number == 2
                savefig(sqrt_fig, main_folder * "linear_and_sqrt_fit_$(pit_number).png")
            end
        end
        
        # --------------------------------------------------------------------
        # MEAN WIDTH ANALYSIS
        # --------------------------------------------------------------------
        
        if plot_mean_widths
            process_mean_width_analysis!(
                mean_fig,
                means_pit, stds_pit, IDs_pit, times, tf,
                xlog_lims, x_lims
            )
            
            display(mean_fig)
        end
    end
end
