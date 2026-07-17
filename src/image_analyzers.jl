using Images               # https://juliaimages.org/latest/tutorials/quickstart/
using ImageFiltering       # https://juliaimages.org/ImageFiltering.jl/stable/function_reference/#ImageFiltering.imfilter
using OffsetArrays
using Statistics
using ImageSegmentation
using ImageMorphology      # https://juliaimages.org/latest/pkgs/segmentation/
using StatsBase
using Logging
using Plots

function threshold_colors(color, mean_color)
    """
    Bin every pixel in the image into 3 categories, 
    with some thresholds for its grayscale color.
    
    Threshold 1, 0.9, is for the lens' shadow. 
    Threshold 2, mean_color, is for the cells. 
    Below threshold 2, we have the wound.
    """
    
    if color >= 0.9
        return 1.0
    elseif color > mean_color
        return 0.5
    else
        return 0.0
    end
end


function blur_image!(im,r,n)
    """
    Apply n times a Gaussian kernel with radius r to im.
    """
    
    G = Kernel.gaussian(r)
    for i in 1:n
        im = abs.(imfilter(im, G))
    end
    
    return im
end


function detect_edges(im)
    """
    Apply a Laplacian kernel to im, and blur it once.
    """
    
    edge_detect = convert(AbstractArray, Kernel.Laplacian() )
    
    filtered_gray_im = Gray.(abs.(imfilter(blur_image!(Gray.(im),1,1), edge_detect)))
    return filtered_gray_im ./ maximum(filtered_gray_im)
end



# Used for uneven illuminated background
function rolling_ball_background_subtraction(img, radius=100)
    background = imfilter(img, Kernel.gaussian(radius))
    #return clamp(img .- background, 0, 1)  # Ensure non-negative values
    return img.-background
end






function bandpass_filter(img, filter_small, filter_large)
    """
    - Difference of Gaussians (DoG) bandpass
    - Autoscale after filtering (min→0, max→255)
    - Saturate image (clip to 0–255)
    
    Filter small structures down to $(filter_small) pixels
    and filter large structures up to $(filter_large) pixels.
    """

    # Ensure float for processing
    imgf = float.(img)

    # --------------------------------------------------------
    # Difference-of-Gaussians for background smoothing: 
    # smaller radius keeps image mostly intact, while bigger radius 
    # blurs it. The difference evidences homogeneous zones, and 
    # we will substract them.
    # -------------------------------------------------------- #**es un poco más sofisticado que el filtro laplaciano
    σ_small = filter_small 
    σ_large = filter_large

    blur_small = imfilter(imgf, Kernel.gaussian(σ_small))
    blur_large = imfilter(imgf, Kernel.gaussian(σ_large))

    band = blur_small .- blur_large   # Values closer to zero mean more homogeneous zones

    #**deberían ser dos funciones separadas esto no?
     
    # --------------------------------------------------------
    # Find the least common intensity values and saturate image
    # --------------------------------------------------------
    
    #h = plot(legend=:outertopright, grid=true)
    #histogram!(h,vec(rawview(channelview((band)))))
    #display(h)

    data = vec(rawview(channelview((band))))
    h = fit(Histogram, data)
    max_count, max_idx = findmax(h.weights)
    max_bin_edge = h.edges[1][max_idx+1] 
    
    # Calculate the 10th, 20th, ..., 90th percentiles
    deciles = quantile(data, 0.01:0.01:0.99)
    
    # The 10th decile (10th percentile)
    decile_1 = deciles[1]
    # The 90th decile 
    decile_9 = deciles[99]

    clipped = clamp.(band, decile_1, decile_9) # te sacás de encima los px con intensidades extremas y aumentás el contraste de todo lo del medio
    
    return (clipped.-decile_1)./(decile_9-decile_1)
    
end




"""
    fiji_subtract_background_merged(img; radius=50, light_background=true, smoothing=true)

Tile-based sliding-paraboloid background subtraction closely matching FIJI's
Subtract Background defaults (sliding paraboloid + tile/block behaviour).

Arguments:
- img : 2D image (Array). Works with UInt8 or Float images. Values assumed 0..255 for 8-bit.
- radius : integer radius in pixels (recommended e.g. 50).
- light_background : if true, behave like FIJI's "Light background" (invert before & after).
- smoothing : apply a small 3x3 mean smoothing to the stitched background (FIJI default).

Returns:
- UInt8 image (Array{UInt8,2}) with background subtracted and clamped 0..255.
"""
#sliding paraboloid algorithm
function fiji_subtract_background_merged(img; radius::Integer=50, light_background::Bool=true, smoothing::Bool=true)
    # Convert to float in 0..255 range for processing
    imgf = Array{Float64}(img)
    # If input was normalized (0..1), rescale to 0..255
    if maximum(imgf) ≤ 1.0
        imgf .*= 255.0
    end

    # Invert first if light background (FIJI strategy)
    proc_img = light_background ? (255.0 .- imgf) : imgf

    h, w = size(proc_img)

    # tile size and number of tiles
    tile = 2 * radius + 1
    ntx = cld(w, tile)
    nty = cld(h, tile)

    # Precompute paraboloid for offsets -radius:radius
    arc = collect(-radius:radius)
    parab = (arc .^ 2) ./ (2.0 * max(radius, 1))  # avoid div by zero if radius==0

    # Prepare background
    background = zeros(Float64, h, w)

    # Process tiles (non-overlapping) and use padded neighbourhood of +/- radius
    for ty in 0:(nty-1)
        y0 = ty * tile + 1
        y1 = min(y0 + tile - 1, h)
        ey0 = max(1, y0 - radius)
        ey1 = min(h, y1 + radius)

        for tx in 0:(ntx-1)
            x0 = tx * tile + 1
            x1 = min(x0 + tile - 1, w)
            ex0 = max(1, x0 - radius)
            ex1 = min(w, x1 + radius)

            # padded subimage (view)
            sub = @view proc_img[ey0:ey1, ex0:ex1]
            sh, sw = size(sub)

            # First pass: vertical 1-D paraboloid minimum for each column
            tmp = fill(Inf, sh, sw)
            @inbounds for c in 1:sw
                # column view
                col = @view sub[:, c]
                tmpcol = @view tmp[:, c]
                for r in 1:sh
                    # compute r-range in sub coordinates
                    rmin = max(1, r - radius)
                    rmax = min(sh, r + radius)
                    localmin = Inf
                    # loop over ry indices
                    for ry in rmin:rmax
                        dy = ry - r
                        pval = parab[dy + radius + 1]   # +1 for 1-based indexing
                        v = col[ry] + pval
                        if v < localmin
                            localmin = v
                        end
                    end
                    tmpcol[r] = localmin
                end
            end

            # Second pass: horizontal 1-D paraboloid minimum for each row
            bg_sub = fill(Inf, sh, sw)
            @inbounds for r in 1:sh
                rowtmp = @view tmp[r, :]
                bgrow = @view bg_sub[r, :]
                for c in 1:sw
                    cmin = max(1, c - radius)
                    cmax = min(sw, c + radius)
                    localmin = Inf
                    for cx in cmin:cmax
                        dx = cx - c
                        pval = parab[dx + radius + 1]
                        v = rowtmp[cx] + pval
                        if v < localmin
                            localmin = v
                        end
                    end
                    bgrow[c] = localmin
                end
            end

            # Extract central tile region (exclude padded borders) and place into background
            ry0 = y0 - ey0 + 1
            ry1 = y1 - ey0 + 1
            rx0 = x0 - ex0 + 1
            rx1 = x1 - ex0 + 1

            background[y0:y1, x0:x1] = bg_sub[ry0:ry1, rx0:rx1]
        end
    end

    # Optional gentle smoothing across stitched background to emulate FIJI smoothing
    #if smoothing
        # 3x3 mean filter
    #    background = imfilter(background, KernelFactors.mean(3))
    #end
    
    # Subtract background from processed image
    result = proc_img .- background

    #***tbh no me acuerdo por qué tenía que ir mirando estos
    #println(maximum(result),"  ",minimum(result))
    #println(maximum(proc_img),"  ",minimum(proc_img))
    #println(maximum(background),"  ",minimum(background))

    # If light background was used (we inverted at start), invert result back
    if light_background
        result = 255.0 .- result
    end

    # Clamp to [0,255] and convert to UInt8
    result = clamp.(result, 0.0, 255.0)
    return Array{UInt8}(round.(result))
end




function get_scale_from_white_bar(im, scale_ref)
    """
    If a scale is present in the image im, find how many pixels long it is
    and return scaling constant relative to scale_ref in μm.
    """
    
    scale = findall(x->x==1.0, im) # Find all white pixels
    
    if isempty(scale)==false
        cols_scale = [i[2] for i in scale] # primer índice es fila, segundo es columna ("igual que como se lee")
        rows_scale = [i[1] for i in scale]
        max_col_scale = maximum(cols_scale)+1 
        min_col_scale = minimum(cols_scale)-1 
        max_row_scale = maximum(rows_scale)+1 
        min_row_scale = minimum(rows_scale)-1 
        
        scale_width = max_col_scale-min_col_scale
    
        println("Ancho de escala: ",scale_ref," μm = ",scale_width-1," px") #+-1px
    else
        println("No scale found")
        return nothing
    end
    
    return scale_ref/(scale_width-1)
end

function get_scale_from_ref(im, scale_ref)# If there is no scale in the image
    """
    Calculate scaling constant, if you know the widths of the image im in μm (scale_ref).
    """
    
    (h,w) = size(im)
    max_col_scale = w
    min_col_scale = 1
    max_row_scale = h
    min_row_scale = 1

    scale_width = max_col_scale-min_col_scale+1
    println("Ancho de escala estimado: ",scale_ref," μm = ",scale_width," px") #+-1px
    return scale_ref/scale_width
end


function isolate_lens_shadow(gray_im, threshold_background)
    """
    Find lens shadow (where gray_im has pixels with a color darker than threshold background).
    Return indices to cut the image, so that the lens shadow is excluded.
    """
    
    (h,w) = size(gray_im)
    # Separo las zonas del lunar negro y el borde del lente (más oscuras) y las pongo en blanco para distinguirlas bien de la herida
    lens_shadow_indices = gray_im .<= threshold_background
   
    min_w = w
    max_w = 0
    min_h = h
    max_h = 0
    for i in 1:h
        x = lens_shadow_indices[i,:]
        if 1 in x
            
            # Ignore isolated pixels
            for (j,y) in enumerate(x)
                if j!=1 && j!=length(x) && x[j-1]==0 && x[j] == 1 && x[j+1]==0
                    x[j] = 0
                end
                if j!=1 && j!=length(x) && j!=length(x)-1 && x[j-1]==0 && x[j]==1 && x[j+1] == 1 && x[j+2]==0
                    x[j] = 0
                end
            end

            # Get maximum and minimum height of lens shadow
            if 1 in x
                if i <min_h
                    min_h = i
                end
                if i>max_h
                    max_h = i
                end

                # Get maximum and minimum width of lens shadow
                where_lens = findall(>(0), x)
                if minimum(where_lens)< min_w
                    min_w = minimum(where_lens)
                end
                if maximum(where_lens) > max_w
                    max_w = maximum(where_lens)
                end
            end
        end
    end

    if min_w >= max_w || min_h >= max_h
        min_w = 1
        max_w = w
        min_h = 1
        max_h = h
    end

    #println(min_w," ",max_w," ",min_h," ",max_h)

    return lens_shadow_indices, min_h, max_h, min_w, max_w 
end


function find_biggest_dark_area(segments, value, key_larger_area=0) 
    """
    Find in segments which one has the biggest area with the color value.
    If the wound is broken into more than one region, find the second largest one
    indicating with key_larger_area what the key in segments is for the biggest one.
    Return the area of the wound.
    """
    
    dict::Dict = segment_mean(segments)
    keys = []
    
    for (key, val) in dict
        if val <= value && key != key_larger_area
            push!(keys,key) 
        end 
    end 
    
    if isempty(keys) 
        error("Value not found in dictionary.") 
    else
        length_keys = []
        for k in keys
            push!(length_keys,length(labels_map(segments)[labels_map(segments) .==k]))
        end
        
        return keys[findfirst(item -> item == maximum(values(length_keys)), length_keys)]
    end
    
end 


function find_leftmost_and_rightmost(mat, value)
    zero_indices = findall(x -> x == value, mat)  # Find all zeros in the matrix
    
    if isempty(zero_indices)
        return nothing, nothing  # If no zeros, return nothing
    end
    
    # Find leftmost (smallest column index)
    leftmost_zero = argmin(x -> x[2], zero_indices)
    
    # Find rightmost (largest column index)
    rightmost_zero = argmax(x -> x[2], zero_indices)
    
    return leftmost_zero, rightmost_zero
end

function find_topmost_and_bottommost(mat,value)
    zero_indices = findall(x -> x == value, mat)  # Find all zeros in the matrix
    
    if isempty(zero_indices)
        return nothing, nothing  # If no zeros, return nothing
    end
    
    # Find topmost (smallest row index)
    topmost_zero = argmin(x -> x[1], zero_indices)
    
    # Find bottommost (largest row index)
    bottommost_zero = argmax(x -> x[1], zero_indices)
    
    return topmost_zero, bottommost_zero
end

function lowest_y_for_x_and_value(mat, x_target, val)
    rows = size(mat, 1)
    ys = []

    for y in 1:rows
        if mat[y, x_target] == val
            push!(ys, y)
        end
    end

    return isempty(ys) ? nothing : minimum(ys)
end

function highest_y_for_x_and_value(mat, x_target, val)
    rows = size(mat, 1)
    ys = []

    for y in 1:rows
        if mat[y, x_target] == val
            push!(ys, y)
        end
    end

    return isempty(ys) ? nothing : maximum(ys)
end

function get_mean_vertical_axis(labels_im, h,w, segment_index)
    count_y = 0
    count_k = 0
    for row in 1:h
        for y in 1:w
            if labels_im[row,y] == segment_index
                count_y += y
                count_k +=1
            end
        end
    end
    return count_y/count_k
end

