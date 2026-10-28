using Images #: load, Kernel, imfilter, rawview #*****ver bien cuáles de estos uso dónde (shadow_detection, segment_positions_detection, segment_image)
using ImageFiltering #: imfilter
using OffsetArrays
using Statistics
using ImageSegmentation
using ImageMorphology
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
    
    if color >= 0.9 #***************************************
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
    
    filtered_gray_im = Gray.(abs.(imfilter(blur_image!(Gray.(im),1,1), edge_detect))) #***************no me acordaba que había este paso. ver si se puede saltear
    return filtered_gray_im ./ maximum(filtered_gray_im)
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
    # -------------------------------------------------------- #******es un poco más sofisticado que el filtro laplaciano
    σ_small = filter_small 
    σ_large = filter_large

    blur_small = imfilter(imgf, Kernel.gaussian(σ_small))
    blur_large = imfilter(imgf, Kernel.gaussian(σ_large))

    band = blur_small .- blur_large   # Values closer to zero mean more homogeneous zones

    #*******deberían ser dos funciones separadas esto no?
     
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




