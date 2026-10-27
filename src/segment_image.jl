
code_folder = @__DIR__
include(joinpath(code_folder,"image_analyzers.jl"))
include(joinpath(code_folder,"shadow_detection.jl"))
include(joinpath(code_folder,"segment_positions_detection.jl"))
#**otra cosa que tengo que revisar es el ignore_pits... me da un poco de fiaca que sea algo que haya que cargar manualmente. creo que debería poder ignorar automáticamente esos archivos si no existen, y cargarlos
# capaz en un .txt de ignore_pits para que busque ahí directamente los que ignoró y se los saltee en vez de intentar cargar todos cada vez


const DEFAULT_PREPROCESSING_STEPS = [
    "bandpass",
    "equalize",
    "detect_edges",
    "blur",
    "threshold",
]


function load_preprocessing_pipeline(path::AbstractString)
    config = TOML.parsefile(path)
    return String.(config["steps"])
end

function save_preprocessing_pipeline(steps, path::AbstractString)
    open(path, "w") do io
        TOML.print(io, Dict("steps" => String.(steps)))
    end
    return nothing
end

"""
One important limitation: not every arbitrary reordering is valid. For example, equalize requires bandpass to have run first, and segment_mean requires map_labels. If a required step hasn't run, Julia will raise a missing-key error. That's useful initially, because it helps identify invalid sequences; we can make those error messages friendlier later.
"""
function apply_preprocessing_step!(name::String, ctx::Dict{Symbol, Any}, im, hp)
    if name == "bandpass"
        ctx[:filter_im] = Gray.(1.0 .- bandpass_filter(
            ctx[:filter_im],
            hp.min_bp_filter,
            hp.max_bp_filter,
        ))

    elseif name == "equalize"
        ctx[:filter_im] = adjust_histogram(
            ctx[:filter_im],
            Equalization(nbins=256),
        )

    elseif name == "detect_edges"
        ctx[:filter_im1] = detect_edges(ctx[:filter_im])

    elseif name == "blur"
        ctx[:filter_im2] = blur_image!(
            ctx[:filter_im1],
            hp.gaussian_blur_sigma,
            1,
        )

    elseif name == "threshold"
        threshold = mean(ctx[:filter_im1]) * hp.wound_threshold

        ctx[:filter_im] = Gray.(
            threshold_colors.(
                ctx[:filter_im2],
                threshold,
            )
        )

    else
        error("Unknown preprocessing step: '$name'")
    end

    return nothing
end


function segmentation_steps(image_path, image_filename, segmentation_hyperparams; steps = DEFAULT_PREPROCESSING_STEPS, display_steps = false,)
    im = load(joinpath(image_path, image_filename))

    ctx = Dict{Symbol, Any}()

    # Grayscale
    ctx[:filter_im] = Gray.(im)

    # Locate lens shadow (if present)
    ctx[:lens_shadow_indices],
    ctx[:min_h],
    ctx[:max_h],
    ctx[:min_w],
    ctx[:max_w] =
        isolate_lens_shadow(
            ctx[:filter_im],
            segmentation_hyperparams.shadow_threshold_color,
        )

    for step in steps
        apply_preprocessing_step!(step, ctx,
            ctx[:filter_im],
            segmentation_hyperparams,
        )
        if display_steps
            #display(ctx[:filter_im])
            save(joinpath(image_path,"$(image_filename)_$(step).png"), ctx[:filter_im])
        end
    end

    # Filter lens shadow
    ctx[:filter_im][ctx[:lens_shadow_indices]] .= 1.0

    # Apply felzenszwalb segmentation
    ctx[:segments] = felzenszwalb(
        ctx[:filter_im],
        segmentation_hyperparams.meansize_wound,
        segmentation_hyperparams.minsize_wound,
    )
    # Save segmentation results
    ctx[:labels_segments] = labels_map(ctx[:segments])
    ctx[:segmented_final_im] = map(
        i -> segment_mean(ctx[:segments], i),
        ctx[:labels_segments],
    )

    if display_steps
        #display(ctx[:segmented_final_im])
        save(joinpath(image_path,"$(image_filename)_$(step).png"), ctx[:segmented_final_im])
    end

    result = (
        ctx[:segments],
        ctx[:labels_segments],
        ctx[:segmented_final_im],
    )

    return result
end




function segmentation_steps(im, segmentation_hyperparams)

    gray_im = Gray.(im)

    # Locate lens shadow (if present)
    lens_shadow_indices, min_h, max_h, min_w, max_w = isolate_lens_shadow(gray_im, segmentation_hyperparams.shadow_threshold_color)
    
    bp = Gray.(1.0.-(bandpass_filter(gray_im, segmentation_hyperparams.min_bp_filter, segmentation_hyperparams.max_bp_filter)))
    equalized = adjust_histogram(bp, Equalization(nbins = 256))
    
    # Detect edges with Laplacian Kernel
    filtered_gray_im = detect_edges(equalized)
    
    # Blur the image
    discrete_filtered_gray_im = blur_image!(filtered_gray_im, segmentation_hyperparams.gaussian_blur_sigma, 1) # **por qué los pixeles blancos no vuelven blancos ??
    
    corrected_img = Gray.(threshold_colors.(discrete_filtered_gray_im,mean(filtered_gray_im)*segmentation_hyperparams.wound_threshold))
    
    # Separate lens shadow (if present) # **podría restar los 1.0 que no quiero ?
    corrected_img[lens_shadow_indices] .= 1.0
    
    discrete_final_im = corrected_img
    
    # Get image segmentation
    segments = felzenszwalb(discrete_final_im, segmentation_hyperparams.meansize_wound, segmentation_hyperparams.minsize_wound) #imagen, tamaño aprox de clusters, mínimos píxeles para hacer sección
    labels_segments = labels_map(segments)
    segmented_final_im = map(i->segment_mean(segments,i), labels_segments) # segment_mean busca el color promedio del segmento y lo asigna a todo el segmento

    return segments, labels_segments, segmented_final_im
end


# Display image steps **agregar títulos o labels ?
    #display([im filtered_gray_im
    #discrete_filtered_gray_im corrected_img
    #discrete_final_im segmented_final_im ])

function get_felzenszwalb_hyperparams(k, meansize, minsize, inv_max_tries)
    """
    Run to try smaller and smaller segments while segmentation tries keep failing. Stop after max_tries attempts.
    """
    meansize_segmentation = meansize - k*meansize*inv_max_tries
    minsize_segmentation = minsize - k*minsize*inv_max_tries
    
    return meansize_segmentation, minsize_segmentation
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






function find_probable_wound_segment(w, segmented_final_im, segments, value_scratch, area_scratch_index, probable_wound_region, k, max_tries)

    # The probable_wound_region is to check if the wound found is away from the center or if it's too close to the border of the image
    max_left_wall_wound_position = probable_wound_region[1]*w
    wound_centre = probable_wound_region[2]*w
    max_right_wall_wound_position = probable_wound_region[3]*w
    
    lm1,rm1 = find_leftmost_and_rightmost(segmented_final_im,value_scratch)
    
    # If biggest darkest area is far away from the center, it might not be the wound
    if rm1[2]<wound_centre || lm1[2]>wound_centre

        # We loop over every segment in case one of them might be the wound
        for s in segment_labels(segments) 
            #println("Try other segments: ",s," Color value: ",segment_mean(segments,s))
            lms,rms = find_leftmost_and_rightmost(segmented_final_im,segment_mean(segments,s))

            # If biggest darkest area is very close to the border of the image, it might not be the wound
            if lms[2]> max_left_wall_wound_position && rms[2] < max_right_wall_wound_position
                #println("Leftmost and rightmost px values for this segment: ",lms[2],"  ",rms[2])
                value_scratch = segment_mean(segments,s)
                area_scratch_index = s
                lm1 = lms
                rm1 = rms
                return true
            end
        end
        
        if k == (max_tries-1)  
            @info ("!!!!!!!!!!!!!!!! The last try didn't work, no wound found !!!!!!!!!!!!!!!!")
            return false
        else
            @info ("COULDN'T FIND WOUND. Redoing steps for smaller segmentation hyperparameters.")
            return true
        end
        
    else
        return true
    end
end


function is_wound_bigger_than_max_area(max_area,area_scratch, k, max_tries)

    # If the segmentation was not right:
    if area_scratch >= max_area 
        if k == (max_tries-1)
            #y_cut = (1,h) #** me sirve devolverlo así ?
            @info ("!!!!!!!!!!!!!!!! The last try didn't work, segmentation failed !!!!!!!!!!!!!!!!") #@warn
            return true, false # Si el último es false, deja de loopear, si es true, sigue intentando segmentar con otros hyperparams. Si el segundo es false, la segmentación anduvo bien
        else
            @info ("COULDN'T DO SEGMENTATION. Redoing steps for smaller segmentation hyperparameters.")
            return true, true
        end
    else 
        return false, false
    end
end

function count_label(idx, labels)
    c = 0
    @inbounds for x in labels
        c += (x == idx)
    end
    return c
end

@inline function spans_are_close(lm1, rm1, lm2, rm2, distance)
    abs(lm2[2] - lm1[2]) ≤ distance &&
    abs(rm2[2] - rm1[2]) ≤ distance
end

function find_second_darkest_area(segmented_final_im, segments, labels_segments, value_scratch, area_scratch, area_scratch_index, merge_distance)  
    lm1,rm1 = find_leftmost_and_rightmost(segmented_final_im,value_scratch) #################
    
    # In case the area is found but a part has already closed
    value_scratch2 =  minimum([v for (k,v) in segment_mean(segments) if k!= area_scratch_index]) 
    
    lm2,rm2 = find_leftmost_and_rightmost(segmented_final_im,value_scratch2)

    if spans_are_close(lm1,rm1,lm2,rm2,merge_distance) #(lm2[2] <= lm1[2]+merge_distance) && (lm2[2] >= lm1[2]-merge_distance) && (rm2[2] <= rm1[2]+merge_distance) && (rm2[2] >= rm1[2]-merge_distance)
        area_scratch_index2 = find_biggest_dark_area(segments, value_scratch2, area_scratch_index)
        area_scratch = area_scratch + count_label(area_scratch_index2, labels_segments)
        @info ("First area: ",area_scratch_index, ", Second area: ", area_scratch_index2)
        
        return area_scratch, area_scratch_index2, true
    else
        return area_scratch, 0, false
    end
    #end
    
end


function compute_edges!(edges::Matrix{UInt8},labels::Matrix{Int},wound_idx::Int,wound_idx2::Union{Int,Nothing})
    h, w = size(labels)

    @inbounds for i in 2:h-1, j in 2:w-1
        seg = labels[i,j]

        up    = labels[i-1,j]
        down  = labels[i+1,j]
        left  = labels[i,j-1]
        right = labels[i,j+1]

        # helper
        is_wound(x) = x == wound_idx || x == wound_idx2

        if up != seg
            edges[i,j] = is_wound(up) ? 2 : 1
        elseif down != seg
            edges[i,j] = is_wound(down) ? 2 : 1
        elseif left != seg
            edges[i,j] = is_wound(left) ? 2 : 1
        elseif right != seg
            edges[i,j] = is_wound(right) ? 2 : 1
        end
    end

    #@info ("COUNT ",count(x->x==1,edges),"  ",count(x->x==2,edges)) #**esto para tener en cuenta si un lado es mayor que el otro? pero podría ser por el corte de la lente
    return edges
end

function paint_edges(im, edges)
    out = RGB.(im)  # one allocation

    @inbounds for idx in eachindex(edges)
        e = edges[idx]
        if e == 2
            out[idx] = RGB(1,0,0)
        elseif e == 1
            out[idx] = RGB(0,0,1)
        end
    end

    return out
end


function classify_segments!(
    edge_img,
    segmented_img,
    segments,
    labels,
    wound_idx,
    wound_idx2,
    w,
    mean_y
)
    for seg in segment_labels(segments)
        if seg == wound_idx || seg == wound_idx2
            paint_segment!(edge_img, labels, seg, RGB(0,0,0))
            continue
        end

        lm, rm = find_leftmost_and_rightmost(segmented_img, segment_mean(segments,seg))

        if lm[2] == 1 && rm[2] != w
            paint_segment!(edge_img, labels, seg, RGB(1,0,0))

        elseif lm[2] != 1 && rm[2] == w
            paint_segment!(edge_img, labels, seg, RGB(0,1,0))

        else
            split_segment!(edge_img, labels, seg, mean_y)
        end
    end
end

function paint_segment!(img, labels, seg, color)
    @inbounds for i in eachindex(labels)
        if labels[i] == seg
            img[i] = color
        end
    end
end

function split_segment!(img, labels, seg, mean_y)
    h,w = size(labels)
    @inbounds for i in 1:h, j in 1:w
        if labels[i,j] == seg
            img[i,j] = j < mean_y ? RGB(1,0,0) : RGB(0,1,0)
        end
    end
end
    




function run_segmentation(main_folder, ID, im_file, report_file, save_images, y_cut, segmentation_hyperparams)
    """
    report_file = 0 if you don't want to save areas in a .txt; else, should be the string with the name of the file
    """
    
    """
    Params for segmentation: an external loop runs this segmentation code and keeps making the felzenszwalb parameters smaller any time the segmentation fails
    """
    meansize_segmentation, minsize_segmentation = get_felzenszwalb_hyperparams(segmentation_hyperparams.k, segmentation_hyperparams.meansize_wound, segmentation_hyperparams.minsize_wound, segmentation_hyperparams.inv_max_tries)

    """
    If y_cut is not [nothing,nothing], cut the image where shadows were previously found.
    """
    im, h, w, y_cut, keep_running = cut_big_shadows(im_file, y_cut)
    # If a cut was needed to remove shadows from the image, the segmentation stops here and is run again on the cut image in an external loop. 
    if !keep_running
        return y_cut, keep_running
    end

    """
    Run preprocessing of the image and segmentation algorithm.
    """
    segments, labels_segments, segmented_final_im = segmentation_steps(im, segmentation_hyperparams)

    """
    Find if there are any shadows present in the image (this is analyzed from the segmented image), and escape the function to redo segmentation if a cut is needed.
    """
    #***esto debería tener una lógica distinta porque sólo va a correr en el primer paso pero no queda super explícito como está ahora (y_cut=(nothing,nothing) sólo en T=001)
    y_cut, cut_made = identify_and_cut_shadow(h,w, y_cut, segments, segmented_final_im, segmentation_hyperparams.shadow_segment_threshold_color, segmentation_hyperparams.margin, segmentation_hyperparams.image_partition)
    # Run segmentation again if a shadow was identified, now with the image cut where the shadow ends.
    if cut_made
        return y_cut,cut_made 
    end

    @info ("Number of segments detected: ", length(segment_labels(segments)))


    # Calculate wound area
    value_scratch = minimum(values(segment_mean(segments)))
    area_scratch_index = find_biggest_dark_area(segments, value_scratch)
    area_scratch = count_label(area_scratch_index, labels_segments)

    # Post-processing
    found_wound = find_probable_wound_segment(w, segmented_final_im, segments, value_scratch, area_scratch_index, (segmentation_hyperparams.max_left_wall_wound_position, segmentation_hyperparams.wound_centre, segmentation_hyperparams.max_right_wall_wound_position), segmentation_hyperparams.k, segmentation_hyperparams.max_tries)

    if !found_wound
        return y_cut, found_wound
    end

    max_area = segmentation_hyperparams.max_area_fraction*w*h
    two_areas = false

    # Maybe the biggest darkest area is not the real wound
    #println(value_scratch)    

    area_is_bigger, keep_running = is_wound_bigger_than_max_area(max_area,area_scratch, segmentation_hyperparams.k, segmentation_hyperparams.max_tries)
    if area_is_bigger
        return y_cut,keep_running
    end

    area_scratch, area_scratch_index2, two_areas = find_second_darkest_area(segmented_final_im, segments, labels_segments, value_scratch, area_scratch, area_scratch_index, segmentation_hyperparams.merge_distance)

    

    edges = zeros(UInt8,h,w)
    compute_edges!(edges, labels_segments, area_scratch_index, area_scratch_index2) 
    edges_image = paint_edges(im, edges)
    #wound_mask = build_wound_mask(labels_segments, h, w, area_scratch, area_scratch_index2)
    wound_edge_img = RGB.(segmented_final_im) # fill(Gray(1.0), h, w)
    mean_y = get_mean_vertical_axis(labels_segments, h,w, area_scratch_index)
    classify_segments!(wound_edge_img, segmented_final_im, segments, labels_segments, area_scratch_index, area_scratch_index2, w, mean_y)

    
    if save_images
        # Save segmented image
        #save(main_folder*"segmented_image_"*ID*".png",colorview(Gray,segmented_final_im))
        # Save original image with edges from segments drawn on top
        save(main_folder*"image_"*ID*".jpg",edges_image)
        # Save wound image with edges
        save(main_folder*"wound_edges_image_"*ID*".png",wound_edge_img)
    end

    if report_file != 0
        #**cuidado si la fila no existe la tengo que agregar, y idem la columna (pero capaz eso desde afuera)
        row = findfirst(report_file.ID .== ID[1:end-12]) # pit name
        col = parse(Int64,ID[end-2:end]) # time value
        #@info ("Row: ",row," Col: ",col)

        report_file[row, col+1] = area_scratch
        #print(report_file,"\t",length(findall(item -> item == area_scratch, labels_segments)))
    end
    #@info ("Area for ",ID,": ",area_scratch)

    return y_cut, false
end

