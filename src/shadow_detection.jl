


function cut_big_shadows(im_file, y_cut)

    # Load image and turn to Gray
    im = load(im_file) #************** esto necesita Images mepa
    (h,w) = size(im)
    #@info ("cut: ",y_cut)

    # The cut was flagged in a previous segmentation as a big section of the image that is mostly dark (shadow)
    if y_cut != (nothing,nothing)
        im = im[y_cut[1]:y_cut[2], 1:end]
        (h,w) = size(im)
    end

    # If most of the image is shadow, then no segmentation will be possible "***creo que esto lo puse afuera en first_pass_collect_cuts de segmentation_pipeline.jl
    if h==0 || w ==0
        @info ("!!!!!!!!!!!!!!!! No segmentation was possible. Seems like the whole image is in shadow. !!!!!!!!!!!!!!!!")
        return im, h, w, (-1,-1), false
    end
    #@info ("Image height and length: ",h, ", ",w)
    return im, h, w, y_cut, true
end


function cut_at_top_and_bottom(segmented_final_im, value_shadow, image_partition, h)
    y_23h = highest_y_for_x_and_value(segmented_final_im, image_partition[2], value_shadow)
    y_13h = highest_y_for_x_and_value(segmented_final_im, image_partition[1], value_shadow)
    y_23l = lowest_y_for_x_and_value(segmented_final_im, image_partition[2], value_shadow)
    y_13l = lowest_y_for_x_and_value(segmented_final_im, image_partition[1], value_shadow)
    #@info ("Where to cut: top-right corner: ",y_23h," top-left corner: ",y_13h," bottom-right: ",y_23l," bottom-left: ",y_13l)

    if y_23h == h || y_13h == h
        if y_13l != nothing && (y_23l == nothing || y_23l>y_13l)
            y_cut2 = y_13l
        elseif y_23l != nothing && (y_13l == nothing || y_13l>y_23l)
            y_cut2 = y_23l
        else
            return (nothing,nothing)
        end
        y_cut1 = 2
    elseif y_23l == 1 || y_13l == 1
        if y_23h != nothing && (y_13h == nothing || y_23h>y_13h)
            y_cut1 = y_23h
        elseif y_13h != nothing && (y_23h == nothing || y_13h>y_23h)
            y_cut1 = y_13h
        else
            return (nothing,nothing)
        end
        y_cut2 = h-1
    end
    return (y_cut1,y_cut2)
end
function cut_at_top(segmented_final_im, value_shadow, image_partition, bm, lm, rm, h, w, margin)
    y_cut1 = bm[1]
    y_cut2 = h-1
    y_23 = highest_y_for_x_and_value(segmented_final_im, image_partition[2], value_shadow)
    y_13 = highest_y_for_x_and_value(segmented_final_im, image_partition[1], value_shadow)
    #@info ("Where to cut in y: if lens shadow is on the right: ",y_23," if it's on the left: ",y_13)
    if y_23 != nothing && (y_13 == nothing || y_23>y_13) && image_partition[2]<bm[2]
        y_cut1 = y_23
    elseif y_13 != nothing && (y_23 == nothing || y_13>y_23) && image_partition[2]>bm[2]
        y_cut1 = y_13
    elseif y_23 == nothing && y_13 == nothing && (lm[2]==0 || rm[2]==w)
        y_cut1 = 1
    end

    if y_cut1 < h-margin #**me acabo de dar cuenta de que si no hay corte igual corto ese margen mmmm
        y_cut1 = y_cut1+margin
    end
    return (y_cut1,y_cut2)
end
function cut_at_bottom(segmented_final_im, value_shadow, image_partition, tm, lm, rm, h, w, margin)
    y_cut1 = 2
    y_cut2 = tm[1]
    y_23 = lowest_y_for_x_and_value(segmented_final_im, image_partition[2], value_shadow)
    y_13 = lowest_y_for_x_and_value(segmented_final_im, image_partition[1], value_shadow)
    if y_23 != nothing && (y_13 == nothing || y_23<y_13) && image_partition[2]<tm[2]
        y_cut2 = y_23
    elseif y_13 != nothing && (y_23 == nothing || y_13<y_23) && image_partition[2]>tm[2]
        y_cut2 = y_13
    elseif y_23 == nothing && y_13 == nothing && (lm[2]==0 || rm[2]==w)
        y_cut2 = h
    end
    if y_cut2 > margin
        y_cut2 = y_cut2-margin
    end
    return (y_cut1,y_cut2)
end


function identify_and_cut_shadow(h, w, y_cut, segments, segmented_final_im, shadow_segment_threshold_color, margin, fraction_of_image)

    # Calculate area of shadow if present and cut it out of the image in T001
    value_shadow = maximum(values(segment_mean(segments)))
    #@info ("Shadow color: ",value_shadow)

    shadow_present = value_shadow >=shadow_segment_threshold_color

    if !shadow_present || y_cut != (nothing,nothing)# y_cut == (nothing,nothing)only on first timeframe
        #@info ("No cut: ",y_cut)
        return (1,h), false
    else
        image_partition = (Int(w*fraction_of_image[1]),Int(w*fraction_of_image[2]))

        lm,rm = find_leftmost_and_rightmost(segmented_final_im,value_shadow) # index 1 is vertical, 2 is horizontal
        tm,bm = find_topmost_and_bottommost(segmented_final_im,value_shadow)
        #@info ("Shadow position: leftmost px: ",lm," rightmost px: ",rm," topmost px: ",tm," bottommost px: ",bm)

        y_cut1 = 2
        y_cut2 = h-1

        if tm[1] == 1 && bm[1] == h
            y_cut1, y_cut2 = cut_at_top_and_bottom(
                segmented_final_im, value_shadow, image_partition, h
            )

        elseif tm[1] == 1
            y_cut1, y_cut2 = cut_at_top(
                segmented_final_im, value_shadow, image_partition, bm, lm, rm, h, w, margin
            )

        elseif bm[1] == h
            y_cut1, y_cut2 = cut_at_bottom(
                segmented_final_im, value_shadow, image_partition, tm, lm, rm, h, w, margin
            )
        end


        @info ("Where to cut: ",(y_cut1,y_cut2))
        return (y_cut1,y_cut2), true
    end

end

