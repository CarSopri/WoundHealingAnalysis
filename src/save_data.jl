using CSV
using DataFrames

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

