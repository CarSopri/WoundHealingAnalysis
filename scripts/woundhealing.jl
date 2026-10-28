using WoundHealingAnalysis

function main(args)
    command = isempty(args) ? "help" : args[1]

    if command == "playground"
        run_playground()

    elseif command == "help"
        println("""
        WoundHealingAnalysis

        Usage:
          julia --project=. scripts/woundhealing.jl playground

        Commands:
          playground    Open the segmentation playground
          help          Show this help message
        """)

    else
        println("Unknown command: $command")
        println("Run with 'help' to see available commands.")
    end
end

main(ARGS)
