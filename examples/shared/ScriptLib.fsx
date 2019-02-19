namespace global

[<AutoOpen>]
module ScriptLib =

    let time f = 
        let sw = System.Diagnostics.Stopwatch()
        sw.Start()
        let res = f()
        sw.Stop()
        printfn "elapse: %A" sw.Elapsed
        res

