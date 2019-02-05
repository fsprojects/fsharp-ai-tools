(*

fsi.AddPrintTransformer(fun (x:obj) ->
    match x with
    | :? IFSIPrint as x -> x.ToFSIString()
    | _ -> null)

*)
