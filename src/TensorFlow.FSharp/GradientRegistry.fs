module GradientRegistry

//// This prevents overwriting of a mapping and provides for the locaion for the previous registration
//open System
//
//type Registry<'a>(name:string) =
//    let mutable registry = Map.empty<string,'a*string*int>
//    let local_name = name
//    with 
//        member this.Add(name:string,x:'a) =
//            match registry.TryFind(name) with
//            | Some(_,file,line) -> 
//                failwithf "Registring two &s with name '%s' !\n(Previous registration was in %s line %i"
//                    name file line 
//            | None ->  
//                let st = System.Diagnostics.StackTrace()
//                // TODO: Make sure we're getting the right frame
//                let (file,line) = st.GetFrames() |> Array.last |> fun x -> (x.GetFileName(), x.GetFileLineNumber())
//                registry  <- registry.Add(name, (x,file,line))
//
//        member this.Add(x:'a) = this.Add(x.ToString(), x)
//        member this.Keys = [|for KeyValue(k,_) in registry -> k|]
//        member this.TryFind(name:string) = registry.TryFind(name) |> Option.map (fun (x,_,_) -> x)
//        member this.Item(name:string) : 'a = 
//            match registry.TryFind(name) with
//            | Some(x,_,_) -> x
//            | _ -> failwithf "%s registry has no entry for: %s" local_name name
//
//
////NOTE: The behaviour here is slightly different, hopefully that difference is unimportant
//let get_gradient_function(op:TFOperation) : GradientFunction option =
//     if op.NumInputs = 0 then None
//     else
//        let op_type = 
//            match op.get_attr("_gradient_op_type") with
//            | Some(AttrValue.Bytes(s)) -> compat.bytes_to_string(s)
//            | None -> op.OpType
//            | _ -> failwith "AttrValue is of an unexpected type"
//        gradient_registry.TryFind(op_type)
