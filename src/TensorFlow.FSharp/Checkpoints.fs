[<AutoOpen>]
module TensorFlow.FSharp.Checkpoints

open TensorFlow.FSharp
open TensorFlow.FSharp.Proto
open System
open System.IO

// NOTE: We should probably closely emultate existing saver / loader ops
//       For now we're doing the simple thing to enable loading of variables

// Given that it's unlikely we'll have a 1:1 emulation of saving and loading we should assume
// that we will not be able 'resume' training from a graph defined in python (and visa versa)
// but we will be able to initialize the variables to the checkpointed weights.
// This means the optimizer variables will need to be restarted. This shouldn't affect
// training performance that much.

// NOTE: Technically variable types can change during operation though in practice this is rare
//       In order to 

// NOTE: GetVariableShapeMapping and GetVariableDataTypeMapping are only available in C++
//       A more complete implementaiton will require these functions. It's possible to use SWIG
//       or to create a Bazel Build.

type TFGraph with

    member this.RestoreVariables(path : string, ?variableNames : string[], ?name : string, ?nameMapping : string -> string) : TFOutput[] = 
        let variableNames = variableNames |> Option.defaultWith (fun _ -> this.AllVariableNames())
        let slices = variableNames |> Array.map (fun _ -> "")
        let nameMapping = nameMapping |> Option.defaultValue id
        let dataTypes = variableNames |> Array.map (fun x -> this.GetOperationByName(nameMapping(x)).[0].TFDataType.BaseType)
        this.RestoreV2(this.Const(TFTensor.CreateString(path)), 
                                  this.Const(TFTensor.CreateString(variableNames)), 
                                  this.Const(TFTensor.CreateString(slices)),dataTypes, ?name = name)

    member this.SaveVariablesOp(path : string, ?variableNames : string[], ?name : string) : TFOperation = 
        let variableNames = variableNames |> Option.defaultWith (fun _ -> this.AllVariableNames())
        let variables= variableNames |> Array.map (fun x -> this.GetOperationByName(x).[0])
        let slices = variables |> Array.map (fun _ -> "")
        this.SaveV2(this.Const(TFTensor.CreateString(path)), 
                               this.Const(TFTensor.CreateString(variableNames)), 
                               this.Const(TFTensor.CreateString(slices)), variables , ?name = name)

    member this.Restore(path : string, ?variableNames : string[])  =
        let variableNames= variableNames |> Option.defaultWith (fun _ -> this.AllVariableNames())
        let variables= variableNames |> Array.map (fun x -> this.GetOperationByName(x).[0])
        let restoreOps = this.RestoreVariables(path,variableNames)
        let assignOps = [|for v,r in (variables, restoreOps) ||> Array.zip -> this.Assign(v,r).Op|]
        use depScope = this.WithDependencies(assignOps)
        this.NoOp("assign_all")

//    member this.Save(path : string, ?variableNames : string[]) = 
//        failwith "todo"

    /// NOTE: Use the result from this to Import Graph from MetaGraph using the Import functions
    static member ExtractGraphFromMetaGraph(path : string) : byte[] = 
        failwith "untested"
        let metaGraph = deserialize<MetaGraphDef>(new MemoryStream(File.ReadAllBytes(path)))
        use ms = new MemoryStream()
        ProtoBuf.Serializer.Serialize(ms, metaGraph.GraphDef)
        ms.ToArray()

    /// This would load the graph from the MetaGraph then load the latest checkpoint
    member this.FromCheckpoint(path : string, ?prefix : string) = 
        // Figure out from path if it's 'fully specified' or 
        failwith "todo"

// TODO Some helper functions for MetaGraphDef

(*
/// MetaGraph exploring 
let xx = TensorFlow.FSharp.CheckpointReader.loadMetaGraph(@"G:\checkpoints\minutiae_train3\model.ckpt-64297.meta")


xx.SaverDef.FilenameTensorName
let graphDef = xx.GraphDef

// saveConst is "model"
let saveConst = graphDef.Nodes |> Seq.tryFind (fun x -> x.Name = "save/Const") |> Option.get


// Find all nodes with this save node as an input
graphDef.Nodes |> Seq.tryFind (fun x -> x.Inputs.Contains("save/Const")) |> Option.get
graphDef.Nodes |> Seq.tryFind (fun x -> x.Inputs.Contains("save/StringJoin")) |> Option.get
graphDef.Nodes |> Seq.tryFind (fun x -> x.Inputs.Contains("save/ShardedFilename")) |> Option.get

graphDef.Nodes |> Seq.tryFind (fun x -> x.Name = "save/SaveV2") 

let getStringFromConstOp(x : NodeDef) = 
    if x.Attrs.["dtype"].Type = Proto.DataType.DtString then
        [| for bytes in x.Attrs.["value"].Tensor.StringVals -> Encoding.UTF8.GetString(bytes)|]
    else [||]


graphDef.Nodes |> Seq.tryFind (fun x -> x.Name = "save/SaveV2/tensor_names") 

graphDef.Nodes |> Seq.tryFind (fun x -> x.Name = "save/StringJoin")

//NOTE: it appears that "save/StringJoin/inputs_1" is overriden by a feed_dict...

let inputs_1_const = graphDef.Nodes |> Seq.tryFind (fun x -> x.Name = "save/StringJoin/inputs_1") |> Option.get

graphDef.Nodes 
|> Seq.choose (fun x -> 
    if x.Op = "Const" then
        match getStringFromConstOp(x) with 
        | [||] -> None
        | xs -> Some(x.Name,xs)
    else None )
|> Seq.toArray



Encoding.UTF8.GetString(inputs_1_const.Attrs.["value"].Tensor.StringVals.[0]) // "_temp_bf99d807a20f4fb0bb1d5dba8d40e4e6/part"

[|for node in graphDef.Nodes do if node.Op = "VariableV2" then yield node.Name|]

xx.CollectionDefs
// TensorFlow.FSharp.Proto.MetaGraphDef+MetaInfoDef
//    {AnyInfo = null;
//     MetaGraphVersion = "";
//     StrippedDefaultAttrs = false;
//     StrippedOpList = TensorFlow.FSharp.Proto.OpList;
//     Tags = seq [];
//     TensorflowGitVersion = "b'v1.12.0-0-ga6d8ffae09'";
//     TensorflowVersion = "1.12.0";}

// TODO figure out what goes into StrippedOpList
//      there seems to be one op of each type, it seems to be 

xx.meta_info_def.StrippedOpList.Ops |> Seq.map (fun x -> x.Name) |> Seq.toArray
xx.meta_info_def.StrippedOpList.Ops.[3].Attrs

// TODO SaverDef FilenameTensorName save/Const:0
// KeepCheckpointEveryNHours = 10000.0f
MaxToKeep
xx.SaverDef
// TensorFlow.FSharp.Proto.SaverDef {FilenameTensorName = "save/Const:0";
//                                    KeepCheckpointEveryNHours = 10000.0f;
//                                    MaxToKeep = 5;
//                                    RestoreOpName = "save/restore_all";
//                                    SaveTensorName = "save/Identity:0";
//                                    Sharded = true;
//                                    Version = V2;}

//xx.SignatureDefs  empty dict
//xx.AssetFileDefs empty seq

//[|("summary_op", "Nodes"); ("trainable_variables", "Bytes");
//    ("summaries", "Nodes"); ("init_op", "Nodes");
//    ("ready_for_local_init_op", "Nodes"); ("cond_context", "Bytes");
//    ("local_init_op", "Nodes"); ("variables", "Bytes"); ("ready_op", "Nodes");
//    ("savers", "Bytes"); ("train_op", "Nodes"); ("global_step", "Bytes");
//    ("global_step_read_op_cache", "Nodes"); ("iterators", "Nodes")|]
[| for KeyValue(k,v) in xx.CollectionDefs -> 
    let res = 
        if v.any_list <> null then "Any"
        elif v.bytes_list <> null then "Bytes"
        elif v.float_list <> null then "Float"
        elif v.int64_list <> null then "Int64"
        elif v.node_list <> null then "Nodes"
        else ""
    (k,res) |]

// NOTE: trainable_variables represent the variables which
xx.CollectionDefs.["trainable_variables"].bytes_list.Values |> Seq.map (fun x -> System.Text.UTF8Encoding.UTF8.GetString(x)) |> Seq.toArray
// NOTE: variables collection
xx.CollectionDefs.["variables"].bytes_list.Values |> Seq.map (fun x -> System.Text.UTF8Encoding.UTF8.GetString(x)) |> Seq.toArray
xx.CollectionDefs.["global_step"].bytes_list.Values |> Seq.map (fun x -> System.Text.UTF8Encoding.UTF8.GetString(x)) |> Seq.toArray
xx.CollectionDefs.["savers"].bytes_list.Values |> Seq.map (fun x -> System.Text.UTF8Encoding.UTF8.GetString(x)) |> Seq.toArray
xx.CollectionDefs.["cond_context"].bytes_list.Values |> Seq.map (fun x -> System.Text.UTF8Encoding.UTF8.GetString(x)) |> Seq.toArray
*)
