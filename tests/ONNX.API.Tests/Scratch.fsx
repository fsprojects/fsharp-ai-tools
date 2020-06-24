#load "Base.fsx"
open Base
let mnistG = MNIST()

open Microsoft.FSharp.Reflection
open FSharp.ML.Onnx.Protobuf
open Microsoft.ML.OnnxRuntime.Tensors
open System.IO
open Microsoft.ML.OnnxRuntime.Tensors
open Onnx
open FSharp.ML.Onnx.Protobuf
open System
open System.IO
open Microsoft.ML.OnnxRuntime
open Microsoft.FSharp.Quotations
open FSharp.ML.Onnx.Utils
open FSharp.ML.Onnx.Utils.Expr
open FSharp.ML.Onnx.Expr
open FSharp.ML.Onnx.Extensions
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.ExprShape
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime
open FSharp.Quotations.Evaluator
open FSharp.ML.Onnx.Utils.Expr
open FSharp.ML.Onnx.Utils


// I know about the new Type mismatch bug, I will work on it
//use graphFunction : DV<Tensor<float32> -> DV<Tensor<float32>>> = toOnnxGraph(<@ mnistG.Forward2 @>)

let graphFunction : DV<Tensor<float32> -> DV<Tensor<float32>>> = toOnnxGraph(<@ mnistG.Forward @>)
let graphFunction2 : DV<Tensor<float32> -> DV<Tensor<float32>>> = toOnnxGraph(<@ mnistG.Forward2 @>)
let graphFunction3 : DV<Tensor<float32> -> DV<Tensor<float32>>> = toOnnxGraph(<@ mnistG.Forward3 @>)
let graphFunction4 : DV<Tensor<float32> -> DV<Tensor<float32>>> = toOnnxGraph(<@ mnistG.Forward4 @>)

let graph = Graph.Default()

((<@ mnistG.Forward5 @> |> simplify) |> processExpr <@ graph @> ).ToString()

(<@ mnistG.Forward5 @> |> simplify).ToString()

let xs = 
    <@ mnistG.Forward5 @> 
    |> simplify 
    |> Expr.flatten 
    |> Seq.choose (function | Application(Var(v),_) -> Some(v) | _ -> None) 
    |> Seq.toArray

xs.[0].Type = xs.[1].Type


(*
let mapType (f:Type -> Type option) (t:Type) : Type option = 
    let rec mapType  (t:Type) : Type option = 
        let f (xs: Type[]) = xs |> Array.map mapType |> Option.all
        if t.IsArray then
            t.GetElementType() |> mapType |> Option.map (fun t2 -> t2.MakeArrayType())
        elif t |> FSharpType.IsTuple then
            t |> FSharpType.GetTupleElements |> f |> Option.map FSharpType.MakeTupleType
        elif FSharpType.IsUnion t || FSharpType.IsRecord t then
            t.GetGenericArguments() |> f |> Option.map (fun xs -> t.GetGenericTypeDefinition().MakeGenericType(xs))
        elif suportedBaseTypes |> Array.exists (fun x -> x.IsAssignableFrom(t)) then
            Some(typeof<ValueInfo>)
        elif FSharpType.IsFunction t then
            let (x,y) = FSharpType.GetFunctionElements t
            match mapType x, mapType y with
            | Some(x),Some(y) -> Some(FSharpType.MakeFunctionType(x,y))
            | _,_ -> None
        elif t.FullName = "System.Int32" then  
            //printfn "System.Int64"
            Some(typeof<float32>)
        else
            None
    mapType t


let mapVar (v1:Var) = 
    match mapType (fun x -> None) (v1.Type) with 
    | Some(t) -> Var(v1.Name,t)
    | None ->  
        printfn "Var %s has type %s which is not mappable" v1.Name v1.Type.FullName
        v1


let processExpr (expr: Expr) : Expr = 
    let rec processExpr (varMap: Map<Var,Var>) (expr: Expr) : Expr = 
        match expr with
        | Var v -> 
            match varMap.TryFind(v) with 
            | Some(v) -> Expr.Var(v) 
            | _ -> failwithf "Var %s not found %s" v.Name v.Type.FullName
        | Lambda(v1,body) -> v1 |> mapVar |> fun v2 -> Expr.Lambda(v2,processExpr (varMap.Add(v1,v2)) body)
        | Application(expr1, expr2) -> 
            Expr.Application(
                expr1 |> processExpr varMap,
                expr2 |> processExpr varMap)
        | Value(o,t) -> if t.FullName = "System.Int32" then Expr.Value(10.f) else failwith "err"
        | Let(v1,exp1,exp2) -> v1 |> mapVar |> fun v2 -> Expr.Let(v2, processExpr varMap exp1, processExpr (varMap.Add(v1,v2)) exp2)
        | TupleGet(x,i) -> Expr.TupleGet(processExpr varMap x,i)
        | NewTuple(xs) -> Expr.NewTuple(xs |> List.map (processExpr varMap))
        | _ -> failwithf "err %A" expr
    processExpr Map.empty expr



( <@ let f (x:int,z:int) (y:int) = x,y,z 
    let g = f(10,20)
    in g  10  @> |> processExpr).EvaluateUntyped()
*)

//let x = 0L
//typeof<float32>.FullName


//[<ReflectedDefinition>]
//module X = 
//    let foo x = x
