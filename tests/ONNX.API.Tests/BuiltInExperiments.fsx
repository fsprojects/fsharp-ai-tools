(*
    Experiments for replacing method calls to built in functions with the quotation equivilant
    I don't have a prefered method at this stage. This is not being used at this stage as it's fine
    to run the with Call for the purposes of building a ONNX Graph
*)
#load "Base.fsx"
open Common
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.Patterns
open System.Reflection

// TODO replace built in operators with 
// NOTE: This is not needed as we can execute built-ins during graph construction

// NOTES On generic type substitution on Expressions
// Functions that use generics have a type of the name of the generic, these are knowable ahead of time
// In theory it should be possible to substitute the variable assignments with variables of the replaced types
// It would have to thread through structual typeing, and perhaps even NewObject
// It could go a long way
[<ReflectedDefinition>]
module BuiltIns = 
    let inline fst (a, _) = a
    let inline snd (_, b) = b
    let inline ignore _ = ()
    let ref value = { contents = value }
    let (:=) cell value = cell.contents <- value
    let (!) cell = cell.contents
    let (|>) arg func = func arg
    let (||>) (arg1, arg2) func = func arg1 arg2
    let (|||>) (arg1, arg2, arg3) func = func arg1 arg2 arg3
    let (<|) func arg1 = func arg1
    let (<||) func (arg1, arg2) = func arg1 arg2
    let (<|||) func (arg1, arg2, arg3) = func arg1 arg2 arg3
    let (>>) func1 func2 x = func2 (func1 x)
    let (<<) func2 func1 x = func2 (func1 x)
    let id x = x
    type internal Marker = interface end
    let t = typeof<Marker>.DeclaringType

let builtInMap = 
    BuiltIns.t.GetMethods() 
    |> Array.filter (fun x -> match x.Name with | "GetType" | "get_t" | "ToString" | "GetHashCode" | "Equals" -> false | _ -> true)
    |> Array.choose (fun x -> Expr.TryGetReflectedDefinition (x) |> Option.map (fun y -> x.Name, y)) |> Map.ofArray

let expand (g: MethodInfo -> Expr option) (expr: Expr) = 
    let f ((instanceO : Expr option),(yss:Var list list),rd, zs) =
        // Reshape will stop short of filling new 'shape' when there are not enough elements
        // This happens when Lambdas pattern matches with more Lambdas than there are Call parameters 
        // This is because there is no early stopping on the Lambdas pattern
        let reshape  (shape: int list) (xs: 'a list) : 'a list list =
            (([],xs),shape) ||> List.fold (fun (acc,xs) count -> 
                if xs.Length = 0 then (acc,[])
                elif xs.Length < count then (xs::acc,[])
                else List.chop count  xs |> fun (head,xs) -> (head::acc,xs))
            |> fst |> List.rev
        match instanceO,zs,yss with
        // special case single unit parameters which are empty
        | None,[], [y]::_ when y.Type = typeof<unit> -> Expr.Applications(rd,[[]])
        | Some(instance),[], [x]::[y]::_ when instance.Type = x.Type && y.Type = typeof<unit> -> Expr.Applications(rd,[[instance];[]])
        | None,[], _ -> failwithf "found an unexpected value %A" rd
        | Some(instance),_,(_::yss) -> Expr.Applications(rd,[instance] :: (zs |> reshape [for ys in yss -> ys.Length]))
        | None,_,_ -> Expr.Applications(rd,zs |> reshape [for ys in yss -> ys.Length])
        | _,_,_ -> failwithf "found an unexpected value %A " rd
        |> Some
    match expr with
    | Call(None, TryFunc g (Lambdas (yss,_) as rd), zs) -> 
        f (None, yss, rd, zs)
    | PropertyGet(instanceO,PropertyGetterWithReflectedDefinition (Lambdas(yss,_) as rd),zs) 
    | Patterns.Call(instanceO, MethodWithReflectedDefinition (Lambdas(yss,_) as rd), zs) ->
        f (instanceO, yss, rd, zs)
    //| Patterns.Call(instanceO, (TryFunc f (Lambdas(yss,_) as rd)), zs) ->
    | _ -> None

