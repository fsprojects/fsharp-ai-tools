module FSharp.ML.Onnx.Utils.Expr

open FSharp.ML.Onnx.Utils
open Microsoft.FSharp.Reflection
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open Microsoft.FSharp.Quotations.Patterns

module Expr = 
    let rec flatten (expr: Expr) : Expr seq = 
        match expr with
        | ShapeVar _ -> Seq.singleton expr
        | Let(_,exp1,exp2) -> seq {yield expr; yield! flatten exp1; yield! flatten exp2}
        | LetRecursive(xs,body) -> seq {yield expr; yield! xs |> Seq.collect (snd >> flatten); yield! flatten body}
        | ShapeLambda (_, exp) -> seq { yield expr; yield! flatten exp}
        | ShapeCombination (_, exprs) ->  seq { yield expr; yield! exprs |> List.toSeq |> Seq.collect flatten}

    let rec visitor (f: 'a -> Expr -> ('a*Expr)) (state: 'a) (expr: Expr) : ('a*Expr) =
        let swap (x,y) = (y,x)
        let fold (acc:'a) (xs:Expr list) = xs |> List.mapFold (fun acc e -> visitor f acc e |> swap) acc |> swap
        let (acc,expr) = f state expr
        match expr with
        | ShapeVar _ -> (acc,expr)
        | Let(v,exp1,exp2) -> [exp1; exp2] |> fold acc |> fun (acc,xs) -> (acc, Expr.Let(v,xs.[0],xs.[1]))
        | LetRecursive(xs,body) ->  
            let (acc, ys) = xs |> List.map snd |> fold acc 
            let (acc, body) = visitor f acc body
            (acc,Expr.LetRecursive((xs |> List.map fst, ys) ||> List.zip, body))
        | ShapeLambda (v, expr) -> visitor f acc expr |> fun (x,y) -> (x, Expr.Lambda (v, y))
        | ShapeCombination (o, xs) -> 
            let (ys,acc) = xs |> List.mapFold (fun (acc:'a) (e:Expr) -> visitor f acc e |> swap) acc
            (acc, RebuildShapeCombination (o, ys))

    // TODO enable early stopping of visitor with a flag?
    let rec earyStopVisitor (f: 'a -> Expr -> ('a*Expr*bool)) (state: 'a) (expr: Expr) : ('a*Expr) =
        let swap (x,y) = (y,x)
        let fold (acc:'a) (xs:Expr list) = xs |> List.mapFold (fun acc e -> earyStopVisitor f acc e |> swap) acc |> swap
        let (acc,expr,earyStop) = f state expr
        if earyStop then (acc,expr) 
        else
            match expr with
            | ShapeVar _ -> (acc,expr)
            | Let(v,exp1,exp2) -> [exp1; exp2] |> fold acc |> fun (acc,xs) -> (acc, Expr.Let(v,xs.[0],xs.[1]))
            | LetRecursive(xs,body) ->  
                let (acc, ys) = xs |> List.map snd |> fold acc 
                let (acc, body) = earyStopVisitor f acc body
                (acc,Expr.LetRecursive((xs |> List.map fst, ys) ||> List.zip, body))
            | ShapeLambda (v, expr) -> earyStopVisitor f acc expr |> fun (x,y) -> (x, Expr.Lambda (v, y))
            | ShapeCombination (o, xs) -> 
                let (ys,acc) = xs |> List.mapFold (fun (acc:'a) (e:Expr) -> earyStopVisitor f acc e |> swap) acc
                (acc, RebuildShapeCombination (o, ys))

    let map (f: Expr -> Expr) (expr: Expr) = visitor (fun _ e -> ((),f(e))) () expr |> snd

    let unfoldWhileChangedWithEarlyStop (f: Expr -> bool*Expr option) (expr: Expr) =
        Seq.unfold (fun expr -> 
            earyStopVisitor (fun state expr -> match f(expr) with | r,Some(e) -> (true,e,r) | r,_ -> (state,expr,r)) false expr 
            |> function
            | false, _ -> None
            | true, expr -> Some(expr,expr)) expr

    let unfoldWhileChanged (f: Expr -> Expr option) (expr: Expr) =
        Seq.unfold (fun expr -> 
            visitor (fun state expr -> 
                match f(expr) with | Some(e) -> (true,e) | _ -> (state,expr)) false expr 
            |> function
            | false, _ -> None
            | true, expr -> Some(expr,expr)) expr

    let rec count (f: Expr -> bool) (expr: Expr) : int = 
        expr |> flatten |> Seq.sumBy (fun x -> if f x then 1 else 0)

    let replace (x: Expr) (y:Expr) (expr:Expr) = 
        map (fun t -> if t = x then y else t) expr

    let isApplication (x:Expr) = 
        match x with | Application(_,_) -> true | _ -> false

    let isLambda (x:Expr) = 
        match x with | Lambda(_,_) -> true | _ -> false

    let getVarsUsage (expr: Expr) = 
        expr |> flatten |> Seq.choose (function | Var(v) -> Some(v) | _ -> None) |> Seq.toArray

    /// To find expressions that are assigned but are not used
    //let getVarsAssignment (expr: Expr) = 
    //    expr |> flatten |> Seq.choose (function | Lambda(v,_) | Let(v,_,_) -> Some(v) | _ -> None) |> Seq.toArray

    let getCallNames (expr: Expr) = 
        expr |> flatten |> Seq.choose (function | Call(_,m,_) -> Some(m.Name) | _ -> None) |> Set

    let tryFirstCall (expr: Expr) = 
        expr |> flatten |> Seq.tryFind (function | Call(_,_,_) -> true | _ -> false)

    let firstCall (expr: Expr) = 
        expr |> flatten |> Seq.tryFind (function | Call(_,_,_) -> true | _ -> false) |> Option.get

    let countVar (v: Var) expr = 
        count (function | ShapeVar v2 -> v = v2 | _ -> false) expr

    /// This gets the method and makes it generic 
    let tryGetGenericMethod (expr: Expr) = 
        expr 
        |> tryFirstCall
        |> Option.bind (function | Call(_,mi,_) -> Some(mi) | _ -> None)
        |> Option.map (fun mi -> if mi.IsGenericMethod then mi.GetGenericMethodDefinition() else mi)

    let applyTransform (f: Expr -> Expr option) (expr: Expr) =
        expr 
        |> unfoldWhileChanged f
        |> Seq.tryLast 
        |> Option.defaultValue expr

    // TODO better name?
    let concat<'a>(xs: Expr list) = Expr.Cast<'a[]>(Expr.NewArray(typeof<'a>, xs))

    let applyTransforms (fs:(Expr -> Expr option)[])  = 
        let cmb (fs:(Expr -> Expr option)[]) (expr: Expr) = 
            seq { for f in fs do match f expr with | Some(x) -> yield x | _ -> ()} |> Seq.tryHead
        applyTransform (cmb fs)

type Expr with
    member this.TryGetLocation() =
        this.CustomAttributes 
        |> List.choose (function 
            | NewTuple 
                [
                    String "DebugRange"; 
                    NewTuple [String file; Int32 a; Int32 b; Int32 c; Int32 d]
                ] -> Some(file, a, b, c, d) 
            | _ -> None)
        |> List.tryHead

    static member Lambdas(vars: Var list list, body: Expr) : Expr = 
        let makeTupledLambda (vars : Var list) (body: Expr) : Expr =
            match vars with
            | [] ->  Expr.Lambda(Var("unitVar", typeof<unit>), body)
            | [x] -> Expr.Lambda(x,body)
            | _ -> 
                let tuple = Var("tupledArg",FSharpType.MakeTupleType([|for v in vars -> v.Type|]))
                (vars |> List.indexed, body)
                ||> List.foldBack (fun (i,v) state -> Expr.Let(v,Expr.TupleGet(Expr.Var(tuple),i),state))
                |> fun body -> Expr.Lambda(tuple, body)
        (vars,  body) ||> List.foldBack makeTupledLambda

module ExprTransforms = 

    /// A collection of simple transforms
    /// These are probably superseded by more advanced transforms
    module Simple = 
        /// Simplifies a common binding patterns
        let bindings = 
            function 
            | Patterns.Application(ExprShape.ShapeLambda(v, body), assign) 
            | Patterns.Let(v, assign, body) ->
                match Expr.countVar v body with
                | 0 -> Some(body)
                | 1 -> Some(Expr.map (function | ShapeVar v2 when v = v2 -> assign | x -> x) body)
                | _ -> None
            | _ -> None

        /// Removes construcing and deconstructing a tuple which is only used once
        /// This can happen when a Expr.Var is substitued by an Expr.NewTuple and only one item from the tuple is used
        let tuples = 
            function 
            | TupleGet(NewTuple(exp),index) when index < exp.Length -> Some(exp.[index]) 
            | _ -> None

        /// TODO generalize to beyond immediate Tuple decomposition
        let newTuple = 
            function
            | Application(NewTuple(args),Lambda(patternInput,body))
            | Let(patternInput,NewTuple(args),body) ->
                let rec f =
                    function
                    | Let(v2,TupleGet(Var v3 ,index ),body) 
                        when  v3 = patternInput && index < args.Length -> 
                        (v2,args.[index],body) :: f body 
                    | _ -> []
                match f body with
                | [] -> None
                | ys ->
                    let (_,_,body) = ys |> List.last 
                    if Expr.countVar patternInput body > 0 then None
                    else 
                        (ys, body) 
                        ||> List.foldBack (fun (v,e,_) body -> Expr.Let(v,e,body)) |> Some
            | _ -> None

        /// TODO perf test to find out if this style of SpecificCall is slow
        let builtIns = function
            | SpecificCall <@ (|>) @> (None,_,[arg;func]) 
            | SpecificCall <@ (<|) @> (None,_,[func;arg]) -> Some(Expr.Application(func,arg))
            | SpecificCall <@ (||>) @> (None,_,[arg1;arg2;func]) 
            | SpecificCall <@ (<||) @> (None,_,[func;arg1;arg2]) -> Some(Expr.Applications(func,[[arg1];[arg2]]))
            | SpecificCall <@ (|||>) @> (None,_,[arg1;arg2;arg3;func]) 
            | SpecificCall <@ (<|||) @> (None,_,[func;arg1;arg2;arg3]) -> Some(<@@ (%%func) %%arg1 %%arg2 %%arg3 @@>) 
            | SpecificCall <@ id @> (None,_,[arg]) -> Some(arg)
            | SpecificCall <@ fst @> (None,_,[arg]) -> Some(Expr.TupleGet(arg,0))
            | SpecificCall <@ snd @> (None,_,[arg]) -> Some(Expr.TupleGet(arg,1))
            //| SpecificCall <@ ignore @> (None,_,[arg]) -> Some(Expr.Lambda( Value.n)) // Lambda(_arg1, Value (<null>))
            | _ -> None

        let selfMatch (expr: Expr) = 
            match expr with
            //| Lambda(v1,Var(v2)) when v1 = v2 -> Some(Expr.Var(v1))
            | NewTuple([]) -> None
            | NewTuple(xs) ->
                xs 
                |> List.map (function | TupleGet(x,i) -> Some(x,i) | _ -> None) 
                |> List.toArray
                |> Option.all
                |> Option.bind (fun xs -> 
                    let isSelf = 
                        ((true,0),xs) 
                        ||> Array.fold (fun (v,i1) (x,i2) -> (v && i1 = i2 && x = fst xs.[0],i1+1)) |> fst
                    if isSelf then Some(fst xs.[0]) else None)
            | _ -> None

    /// Combines an array of transform functions and performs the transforms recursively at the same time in the order given
    //let groupUnfold (fs: (Expr -> Expr option)[]) (expr: Expr) =
    //    expr |> Expr.unfoldWhileChanged (fun x -> fs |> Seq.choose (fun f -> f(x)) |> Seq.tryHead)

    /// This replaces function calls with the reflected definition
    /// TODO extend this to include PropertyGet and PropertySet (?)
//    let expandCalls  = 
//        function 
//        | Patterns.Call(instanceO, MethodWithReflectedDefinition (Lambdas(yss,_) as rd), zs) ->
//            // WARN instanceO is not handled and 
//            Some(Expr.Applications(rd,zs |> List.reshape [for ys in yss -> ys.Length] 
//                        |> List.map (List.filter (fun z -> z.Type <> typeof<unit>))))
//        | _ -> None


    let expandWithReflectedDefinition  = 
        function
        | PropertyGet(instanceO,PropertyGetterWithReflectedDefinition (Lambdas(yss,_) as rd),zs) 
        //| Call(None, TryFunc f (Lambdas (yss,_) as rd), zs) 
        | Patterns.Call(instanceO, MethodWithReflectedDefinition (Lambdas(yss,_) as rd), zs) ->
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
            | Some(instance),[], [x]::[y]::_ 
                when instance.Type = x.Type && y.Type = typeof<unit> -> 
                    Expr.Applications(rd,[[instance];[]])
            | None,[], _ -> failwithf "found an unexpected value %A" rd
            | Some(instance),_,(_::yss) -> 
                Expr.Applications(rd,[instance] :: (zs |> reshape [for ys in yss -> ys.Length]))
            | None,_,_ -> Expr.Applications(rd,zs |> reshape [for ys in yss -> ys.Length])
            | _,_,_ -> failwithf "found an unexpected value %A " rd
            |> Some
        | _ -> None

//        | PropertyGet(instanceO,PropertyGetterWithReflectedDefinition (Lambdas(yss,_) as rd),zs) 
//        //| Call(None, TryFunc f (Lambdas (yss,_) as rd), zs) 
//        | Patterns.Call(instanceO, MethodWithReflectedDefinition (Lambdas(yss,_) as rd), zs) ->

    /// This simplifies applying exprs into lambdas vars where vars are either not used or only used once
    /// TODO / WARN we're removing unit applications which will cause issues when there is side-effectful code
    /// It could be possible to check if side-effectful code is in the body and use this to determine if to keep
    let reduceApplicationsAndLambdas removeUnits  (expr: Expr) =
        let f (xss : Expr list list,yss,body) =
            // TODO check if any changes can be made, if not then short circuit back
            let (yss, remaining) = yss |> List.chop xss.Length
            let body = Expr.Lambdas(remaining,body)
            let pairs = (xss,yss) ||> List.zip 
            let checkMatch = 
                pairs |> List.exists (function 
                    | ([],[_]) -> false 
                    | (xs,ys) -> 
                        (xs.Length = ys.Length) && ((xs,ys) 
                        ||> List.zip |> List.exists (fun (x,y) -> x.Type <> y.Type))) 
            if checkMatch then failwith "Non matching Applications and Lambdas - Not sure if this will ever happen"
            let varCounts = body |> Expr.getVarsUsage |> Array.countBy id |> Map.ofArray
            let varMapping = 
                pairs |> List.collect (function 
                                       | ([],[_]) -> [] 
                                       | x -> x ||> List.zip |> List.map swap ) |> Map.ofList

            let singleVars = varMapping |> Map.filter (fun k _ -> varCounts.TryFind(k) = Some(1))
            let body = body |> Expr.map (function | ShapeVar (Found singleVars x) ->  x | x -> x)
            let filterVars = 
                set [ 
                        for KeyValue(k,_) in varMapping do 
                            match varCounts.TryFind(k) with | None | Some(1) -> yield k | _ -> () 
                    ]
            let (xssCount,yssCount) = (xss |> List.sumBy List.length,yss |> List.sumBy List.length)
            let (xss,yss) = 
                pairs 
                |> List.map (function 
                    | [],[_] as x -> (if removeUnits then [],[] else x) 
                    | zs -> zs ||> List.zip |> List.filter (fun (_,v) -> filterVars.Contains v |> not) |> List.unzip)
                |> List.filter (function | [],[] -> false | _ -> true)
                |> List.unzip
            let (xssCount1,yssCount1) = (xss |> List.sumBy List.length,yss |> List.sumBy List.length)
            if (xssCount,yssCount) = (xssCount1,yssCount1) then None else Some (xss,yss,body)

        match expr with
        //| Applications(Lambdas(yss,body),xss) ->  
        | Applications(Let(objectArg, expr,Lambdas(yss,body)),xss) ->  
            f (xss, yss, body) 
            |> Option.map (fun (xss,yss,body) -> 
                Expr.Applications(Expr.Let(objectArg,expr,Expr.Lambdas(yss,  body)),xss))
        | Applications(Lambdas(yss,body),xss) ->  
            f (xss, yss, body) 
            |> Option.map (fun (xss,yss,body) -> Expr.Applications(Expr.Lambdas(yss,  body) ,xss))
        | _ -> None

    /// This takes a tupled function application and turns it into a curried function
    /// This was used expanding calls without tuples.
    /// After reduceApplicationAndLambdas was implemented this function was no longer needed
    let toCurried =
        function
        | Lambdas(xss : Var list list,body) when xss.Length > 1 -> 
            Some(Expr.Lambdas([for xs in xss do for x in xs -> [x]],body))
        | _ -> None

    /// This pre-computes Math when used on literals
    /// NOTE Will consider expanding this if it could be useful
//    let evaulateMath =
//        function 
//        | SpecificCall <@ (*) @> (None,_,[exp1;exp2]) as exp -> 
//            match exp1,exp2 with 
//            | Int32(a),Int32(b) -> Some(Expr.Value(a * b)) 
//            | Int64(a),Int64(b) -> Some(Expr.Value(a * b)) 
//            | Single(a),Single(b) -> Some(Expr.Value(a * b)) 
//            | Double(a),Double(b) -> Some(Expr.Value(a * b)) 
//            | Value(_),Value(_) -> Expr.Value(exp.EvaluateUntyped())
//            | _ -> None
//        | _ -> None


       /// Alternative method to simplifying values would be to have a white list of functions where every 
       /// expression for which the sub expression is white list is evaluated
//       let evaluateMath2 = 
//            function
//            | Value v -> None
//            | _ when not (expr |> Expr.flatten |> Seq.map (function | SpecificCall <@ (+) @> _ -> true 
//                    | SpecificCall <@ (*) @> _ -> false | _ -> false ) |> Seq.exists id) -> 
//                Some(Expr.Value(expr.EvaluateUntyped(),expr.Type))
//            | _ -> None)  
//            |> Seq.map sprintExpr

// TODO move and rename
// TODO also add some logging to make debugging easier
let simplify (expr: Expr) = 
    expr
    |> Expr.applyTransform ExprTransforms.expandWithReflectedDefinition
    |> Expr.applyTransform ExprTransforms.Simple.builtIns
    |> Expr.applyTransforms 
        [|
            ExprTransforms.reduceApplicationsAndLambdas true
            ExprTransforms.Simple.tuples
            ExprTransforms.Simple.newTuple
        |]
    |> Expr.applyTransform ExprTransforms.Simple.bindings
    |> Expr.applyTransform ExprTransforms.Simple.selfMatch

