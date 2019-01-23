[<AutoOpen>]
module TensorFlow.NNOps

#I __SOURCE_DIRECTORY__
#r "netstandard"
#r "../../tests/bin/Debug/net472/TensorFlow.FSharp.dll"

open System
open TensorFlow.FSharp

type TFGraph with
    /// https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/nn_impl.py
    member this.Moments(x:TFOutput, ?axes:TFOutput, ?shift, ?name, ?keep_dims) =
        let keep_dimsV = defaultArg keep_dims false
        use name = this.WithScope("moments") // NOTE: this needs control parameters

        let y = if x.OutputType = TFDataType.Half then this.Cast(x,TFDataType.Float) else x
        let mean = this.ReduceMean(y, axes |> Option.toNullable, keep_dims=Nullable(true), operName ="mean")
        let variance = this.ReduceMean(
                         this.SquaredDifference(y, this.StopGradient(mean)), 
                         axes |> Option.toNullable,
                         keep_dims=Nullable(true),
                         operName="variance")

        let maybeSqueezeAndCast (y:TFOutput) = 
            let y = if keep_dimsV then y else this.Squeeze(y)
            if x.OutputType = TFDataType.Half then this.Cast(y,TFDataType.Half) else y
        (mean |> maybeSqueezeAndCast, variance |> maybeSqueezeAndCast)
