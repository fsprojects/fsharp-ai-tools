[<AutoOpen>]
module TensorFlow.NNImpl
#r "netstandard"
#r "../lib/TensorFlowSharp.dll"

open TensorFlow

type TFGraph with
    member this.Conv2DTranspose(value,filter, output_shape:int64[], strides:int64[], ?padding:string, ?data_format:string,?operName:string) = 
        let paddingV     = defaultArg padding "SAME"
        let data_formatV = defaultArg data_format "NHWC"
        use name = this.WithScope("conv2d_transpose") // NOTE: this needs control parameters
        if not (data_formatV = "NCHW" || data_formatV = "NHWC") then 
            failwith "dataformat has to be either NCHW or NHWC."
        let axis = if data_formatV = "NHWC" then 3 else 1
        let value_shape = this.GetShape(value)
        let filter_shape = this.GetShape(filter)
        if output_shape.Length <> 4 then
            failwithf "output_shape must have shape (4,) got %A" output_shape
        if value_shape.[axis] <> filter_shape.[3] then
            failwithf "input channels does not match filter's input channels, \n %i != %i" 
                value_shape.[axis]
                filter_shape.[3]
        if output_shape.[3] <> filter_shape.[2] then
            failwithf "output_shape does does not match filter's output channels, \n %i != %i" 
                value_shape.[axis]
                filter_shape.[3]
        if paddingV <> "VALID" && paddingV <> "SAME" then
            failwithf "padding must be either VALID or SAME: %s" paddingV

        this.Conv2DBackpropInput(
            input_sizes = this.Const(TFShape(output_shape).AsTensor()),
            filter = filter,
            out_backprop = value,
            strides = strides,
            padding = paddingV,
            data_format = data_formatV//,
            //?operName = operName // The name pass through does not seem to be working here for some reason
        )