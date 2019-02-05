namespace TensorFlow.FSharp
// TODO: (Miguel) need an overload that allows minval/maxval to be float16, float32, float64, bfloat16, int32, int64
// Or perhaps TFTensor overload.


// NOTE: (Matt), in general 32bit is flexible enough and fast enought to work with that we can set the expectation that 
// random numbers are generated in Float32 and you're free to cast it to the other data formats

// TODO 

//[<AutoOpen>]
//module RandomOps =
//    type TFGraph with
//        /// <summary>
//        /// Outputs random values from a normal distribution
//        /// </summary>
//        /// <returns>A tensor of the specified shape filled with random normal values.</returns>
//        /// <param name="shape">Shape of the output tensor.</param>
//        /// <param name="mean">The mean of the standard distribution.</param>
//        /// <param name="stddev">The standard deviation of the normal distribution.</param>
//        /// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
//        /// <param name="operName">Operation name, optional.</param>
//        member graph.RandomNormal (shape : TFShape, ?mean : float32, ?stddev : float32, ?seed : int, ?name : string) =
//            let mean = defaultArg mean 0.f
//            let stddev = defaultArg stddev 1.f
//            use ns = graph.NameScope(name, "RandomNormal")
//            let shapeTensor = graph.ShapeTensorOutput (shape)
//            let tmean = graph.Const(new TFTensor(mean), "mean")
//            let tstddev = graph.Const(new TFTensor(stddev), "stddev")
//            let graphSeed, localSeed = graph.GetRandomSeeds(?operationSeed=seed)
//            let rnd = graph.RandomStandardNormal(shapeTensor, TFDataType.Float32, int64 graphSeed, int64 localSeed)
//            let mul = graph.Mul(rnd, tstddev)
//            graph.Add(mul, tmean,name= string ns)
//
//        /// <summary>
//        /// Randoms the uniform.
//        /// </summary>
//        /// <returns>The uniform.</returns>
//        /// <param name="shape">Shape.</param>
//        /// <param name="minval">Minval.</param>
//        /// <param name="maxval">Maxval.</param>
//        /// <param name="seed">Seed.</param>
//        /// <param name="operName">Oper name.</param>
//        member graph.RandomUniform (shape : TFShape, ?minval : float32, ?maxval : float32, ?seed : int, ?name : string) =
//            let minval = defaultArg minval 0.f
//            let maxval = defaultArg maxval 1.f
//            use ns = graph.NameScope(name, "RandomUniform")
//            let shapeTensor = graph.ShapeTensorOutput (shape)
//            let minvalTensor = graph.Const (new TFTensor(minval), "minval")
//            let maxvalTensor = graph.Const (new TFTensor(maxval), "maxval")
//            let graphSeed, localSeed = graph.GetRandomSeeds(?operationSeed=seed)
//            let rnd = graph.RandomUniform(shapeTensor, TFDataType.Float32, int64 graphSeed, int64 localSeed)
//            let mul = graph.Mul(rnd, graph.Sub(maxvalTensor, minvalTensor))
//            graph.Add(mul, minvalTensor, name = string ns)
