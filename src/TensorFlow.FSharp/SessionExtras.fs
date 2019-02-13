[<AutoOpen>]
module TensorFlow.FSharp.SessionExtras

open System
open System.Text

type TFSession with
     /// <summary>
     /// Restores a tensor from a serialized tensorflor file.
     /// </summary>
     /// <returns>The deserialized tensor from the file.</returns>
     /// <param name="filename">File containing your saved tensors.</param>
     /// <param name="tensor">The name that was used to save the tensor.</param>
     /// <param name="type">The data type for the tensor.</param>
     /// <code>
     /// using (var session = new Session()){
     ///   var a = session.Graph.Const(30, "a");
     ///   var b = session.Graph.Const(12, "b");
     ///   var multiplyResults = session.GetRunner().Run(session.Graph.Add(a, b));
     ///   var multiplyResultValue = multiplyResults.GetValue();
     ///   Console.WriteLine("a*b={0}", multiplyResultValue);
     ///   session.SaveTensors($"saved.tsf", ("a", a), ("b", b));
     /// }
     /// </code>
     member this.RestoreTensor(filename : string, tensor : string, _type : TFDataType) : TFOutput =
         let graph = this.Graph
         graph.Restore (graph.Const (TFTensor.CreateString (Encoding.UTF8.GetBytes (filename))),
                            graph.Const (TFTensor.CreateString (Encoding.UTF8.GetBytes (tensor))),
                            _type);

     /// <summary>
     /// Saves the tensors in the session to a file.
     /// </summary>
     /// <returns>The tensors.</returns>
     /// <param name="filename">File to store the tensors in (for example: tensors.tsf).</param>
     /// <param name="tensors">An array of tuples that include the name you want to give the tensor on output, and the tensor you want to save.</param>
     /// <remarks>
     /// <para>
     /// Tensors saved with this method can be loaded by calling <see cref="M:RestoreTensor"/>.
     /// </para>
     /// <code>
     /// using (var session = new Session ()) {
     ///   var a = session.Graph.Const(30, "a");
     ///   var b = session.Graph.Const(12, "b");
     ///   var multiplyResults = session.GetRunner().Run(session.Graph.Add(a, b));
     ///   var multiplyResultValue = multiplyResults.GetValue();
     ///   Console.WriteLine("a*b={0}", multiplyResultValue);
     ///   session.SaveTensors($"saved.tsf", ("a", a), ("b", b));
     /// }
     /// </code>
     /// </remarks>
     member this.SaveTensors(filename : string, [<ParamArray>] tensors : (string*TFOutput) []) : TFTensor [] =
//            return GetRunner ().AddTarget (Graph.Save (Graph.Const (TFTensor.CreateString (Encoding.UTF8.GetBytes (filename)), TFDataType.String),
//                      Graph.Concat (Graph.Const (0), tensors.Select (T => {
//                          TFTensor clone = TFTensor.CreateString (Encoding.UTF8.GetBytes (T.Item1));
//                          return Graph.Const (new TFTensor (TFDataType.String,
//                                              new long [] { 1 },
//                                              clone.Data,
//                                              clone.TensorByteSize,
//                                              null, IntPtr.Zero));
//                      }).ToArray ()), tensors.Select (d => d.Item2).ToArray ())).Run ();
         let graph = this.Graph
         let clonedTensors = 
                     tensors |> Array.map (fun (x,_) ->  
                         let clone = TFTensor.CreateString (System.Text.Encoding.UTF8.GetBytes (x))
                         graph.Const (new TFTensor(TFDataType.String,  [|1L|], clone.Data, clone.TensorByteSize, null, IntPtr.Zero)))

         let save : TFOperation = graph.Save (graph.Const (TFTensor.CreateString (Encoding.UTF8.GetBytes (filename)), TFDataType.String), graph.Concat (graph.Const (new TFTensor(0)), clonedTensors), tensors |> Array.map snd)

         this.GetRunner().AddTarget(save).Run ()

