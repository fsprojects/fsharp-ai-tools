namespace TensorFlow.FSharp
/// TODO consider deleteing if this remains unused

//open System
//open Utils
//open TensorFlow
////open Common
//open System
//open Utils
//
//
/////  """Decorator used to define TensorFlow functions.
/////
/////  Use this decorator to make a Python function usable directly as a TensorFlow
/////  function.
/////
/////  The decorated function must add ops to the default graph and return zero or
/////  more `Tensor` objects.  Call the decorator with named arguments, one for each
/////  argument of the function to decorate, with the expected type of the argument
/////  as value.
/////
/////  For example if the function to decorate accepts two `tf.float32` arguments
/////  named `x` and `y`, call the decorator with:
/////
/////      @Defun(tf.float32, tf.float32)
/////      def foo(x, y):
/////        ...
/////
/////  When you call the decorated function it will add `call` ops to the
/////  default graph and adds the definition of the function into the
/////  default graph. Because the addition of the function into the graph
/////  is deferred, the decorator can be used anywhere in the program.
/////
/////  Any variables created inside of the function are hoisted into the outer graph.
/////  Note that the variables are created in the variable scope that was active
/////  during the first call to the function. Subsequent function calls will refer to
/////  the same set of variables.
/////
/////  Definitions of functions in a graph are frozen as soon as the graph is used to
/////  create a session. However, new functions and new calls to existing functions
/////  may be added to the graph, with the new functions themselves becoming
/////  immediately frozen.
/////
/////  Example, but also see the [How To on functions](link_needed).
/////
/////  ```python
/////  # Defining the function.
/////  @tf.Defun(tf.float32, tf.float32)
/////  def MyFunc(x, y):
/////    return x + y, x - y
/////
/////  # Building the graph.
/////  a = tf.constant([1.0])
/////  b = tf.constant([2.0])
/////  c, d = MyFunc(a, b, name='mycall')
/////  ```
/////  """
/////    """Create a `Defun` decorator.
/////
/////    Args:
/////      *input_types: A list of `tf.DType`
/////      **kwargs: Optional keyword arguments, including
/////         func_name - (optional).  A python string, the name to use to
/////           declare this `Function` in the graph.
/////
/////         grad_func - (optional).  A function implementing the gradient
/////           of the function-to-register.  This is must be a
/////           `_DefinedFunction` object. The gradient
/////           function must satisfy the criterion defined in
/////           function.proto:GradientDef.
/////
/////         python_grad_func - (optional).  A function implementing the
/////           gradient of the function python-side. This function must
/////           take the current op and the gradients w.r.t. its outputs,
/////           and return the gradients w.r.t. the inputs. That is it must
/////           implement the interface expected by `tf.RegisterGradient`).
/////           This will be called by tf.gradients to add the gradient ops
/////           to the graph. At most one of grad_func and python_grad_func
/////           can be specified.
/////
/////         out_names = (optional). A list of strings, one per output
/////           tensor.
/////
/////         shape_func - (optional). A function taking the op and returning a list
/////           of static shapes to set for the function's outputs.
/////    """
///// NOTE: this was done with **kwargs but it should be possible to use optional arguments
///// TODO 
//type Defun(input_types:TFDataType[],
//             ?func_name:string,
//             ?grad_func:string,
//             ?python_grad_func:string,
//             ?out_names:string[],
//             ?extra_kwargs:KWArgs
//             ) = 
//    member private this.call(func:#Delegate) =
//        failwith "todo"
//        //match func with
//        //| :? Delegate -> failwith "todo"
//        //| _ -> failwith "todo"
//        // LINE 132 framework/function.py
//
//type FuncGraph() = 
//    inherit ops.TFFGraph()
//    // TODO function.py
//    /// Maps external tensor -> internal tensor (e.g. input placeholder).
//    let mutable captured = Map.empty<TFOutput,TFOutput>
//    member this._captured with get() = captured
//    member this.captures with get() = captured
//
//    
