namespace TensorFlow.FSharp
/// NOTE: This will probably need to be fundamentally reworked
//
// TensorFlow.cs; Bindings to the TensorFlow C API for .NET
// 
// Authors:
//   Miguel de Icaza (miguel@microsoft.com)
//
open System
open System.Numerics
open System.Runtime.InteropServices
open System.Text


/// <summary>
/// The Variable class holds the Output nodes that are used to initialize, read and assign a value to a variable.   
/// </summary>
/// <remarks>
/// A variable maintains state in the graph across calls to `run()`. You add a
/// variable to the graph by constructing an instance of the class `Variable`.
/// 
/// The `Variable()` constructor requires an initial value for the variable,
/// which can be a `Tensor` of any type and shape. The initial value defines the
/// type and shape of the variable. After construction, the type and shape of
/// the variable are fixed. The value can be changed using one of the assign
/// methods.
/// 
/// When a variable is created a VarHandleOp is created which is returned as 
/// the VariableOp property, an assign operation is created that can be accessed
/// using the assignHandle and you can read the value of the variable using the
/// ReadHandle.
/// 
/// When you launch the graph, variables have to be explicitly initialized before
/// you can run Ops that use their value. You can initialize a variable by
/// running its *initializer op*, restoring the variable from a save file, or
/// simply running an `assign` Op that assigns a value to the variable. In fact,
/// the variable *initializer op* is just an `assign` Op that assigns the
/// variable's initial value to the variable itself.
/// 
/// There is an implicit conversion from the Variable into the VarHandleOp if
/// used.
/// </remarks>
type TFVariable internal (variableHandle : TF_Output, readHandle : TF_Output, assignOp : TF_Operation) = 

    /// <summary>
    /// Returns the ReadVariableOp that is used to fetch the value of the variable from the graph.
    /// </summary>
    /// <value>The read op.</value>
    member __.Read = new TFOutput(readHandle)

    /// <summary>
    /// Returns the AssignVariableOp that is used to assign the initial value to the variable from the graph.
    /// </summary>
    /// <value>The assign op.</value>
    member __.Assign = new TFOperation(assignOp)

    /// <summary>
    /// Returns the VarHandleOp that was created using the shape of the initial value.
    /// </summary>
    /// <value>The variable op.</value>
    member __.VariableOp = new TFOutput(variableHandle);

    /// <summary>
    /// Returns the VarHandleOp (the VariableOp property).
    /// </summary>
    /// <returns>The variable handle created for the variable.</returns>
    /// <param name="variable">Variable reference.</param>
    /// <remarks>
    /// This implicit operator exists to preserve the compatibility with code that
    /// created Variables and expected the result to be the VariableOp.
    /// </remarks>
    static member op_Implicit (variable : TFVariable) : TFOutput = new TFOutput(variable.VariableOp)

    interface IComparable with 
        member this.CompareTo(x : obj) = 
            if (x.GetType() <> this.GetType()) then -1
            else (this :> IComparable<TFVariable>).CompareTo(x :?> TFVariable)

    interface IComparable<TFVariable> with 
        member this.CompareTo(other : TFVariable) =
            let left = this.VariableOp.Struct.handle.ToInt64()
            let right = other.VariableOp.Struct.handle.ToInt64()
            if left <> right then
                left.CompareTo(right)
            else this.VariableOp.Struct.index.CompareTo(other.VariableOp.Struct.index)

    override this.Equals(x:obj) = 
        match x with
        | :? TFVariable as other -> (this.VariableOp.Struct = other.VariableOp.Struct)
        | _ -> false
    
    override this.GetHashCode() = this.VariableOp.Struct.GetHashCode()