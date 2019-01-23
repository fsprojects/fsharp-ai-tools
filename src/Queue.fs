namespace TensorFlow.FSharp

/// Base class for queue implementations
/// Port of Python implementation https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/data_flow_ops.py
/// <summary>
/// A queue is a TensorFlow data structure that stores tensors across
/// multiple steps, and exposes operations that enqueue and dequeue
/// tensors.
/// Each queue element is a tuple of one or more tensors, where each
/// tuple component has a static dtype, and may have a static shape.The
/// queue implementations support versions of enqueue and dequeue that
/// handle single elements, versions that support enqueuing and
/// dequeuing a batch of elements at once.
/// </summary>
/// <param name="session">Session instance</param>
[<AbstractClass>]
type  QueueBase(session : TFSession) =

    /// <summary>
    /// The session that this QueueBased was created for.
    /// </summary>
    /// <value>The session.</value>
    member internal this.Session = session

    /// <summary>
    ///   Enqueues a tuple of one or more tensors in this queue.
    /// </summary>
    /// <param name="components">
    ///   One or more tensors from which the enqueued tensors should be taken.
    /// </param>
    /// <param name="operationName">
    ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueEnqueueV2'.
    /// </param>
    /// <param name="timeout_ms">
    ///   Optional argument
    ///   If the queue is full, this operation will block for up to
    ///   timeout_ms milliseconds.
    ///   Note: This option is not supported yet.
    /// </param>
    /// <returns>
    ///   Returns the description of the operation
    /// </returns>
    /// <remarks>
    ///   The components input has k elements, which correspond to the components of
    ///   tuples stored in the given queue.
    /// </remarks>
    abstract member Enqueue : components : TFOutput [] * ?timeout_ms : int64 * ?operationName : string -> TFOperation 

    /// <summary>
    ///   Dequeues a tuple of one or more tensors from this queue.
    /// </summary>
    /// <param name="operationName">
    ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueDequeueV2'.
    /// </param>
    /// <param name="timeout_ms">
    ///   Optional argument
    ///   If the queue is empty, this operation will block for up to
    ///   timeout_ms milliseconds.
    ///   Note: This option is not supported yet.
    /// </param>
    /// <returns>
    ///   One or more tensors that were dequeued as a tuple.
    ///   The Operation can be fetched from the resulting Output, by fethching the Operation property from the result.
    /// </returns>
    /// <remarks>
    ///   This operation has k outputs, where k is the number of components
    ///   in the tuples stored in the given queue, and output i is the ith
    ///   component of the dequeued tuple.
    /// </remarks>
    abstract member Dequeue : ?timeout_ms : int64 * ?operationName : string -> TFOutput [] 

    /// <summary>
    /// Gets the size of this queue.
    /// </summary>
    /// <param name="operationName"></param>
    /// <returns>queue size</returns>
    abstract member GetSize : ?operationName : string -> TFOutput 

/// <summary>
/// A FIFOQueue that supports batching variable-sized tensors by padding.
/// Port of Python implementation https://github.com/tensorflow/tensorflow/blob/b46340f40fe5e2ec9bfcd385b07cfb914055fb51/tensorflow/python/ops/data_flow_ops.py#L697
/// </summary>
/// <summary>
/// Creates a queue that dequeues elements in a first-in first-out order.
/// A `PaddingFIFOQueue` has bounded capacity; supports multiple concurrent
/// producers and consumers; and provides exactly-once delivery.
/// A `PaddingFIFOQueue` holds a list of up to `capacity` elements.Each
/// element is a fixed-length tuple of tensors whose dtypes are
/// described by `dtypes`, and whose shapes are described by the `shapes`
/// </summary>
/// <param name="session"></param>
/// <param name="componentTypes">The type of each component in a tuple.</param>
/// <param name="shapes">
///   Optional argument
///   The shape of each component in a value. The length of this attr must
///   be either 0 or the same as the length of component_types.
///   Shapes of fixed rank but variable size are allowed by setting
///   any shape dimension to -1.  In this case, the inputs' shape may vary along
///   the given dimension, and DequeueMany will pad the given dimension with
///   zeros up to the maximum shape of all elements in the given batch.
///   If the length of this attr is 0, different queue elements may have
///   different ranks and shapes, but only one element may be dequeued at a time.</param>
/// <param name="capacity"> Optional argument. The upper bound on the number of elements in this queue. Negative numbers mean no limit.</param>
/// <param name="container"> Optional argument. If non-empty, this queue is placed in the given container. Otherwise, a default container is used.</param>
/// <param name="operationName"> If specified, the created operation in the graph will be this one, otherwise it will be named 'PaddingFIFOQueueV2'.</param>
type PaddingFIFOQueue(session : TFSession, componentTypes : TFDataType [], shapes : TFShape [], ?capacity : int, ?container : string, ?operationName : string) =
    inherit QueueBase(session)
    let graph = session.Graph
    let handle = graph.PaddingFIFOQueueV2 ( componentTypes, shapes, ?capacity = (capacity |> Option.map int64), ?container = container, ?name=operationName)

    /// <summary>
    ///   Enqueues a tuple of one or more tensors in this queue.
    /// </summary>
    /// <param name="components">
    ///   One or more tensors from which the enqueued tensors should be taken.
    /// </param>
    /// <param name="operationName">
    ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueEnqueueV2'.
    /// </param>
    /// <param name="timeout_ms">
    ///   Optional argument
    ///   If the queue is full, this operation will block for up to
    ///   timeout_ms milliseconds.
    ///   Note: This option is not supported yet.
    /// </param>
    /// <returns>
    ///   Returns the description of the operation
    /// </returns>
    /// <remarks>
    ///   The components input has k elements, which correspond to the components of
    ///   tuples stored in the given queue.
    ///   
    ///   N.B. If the queue is full, this operation will block until the given
    ///   element has been enqueued (or 'timeout_ms' elapses, if specified).
    /// </remarks>
    override this.Enqueue (components : TFOutput [], ?timeout_ms : int64, ?operationName : string) =
        graph.QueueEnqueueV2 (handle, components, ?timeout_ms = timeout_ms, ?name = operationName)

    /// <summary>
    ///   Enqueues a tuple of one or more tensors in this queue and runs the session.
    /// </summary>
    /// <param name="components">
    ///   One or more tensors from which the enqueued tensors should be taken.
    /// </param>
    /// <param name="operationName">
    ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueEnqueueV2'.
    /// </param>
    /// <param name="inputValues">
    ///   Values to enqueue
    /// </param>
    /// <param name="timeout_ms">
    ///   Optional argument
    ///   If the queue is full, this operation will block for up to
    ///   timeout_ms milliseconds.
    ///   Note: This option is not supported yet.
    /// </param>
    /// <returns>
    ///   Returns the description of the operation
    /// </returns>
    /// <remarks>
    ///   The components input has k elements, which correspond to the components of
    ///   tuples stored in the given queue.
    /// </remarks>
    member this.EnqueueExecute (components : TFOutput [], inputValues : TFTensor [], ?timeout_ms : int64, ?operationName : string) : TFTensor [] =
        let enqueueOp = this.Enqueue (components, ?timeout_ms=timeout_ms, ?operationName=operationName)
        session.Run (components, inputValues, [||], [|enqueueOp|])

    /// <summary>
    ///   Dequeues a tuple of one or more tensors from the given queue.
    /// </summary>
    /// <param name="operationName">
    ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueDequeueV2'.
    /// </param>
    /// <param name="timeout_ms">
    ///   Optional argument
    ///   If the queue is empty, this operation will block for up to
    ///   timeout_ms milliseconds.
    ///   Note: This option is not supported yet.
    /// </param>
    /// <returns>
    ///   One or more tensors that were dequeued as a tuple.
    ///   The Operation can be fetched from the resulting Output, by fethching the Operation property from the result.
    /// </returns>
    /// <remarks>
    ///   This operation has k outputs, where k is the number of components
    ///   in the tuples stored in the given queue, and output i is the ith
    ///   component of the dequeued tuple.
    /// </remarks>
    override this.Dequeue (?timeout_ms : int64, ?operationName : string) : TFOutput [] =
        graph.QueueDequeueV2 (handle, componentTypes, ?timeout_ms=timeout_ms, ?name=operationName)

    /// <summary>
    ///   Dequeues a tuple of one or more tensors from this queue and runs the session.
    /// </summary>
    /// <param name="operationName">
    ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueDequeueV2'.
    /// </param>
    /// <param name="timeout_ms">
    ///   Optional argument
    ///   If the queue is empty, this operation will block for up to
    ///   timeout_ms milliseconds.
    ///   Note: This option is not supported yet.
    /// </param>
    /// <returns>
    ///   One or more tensors that were dequeued as a tuple.
    ///   The Operation can be fetched from the resulting Output, by fethching the Operation property from the result.
    /// </returns>
    /// <remarks>
    ///   This operation has k outputs, where k is the number of components
    ///   in the tuples stored in the given queue, and output i is the ith
    ///   component of the dequeued tuple.
    /// </remarks>
    member this.DequeueExecute (?timeout_ms : int64, ?operationName : string) : TFTensor [] =
        let values = graph.QueueDequeueV2 (handle, componentTypes, ?timeout_ms=timeout_ms, ?name=operationName)
        session.Run ([||],[||], values)

    /// <summary>
    ///   Dequeues elements from this queue and cast all elements to specific T type. It can be use when all elements in the queue of the same T type
    /// </summary>
    /// <param name="operationName">
    ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueDequeueV2'.
    /// </param>
    /// <param name="timeout_ms">
    ///   Optional argument
    ///   If the queue is empty, this operation will block for up to
    ///   timeout_ms milliseconds.
    ///   Note: This option is not supported yet.
    /// </param>
    /// <returns>
    ///   
    /// </returns>
    member this.DequeueExecute<'T when 'T :> TFTensor > (?timeout_ms : int64, ?operationName : string) =
        this.DequeueExecute (?timeout_ms=timeout_ms, ?operationName=operationName) |> Array.cast<'T>

    /// <summary>
    /// Gets the size of this queue.
    /// </summary>
    /// <param name="operationName">If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueSizeV2'.</param>
    /// <returns>queue size</returns>
    override this.GetSize (?operationName : string) : TFOutput =
        graph.QueueSizeV2 (handle, ?name = operationName)

    /// <summary>
    /// Uses provided session instance to obtain the size of this queue
    /// </summary>
    /// <param name="operationName">If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueSizeV2'.</param>
    /// <returns>number of elements in the queue</returns>
    member this.GetSizeExecute (?operationName : string) : int =
        let sizeOutput = this.GetSize (?operationName=operationName)
        let x = session.Run ([||], [||], [|sizeOutput|]) |> Seq.head 
        x.GetValue() :?> int
