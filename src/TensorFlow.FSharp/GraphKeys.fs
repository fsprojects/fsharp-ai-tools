namespace TensorFlow.FSharp
// TODO this is from framework/ops.py, consider folding it into Graph or to some other location

///  Standard names to use for graph collections.
///
///  The standard library uses various well-known names to collect and
///  retrieve values associated with a graph. For example, the
///  `tf.Optimizer` subclasses default to optimizing the variables
///  collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
///  specified, but it is also possible to pass an explicit list of
///  variables.
///
///  The following standard keys are defined:
///
///  * `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared
///    across distributed environment (model variables are subset of these). See
///    `tf.global_variables`
///    for more details.
///    Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`,
///    and all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`.
///  * `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each
///    machine. Usually used for temporarily variables, like counters.
///    Note: use `tf.contrib.framework.local_variable` to add to this collection.
///  * `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the
///    model for inference (feed forward). Note: use
///    `tf.contrib.framework.model_variable` to add to this collection.
///  * `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
///    be trained by an optimizer. See
///    `tf.trainable_variables`
///    for more details.
///  * `SUMMARIES`: the summary `Tensor` objects that have been created in the
///    graph. See
///    `tf.summary.merge_all`
///    for more details.
///  * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
///    produce input for a computation. See
///    `tf.train.start_queue_runners`
///    for more details.
///  * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
///    keep moving averages.  See
///    `tf.moving_average_variables`
///    for more details.
///  * `REGULARIZATION_LOSSES`: regularization losses collected during graph
///    construction.
///
///  The following standard keys are _defined_, but their collections are **not**
///  automatically populated as many of the others are:
///
///  * `WEIGHTS`
///  * `BIASES`
///  * `ACTIVATIONS`
module GraphKeys =

    /// Key to collect Variable objects that are global (shared across machines).
    /// Default collection for all variables, except local ones.
    let GLOBAL_VARIABLES = "variables"
    /// Key to collect local variables that are local to the machine and are not
    /// saved/restored.
    let LOCAL_VARIABLES = "local_variables"
    /// Key to collect local variables which are used to accumulate interal state
    /// to be used in tf.metrics.*.
    let METRIC_VARIABLES = "metric_variables"
    /// Key to collect model variables defined by layers.
    let MODEL_VARIABLES = "model_variables"
    /// Key to collect Variable objects that will be trained by the
    /// optimizers.
    let TRAINABLE_VARIABLES = "trainable_variables"
    /// Key to collect summaries.
    let SUMMARIES = "summaries"
    /// Key to collect QueueRunners.
    let QUEUE_RUNNERS = "queue_runners"
    /// Key to collect table initializers.
    let TABLE_INITIALIZERS = "table_initializer"
    /// Key to collect asset filepaths. An asset represents an external resource
    /// like a vocabulary file.
    let ASSET_FILEPATHS = "asset_filepaths"
    /// Key to collect Variable objects that keep moving averages.
    let MOVING_AVERAGE_VARIABLES = "moving_average_variables"
    /// Key to collect regularization losses at graph construction.
    let REGULARIZATION_LOSSES = "regularization_losses"
    /// Key to collect concatenated sharded variables.
    let CONCATENATED_VARIABLES = "concatenated_variables"
    /// Key to collect savers.
    let SAVERS = "savers"
    /// Key to collect weights
    let WEIGHTS = "weights"
    /// Key to collect biases
    let BIASES = "biases"
    /// Key to collect activations
    let ACTIVATIONS = "activations"
    /// Key to collect update_ops
    let UPDATE_OPS = "update_ops"
    /// Key to collect losses
    let LOSSES = "losses"
    /// Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
    let SAVEABLE_OBJECTS = "saveable_objects"
    /// Key to collect all shared resources used by the graph which need to be
    /// initialized once per cluster.
    let RESOURCES = "resources"
    /// Key to collect all shared resources used in this graph which need to be
    /// initialized once per session.
    let LOCAL_RESOURCES = "local_resources"
    /// Trainable resource-style variables.
    let TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables"

    /// Key to indicate various ops.
    let INIT_OP = "init_op"
    let LOCAL_INIT_OP = "local_init_op"
    let READY_OP = "ready_op"
    let READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op"
    let SUMMARY_OP = "summary_op"
    let GLOBAL_STEP = "global_step"

    /// Used to count the number of evaluations performed during a single evaluation
    /// run.
    let EVAL_STEP = "eval_step"
    let TRAIN_OP = "train_op"

    /// Key for control flow context.
    let COND_CONTEXT = "cond_context"
    let WHILE_CONTEXT = "while_context"

    /// Used to store v2 summary names.
    let internal SUMMARY_COLLECTION = "_SUMMARY_V2"

    /// List of all collections that keep track of variables.
    let internal VARIABLE_COLLECTIONS = 
                  [|
                      GLOBAL_VARIABLES;
                      LOCAL_VARIABLES;
                      METRIC_VARIABLES;
                      MODEL_VARIABLES;
                      TRAINABLE_VARIABLES;
                      MOVING_AVERAGE_VARIABLES;
                      CONCATENATED_VARIABLES;
                      TRAINABLE_RESOURCE_VARIABLES
                  |]

    /// Key for streaming model ports.
    /// NOTE(yuanbyu): internal and experimental.
    let internal STREAMING_MODEL_PORTS = "streaming_model_ports"
