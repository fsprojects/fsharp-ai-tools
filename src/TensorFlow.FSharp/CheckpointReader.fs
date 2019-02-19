module CheckpointReader

open System
open System.IO

// NOTE: Tensorlfow Scala uses the underlining C++ RecordReader/Checkpoint/

// See training/checkpointable/util.py

// Functions possibly needed from


// 

// tensorflow/c/checkpoint_reader
// NewCheckpointReader(file_pattern : string) : CheckpointReader
// reader.HasTensor(name : string) : bool
// reader.GetTensor(name) : TFTensor
// reader.GetVariableToShapeMap() : Map<string,VariableShapes>
// reader.GetVariableToDataTypeMap() : Map<string,TFDataType>


//import tensorflow as tf
//import numpy as np
//
//sess = tf.Session()
//
//A = tf.get_variable("A",shape=(3,4),dtype=tf.float32,initializer=tf.zeros_initializer())
//B = tf.get_variable("B",shape=(3,4),dtype=tf.float32,initializer=tf.zeros_initializer())
//C = A + B
//
//restoreOp = io_ops.restore_v2("./my_test_model/my_test_model-10", ["A","B"],["",""],[tf.float32,tf.float32])
//ass1 = tf.assign(A,restoreOp[0],validate_shape=False)
//ass2 = tf.assign(B,restoreOp[1],validate_shape=False)
//
//def assign_all(xs):
//    with tf.control_dependencies(xs):
//        return tf.no_op("assign_all")
//
//restore_all = assign_all([ass1,ass2])
//
//sess.run(C) # this raises an uninitialized value error
//sess.run(restore_all)
//sess.run(C) # this works
//
//open ProtoBuf
//
//[<ProtoContract>]
//type Any = {
//    [<ProtoMember(1)>]
//    type_url : string
//    [<ProtoMember(2)>]
//    value : byte[]
//}
//
//ProtoBuf.Meta.RuntimeTypeModel.Default.Add(typeof<Google.Protobuf.WellKnownTypes.Any>,false).SetSurrogate(typeof<Any>)
//
//let deserialize<'a> = Serializer.Deserialize<'a>
//
//let graph = 
//    deserialize<TensorFlow.FSharp.Proto.MetaGraphDef>(
//        new MemoryStream(File.ReadAllBytes(@"G:\checkpoints\minutiae_train3\model.ckpt-64297.meta")))
//
//
//[| for KeyValue(x,v) in graph.CollectionDefs -> x,v.node_list|]
//
//// This is a bit weird as they are strings
//graph.CollectionDefs.["trainable_variables"].bytes_list.Values 
//|> Seq.map (fun x -> System.Text.UTF8Encoding.UTF8.GetString(x))
//|> Seq.toArray
//
//graph.CollectionDefs
