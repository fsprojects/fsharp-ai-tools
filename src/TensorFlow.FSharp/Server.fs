namespace TensorFlow.FSharp


// NOTE: The TF_Server functions do not appear to be missing in the pre-build binaries we are currently using
//       The following functions are form c_api.h

//// Creates a new in-process TensorFlow server configured using a serialized
//// ServerDef protocol buffer provided via `proto` and `proto_len`.
////
//// The server will not serve any requests until TF_ServerStart is invoked.
//// The server will stop serving requests once TF_ServerStop or
//// TF_DeleteServer is invoked.
//TF_CAPI_EXPORT extern TF_Server* TF_NewServer(const void* proto,
//                                              size_t proto_len,
//                                              TF_Status* status);
//
//// Starts an in-process TensorFlow server.
//TF_CAPI_EXPORT extern void TF_ServerStart(TF_Server* server, TF_Status* status);
//
//// Stops an in-process TensorFlow server.
//TF_CAPI_EXPORT extern void TF_ServerStop(TF_Server* server, TF_Status* status);
//
//// Blocks until the server has been successfully stopped (via TF_ServerStop or
//// TF_ServerClose).
//TF_CAPI_EXPORT extern void TF_ServerJoin(TF_Server* server, TF_Status* status);
//
//// Returns the target string that can be provided to TF_SetTarget() to connect
//// a TF_Session to `server`.
////
//// The returned string is valid only until TF_DeleteServer is invoked.
//TF_CAPI_EXPORT extern const char* TF_ServerTarget(TF_Server* server);
//
//// Destroy an in-process TensorFlow server, frees memory. If server is running
//// it will be stopped and joined.
//TF_CAPI_EXPORT extern void TF_DeleteServer(TF_Server* server);
//