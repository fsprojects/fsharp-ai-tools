// TODO: all functions with heuristically determined primary input will be modified to enable currying 
//  e.g. input |> TF.Conv2D (filters=filters)
// TODO: extend defaults to sensible things e.g. strides and dilations of [1,1,1,1]. This extends the built in ops
// TODO: unary functions as standalone functions without properties
// TODO: infix opeartions as overloaded
//       include conversions to constants for TFTensor conversions 
// e.g. - + * *. 

// Enable fluent design pattern by extending TFOutput

// input.Conv2D(filters1,strides=[1,2,2,1])
//      .Conv2D(filters2,strides=[1,1,1,1])
//      .Relu()

// Also
// M.dot(N)
