
# TensorFlow Examples in F#

List of examples
* ImageClassifier
   * Uses Resnet50 to classifiy an image into one of 1000 categories
* NeuralStyleTransfer
   * Uses Fast Style Transfer to transfer image styles
   * The pretrained weights are for style of Rain Princess by Lenoid Afremov
   * Different weights can be loaded to apply different styles
* AttGAN
   * Using a GAN to visually change the attributs of faces
   * This is a work in progress


Setup
* Make sure FSI is 64bit and does not shadow assemblies<sup>1</sup>
* run script setup.fsx

Run
* Scripts can be run by either "Send to FSI" or by FSI. For example `FSI.exe NeuralStyleTransfer.fsx --style wave`


### 1: Making sure FSI is 64bit and does not shadow
For Visual Studio Code:
* On Ubuntu: it should work with default settings
* On Windows: if FSI is defaulting to 32bit this will need to be changed to 64bit by setting the Ionide property for fsiFilePath to the full path to the fsiAnyCpu.exe.

`"FSharp.fsiFilePath": "C:\..fsiAnyCpu.exe"`
    
    See Ionide issue [956](https://github.com/ionide/ionide-vscode-fsharp/issues/956)

* On Mac: this is currently untested

For Visual Studio both 64bit needs to be enabled and shadow copy assemblies needs to be switched off. Tools -> Options -> F# Tools -> F# Interactive





