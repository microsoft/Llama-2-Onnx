# **Running Llama 2 With ONNX**

This file will help you get started with the Llama 2 models using ONNX.

Pre-requisites:
* Python, pip
* git-lfs

Dependencies:
* torch
* onnxruntime-gpu (or onnxruntime for CPU only systems)
* numpy
* sentencepiece

``` bash
pip install torch onnxruntime-gpu numpy sentencepiece
```

To avoid requiring the end user to clone all versions of the model, each version is contained inside a git submodule. 
You can choose from: 
* 7B_FT_float16
* 7B_FT_float32
* 7B_float16
* 7B_float32
* 13B_FT_float16
* 13B_FT_float32
* 13B_float16
* 13B_float32

Use one of the above strings in the <chosen_submodule> placeholders below. You can initialze multiple submodules by repeating the init command with a different submodule name. 

``` bash
git clone https://github.com/microsoft/UploadTestTop.git
cd UploadTestTop
git submodule init <chosen_submodule> 
git submodule update
python Example_ONNX_LlamaV2.py --ONNX_file <chosen_submodule>/ONNX/LlamaV2_<chosen_submodule>.onnx --embedding_file <chosen_submodule>/embeddings.pth --TokenizerPath tokenizer.model --prompt "What is the lightest element?"
```
