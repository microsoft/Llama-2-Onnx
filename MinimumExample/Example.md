# **Running Llama 2 With ONNX**

Pre-requisites:
* Python, pip
* git-lfs

Dependencies:
* torch
* onnxruntime-gpu (or onnxruntime for CPU only systems)
* numpy
* sentencepiece

You can install the dependencies with the following command:
``` bash
pip install torch onnxruntime-gpu numpy sentencepiece
```

## **Cloning The Repository and Sub-Modules**
To avoid requiring the end user to clone all versions of the model, each version is contained inside a git submodule. 
You can choose from: 
* 7B_CHAT_HF_CUDA_OPT_float16
* 7B_CHAT_HF_CUDA_OPT_float32
* 7B_HF_CUDA_OPT_float16
* 7B_HF_CUDA_OPT_float32

Use one of the above strings in the <chosen_submodule> placeholders below. You can initialize multiple submodules by repeating the init command with a different submodule name. 

``` bash
git clone https://github.com/microsoft/Llama-2-Onnx.git
cd Llama-2-Onnx
git submodule init <chosen_submodule> 
git submodule update
```

## **Running The Minimum Example**
The minimum example is a simple command line program that will complete some text with the chosen version of Llama 2. You can run it with the following command:

``` bash
python MinimumExample/Example_ONNX_LlamaV2.py --ONNX_file <chosen_submodule>/ONNX/<chosen_submodule>.onnx --TokenizerPath tokenizer.model --prompt "What is the lightest element?"
```

Output:
```
The lightest element is hydrogen. Hydrogen is the lightest element on the periodic table, with an atomic mass of 1.00794 u (unified atomic mass units).
```
