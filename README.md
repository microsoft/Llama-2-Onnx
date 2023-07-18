**This is an optimized version of the Llama 2 model, available from Meta under the Llama Community License Agreement found on this repository. Microsoft permits you to use, modify, redistribute and create derivatives of Microsoft's contributions to the optimized version subject to the restrictions and disclaimers of warranty and liability in the Llama Community License agreement.**

Llama 2 is a collection of pretrained and fine-tuned generative text models. To learn more about Llama 2, review the [Llama 2 model card](https://github.com/microsoft/Llama-2-Onnx/blob/main/MODEL-CARD-META-LLAMA-2.md).


In order to help developers innovate responsibly, Meta encourages you to review the [Responsible Use Guide](https://ai.meta.com/llama/responsible-use-guide/) for the Llama 2 models.

Microsoft encourages you to learn more about its [Responsible AI approach](https://aka.ms/rai), including many publicly available resources and tools for developers.

 

    

**Fine-tuned Chat Models**

The fine-tuned models were trained for dialogue applications.

To get the expected features and performance for them, a specific formatting needs to be followed, including the `INST` tag, `BOS` and `EOS` tokens, and the whitespaces and breaklines in between (we recommend calling `strip()` on inputs to avoid double-spaces).

 This enables models in chat mode as well as additional safeguards  to reduce potentially undesirable output.

 

**FAQ**

 

**Why is the first inference session slow?** 

ONNX runtime execution provider might need to generate JIT binaries for the underlying hardware, typically the binary is cache and will be loaded directly in the subsequent runs to reduce the overhead. 

 

**Why is FP16 ONNX slower than ONNX FP32 on my device?** 

It is possible that your device does not support native FP16 math, therefore weights will be cast to FP32 at runtime. 

 

**How do I get better inference speed?** 

It is recommended that inputs/outputs are put on target device to avoid expensive data copies, please refer to the following document for details.  

[I/O Binding | onnxruntime](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) 

 

**What parameters to test with?** 

Users can perform temperature and top-p sampling using the model’s output logits. Please refer to Meta’s guidance for the best parameters combination; an example is located [here.](https://github.com/facebookresearch/llama/)

 

**Is there an example on how to run this?**

To run an example with this repository, please see [Llama-2-Onnx/Example.md](https://github.com/microsoft/Llama-2-Onnx/blob/main/Example.md)
