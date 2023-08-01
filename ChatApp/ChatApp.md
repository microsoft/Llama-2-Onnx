# **Chat App Example**

This is a more complete example of how to use the Llama 2 models with ONNX. This is a python program based on the popular Gradio package for making web interfaces for machine learning demonstrations.

Some experience in setting up Python environments is useful, for example we would recommend running this example with a conda environment.

All of the required packages are listed in the [requirements.txt](requirements.txt) file. You can install them with the following command:

```bash
pip install -r requirements.txt
```

You should set the python path to the root directory of the repository so that the modules are found properly.

You should then be able to run the example like this from the root of the repository:

```bash
python ChatApp/app.py
```

This will start a web server on a particular port, which you can access in your browser. The address will be printed in the terminal, but it will be something like this:

```
Running on local URL:  http://127.0.0.1:7860
```

When this address is opened in a browser, you should see a page like this:

![ChatApp](../Images/ChatAppExample.png)

## **Using The Chat App**
You first must pick the model you want to use, there is a drop down list showing the options. You should ensure that the model you choose is downloaded using the git submodule commands in the [README.md](../README.md) file, otherwise attempting to load it will return an error.

You can then select some parameters using the sliders. You should refer to Meta's guidance [here.](https://github.com/facebookresearch/llama/). The defaults are appropriate to get started.

Now you can type some text into the chat window, and Llama 2 will respond. You can start a new conversation, regenerate the last response, or remove the last user-bot turn from the conversation history by pressing the appropriate buttons.