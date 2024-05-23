from llama_cpp import Llama

model = Llama(
    "/Your/Path/To/GGUF/File",
    verbose=False,
    n_gpu_layers=-1,
)

system_prompt = "You are a helpful assistant."

def generate_reponse(_model, _messages, _max_tokens=8192):
    _output = _model.create_chat_completion(
        _messages,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=_max_tokens,
    )["choices"][0]["message"]["content"]
    return _output

# The following are some examples

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {"role": "user", "content": "写一首诗吧"},
]

print(generate_reponse(model, messages))