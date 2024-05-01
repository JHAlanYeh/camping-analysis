from llama_cpp import Llama

model_id = "../taide_Llama3_TAIDE_LX_8B_Chat_Alpha1_4bit/taide-8b-a.3-q4_k_m.gguf"

llm = Llama(
  model_path=model_id,  # path to GGUF file
  n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8, # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=35, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
)

prompt = "How to explain Internet to a medieval knight?"

# Simple inference example
output = llm(
  f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
  max_tokens=256,  # Generate up to 256 tokens
  stop=["<|end|>"], 
  echo=True,  # Whether to echo the prompt
)

print(output['choices'][0]['text'])
