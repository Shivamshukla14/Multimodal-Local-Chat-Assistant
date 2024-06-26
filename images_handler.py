from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64

def convert_bytes_to_base64(image_bytes):
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return "data:image/jpeg;base64," + encoded_image


def handle_image(image_bytes, user_msg):
    chat_handler = Llava15ChatHandler(clip_model_path="models/llava/mmproj-model-f16.gguf")

    llm = Llama(
        model_path="models/llava/ggml-model-q5_k.gguf",
        chat_handler=chat_handler,
        n_ctx=1024, # n_ctx should be increased to accomodate the image embedding
        logits_all=True,# needed to make llava work
    )

    image_base64 = convert_bytes_to_base64(image_bytes)

    output = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type" : "text", "text": user_msg}
                ]
            }
        ]
    )


    print(output)
    return output["choices"][0]["message"]["content"]