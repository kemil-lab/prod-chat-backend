# from google import genai
from app.core.config import settings

# client = genai.Client(api_key=settings.GEMINI_API_KEY)

# def generate_answer(prompt: str) -> str:
#     response = client.models.generate_content(
#         model=settings.GEMINI_MODEL,
#         contents=prompt,
        
#     )
#     return response.text


from huggingface_hub import InferenceClient

client = InferenceClient(api_key=settings.HUGGINGFACE_API_KEY)

def generate_answer(prompt: str) -> str:
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content