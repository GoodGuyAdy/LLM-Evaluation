import os
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    def __init__(self, model_name="gemini-2.5-flash"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate(self, system_prompt: str, user_prompt: str):
        prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.2},
        )
        return response.text.strip()


class OpenAIClient:
    def __init__(self, model_name="gpt-5.4-nano"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str):
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()
