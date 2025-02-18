import openai
import os
import dotenv

dotenv.load_dotenv()

class llm:
    def __init__(self):
        self.llm = llm
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def describe(self, df):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Write a very succinct and concise description of the following csv file. This description will be used to find relevant csv files using semantic search so make sure it has keywords. Describe the data in the following CSV file: " + df.head(3).to_csv()}],
            max_tokens=1000
        )
        return response.choices[0].message.content
