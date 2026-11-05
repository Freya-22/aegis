from openai import OpenAI

# The ONLY change from a normal OpenAI app: base_url points at Aegis.
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed-locally")

reply = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Reply with exactly: Aegis online"}],
)
print(reply.choices[0].message.content)
