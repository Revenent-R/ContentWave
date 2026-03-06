#
# from google import genai
# from PIL import Image
# from io import BytesIO
#
# client = genai.Client(api_key="AIzaSyBtyrbKyUsa_X2BKcvSEj2z0gAGicHkaXg")
#
# response = client.models.generate_content(
#     model="gemini-2.0-flash-exp",
#     contents="Generate an image of a dark fantasy phoenix rising from ashes"
# )
#
# print(response)


from google import genai
from PIL import Image
from io import BytesIO

client = genai.Client(api_key="AIzaSyBtyrbKyUsa_X2BKcvSEj2z0gAGicHkaXg")

response = client.models.generate_content(
    model="nano-banana-pro-preview",
    contents="ultra realistic cyberpunk crow warrior"
)

for part in response.candidates[0].content.parts:
    if hasattr(part, "inline_data") and part.inline_data:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("banana.png")