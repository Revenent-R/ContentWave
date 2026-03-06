from huggingface_hub import InferenceClient
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you'd put your frontend's URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("API_TOKEN", None)
MODEL = os.environ.get("MODEL", None)

model = InferenceClient(token=API_KEY)

class RateRequest(BaseModel):
    word: str

class PostRequest(BaseModel):
    prompt: str


@app.get("/")
def health_check():
    return {"status": "healthy", "message": "API is running"},200

@app.head("/")
def health_check():
    return {"status": "healthy", "message": "API is running"},200


@app.post("/rate")
def rate(word: RateRequest):

    messages = [
        {"role": "system",
         "content":
         """
         Rate the given word or phrase on a scale from **1.0 to 10.0** based on **semantic specificity**, meaning how uniquely and narrowly it identifies a concept, event, object, or category.
        
        ## Core Principle:
        
        The score must increase whenever the phrase adds narrowing information such as:
        
        * Named entities
        * Dates
        * Locations
        * Unique event names
        * Specific actions
        * Identifiable subtypes
        
        ## Rating Rules:
        
        * **1.0–1.9 = Extremely broad umbrella term**
          Covers massive conceptual space.
          Examples: war, conflict, object, animal
        
        * **2.0–3.9 = Broad category**
          Large category with many subtypes.
          Examples: civil war, military conflict, vehicle
        
        * **4.0–5.9 = Partially narrowed concept**
          Includes named domain or actors but still broad.
          Examples: Iran-Israel conflict
        
        * **6.0–7.9 = Clearly narrowed identifiable topic**
          Refers to a particular ongoing situation or limited category.
          Examples: ongoing Iran-Israel war
        
        * **8.0–9.4 = Highly specific identifiable entity/event**
          Named operation, subtype, or clearly unique target.
          Examples: Operation Lion’s Roar
        
        * **9.5–10.0 = Extremely precise unique event**
          Exact action + date + location + actor combination.
          Examples: Israeli airstrike on Tehran Feb 28 2026
        
        ## Critical Scoring Rules:
        
        1. **Named operations automatically score at least 8.5 unless they still cover multiple phases.**
        
        2. **Date + location + action together usually push score above 9.3**
        
        3. **Words like war/conflict should NOT automatically score low if modified by strong narrowing context**
        
        4. **Always judge full phrase, not individual words**
        
        5. **A unique real-world event must score higher than a category**
        
        6. Never assign high specificity to a generic head noun unless modifiers uniquely identify it.

        Examples:
        
        * "Operation" = broad generic term → low score
        * "Operation Lion's Roar" = named unique event → high score
        
        A generic word alone must stay low even if capitalized.

        
        ## Output Format (STRICT JSON ONLY):
        
        {
        "Word": "[phrase]",
        "Rating": [1.0-10.0],
        "Reason": "[short explanation]"
        }
        
        ## Examples:
        
        {
        "Word": "war",
        "Rating": 1.0,
        "Reason": "Extremely broad umbrella term covering all wars."
        }
        
        {
        "Word": "Iran-Israel conflict",
        "Rating": 4.8,
        "Reason": "Named geopolitical conflict but still includes many events."
        }
        
        {
        "Word": "Operation Lion’s Roar",
        "Rating": 8.7,
        "Reason": "Named military operation identifying a unique campaign."
        }
        
        {
        "Word": "Israeli airstrike on Tehran Feb 28 2026",
        "Rating": 9.7,
        "Reason": "Exact military event with actor, action, location, and date."
        }        
         """
         },
        {"role": "user", "content": f"{word.word}"}
    ]

    response = model.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.1
    )

    print(response)

    final_text = response.choices[0].message.content

    return {'rating': final_text}


@app.post("/posts")
def posts(prompt: PostRequest):

    messages = [
        {"role": "system",
         "content": """
You are a social media content generator.

Input:
- Topic: [USER PROMPT]
- Number of outputs: X

Task:
Generate X high-quality social media posts.

Rules:
- Every output must differ clearly from others.
- Mix tone, rhythm, and sentence patterns.
- Preserve topic relevance.
- Some posts should be short and sharp.
- Some should be detailed and persuasive.
- Some should feel premium and brand-ready.
- No duplicate ideas.
- Platform-neutral unless specified.

Return output in STRICT JSON ONLY using this format:

{
  "Topic": "[USER PROMPT]",
  "Outputs": [
    {
      "PostNumber": 1,
      "Content": "Generated post text"
    },
    {
      "PostNumber": 2,
      "Content": "Generated post text"
    }
  ]
}

Rules:
- Return only valid JSON
- Use double quotes
- No markdown
- No extra explanation
- Output exactly X posts inside Outputs array
         """
        },
        {
            'role': 'user',
            'content': f'{prompt.prompt}'
        }
    ]

    response = model.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.1
    )

    print(response)

    final_text = response.choices[0].message.content

    return {'posts': final_text}


