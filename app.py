from huggingface_hub import InferenceClient
from fastapi import FastAPI, Request, Response
import os


app = FastAPI()


API_KEY = os.environ.get("API_KEY", None)
MODEL = os.environ.get("MODEL", None)

model = InferenceClient(token=API_KEY)


@app.get("/")
def health_check():
    return {"status": "healthy", "message": "API is running"}


@app.post("/rate")
def rate(word):

    messages = [
        {"role": "system",
         "content":
             """
             Rate the given word on a scale from **1.0 to 10.0** based on how broad or specific its meaning is.

### Rating Rules:

* **1.0 = Extremely broad umbrella term** (covers vast categories, highly general, highly vague)
* **2.0–3.9 = Broad term** (large conceptual coverage, many subcategories)
* **4.0–5.9 = Moderately broad** (general but partially narrowed)
* **6.0–7.9 = Moderately specific** (clear category, limited scope)
* **8.0–9.4 = Highly specific** (narrow meaning, precise category)
* **9.5–10.0 = Extremely specific** (very exact object, instance, or narrowly defined concept)

### Important Rule:

If the word is an **umbrella term**, reduce the score according to how many meanings, categories, or subtypes it includes.

### Decimal Precision:

Use decimals when needed to reflect subtle differences in specificity.

### Consider:

* Number of subcategories included
* Breadth of interpretation
* Vagueness vs precision
* Context flexibility
* Conceptual scope

### Output Format:

Word: [word]
Rating: [1.0–10.0]
Reason: [short explanation]

### Examples:

Word: Entity
Rating: 1.2
Reason: Extremely broad; can refer to almost anything conceptual or physical.

Word: Animal
Rating: 1.8
Reason: Covers all animal species.

Word: Vehicle
Rating: 2.4
Reason: Includes many transport categories.

Word: Car
Rating: 6.3
Reason: Specific transport type but contains many variants.

Word: Sedan
Rating: 8.4
Reason: Narrow subtype of car.

Word: Red Ferrari 488
Rating: 9.8
Reason: Very precise and highly specific.
             """
         },
        {"role": "user", "content": f"{word}"}
    ]

    response = model.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.1
    )

    print(response)

    return {'rating': response}


@app.post("/posts")
def posts(prompt):

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

Return only final posts.
         """
        },
        {
            'role': 'user',
            'content': f'{prompt}'
        }
    ]

    response = model.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.1
    )

    print(response)

    return {'posts': response}


