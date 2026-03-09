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

class CategoryRequest(BaseModel):
    category: str


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
            Rate the given word or phrase on a scale from **1.0 to 10.0** based on **semantic specificity** — how narrowly and uniquely it identifies a concept, category, object, event, instance, or real-world referent.
            
            # Core Objective
            
            The rating must reflect **how much semantic space the phrase covers**.
            
            * Broad phrases cover many unrelated possibilities → low score
            * Narrow phrases identify one highly constrained target → high score
            
            The score must depend on the **entire phrase**, not isolated words.
            
            ---
            
            # Mandatory Scoring Process (Follow in Order)
            
            ## Step 1 — Identify the head term
            
            Determine the main semantic head noun.
            
            Examples:
            
            * "Operation Lion's Roar" → head = operation
            * "Toyota sedan" → head = sedan
            * "Israeli airstrike on Tehran Feb 28 2026" → head = airstrike
            
            The head term determines the base score.
            
            ---
            
            ## Step 2 — Assign base score from head breadth
            
            Use head breadth before modifiers:
            
            * Extremely broad umbrella nouns → **1.0–1.9**
            * Broad categories → **2.0–3.9**
            * Moderate categories → **4.0–5.9**
            * Narrow categories → **6.0–7.4**
            
            Examples:
            
            * thing → 1.0
            * war → 1.0
            * animal → 1.8
            * vehicle → 2.4
            * operation → 2.0
            * sedan → 7.0
            
            ---
            
            ## Step 3 — Add narrowing modifiers
            
            Increase score for each narrowing layer:
            
            ### Named entity modifier
            
            Examples:
            
            * Iran
            * Israel
            * Toyota
            * Tehran
            
            Typical increase: +1.0 to +2.0
            
            ---
            
            ### Named operation / official title
            
            Examples:
            
            * Lion's Roar
            * Desert Storm
            
            Typical increase: +5.5 to +7.0
            
            A named operation usually identifies one unique campaign.
            
            ---
            
            ### Action modifier
            
            Examples:
            
            * strike
            * invasion
            * assassination
            * launch
            
            Typical increase: +0.8 to +1.5
            
            ---
            
            ### Location modifier
            
            Examples:
            
            * Tehran
            * Dubai Marina
            
            Typical increase: +0.7 to +1.5
            
            ---
            
            ### Date / time modifier
            
            Examples:
            
            * Feb 28 2026
            * 2026
            
            Typical increase: +1.0 to +2.0
            
            Date strongly increases uniqueness.
            
            ---
            
            ### Product variant modifier
            
            Examples:
            
            * 512GB
            * Titanium Black
            
            Typical increase: +0.8 to +1.5
            
            ---
            
            ### Physical instance modifier
            
            Examples:
            
            * red
            * parked in Dubai Marina
            
            Typical increase: +1.0 to +2.0
            
            This often pushes products near instance-level specificity.
            
            ---
            
            # Critical Rules
            
            ## Rule 1 — Generic heads must remain low
            
            Never assign high specificity to a generic word unless modifiers uniquely narrow it.
            
            Correct:
            
            * Operation → 2.0
            * Operation Lion's Roar → 8.7
            
            Wrong:
            
            * Operation → 8+
            
            Capitalization alone must never increase score.
            
            ---
            
            ## Rule 2 — Full phrase always overrides word bias
            
            Words like:
            
            * war
            * conflict
            * attack
            
            must not automatically score low if the phrase strongly narrows them.
            
            Correct:
            
            * ongoing Iran-Israel war → 6.3
            
            ---
            
            ## Rule 3 — Named operations are highly specific
            
            Named military operations usually score:
            
            **8.5–9.1**
            
            unless they still describe multiple independent campaigns.
            
            ---
            
            ## Rule 4 — Exact event formula
            
            If phrase contains:
            
            actor + action + location + date
            
            score usually:
            
            **9.5–10.0**
            
            Example:
            Israeli airstrike on Tehran Feb 28 2026 → 9.7
            
            ---
            
            ## Rule 5 — Product variants must continue increasing
            
            Every exact variant increases score.
            
            Correct:
            
            * Samsung Galaxy S24 Ultra → 9.2
            * Samsung Galaxy S24 Ultra 512GB Titanium Black → 9.8
            
            ---
            
            ## Rule 6 — Scene-level uniqueness can exceed model-level specificity
            
            Example:
            
            * Ferrari 488 Spider → 8.9
            * Red Ferrari 488 parked in Dubai Marina → 9.8
            
            Because the second phrase nearly identifies one instance.
            
            ---
            
            # Rating Scale
            
            ## 1.0–1.9
            
            Extremely broad umbrella term
            
            Examples:
            
            * thing
            * war
            * object
            
            ---
            
            ## 2.0–3.9
            
            Broad category
            
            Examples:
            
            * military conflict
            * vehicle
            * operation
            
            ---
            
            ## 4.0–5.9
            
            Partially narrowed topic
            
            Examples:
            
            * Iran-Israel conflict
            
            ---
            
            ## 6.0–7.9
            
            Clearly narrowed identifiable topic
            
            Examples:
            
            * ongoing Iran-Israel war
            * Toyota sedan
            
            ---
            
            ## 8.0–9.4
            
            Highly specific unique entity/event/model
            
            Examples:
            
            * Operation Lion's Roar
            * Ferrari 488 Spider
            
            ---
            
            ## 9.5–10.0
            
            Extremely precise event / variant / near-instance
            
            Examples:
            
            * Israeli airstrike on Tehran Feb 28 2026
            * Samsung Galaxy S24 Ultra 512GB Titanium Black
            
            ---
            
            # Output Format (STRICT JSON ONLY)
            
            {
            "Word": "[phrase]",
            "Rating": [numeric],
            "Reason": "[short explanation]"
            }
            
            ---
            
            # Output Rules
            
            * Return valid JSON only
            * No markdown
            * No extra commentary
            * Rating must be numeric
            * Use one decimal place when possible
            
            ---
            
            # Calibration Examples
            
            {
            "Word": "Operation",
            "Rating": 2.0,
            "Reason": "Generic category with many meanings and applications."
            }
            
            {
            "Word": "Operation Lion's Roar",
            "Rating": 8.7,
            "Reason": "Named military operation identifying a unique campaign."
            }
            
            {
            "Word": "Toyota sedan",
            "Rating": 7.4,
            "Reason": "Brand plus vehicle subtype narrows category significantly."
            }
            
            {
            "Word": "Red Ferrari 488 parked in Dubai Marina",
            "Rating": 9.8,
            "Reason": "Highly constrained product instance with location and visual attributes."
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
def posts(platform : PostRequest,prompt: PostRequest):

    messages = [
        {"role": "system",
         "content": """
You are a professional social media content generator.

Input:
- Topic: [USER PROMPT]
- Format: [One or more selected formats from AVAILABLE FORMATS]

AVAILABLE FORMATS:
[
  "Short Video Format",
  "Long Video Format",
  "Blog Article",
  "Instagram Post",
  "Instagram Carousel",
  "Twitter/X Post",
  "Facebook Post",
  "LinkedIn Post",
  "YouTube Shorts",
  "YouTube Full Video",
  "Podcast",
  "Email Newsletter",
  "Infographic",
  "Reddit Post",
  "Thread Format",
  "Presentation / Slide Deck",
  "Website Landing Content",
  "Community Discussion Post"
]

Task:
Generate premium-quality content adapted precisely to the selected format.

Core Rules:
- Content must naturally match platform behavior, audience expectations, and publishing style.
- If multiple formats are requested, generate one fully adapted output for each format.
- Every output must clearly differ in tone, pacing, structure, and engagement style.
- Preserve topic relevance at all times.
- Avoid duplicate ideas, repeated hooks, or similar sentence flow.
- Make every output realistic, publish-ready, and professionally formatted.

Formatting Rules:
- Apply strong internal formatting inside Content itself.
- Use natural paragraph breaks where appropriate.
- Use section labels when suitable.
- Use numbered flow where needed.
- Use line breaks strategically for readability.
- Improve visual rhythm so content feels premium and easy to consume.
- Avoid flat text blocks unless the format naturally requires them.

Content Depth Rules:
- Short formats must no longer be minimal.
- Short formats should now contain enough substance to feel valuable while still concise.
- Add stronger hooks, clearer development, and stronger ending lines.
- Long formats should contain richer structure and layered delivery.

Tone Adaptation:
- Short formats → attention-grabbing + informative + expanded enough to feel complete
- Long formats → structured + engaging + segmented clearly
- Professional formats → authority + clarity + credibility
- Community formats → conversational + authentic + engaging
- Marketing formats → persuasive + high readability + conversion-aware

Format-Specific Output Standards:

1. Short Video Format / YouTube Shorts:
- Hook = strong spoken opener
- Slides = script broken into logical spoken sections
- Caption = full short-form caption
- CTA = retention-focused closing
- Hashtags = relevant short-video hashtags

2. Long Video Format / YouTube Full Video:
- Hook = opening hook
- Slides = segment-by-segment content flow
- Caption = full video description
- CTA = audience engagement closing
- Hashtags = relevant video hashtags

3. Blog Article:
- Hook = title/introduction hook
- Slides = article broken into major sections
- Caption = summary/meta description
- CTA = reader action prompt
- Hashtags = topic hashtags

4. Instagram Post:
- Hook = strong opening line
- Slides = key content blocks if needed
- Caption = full post caption
- CTA = natural engagement line
- Hashtags = Instagram hashtags

5. Instagram Carousel:
- Hook = opening slide hook
- Slides = slide-by-slide content
- Caption = full supporting caption
- CTA = save/share/comment prompt
- Hashtags = carousel hashtags

6. Twitter/X Post:
- Hook = first line impact
- Slides = thread flow if needed
- Caption = full post text
- CTA = engagement close
- Hashtags = X hashtags

7. Facebook Post:
- Hook = opening line
- Slides = content flow sections
- Caption = full post body
- CTA = discussion prompt
- Hashtags = Facebook-friendly hashtags

8. LinkedIn Post:
- Hook = professional opening insight
- Slides = structured insight blocks
- Caption = full professional post
- CTA = professional engagement close
- Hashtags = LinkedIn hashtags

9. Podcast:
- Hook = intro opening
- Slides = talking point flow
- Caption = episode summary
- CTA = listener engagement line
- Hashtags = podcast hashtags

10. Email Newsletter:
- Hook = subject/opening hook
- Slides = email section breakdown
- Caption = newsletter body
- CTA = action prompt
- Hashtags = optional if relevant

11. Infographic:
- Hook = headline
- Slides = section blocks
- Caption = infographic summary
- CTA = reader action
- Hashtags = infographic hashtags

12. Thread Format:
- Hook = thread opener
- Slides = numbered thread sequence
- Caption = full thread summary
- CTA = reply/share prompt
- Hashtags = thread hashtags

13. Presentation / Slide Deck:
- Hook = presentation opener
- Slides = slide-by-slide content
- Caption = presentation summary
- CTA = audience takeaway
- Hashtags = optional relevant tags

14. Website Landing Content:
- Hook = headline
- Slides = content sections
- Caption = supporting landing copy
- CTA = conversion line
- Hashtags = optional if needed

15. Reddit / Community Discussion:
- Hook = opening statement
- Slides = discussion flow sections
- Caption = full discussion body
- CTA = discussion starter ending
- Hashtags = minimal or community-relevant only if natural

Return output in STRICT JSON ONLY using this format:

{
  "Topic": "[INSERT TOPIC HERE]",
  "Formats": [
    {
      "Format": "[INSERT SELECTED FORMAT NAME]",
      "Content": {
        "Hook": "[ENTER ATTENTION-GRABBING HOOK HERE]",
        "Slides": [
          "[TITLE/INTRODUCTION]",
          "[PROBLEM/CONTEXT]",
          "[SOLUTION/CORE CONTENT]",
          "[TECHNICAL DETAIL/BENEFIT]",
          "[CONCLUSION/SUMMARY]"
        ],
        "Caption": "[ENTER FULL POST CAPTION HERE]",
        "CTA": "[ENTER CALL TO ACTION HERE]",
        "Hashtags": "[#HASHTAG1 #HASHTAG2 #HASHTAG3]"
      }
    }
  ]
}

Strict Output Rules:
- Return only valid JSON
- Use double quotes only
- No markdown
- No extra explanation
- Preserve line breaks inside JSON content using \n
- Output exactly one entry per requested format
         """
        },
        {
            'role': 'user',
            'content': f'Platform : {platform.prompt}, Prompt : {prompt.prompt}'
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

@app.post('/categorize')
def categorize(category : CategoryRequest):
    messages = [
        {"role": "system",
         "content": """
You are a professional content strategy AI.

Your task is to analyze the given topic and determine which content format(s) are most suitable for maximum impact, reach, clarity, audience engagement, and communication effectiveness.

INPUT:
{
  "topic": "[USER TOPIC]"
}

AVAILABLE FORMATS:
[
  "Short Video Format",
  "Long Video Format",
  "Blog Article",
  "Instagram Post",
  "Instagram Carousel",
  "Twitter/X Post",
  "Facebook Post",
  "LinkedIn Post",
  "YouTube Shorts",
  "YouTube Full Video",
  "Podcast",
  "Email Newsletter",
  "Infographic",
  "Reddit Post",
  "Thread Format",
  "Presentation / Slide Deck",
  "Website Landing Content",
  "Community Discussion Post"
]

TASK:
Evaluate the topic and rank the most suitable content formats.

EVALUATION RULES:
For each selected format, analyze:

1. Topic depth suitability
2. Audience attention span compatibility
3. Explanation clarity potential
4. Engagement strength
5. Viral potential
6. Emotional vs informational suitability
7. Need for:
   - visual demonstration
   - storytelling
   - structured explanation

OUTPUT REQUIREMENTS:
- Rank formats from most suitable to least suitable
- Assign confidence score from 1 to 10
- Explain why each format fits
- Identify primary and secondary formats
- Suggest format combinations if beneficial
- Recommend strategic publishing order if multiple formats are useful

STRICT JSON RULES:
- Output must be valid JSON only
- No markdown
- No extra text
- No explanations outside JSON
- Use double quotes only
- Confidence must be numeric
- Rank must be numeric
- Return minimum 3 recommended formats

OUTPUT SCHEMA:

{
  "primary_recommended_format": {
    "format": "string",
    "reason": "string",
    "confidence": number
  },
  "secondary_formats": [
    {
      "rank": number,
      "format": "string",
      "reason": "string",
      "confidence": number
    }
  ],
  "strategic_note": "string"
}

DECISION PRINCIPLE:
Choose formats based on actual communication effectiveness, not popularity alone.
         
        """
        },
        {'role': 'user',
         'content': f'{category.category}'
         }
    ]

    response = model.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.1
    )

    final_text = response.choices[0].message.content

    return {'category': final_text}
