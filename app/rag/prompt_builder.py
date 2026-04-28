import textwrap


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        [
            f"[Source: {chunk['metadata'].get('source')} | Page: {chunk['metadata'].get('page')}]\n"
            f"{chunk['content']}"
            for chunk in chunks
        ]
    )

    prompt = f"""
You are a helpful medical knowledge assistant.

Constraints:
1. Answer only from the provided context.
2. Do not make up information.
3. If the answer is not in the context, respond exactly with:
   <p>I don't know based on the provided documents.</p>
4. Return the answer as valid HTML only.
5. Do not use Markdown.
6. Use semantic HTML when helpful:
   - Use <p> for paragraphs
   - Use <ul> or <ol> for lists when needed
   - Use <li> for list items
   - Use <strong> for important labels if needed
7. Keep the answer concise, clear, and well-structured.
8. Do not mention anything outside the retrieved documents.
9. Do not include <html>, <head>, or <body> tags.
10. Do not include explanations about formatting.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return prompt

def build_prompt_v2(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        [
            f"[Source: {chunk['metadata'].get('source', 'Unknown')} | Page: {chunk['metadata'].get('page', 'N/A')}]\n{chunk['content'].strip()}"
            for chunk in chunks
        ]
    )

    prompt = textwrap.dedent(f"""
    You are a helpful medical knowledge assistant specialized in Multiple Sclerosis.

    Constraints:
    1. Answer ONLY using the provided context.
    2. Do NOT make up information.
    3. If the answer is not in the context, respond exactly with:
    <p>I don't know based on the provided documents.</p>
    4. Return the answer as valid HTML only.
    5. Do NOT use Markdown.
    6. Use semantic HTML: <p>, <ul>, <li>, <strong>.
    7. Keep answers concise and clear.
    8. Do NOT include unnecessary whitespace or line breaks.
    9. Do NOT include <html>, <head>, or <body> tags.
    10. Do NOT explain formatting.

    Context:
    {context}

    Question: {question}
    Answer:
    """).strip()

    return prompt

def query_analyzer(question: str) -> str:
    prompt = f"""
You are a query analyzer for a pharmaceutical knowledge assistant focused on Multiple Sclerosis (MS).

Your job is to analyze the user's question before retrieval.

You must return ONLY valid JSON.
Do not return Markdown.
Do not return explanations.
Do not answer the user's medical question.
Do not include any text outside the JSON.

Rules:
1. Determine whether the question is relevant to this chatbot's domain.
2. The chatbot domain includes:
   - Multiple Sclerosis
   - MS symptoms
   - MS diagnosis
   - MS treatment
   - MS medicines / drugs / therapies
   - drug safety, monitoring, side effects, contraindications
   - disease-modifying therapies for MS
   - medically relevant pharmaceutical questions connected to MS
3. If the question is NOT relevant to the domain:
   - set is_relevant to false
   - set needs_retrieval to false
   - set needs_query_expansion to false
   - set needs_decomposition to false
   - decomposed_queries must be an empty list
   - provide a short fallback_response:
     "I specialize in Multiple Sclerosis and related pharmaceutical information."
4. If the question IS relevant:
   - set is_relevant to true
   - decide if retrieval is needed
5. Query expansion:
   - set needs_query_expansion to true only if the user question is too short, vague, ambiguous, or poorly formed for retrieval
   - if true, provide one improved expanded_query
   - if false, expanded_query must be an empty string
6. Query decomposition:
   - set needs_decomposition to true only if the question contains multiple sub-questions or is too broad for a single retrieval
   - if true, provide at most 3 decomposed_queries
   - each decomposed query must be clear, retrieval-friendly, and focused
   - if false, decomposed_queries must be an empty list
7. Keep decomposed queries concise and semantically meaningful for RAG retrieval.
8. If a question is relevant but simple, do not decompose unnecessarily.
9. The JSON must follow this exact structure:

{{
  "is_relevant": true,
  "reason": "short reason",
  "needs_retrieval": true,
  "needs_query_expansion": false,
  "expanded_query": "",
  "needs_decomposition": false,
  "decomposed_queries": [],
  "fallback_response": ""
}}

User question:
{question}
""".strip()

    return prompt