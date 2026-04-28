
from app.rag.prompt_builder import  build_prompt_v2, query_analyzer
# from app.services.hf_classifier import classify_query_hf
from app.services.llm_service import generate_answer
import re
# from app.services.rerank_service import rerank
from app.services.retrieval_service_llama import engine, reRanker
import json



THRESHOLD = 0.5
FINAL_TOP_K = 5

def run_rag_pipeline_llamaIndex(question: str) -> dict:
    query_analysis_prompt = query_analyzer(question)
    print("Query Analysis Prompt:", query_analysis_prompt)
    analysis_raw = generate_answer(query_analysis_prompt)

    analysis = parse_json_output(analysis_raw)
    print("Parsed Query Analysis:", analysis)
    if not analysis.get("is_relevant", False):
        return {
            "answer": analysis.get(
                "fallback_response",
                "I specialize in Multiple Sclerosis and related pharmaceutical information.",
            ),
            "sources": [],
            "analysis": analysis,
        }


    if analysis.get("needs_decomposition", False):
        retrieval_queries = analysis.get("decomposed_queries", [])[:3]

    elif analysis.get("needs_query_expansion", False):
        expanded_query = analysis.get("expanded_query", "").strip()
        retrieval_queries = [expanded_query] if expanded_query else [question]

    else:
        retrieval_queries = [question]
   
    all_source_nodes = []
    seen_texts = set()

    for q in retrieval_queries:
        response = engine.query(q)

        if not response.source_nodes:
            continue

        for node in response.source_nodes:
            node_text = getattr(node, "text", "").strip()
            print(f"Retrieved Node Text: '{node_text[:12]}' with score {getattr(node, 'score', 0)}")
            if node_text and node_text not in seen_texts:
                seen_texts.add(node_text)
                all_source_nodes.append(node)

    if not all_source_nodes:
        return {
            "answer": "I couldn't find relevant MS-related information in my database.",
            "sources": [],
            "analysis": analysis,
            
        }


    reranker = reRanker()
    
    if analysis.get("needs_decomposition", False):
        reranker.top_n = FINAL_TOP_K

    reranked_nodes = reranker.postprocess_nodes(
        all_source_nodes,
        query_str=question,
        
    )

    sources = []
    for node in reranked_nodes:
        if getattr(node, "score", 0) >= THRESHOLD:
            sources.append({
                "id": getattr(node, "id_", None),
                "content": node.get_content(),
                "metadata": node.metadata,
                "score": getattr(node, "score", 0),
            })


    if not sources:
        return {
            "answer": "I couldn't find relevant MS-related information in my database.",
            "sources": [],
            "analysis": analysis,
            
        }
  
    prompt = build_prompt_v2(question, sources)
    print(prompt)

    answer = generate_answer(prompt)
    cleaned = clean_text(answer)
    return {
        "answer": cleaned,
        "sources": sources,
        "analysis":analysis,
    }
def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\n+', '\n', text)   # collapse multiple newlines
    text = re.sub(r'\s+', ' ', text)    # collapse spaces
    return text
def parse_json_output(text: str) -> dict:
    if not text:
        raise ValueError("Empty LLM response")

    cleaned = text.strip()

    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Invalid JSON response: {text}")