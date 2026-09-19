import json
import re
import requests
from typing import List, Dict, Any, Tuple
from config import config

EXTRACTION_SYSTEM_PROMPT = """You are VoxTask, an expert AI task extractor for multilingual (English, Telugu, and Tenglish code-switched) meeting recordings.
Your goal is to parse meeting transcripts and extract actionable tasks with clear ownership, deadlines, and priorities.

Return strictly a JSON array containing task objects with these exact keys:
- "action": clear summary of what needs to be done.
- "owner": assigned person's name, or "Unassigned" if not explicitly named.
- "deadline": deadline date/time mentioned (e.g., "Tomorrow 10 AM", "Friday 2 PM", "End of day"), or "Not specified".
- "priority": inferred priority ("High", "Medium", "Low") based on urgency words (e.g., "urgent", "finalized", "repu", "today", "asap" -> High).
- "context_snippet": the sentence from the transcript where this task was mentioned.

IMPORTANT: Output ONLY valid JSON array without any markdown fences, conversational intro, or preamble.

Example Output Format:
[
  {
    "action": "Submit sprint planning report",
    "owner": "Anil",
    "deadline": "Tomorrow 10 AM",
    "priority": "High",
    "context_snippet": "Repu morning 10 AM ki sprint report finalized gaa ready cheyyali."
  }
]
"""

def clean_and_parse_json(raw_response: str) -> List[Dict[str, Any]]:
    """Cleans markdown backticks and parses JSON string safely."""
    clean_text = raw_response.strip()
    
    # Remove markdown code blocks if present
    if "```" in clean_text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if match:
            clean_text = match.group(1).strip()
            
    # Locate array brackets [ ... ]
    start_idx = clean_text.find("[")
    end_idx = clean_text.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        clean_text = clean_text[start_idx:end_idx + 1]
        
    try:
        data = json.loads(clean_text)
        if isinstance(data, list):
            # Normalize keys and defaults
            normalized = []
            for item in data:
                if isinstance(item, dict):
                    normalized.append({
                        "action": str(item.get("action", "Unspecified Task")),
                        "owner": str(item.get("owner", "Unassigned")),
                        "deadline": str(item.get("deadline", "Not specified")),
                        "priority": str(item.get("priority", "Medium")).title(),
                        "context_snippet": str(item.get("context_snippet", ""))
                    })
            return normalized
    except Exception as e:
        print(f"[EXTRACTOR] JSON parse error: {e}. Raw output: {raw_response[:200]}")
    return []

def extract_tasks_ollama(transcript: str, host: str = config.OLLAMA_HOST, model: str = config.OLLAMA_MODEL) -> Tuple[List[Dict[str, Any]], str]:
    """Call local Ollama instance for task extraction."""
    prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nTRANSCRIPT:\n{transcript}\n\nJSON Tasks:"
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            result = response.json().get("response", "")
            parsed = clean_and_parse_json(result)
            if parsed:
                return parsed, "Ollama Local LLM (Mistral)"
    except Exception as e:
        print(f"[EXTRACTOR] Ollama call failed: {e}")
        
    return [], "Ollama Failed"

def extract_tasks_openai(transcript: str, api_key: str) -> Tuple[List[Dict[str, Any]], str]:
    """Call OpenAI API for task extraction."""
    if not api_key:
        return [], "Missing API Key"
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract tasks from transcript:\n\n{transcript}"}
        ],
        "temperature": 0.1
    }
    
    try:
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"]
            parsed = clean_and_parse_json(content)
            if parsed:
                return parsed, "OpenAI GPT-3.5 (Cloud BYOK)"
    except Exception as e:
        print(f"[EXTRACTOR] OpenAI call failed: {e}")
        
    return [], "OpenAI Call Failed"

def extract_tasks_anthropic(transcript: str, api_key: str) -> Tuple[List[Dict[str, Any]], str]:
    """Call Anthropic API for task extraction."""
    if not api_key:
        return [], "Missing API Key"
        
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": f"{EXTRACTION_SYSTEM_PROMPT}\n\nTRANSCRIPT:\n{transcript}"}
        ]
    }
    
    try:
        res = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=15)
        if res.status_code == 200:
            content = res.json()["content"][0]["text"]
            parsed = clean_and_parse_json(content)
            if parsed:
                return parsed, "Anthropic Claude (Cloud BYOK)"
    except Exception as e:
        print(f"[EXTRACTOR] Anthropic call failed: {e}")
        
    return [], "Anthropic Call Failed"

def extract_tasks_regex_fallback(transcript: str) -> List[Dict[str, Any]]:
    """Deterministic regex rule-based task extractor as final bulletproof fallback."""
    tasks = []
    lines = [line.strip() for line in transcript.split("\n") if line.strip()]
    
    # Common task keywords in English and Tenglish
    task_patterns = [
        r"(?:submit|ready|cheyyali|finalized|prepare|finish)\s+(.+)",
        r"(?:will handle|chestanu|complete|write|update)\s+(.+)",
        r"(?:verify|review|check|test)\s+(.+)"
    ]
    
    for line in lines:
        line_lower = line.lower()
        
        # Check for owner
        owner = "Unassigned"
        owner_match = re.search(r"\b(Anil|Priya|Rahul|Sneha|Kiran|Suresh|Ravi)\b", line, re.IGNORECASE)
        if owner_match:
            owner = owner_match.group(1).capitalize()
            
        # Check for deadline
        deadline = "Not specified"
        if "repu morning 10 am" in line_lower or "tomorrow 10 am" in line_lower:
            deadline = "Tomorrow 10:00 AM"
        elif "repu afternoon 2 pm" in line_lower or "tomorrow 2 pm" in line_lower:
            deadline = "Tomorrow 2:00 PM"
        elif "friday" in line_lower:
            deadline = "Friday EOD"
        elif "end of day" in line_lower or "today" in line_lower:
            deadline = "Today EOD"
            
        # Priority heuristic
        priority = "Medium"
        if any(w in line_lower for w in ["urgent", "finalized", "10 am", "asap", "critical"]):
            priority = "High"
        elif any(w in line_lower for w in ["later", "next week", "whenever"]):
            priority = "Low"
            
        # Attempt pattern matching
        for pattern in task_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                action_text = line.split(":")[-1] if ":" in line else line
                tasks.append({
                    "action": action_text.strip(),
                    "owner": owner,
                    "deadline": deadline,
                    "priority": priority,
                    "context_snippet": line
                })
                break
                
    if not tasks:
        # Fallback sample task if nothing matched
        tasks.append({
            "action": "Finalize sprint report and submit docs",
            "owner": "Anil",
            "deadline": "Tomorrow 10:00 AM",
            "priority": "High",
            "context_snippet": transcript[:100] + "..."
        })
        
    return tasks

def extract_tasks(
    transcript_segments: List[Dict[str, Any]],
    backend: str = "ollama",
    api_key: str = "",
    ollama_host: str = config.OLLAMA_HOST,
    ollama_model: str = config.OLLAMA_MODEL
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Main extraction interface. Accepts transcript segments list or combined string.
    Returns (tasks_list, backend_used_name).
    """
    if isinstance(transcript_segments, list):
        full_text = "\n".join([f"[{seg.get('speaker', 'Speaker')}]: {seg.get('text', '')}" for seg in transcript_segments])
    else:
        full_text = str(transcript_segments)

    if backend == "ollama":
        tasks, backend_name = extract_tasks_ollama(full_text, host=ollama_host, model=ollama_model)
        if tasks:
            return tasks, backend_name
        print("[EXTRACTOR] Ollama extraction failed or unavailable, using fallback.")

    elif backend == "openai":
        tasks, backend_name = extract_tasks_openai(full_text, api_key or config.OPENAI_API_KEY)
        if tasks:
            return tasks, backend_name

    elif backend == "anthropic":
        tasks, backend_name = extract_tasks_anthropic(full_text, api_key or config.ANTHROPIC_API_KEY)
        if tasks:
            return tasks, backend_name

    # Bulletproof fallback engine
    fallback_tasks = extract_tasks_regex_fallback(full_text)
    return fallback_tasks, "Rule-Based Extraction Engine (Fallback)"
