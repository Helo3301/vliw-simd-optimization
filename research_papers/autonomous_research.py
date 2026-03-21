#!/usr/bin/env python3
"""
Autonomous Research Loop for VLIW SIMD Optimization
Searches arxiv, downloads papers, extracts insights, generates hypotheses
"""

import arxiv
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

PAPERS_DIR = Path(__file__).parent
EXPERIMENTS_DIR = PAPERS_DIR.parent / "experiments"
LOG_FILE = PAPERS_DIR / "research_log.md"

# Search queries targeting our optimization problem
QUERIES = [
    "VLIW instruction scheduling optimization",
    "SIMD gather scatter memory access",
    "software pipelining loop optimization",
    "tree traversal vectorization SIMD",
    "GPU memory coalescing optimization",
    "instruction level parallelism scheduling",
    "register allocation VLIW",
    "modulo scheduling VLIW",
    "loop unrolling SIMD performance",
    "hash function SIMD vectorization",
    "binary tree traversal parallel",
    "memory access pattern optimization",
]

def log(msg):
    """Log to file and stdout"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def search_papers(query, max_results=5):
    """Search arxiv for papers"""
    log(f"Searching: {query}")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "arxiv_id": result.entry_id.split("/")[-1],
            "published": str(result.published)
        })
    return papers

def download_paper(paper):
    """Download PDF and extract text"""
    arxiv_id = paper["arxiv_id"]
    pdf_path = PAPERS_DIR / f"{arxiv_id}.pdf"
    txt_path = PAPERS_DIR / f"{arxiv_id}.txt"

    if txt_path.exists():
        log(f"Already have: {arxiv_id}")
        return txt_path.read_text()

    # Download PDF
    log(f"Downloading: {arxiv_id} - {paper['title'][:50]}...")
    try:
        subprocess.run(
            ["curl", "-sL", "-o", str(pdf_path), paper["pdf_url"]],
            check=True, timeout=60
        )
    except Exception as e:
        log(f"Failed to download {arxiv_id}: {e}")
        return None

    # Extract text
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), str(txt_path)],
            check=True, capture_output=True, timeout=30
        )
        text = txt_path.read_text()
        log(f"Extracted {len(text)} chars from {arxiv_id}")
        return text
    except Exception as e:
        log(f"Failed to extract {arxiv_id}: {e}")
        return None

def extract_techniques(text, paper):
    """Extract optimization techniques mentioned in paper"""
    techniques = []

    # Keywords to look for
    keywords = {
        "loop unrolling": "unroll",
        "software pipelining": "pipeline",
        "modulo scheduling": "modulo",
        "register blocking": "block",
        "prefetching": "prefetch",
        "vectorization": "vector",
        "SIMD": "simd",
        "gather": "gather",
        "scatter": "scatter",
        "coalescing": "coalesce",
        "tiling": "tile",
        "fusion": "fuse",
        "fission": "fission",
        "interchange": "interchange",
        "parallelism": "parallel",
        "ILP": "ilp",
        "dependency": "depend",
        "latency hiding": "latency",
        "double buffering": "buffer",
        "speculative": "specul",
    }

    text_lower = text.lower()
    for technique, keyword in keywords.items():
        if keyword in text_lower:
            # Find context
            idx = text_lower.find(keyword)
            context = text[max(0, idx-200):idx+200]
            techniques.append({
                "technique": technique,
                "keyword": keyword,
                "context": context.replace("\n", " ")[:400]
            })

    return techniques

def main():
    log("=" * 60)
    log("AUTONOMOUS RESEARCH LOOP STARTED")
    log("=" * 60)

    all_papers = []
    all_techniques = []

    # Phase 1: Search and download papers
    for query in QUERIES:
        try:
            papers = search_papers(query, max_results=3)
            for paper in papers:
                # Skip duplicates
                if any(p["arxiv_id"] == paper["arxiv_id"] for p in all_papers):
                    continue

                text = download_paper(paper)
                if text:
                    techniques = extract_techniques(text, paper)
                    paper["techniques"] = techniques
                    all_papers.append(paper)
                    all_techniques.extend(techniques)

                    log(f"  Found {len(techniques)} technique mentions")
        except Exception as e:
            log(f"Error with query '{query}': {e}")

    # Save results
    results = {
        "papers": all_papers,
        "technique_summary": {}
    }

    # Count techniques
    for t in all_techniques:
        name = t["technique"]
        if name not in results["technique_summary"]:
            results["technique_summary"][name] = 0
        results["technique_summary"][name] += 1

    with open(PAPERS_DIR / "research_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nTotal papers downloaded: {len(all_papers)}")
    log(f"Total technique mentions: {len(all_techniques)}")
    log("\nTechnique frequency:")
    for t, count in sorted(results["technique_summary"].items(), key=lambda x: -x[1]):
        log(f"  {t}: {count}")

    return results

if __name__ == "__main__":
    main()
