import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

from ollama_deep_researcher.graph import graph

SOURCE_ID_PATTERN = r"\[(S[0-9A-F]{8})\]"

def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def calculate_chinese_ratio(text: str) -> float:
    chinese_characters = re.findall(
        r"[\u4e00-\u9fff]",
        text,
    )

    language_characters = re.findall(
        r"[A-Za-z\u4e00-\u9fff]",
        text,
    )

    if not language_characters:
        return 0.0

    return len(chinese_characters) / len(language_characters)


def calculate_language_metrics(
    question: str,
    report: str,
) -> dict:
    if "### Sources:" in report:
        body = report.split("### Sources:", 1)[0]
    else:
        body = report

    chinese_ratio = calculate_chinese_ratio(body)

    if contains_chinese(question):
        language_consistency = chinese_ratio >= 0.15
        expected_language = "zh"
    else:
        language_consistency = True
        expected_language = "unknown"

    return {
        "expected_language": expected_language,
        "chinese_character_ratio": round(
            chinese_ratio,
            3,
        ),
        "language_consistency": language_consistency,
    }

def calculate_concept_coverage(
    report: str,
    expected_concepts: list,
) -> dict:
    if not expected_concepts:
        return {
            "concept_coverage": None,
            "covered_concepts": [],
            "missing_concepts": [],
        }

    if "### Sources:" in report:
        body = report.split("### Sources:", 1)[0]
    else:
        body = report

    normalized_body = body.lower()
    covered_concepts = []
    missing_concepts = []

    for concept in expected_concepts:
        if isinstance(concept, str):
            name = concept
            aliases = [concept]
        else:
            name = concept["name"]
            aliases = concept.get("aliases", [name])

        found = any(
            alias.lower() in normalized_body
            for alias in aliases
        )

        if found:
            covered_concepts.append(name)
        else:
            missing_concepts.append(name)

    coverage = len(covered_concepts) / len(expected_concepts)

    return {
        "concept_coverage": round(coverage, 3),
        "covered_concepts": covered_concepts,
        "missing_concepts": missing_concepts,
    }

def calculate_metrics(
    question: str,
    report: str,
    expected_concepts: list | None = None,
) -> dict:
    if "### Sources:" in report:
        body, sources = report.split("### Sources:", 1)
    else:
        body, sources = report, ""

    cited_ids = set(re.findall(SOURCE_ID_PATTERN, body))
    citation_tokens = re.findall(
        r"\[([^\]]*S[0-9A-F]{8}[^\]]*)\]",
        body,
    )

    malformed_citations = sorted({
        token
        for token in citation_tokens
        if not re.fullmatch(r"S[0-9A-F]{8}", token)
    })
    source_id_list = re.findall(
        rf"^\* {SOURCE_ID_PATTERN}",
        sources,
        flags=re.MULTILINE,
    )

    available_ids = set(source_id_list)

    source_id_counts = Counter(source_id_list)

    duplicate_source_ids = sorted(
        source_id
        for source_id, count in source_id_counts.items()
        if count > 1
    )

    primary_ids = set(
        re.findall(
            rf"^\* {SOURCE_ID_PATTERN} \[PRIMARY\]",
            sources,
            flags=re.MULTILINE,
        )
    )

    valid_ids = cited_ids & available_ids

    citation_total = len(cited_ids) + len(malformed_citations)

    citation_validity = (
        len(valid_ids) / citation_total
        if citation_total else 0.0
    )

    source_utilization = (
        len(valid_ids) / len(available_ids)
        if available_ids else 0.0
    )

    primary_source_ratio = (
        len(primary_ids) / len(available_ids)
        if available_ids else 0.0
    )
    concept_metrics = calculate_concept_coverage(
        report,
        expected_concepts or [],
    )
    language_metrics = calculate_language_metrics(
        question,
        report,
    )
    return {
        "citation_count": citation_total,
        "malformed_citations": malformed_citations,
        "source_count": len(available_ids),
        "citation_validity": round(citation_validity, 3),
        "source_utilization": round(source_utilization, 3),
        "primary_source_ratio": round(primary_source_ratio, 3),
        "invalid_citations": sorted(cited_ids - available_ids),
        "unused_sources": sorted(available_ids - cited_ids),
        "duplicate_source_ids": duplicate_source_ids,
        "duplicate_source_free": not duplicate_source_ids,
        **language_metrics,
        **concept_metrics,
    }


def main():
    cases = json.loads(
        (ROOT / "eval" / "questions.json").read_text(encoding="utf-8")
    )
    eval_case = os.getenv("EVAL_CASE")

    if eval_case:
        cases = [
            case for case in cases
            if case["id"] == eval_case
        ]
    limit = min(
        int(os.getenv("EVAL_LIMIT", "3")),
        len(cases),
    )

    os.environ["MAX_WEB_RESEARCH_LOOPS"] = os.getenv(
        "EVAL_LOOPS", "1"
    )
    os.environ["FETCH_FULL_PAGE"] = os.getenv(
        "EVAL_FETCH_FULL_PAGE", "False"
    )

    results = []

    for index, case in enumerate(cases[:limit], start=1):
        print(f"\n[{index}/{limit}] {case['id']}")

        os.environ["PREFERRED_SOURCES"] = case["preferred_sources"]
        started_at = time.perf_counter()

        try:
            output = graph.invoke({
                "research_topic": case["question"]
            })

            report = output["running_summary"]
            latency = time.perf_counter() - started_at
            metrics = calculate_metrics(
                case["question"],
                report,
                case.get("expected_concepts", []),
            )

            result = {
                "id": case["id"],
                "question": case["question"],
                "success": True,
                "latency_seconds": round(latency, 2),
                "metrics": metrics,
                "report": report,
            }

            print(json.dumps(metrics, ensure_ascii=False))

        except Exception as error:
            result = {
                "id": case["id"],
                "question": case["question"],
                "success": False,
                "error": str(error),
            }

            print(f"FAILED: {error}")

        results.append(result)

    successful = [
        result for result in results if result["success"]
    ]

    aggregate = {
        "total": len(results),
        "successful": len(successful),
        "success_rate": round(
            len(successful) / len(results), 3
        ) if results else 0.0,
    }

    if successful:
        aggregate.update({
            "average_latency_seconds": round(
                mean(r["latency_seconds"] for r in successful), 2
            ),
            "average_citation_validity": round(
                mean(
                    r["metrics"]["citation_validity"]
                    for r in successful
                ), 3
            ),
            "average_source_utilization": round(
                mean(
                    r["metrics"]["source_utilization"]
                    for r in successful
                ), 3
            ),
            "average_primary_source_ratio": round(
                mean(
                    r["metrics"]["primary_source_ratio"]
                    for r in successful
                ), 3
            ),
        })
    concept_scores = [
        result["metrics"]["concept_coverage"]
        for result in successful
        if result["metrics"]["concept_coverage"] is not None
    ]

    if concept_scores:
        aggregate["average_concept_coverage"] = round(
            mean(concept_scores),
            3,
        )
    if successful:
        aggregate["language_consistency_rate"] = round(
            mean(
                1.0
                if result["metrics"]["language_consistency"]
                else 0.0
                for result in successful
            ),
            3,
        )

        aggregate["duplicate_source_free_rate"] = round(
            mean(
                1.0
                if result["metrics"]["duplicate_source_free"]
                else 0.0
                for result in successful
            ),
            3,
        )
    output = {
        "aggregate": aggregate,
        "results": results,
    }

    output_path = ROOT / "eval" / "results.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nAggregate:")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()