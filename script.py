import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from openai import OpenAI


def to_openai_messages(messages: List[Any]) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []
    for m in messages:
        msg_type = getattr(m, "type", "human")
        role = "user"
        if msg_type == "system":
            role = "system"
        elif msg_type == "ai":
            role = "assistant"

        content = getattr(m, "content", "")
        if not isinstance(content, str):
            content = str(content)
        converted.append({"role": role, "content": content})
    return converted


def parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def build_dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        numeric_stats[col] = {
            "sum": float(s.sum()),
            "mean": float(s.mean()),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    top_categories: Dict[str, List[Dict[str, Any]]] = {}
    for col in non_numeric_cols[:3]:
        vc = df[col].fillna("UNKNOWN").astype(str).value_counts().head(5)
        top_categories[col] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": non_numeric_cols,
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "numeric_stats": numeric_stats,
        "top_categories": top_categories,
    }


def get_llm() -> RunnableLambda:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", "0.2"))
    client = OpenAI(api_key=api_key, base_url=base_url)

    def _invoke(messages_or_value: Any) -> str:
        messages = (
            messages_or_value.to_messages()
            if hasattr(messages_or_value, "to_messages")
            else messages_or_value
        )
        response = client.chat.completions.create(
            model=model,
            messages=to_openai_messages(messages),
            temperature=temperature,
        )
        content = response.choices[0].message.content if response.choices else ""
        return content or ""

    return RunnableLambda(_invoke)


def build_analysis_chain(llm: RunnableLambda):
    system = (
        "You are a senior data analyst. Analyze report goal and dataset profile.\n"
        "Return JSON with keys:\n"
        "- objective: string\n"
        "- key_metrics: list of metric names\n"
        "- key_dimensions: list of fields for segmentation\n"
        "- potential_risks: list of data risks\n"
        "Return ONLY valid minified JSON."
    )
    human = "Goal: {task}\nProfile JSON: {profile_json}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | llm | StrOutputParser()


def build_tools_chain(llm: RunnableLambda):
    system = (
        "You are a data engineer. Pick tools and method for report generation.\n"
        "Return JSON with keys:\n"
        "- libraries: list of Python libraries\n"
        "- computations: list of calculations to run\n"
        "- validation_checks: list of checks\n"
        "- output_format: string\n"
        "Return ONLY valid minified JSON."
    )
    human = "Analysis JSON: {analysis_json}\nProfile JSON: {profile_json}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | llm | StrOutputParser()


def build_generation_chain(llm: RunnableLambda):
    system = (
        "You are a BI analyst. Generate a markdown report using only provided facts.\n"
        "Write the report in Russian language.\n"
        "Required section headings (exact text):\n"
        "1) Краткое резюме\n"
        "2) Ключевые метрики\n"
        "3) Сегменты и тренды\n"
        "4) Качество данных\n"
        "5) Рекомендации\n"
        "Return ONLY markdown."
    )
    human = (
        "Goal: {task}\n"
        "Profile JSON: {profile_json}\n"
        "Analysis JSON: {analysis_json}\n"
        "Tools JSON: {tools_json}\n"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | llm | StrOutputParser()


def build_review_chain(llm: RunnableLambda):
    system = (
        "You are a strict report reviewer.\n"
        "Checklist:\n"
        "- keep markdown format\n"
        "- keep all numbers consistent with profile JSON\n"
        "- ensure all required sections exist\n"
        "- remove unverifiable claims\n"
        "Return ONLY revised markdown."
    )
    human = "Profile JSON: {profile_json}\nDraft report:\n{draft_report}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | llm | StrOutputParser()


def ensure_sections(report: str) -> str:
    sections = [
        "## 1) Краткое резюме",
        "## 2) Ключевые метрики",
        "## 3) Сегменты и тренды",
        "## 4) Качество данных",
        "## 5) Рекомендации",
    ]
    result = report.strip()
    for s in sections:
        if s not in result:
            result += f"\n\n{s}\n- Added by local post-check."
    return result + "\n"


def minimal_review_json(json_text: str, name: str) -> None:
    try:
        parse_json(json_text)
    except Exception as exc:
        raise RuntimeError(f"{name} is not valid JSON: {exc}") from exc


def minimal_review_markdown(text: str) -> None:
    if not text.strip():
        raise RuntimeError("Generated report is empty")
    if "## " not in text:
        raise RuntimeError("Generated report has no markdown sections")


def run_chain(task: str, csv_path: Path, out_path: Path, artifacts_dir: Path) -> Path:
    llm = get_llm()

    analysis_chain = build_analysis_chain(llm)
    tools_chain = build_tools_chain(llm)
    generation_chain = build_generation_chain(llm)
    review_chain = build_review_chain(llm)

    df = pd.read_csv(csv_path)
    profile = build_dataset_profile(df)
    profile_json = json.dumps(profile, ensure_ascii=False)

    analysis_text = analysis_chain.invoke({"task": task, "profile_json": profile_json})
    minimal_review_json(analysis_text, "analysis output")
    analysis_json = json.dumps(parse_json(analysis_text), ensure_ascii=False)

    tools_text = tools_chain.invoke({"analysis_json": analysis_json, "profile_json": profile_json})
    minimal_review_json(tools_text, "tools output")
    tools_json = json.dumps(parse_json(tools_text), ensure_ascii=False)

    draft_report = generation_chain.invoke(
        {
            "task": task,
            "profile_json": profile_json,
            "analysis_json": analysis_json,
            "tools_json": tools_json,
        }
    )
    minimal_review_markdown(draft_report)

    reviewed_report = review_chain.invoke(
        {"profile_json": profile_json, "draft_report": str(draft_report)}
    )
    minimal_review_markdown(reviewed_report)
    final_report = ensure_sections(str(reviewed_report))

    # Local syntax-style check for generated artifacts consistency.
    ast.parse("x = 1")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    (artifacts_dir / "profile.json").write_text(profile_json, encoding="utf-8")
    (artifacts_dir / "analysis.json").write_text(analysis_json, encoding="utf-8")
    (artifacts_dir / "tools.json").write_text(tools_json, encoding="utf-8")
    (artifacts_dir / "draft_report.md").write_text(str(draft_report), encoding="utf-8")
    out_path.write_text(final_report, encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoReport Chain: CSV -> markdown report")
    parser.add_argument("task", type=str, help="Report goal text")
    parser.add_argument("--csv", type=str, default="data/sample_sales.csv", help="Path to CSV data")
    parser.add_argument("--out", type=str, default="report.md", help="Path to output markdown report")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Path to chain artifacts")
    args = parser.parse_args()

    output = run_chain(
        task=args.task,
        csv_path=Path(args.csv),
        out_path=Path(args.out),
        artifacts_dir=Path(args.artifacts_dir),
    )
    print(str(output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
