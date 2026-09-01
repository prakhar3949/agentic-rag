"""Router-only evaluation against arXiv category ground truth (ROADMAP §Phase 5 router bullet:
"evaluate against arXiv labels - free ground truth").

    python -m rag.router_eval [-n N]

No judge call, no generation, no retrieval - just the router's one LLM call per question, scored
against results/goldset_curated.jsonl's `category` field. Two numbers, in tension with each other:

- hit_rate: fraction of questions where the true category is IN the router's predicted route.
  Recall for the router - can it find the right corpus slice at all?
- mean_route_len: average number of categories the router returns. A router that always returns
  every indexed category gets 100% hit_rate trivially and is worthless - this number is what
  catches that. Compare it against len(config.INDEXED_CATEGORIES) to see how much narrowing the
  router is actually doing.
"""

import argparse
import json
import random
from pathlib import Path

from rag import config
from rag.generation import Generator
from rag.router import router_node


def load_goldset(path: Path = config.GOLDSET_CURATED_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sample_goldset(examples: list[dict], n: int, seed: int = 0) -> list[dict]:
    if n >= len(examples):
        return list(examples)
    return random.Random(seed).sample(examples, n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.router_eval")
    parser.add_argument("-n", "--n", type=int, default=None,
                        help="questions to sample (default: the whole curated goldset)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goldset", default=str(config.GOLDSET_CURATED_PATH))
    args = parser.parse_args(argv)

    examples = load_goldset(Path(args.goldset))
    if args.n:
        examples = sample_goldset(examples, args.n, seed=args.seed)
    print(f"Router eval | n={len(examples)} | indexed categories={config.INDEXED_CATEGORIES}")

    generator = Generator()
    rows = []
    for ex in examples:
        state = {"question": ex["question"]}
        route = router_node(state, generator)["route"]
        hit = ex["category"] in route
        rows.append({
            "question_id": ex["id"], "true_category": ex["category"],
            "route": route, "hit": hit,
        })
        print(f"  {'HIT ' if hit else 'MISS'} true={ex['category']:<8} route={route}")

    hit_rate = sum(r["hit"] for r in rows) / len(rows)
    mean_route_len = sum(len(r["route"]) for r in rows) / len(rows)
    summary = {
        "n": len(rows),
        "hit_rate": hit_rate,
        "mean_route_len": mean_route_len,
        "n_indexed_categories": len(config.INDEXED_CATEGORIES),
        "chance_hit_rate_if_uniform_random": 1 / len(config.INDEXED_CATEGORIES),
    }

    config.ROUTER_EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.ROUTER_EVAL_PATH.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {config.ROUTER_EVAL_PATH}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
