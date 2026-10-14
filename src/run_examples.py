"""Run the full pipeline on examples/sample_inputs.json and save examples/sample_outputs.json.

Run:  python -m src.run_examples
"""
import json

from .config import EXAMPLES_DIR
from .data import load_tickets, split
from .pipeline import TicketAssistant


def main() -> None:
    train, _ = split(load_tickets())
    assistant = TicketAssistant(train)
    inputs = json.loads((EXAMPLES_DIR / "sample_inputs.json").read_text(encoding="utf-8"))
    outputs = [assistant.analyze(t) for t in inputs]
    (EXAMPLES_DIR / "sample_outputs.json").write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for o in outputs:
        print(f"[{o['language']}] {o['category']:<9} {o['priority']:<6} {o['intent']:<20} <- {o['text'][:60]}")
    print(f"\nSaved {len(outputs)} results to {EXAMPLES_DIR / 'sample_outputs.json'}")


if __name__ == "__main__":
    main()
