from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_FILE = "workloads/prompts/benchmark_prompt.txt"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt = f.read()

    tokens = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    print(f"Prompt characters: {len(prompt)}")
    print(f"Prompt tokens: {len(tokens)}")


if __name__ == "__main__":
    main()