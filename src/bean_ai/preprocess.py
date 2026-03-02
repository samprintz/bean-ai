import re
from beancount.parser import parser


def preprocess(input_path: str, output_path: str) -> None:
    """
    Preprocess a beancount file and extract training data.

    Extracts transaction descriptions and their corresponding accounts
    for transactions with exactly 2 postings where the description
    contains no digits.
    """
    entries, errors, options = parser.parse_file(input_path)

    dataset = []

    for entry in entries:
        # Filter to Transaction entries only
        if not hasattr(entry, 'postings') or entry.postings is None:
            continue

        # Filter to transactions with exactly two postings
        if len(entry.postings) != 2:
            print(f"Skip {entry.narration} ({len(entry.postings)} postings)")
            continue

        # Filter to transactions without digits (better AI results)
        narration = entry.narration or ""
        if any(char.isdigit() for char in narration):  # TODO make this configurable
            continue

        # Use the posting that does not start with common source accounts
        posting = entry.postings[0]
        source_prefixes = ("Konto:", "Aktiva:Barvermögen", "Assets:")  # TODO make this configurable
        if any(entry.postings[0].account.startswith(p) for p in source_prefixes):
            posting = entry.postings[1]

        # Sanitize description (remove +word patterns)
        description = re.sub(r'\+\w+\s*', '', narration)

        if description.strip():
            dataset.append((description.strip(), posting.account))

    # Write out the dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('text\tlabel\n')
        for text, label in dataset:
            f.write(f"{text}\t{label}\n")

    print(f"Preprocessed {len(dataset)} transactions from {input_path} -> {output_path}")
