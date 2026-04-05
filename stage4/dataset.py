"""
Dataset functionality for ClassifMLP
"""

from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional
import random
import torch
from torch.utils.data import Dataset

# Canonical reference snippets keyed by question text. These serve as gold answers for
# overlap-based metrics so we never accidentally compare a sample to itself.
REFERENCE_LIBRARY: Dict[str, str] = {
    "How to sort a list in Python?": (
        "numbers = [5, 2, 9, 1]\n"
        "sorted_numbers = sorted(numbers)\n"
        "print(sorted_numbers)"
    ),
    "How to read a file in Python?": (
        "from pathlib import Path\n"
        "text = Path('notes.txt').read_text(encoding='utf-8')\n"
        "print(text)"
    ),
    "How to calculate factorial recursively?": (
        "def factorial(n: int) -> int:\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n"
        "\n"
        "print(factorial(5))"
    ),
    "How to handle exceptions in Python?": (
        "try:\n"
        "    risky_operation()\n"
        "except Exception as exc:\n"
        "    print(f\"operation failed: {exc}\")"
    ),
    "How to use list comprehensions?": (
        "squares = [value * value for value in range(10)]\n"
        "print(squares)"
    ),
    "How to work with dictionaries?": (
        "person = {'name': 'Alice', 'age': 30}\n"
        "for key, value in person.items():\n"
        "    print(f\"{key}: {value}\")"
    ),
    "How to write functions in Python?": (
        "def greet(name: str = 'world') -> str:\n"
        "    return f'Hello, {name}!'\n"
        "\n"
        "print(greet('Ada'))"
    ),
    "How to use classes in Python?": (
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n"
        "\n"
        "    def increment(self, amount: int = 1) -> None:\n"
        "        self.value += amount\n"
        "\n"
        "counter = Counter()\n"
        "counter.increment()\n"
        "print(counter.value)"
    ),
    "How to handle file I/O?": (
        "with open('output.txt', 'w', encoding='utf-8') as fh:\n"
        "    fh.write('sample data')\n"
        "\n"
        "with open('output.txt', 'r', encoding='utf-8') as fh:\n"
        "    print(fh.read())"
    ),
    "How to handle file I/O properly?": (
        "from pathlib import Path\n"
        "path = Path('log.txt')\n"
        "path.write_text('event captured', encoding='utf-8')\n"
        "print(path.read_text(encoding='utf-8'))"
    ),
    "How to use loops in Python?": (
        "total = 0\n"
        "for number in range(5):\n"
        "    total += number\n"
        "print(total)"
    ),
    "How to format strings in Python?": (
        "user = 'Bob'\n"
        "balance = 19.5\n"
        "print(f'{user} has ${balance:.2f} available')"
    ),
    "How to work with dates in Python?": (
        "from datetime import datetime, timedelta\n"
        "today = datetime.now()\n"
        "print(today.strftime('%Y-%m-%d'))\n"
        "print((today + timedelta(days=7)).isoformat())"
    ),
    "How to handle JSON data?": (
        "import json\n"
        "payload = {'name': 'api', 'version': 1}\n"
        "blob = json.dumps(payload)\n"
        "restored = json.loads(blob)\n"
        "print(restored)"
    ),
    "How to create a simple web server?": (
        "from http.server import HTTPServer, SimpleHTTPRequestHandler\n"
        "server = HTTPServer(('0.0.0.0', 8000), SimpleHTTPRequestHandler)\n"
        "server.serve_forever()"
    ),
    "How to use regular expressions?": (
        "import re\n"
        "emails = re.findall(r'[\\w.-]+@[\\w.-]+', 'reach me at dev@example.com')\n"
        "print(emails)"
    ),
    "How to work with NumPy arrays?": (
        "import numpy as np\n"
        "arr = np.array([1, 2, 3, 4])\n"
        "print(arr.mean())"
    ),
    "How to read CSV files?": (
        "import csv\n"
        "with open('data.csv', newline='') as fh:\n"
        "    reader = csv.DictReader(fh)\n"
        "    rows = list(reader)\n"
        "print(rows)"
    ),
    "How to create unit tests?": (
        "import unittest\n"
        "\n"
        "class MathTests(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(2 + 2, 4)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()"
    ),
    "How to use context managers?": (
        "from contextlib import contextmanager\n"
        "\n"
        "@contextmanager\n"
        "def opened(path):\n"
        "    fh = open(path, 'w', encoding='utf-8')\n"
        "    try:\n"
        "        yield fh\n"
        "    finally:\n"
        "        fh.close()\n"
        "\n"
        "with opened('ctx.txt') as fh:\n"
        "    fh.write('ctx')"
    ),
    "How to work with command line arguments?": (
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--name', default='world')\n"
        "args = parser.parse_args([])\n"
        "print(f'Hello {args.name}!')"
    ),
    "How to implement a binary search?": (
        "def binary_search(values, target):\n"
        "    lo, hi = 0, len(values) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if values[mid] == target:\n"
        "            return mid\n"
        "        if values[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1"
    ),
    "How to sort a list?": (
        "items = ['c', 'a', 'b']\n"
        "items.sort()\n"
        "print(items)"
    ),
    "How to read files?": (
        "with open('document.txt', encoding='utf-8') as fh:\n"
        "    for line in fh:\n"
        "        print(line.strip())"
    ),
    "How to write functions?": (
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "print(add(2, 3))"
    ),
    "How to handle errors?": (
        "def safe_div(num, denom):\n"
        "    try:\n"
        "        return num / denom\n"
        "    except ZeroDivisionError:\n"
        "        return None"
    ),
    "How to work with data?": (
        "records = [{'score': 10}, {'score': 7}]\n"
        "avg = sum(r['score'] for r in records) / len(records)\n"
        "print(avg)"
    ),
    "How to create classes?": (
        "class Animal:\n"
        "    def speak(self):\n"
        "        raise NotImplementedError\n"
        "\n"
        "class Dog(Animal):\n"
        "    def speak(self):\n"
        "        return 'woof'\n"
        "\n"
        "print(Dog().speak())"
    ),
    "How to use loops?": (
        "count = 3\n"
        "while count:\n"
        "    print('looping')\n"
        "    count -= 1"
    ),
    "How to format output?": (
        "value = 3.14159\n"
        "print('Pi is approx {:.2f}'.format(value))"
    ),
    "How to manage memory?": (
        "with open('bigfile.bin', 'rb') as fh:\n"
        "    for chunk in iter(lambda: fh.read(4096), b''):\n"
        "        process(chunk)"
    ),
    "How to handle network requests?": (
        "from urllib.request import urlopen\n"
        "with urlopen('https://example.com') as resp:\n"
        "    print(resp.read(100))"
    ),
    "How to process text?": (
        "text = 'Hello, World!'\n"
        "clean = ''.join(ch.lower() for ch in text if ch.isalpha())\n"
        "print(clean)"
    ),
    "How to validate input?": (
        "def read_age(value: str) -> int:\n"
        "    age = int(value)\n"
        "    if age < 0:\n"
        "        raise ValueError('age must be positive')\n"
        "    return age"
    ),
    "How to handle databases?": (
        "import sqlite3\n"
        "conn = sqlite3.connect(':memory:')\n"
        "conn.execute('CREATE TABLE users(id INTEGER, name TEXT)')\n"
        "conn.commit()\n"
        "conn.close()"
    ),
    "How to implement algorithms?": (
        "def insertion_sort(values):\n"
        "    for idx in range(1, len(values)):\n"
        "        key = values[idx]\n"
        "        pos = idx - 1\n"
        "        while pos >= 0 and values[pos] > key:\n"
        "            values[pos + 1] = values[pos]\n"
        "            pos -= 1\n"
        "        values[pos + 1] = key\n"
        "    return values"
    ),
    "How to debug code?": (
        "import logging\n"
        "logging.basicConfig(level=logging.DEBUG)\n"
        "logging.debug('starting computation')"
    ),
    "How to optimize performance?": (
        "values = [i for i in range(1_000_000)]\n"
        "evens = [v for v in values if not v % 2]\n"
        "print(len(evens))"
    ),
    "How to write documentation?": (
        "def area(radius: float) -> float:\n"
        "    \"\"\"Compute the area of a circle.\"\"\"\n"
        "    from math import pi\n"
        "    return pi * radius ** 2"
    ),
    "How to handle concurrency?": (
        "import threading\n"
        "\n"
        "def worker():\n"
        "    print('running in thread')\n"
        "\n"
        "thread = threading.Thread(target=worker)\n"
        "thread.start()\n"
        "thread.join()"
    ),
    "How to manage dependencies?": (
        "from pathlib import Path\n"
        "requirements = Path('requirements.txt').read_text().splitlines()\n"
        "print([pkg for pkg in requirements if pkg])"
    ),
    "How to calculate sum of list?": (
        "values = [1, 2, 3]\n"
        "print(sum(values))"
    ),
    "How to reverse a string?": (
        "text = 'sample'\n"
        "print(text[::-1])"
    ),
    "How to check if number is prime?": (
        "def is_prime(n: int) -> bool:\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for factor in range(2, int(n ** 0.5) + 1):\n"
        "        if n % factor == 0:\n"
        "            return False\n"
        "    return True"
    ),
    "How to merge dictionaries?": (
        "lhs = {'a': 1}\n"
        "rhs = {'b': 2}\n"
        "merged = {**lhs, **rhs}\n"
        "print(merged)"
    ),
    "How to remove duplicates from list?": (
        "names = ['a', 'b', 'a']\n"
        "deduped = list(dict.fromkeys(names))\n"
        "print(deduped)"
    )
}

DEFAULT_REFERENCE = (
    "def reference_placeholder(value):\n"
    "    \"\"\"Fallback canonical snippet when no curated answer exists.\"\"\"\n"
    "    return value\n"
)


class FeedbackClassificationDataset(Dataset):
    """Dataset for code quality classification feedback."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            'question': sample.get('question', ''),
            'answer': sample.get('answer', ''),
            'label': float(sample.get('label', 0)),
            'metadata': sample.get('metadata', {})
        }


def _first_non_empty(source: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = source.get(key)
        if value:
            return str(value).strip()
    return ""


def load_real_datasets(eval_dir: Path, feedback_dir: Path) -> List[Dict[str, Any]]:
    """Load real datasets from CSV files and human feedback JSON files.
    
    FIXED: Proper matching using question ID + answer hash, no collisions.
    """
    import csv
    import json
    import glob
    import os
    import hashlib

    print(f"Loading real datasets from eval_dir: {eval_dir}, feedback_dir: {feedback_dir}")

    samples = []

    # Load human feedback data - FIXED with proper hashing
    feedback_data = {}
    json_pattern = str(feedback_dir / "*.json")
    json_files = glob.glob(json_pattern)
    print(f"Found {len(json_files)} JSON files in feedback_dir")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Process each question in the feedback (usually 2: Left and Right)
            questions_list = data.get('questions_df', [])
            
            for idx, question_data in enumerate(questions_list):
                question_id = question_data.get('ID')
                answer = question_data.get('Answer', '').strip()
                question_text = question_data.get('Question', '').strip()

                if question_id and answer:
                    # FIXED: Use MD5 hash to avoid collisions
                    answer_hash = hashlib.md5(answer.encode('utf-8')).hexdigest()[:16]
                    key = f"{question_id}_{answer_hash}"

                    # Determine if this is Left (idx=0) or Right (idx=1) answer
                    if idx == 0:
                        consistent = (data.get('consistent_L', 0) + 2) / 4  # -2->0, +2->1
                        correct = (data.get('correct_L', 0) + 2) / 4
                        useful = (data.get('useful_L', 0) + 2) / 4
                    else:
                        consistent = (data.get('consistent_R', 0) + 2) / 4
                        correct = (data.get('correct_R', 0) + 2) / 4
                        useful = (data.get('useful_R', 0) + 2) / 4

                    feedback_data[key] = {
                        'consistent': consistent,
                        'correct': correct,
                        'useful': useful,
                        'question': question_text,
                        'answer': answer,
                        'source_file': os.path.basename(json_file)
                    }
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    print(f"Loaded feedback for {len(feedback_data)} samples")

    # First, add all samples directly from feedback data (they already have labels)
    for key, feedback in feedback_data.items():
        question = feedback.get('question', '')
        answer = feedback.get('answer', '')
        
        if question and answer:
            sample = {
                'question': question,
                'answer': answer,
                'labels': {
                    'consistent': feedback['consistent'],
                    'correct': feedback['correct'],
                    'useful': feedback['useful']
                },
                'metadata': {
                    'source_file': feedback.get('source_file', 'feedback'),
                    'has_feedback': True,
                    # FIXED: Don't use answer as its own reference
                    'reference_answer': None
                }
            }
            samples.append(sample)
    
    print(f"Added {len(samples)} samples from feedback JSON files")

    # Now try to find reference answers from CSV files
    csv_pattern = str(eval_dir / "*.csv")
    csv_files = glob.glob(csv_pattern)
    print(f"Found {len(csv_files)} CSV files in eval_dir")
    
    # Build a lookup for reference answers by question ID
    reference_lookup = {}
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question_id = row.get('question_id') or row.get('ID')
                    reference_answer = _first_non_empty(row, [
                        'ans_gt', 'reference_answer'
                    ])
                    
                    if question_id and reference_answer:
                        # Store reference by ID (first one wins)
                        if question_id not in reference_lookup:
                            reference_lookup[question_id] = reference_answer
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    # Update samples with reference answers where available
    updated_count = 0
    for sample in samples:
        q_text = sample['question']
        # Try to extract ID from question or use hash
        for qid, ref in reference_lookup.items():
            if str(qid) in q_text or q_text[:50] in str(reference_lookup.get(qid, '')):
                if sample['metadata'].get('reference_answer') is None:
                    sample['metadata']['reference_answer'] = ref
                    updated_count += 1
                break
    
    print(f"Updated {updated_count} samples with reference answers")
    print(f"Total samples: {len(samples)}")
    return samples


def iter_feedback_samples(feedback_dir: Path) -> List[Dict[str, Any]]:
    """Build a diverse synthetic dataset with richer label distribution."""
    samples: List[Dict[str, Any]] = []

    def add_sample(
        question: str,
        answer: str,
        label: float,
        dataset_tag: str,
        reference_answer: Optional[str] = None
    ) -> None:
        label = float(label)
        question_clean = question.strip()
        answer_clean = answer.strip()
        reference_source = reference_answer.strip() if reference_answer else REFERENCE_LIBRARY.get(question_clean, DEFAULT_REFERENCE)
        reference = reference_source or DEFAULT_REFERENCE
        metadata = {
            'sample_id': len(samples),
            'dataset': dataset_tag,
            'csv_path': str(feedback_dir / 'synthetic_feedback.csv'),
            'quality_score': label,
            'reference_answer': reference
        }
        samples.append({
            'question': question_clean,
            'answer': answer_clean,
            'label': label,
            'labels': {
                'consistent': label,
                'correct': min(1.0, label if label >= 0.5 else max(0.0, label * 0.8)),
                'useful': label
            },
            'metadata': metadata
        })

    good_questions = [
        "How to sort a list in Python?",
        "How to read a file in Python?",
        "How to calculate factorial recursively?",
        "How to use list comprehensions?",
        "How to work with dictionaries?",
        "How to write functions in Python?",
        "How to use classes in Python?",
        "How to handle file I/O properly?",
        "How to use loops in Python?",
        "How to format strings in Python?",
        "How to work with dates in Python?",
        "How to handle JSON data?",
        "How to create a simple web server?",
        "How to use regular expressions?",
        "How to work with NumPy arrays?",
        "How to read CSV files?",
        "How to create unit tests?",
        "How to use context managers?",
        "How to work with command line arguments?",
        "How to implement a binary search?"
    ]

    good_answers = [
        "sorted_list = sorted(my_list)",
        "with open('file.txt', 'r') as f: content = f.read()",
        "def factorial(n): return n * factorial(n-1) if n > 1 else 1",
        "squares = [x**2 for x in range(10)]",
        "my_dict = {'key': 'value'}; value = my_dict.get('key', 'default')",
        "def greet(name: str) -> str: return f'Hello {name}'",
        "class Calculator: def __init__(self): self.value = 0\n    def add(self, x): self.value += x",
        "with open('data.txt', 'w') as f: f.write('content')",
        "for i in range(5): print(f'Number: {i}')",
        "name = 'Alice'; message = f'Hello, {name}!'",
        "from datetime import datetime; now = datetime.now()",
        "import json; data = json.loads('{\"key\": \"value\"}')",
        "from http.server import HTTPServer, BaseHTTPRequestHandler; # Simple server setup",
        "import re; result = re.findall(r'\\d+', 'abc123def456')",
        "import numpy as np; arr = np.array([1, 2, 3, 4, 5])",
        "import csv; with open('data.csv') as f: reader = csv.reader(f)",
        "import unittest; class TestMath(unittest.TestCase): pass",
        "with open('file.txt') as f: content = f.read().strip()",
        "import argparse; parser = argparse.ArgumentParser()",
        "def binary_search(arr, target): left, right = 0, len(arr)-1\n    while left <= right: mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1"
    ]

    base_questions = [
        "How to sort a list in Python?",
        "How to read a file in Python?",
        "How to calculate factorial recursively?",
        "How to handle exceptions in Python?",
        "How to use list comprehensions?",
        "How to work with dictionaries?",
        "How to write functions in Python?",
        "How to use classes in Python?",
        "How to handle file I/O?",
        "How to use loops in Python?"
    ]
    base_answers = [
        "sorted_list = sorted(my_list)",
        "with open('file.txt', 'r') as f: content = f.read()",
        "def factorial(n): return n * factorial(n-1) if n > 1 else 1",
        "try: risky_code() except Exception as e: handle_error(e)",
        "squares = [x**2 for x in range(10)]",
        "my_dict = {'key': 'value'}; value = my_dict.get('key')",
        "def greet(name): return f'Hello {name}'",
        "class Calculator: def add(self, a, b): return a + b",
        "with open('data.txt', 'w') as f: f.write('content')",
        "for i in range(5): print(i)"
    ]
    base_labels = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
    for q, a, label in zip(base_questions, base_answers, base_labels):
        add_sample(q, a, label, 'baseline')

    for q, a in zip(good_questions, good_answers):
        add_sample(q, a, 1.0, 'good_examples')

    bad_questions = [
        "How to handle exceptions in Python?",
        "How to sort a list?",
        "How to read files?",
        "How to write functions?",
        "How to handle errors?",
        "How to work with data?",
        "How to create classes?",
        "How to use loops?",
        "How to format output?",
        "How to manage memory?",
        "How to handle network requests?",
        "How to process text?",
        "How to validate input?",
        "How to handle databases?",
        "How to implement algorithms?",
        "How to debug code?",
        "How to optimize performance?",
        "How to write documentation?",
        "How to handle concurrency?",
        "How to manage dependencies?"
    ]

    bad_answers = [
        "try: risky_code() except: pass  # Bare except is bad practice",
        "my_list.sort()  # Modifies original list unexpectedly",
        "f = open('file.txt'); content = f.read(); f.close()  # No context manager",
        "def func(): return 42  # No type hints or docstring",
        "if error: print('Error!')  # Poor error handling",
        "data = 'some data'  # No validation",
        "class BadClass: pass  # Empty class with no methods",
        "i = 0; while i < 10: print(i); i += 1  # Manual loop control",
        "print('Result: ' + str(result))  # Old-style string formatting",
        "big_list = [i for i in range(1000000)]  # Memory inefficient",
        "import requests; r = requests.get('url')  # No error handling",
        "text = 'hello'; processed = text.upper()  # Minimal processing",
        "user_input = input(); process(user_input)  # No input validation",
        "conn = sqlite3.connect('db.db'); cursor = conn.cursor()  # No cleanup",
        "def sort(arr): return sorted(arr)  # Unnecessarily wraps built-in",
        "print(variable)  # Debugging left in production code",
        "result = slow_function()  # No performance considerations",
        "# TODO: Add documentation  # Missing docstring",
        "import threading; t = threading.Thread(target=func); t.start()  # No join",
        "from some_lib import *  # Pollutes namespace"
    ]

    for q, a in zip(bad_questions, bad_answers):
        add_sample(q, a, 0.0, 'bad_examples')

    medium_questions = [
        "How to calculate sum of list?",
        "How to reverse a string?",
        "How to check if number is prime?",
        "How to merge dictionaries?",
        "How to remove duplicates from list?"
    ]

    medium_answers = [
        "total = 0; for num in numbers: total += num; return total  # Works but not Pythonic",
        "reversed_str = ''; for char in string: reversed_str = char + reversed_str  # Inefficient",
        "def is_prime(n): if n < 2: return False\n    for i in range(2, int(n**0.5)+1): \n        if n % i == 0: return False\n    return True  # Correct but could be optimized",
        "dict1.update(dict2); return dict1  # Modifies original dict",
        "seen = set(); result = []; for item in lst:\n    if item not in seen:\n        seen.add(item); result.append(item)  # Verbose but correct"
    ]

    for q, a in zip(medium_questions, medium_answers):
        add_sample(q, a, 0.5, 'medium_examples')

    random.Random(42).shuffle(samples)
    return samples


def iter_multihead_samples(feedback_dir: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over multi-head feedback samples.

    Simplified version for demonstration.
    """
    # Generate samples with multiple labels (consistent, correct, useful)
    base_samples = iter_feedback_samples(feedback_dir)

    for sample in base_samples:
        # Convert single label to multi-head labels
        base_label = sample['label']
        sample['labels'] = {
            'consistent': base_label,
            'correct': base_label if base_label > 0.5 else 0.0,
            'useful': base_label
        }
        yield sample
