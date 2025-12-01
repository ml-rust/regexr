# Features

This document provides detailed information about the features supported by regexr.

## Pattern Syntax

### Literals

Match literal characters:

```rust
let re = Regex::new("hello").unwrap();
assert!(re.is_match("hello world"));
```

### Character Classes

#### Basic Classes

- `.` - Any character except newline
- `\d` - Digit `[0-9]`
- `\D` - Non-digit `[^0-9]`
- `\w` - Word character `[a-zA-Z0-9_]`
- `\W` - Non-word character `[^a-zA-Z0-9_]`
- `\s` - Whitespace `[ \t\n\r\f\v]`
- `\S` - Non-whitespace

```rust
let re = Regex::new(r"\d+").unwrap();
assert!(re.is_match("123"));
```

#### Custom Classes

```rust
let re = Regex::new(r"[aeiou]").unwrap();
assert!(re.is_match("hello"));

let re = Regex::new(r"[^aeiou]").unwrap(); // Negated
assert!(re.is_match("xyz"));

let re = Regex::new(r"[a-z]").unwrap(); // Range
assert!(re.is_match("hello"));
```

### Quantifiers

#### Greedy Quantifiers

- `*` - Zero or more
- `+` - One or more
- `?` - Zero or one
- `{n}` - Exactly n times
- `{n,}` - n or more times
- `{n,m}` - Between n and m times

```rust
let re = Regex::new(r"\d+").unwrap();
assert_eq!(re.find("abc123def").unwrap().as_str(), "123");

let re = Regex::new(r"\w{3,5}").unwrap();
assert!(re.is_match("hello"));
```

#### Non-Greedy Quantifiers

Add `?` after any quantifier to make it non-greedy:

- `*?` - Zero or more (non-greedy)
- `+?` - One or more (non-greedy)
- `??` - Zero or one (non-greedy)
- `{n,}?` - n or more (non-greedy)
- `{n,m}?` - Between n and m (non-greedy)

```rust
let re = Regex::new(r#"<.*?>"#).unwrap();
assert_eq!(re.find("<a><b>").unwrap().as_str(), "<a>");

let re = Regex::new(r#"<.*>"#).unwrap(); // Greedy version
assert_eq!(re.find("<a><b>").unwrap().as_str(), "<a><b>");
```

### Anchors

- `^` - Start of string
- `$` - End of string
- `\b` - Word boundary
- `\B` - Non-word boundary

```rust
let re = Regex::new(r"^\d+").unwrap();
assert!(re.is_match("123abc"));
assert!(!re.is_match("abc123"));

let re = Regex::new(r"\bword\b").unwrap();
assert!(re.is_match("a word here"));
assert!(!re.is_match("awordhere"));
```

### Alternation

```rust
let re = Regex::new(r"cat|dog|bird").unwrap();
assert!(re.is_match("I have a dog"));
assert!(re.is_match("She has a cat"));
```

### Grouping

#### Capturing Groups

```rust
let re = Regex::new(r"(\w+)@(\w+)").unwrap();
let caps = re.captures("user@example").unwrap();
assert_eq!(&caps[1], "user");
assert_eq!(&caps[2], "example");
```

#### Named Groups

```rust
let re = Regex::new(r"(?P<user>\w+)@(?P<domain>\w+)").unwrap();
let caps = re.captures("user@example").unwrap();
assert_eq!(&caps["user"], "user");
assert_eq!(&caps["domain"], "example");
```

#### Non-Capturing Groups

```rust
let re = Regex::new(r"(?:cat|dog)+").unwrap();
assert!(re.is_match("catdogcat"));
```

### Backreferences

Match the same text as a previous capture group:

```rust
let re = Regex::new(r"(\w+)\s+\1").unwrap();
assert!(re.is_match("hello hello"));
assert!(!re.is_match("hello world"));
```

Backreferences require the BacktrackingVm or BacktrackingJit engine.

### Lookaround Assertions

#### Lookahead

- `(?=...)` - Positive lookahead
- `(?!...)` - Negative lookahead

```rust
let re = Regex::new(r"\w+(?=@)").unwrap();
assert_eq!(re.find("user@example").unwrap().as_str(), "user");

let re = Regex::new(r"\w+(?!@)").unwrap();
assert_eq!(re.find("user example").unwrap().as_str(), "user");
```

#### Lookbehind

- `(?<=...)` - Positive lookbehind
- `(?<!...)` - Negative lookbehind

```rust
let re = Regex::new(r"(?<=@)\w+").unwrap();
assert_eq!(re.find("user@example").unwrap().as_str(), "example");

let re = Regex::new(r"(?<!@)\w+").unwrap();
assert_eq!(re.find("user example").unwrap().as_str(), "user");
```

Lookaround assertions require the PikeVm or TaggedNfa engine.

## Unicode Support

### Unicode Character Classes

Patterns can match Unicode characters using escape sequences:

```rust
let re = Regex::new(r"\w+").unwrap();
assert!(re.is_match("café")); // Matches Unicode word characters
```

### Unicode Properties

Match characters by Unicode properties:

```rust
// Letter category
let re = Regex::new(r"\p{Letter}+").unwrap();
assert!(re.is_match("hello"));
assert!(re.is_match("привет")); // Cyrillic

// Number category
let re = Regex::new(r"\p{Number}+").unwrap();
assert!(re.is_match("123"));
assert!(re.is_match("①②③")); // Unicode numbers

// Script
let re = Regex::new(r"\p{Greek}+").unwrap();
assert!(re.is_match("αβγ"));

let re = Regex::new(r"\p{Cyrillic}+").unwrap();
assert!(re.is_match("привет"));
```

### Supported Unicode Categories

- `Letter` (`L`): All letters
- `Number` (`N`): All numbers
- `Mark` (`M`): Combining marks
- `Punctuation` (`P`): Punctuation characters
- `Symbol` (`S`): Symbols
- `Separator` (`Z`): Separators

### Supported Scripts

- `Arabic`, `Armenian`, `Bengali`, `Cyrillic`, `Devanagari`
- `Georgian`, `Greek`, `Gujarati`, `Gurmukhi`, `Han`
- `Hangul`, `Hebrew`, `Hiragana`, `Kannada`, `Katakana`
- `Khmer`, `Lao`, `Latin`, `Malayalam`, `Myanmar`
- `Oriya`, `Sinhala`, `Tamil`, `Telugu`, `Thai`, `Tibetan`

And many more. See Unicode Character Database for the complete list.

### Case-Insensitive Matching

Unicode-aware case folding:

```rust
let re = Regex::new(r"(?i)hello").unwrap();
assert!(re.is_match("HELLO"));
assert!(re.is_match("Hello"));
assert!(re.is_match("café")); // Unicode case folding
```

## JIT Compilation

### Enabling JIT

Use `RegexBuilder` to enable JIT compilation:

```rust
use regexr::RegexBuilder;

let re = RegexBuilder::new(r"\w+@\w+\.\w+")
    .jit(true)
    .build()
    .unwrap();

assert!(re.is_match("user@example.com"));
```

### When to Use JIT

JIT compilation is beneficial when:

1. **Pattern will be matched many times**: JIT has higher compilation cost but faster execution
2. **Performance is critical**: JIT generates native code for maximum speed
3. **Pattern has effective prefilters**: Combines SIMD literal search with native DFA execution

### JIT Requirements

- Only available on x86-64 architecture
- Requires `jit` feature flag
- Automatically falls back to interpreted engines if compilation fails

### JIT Engine Selection

When JIT is enabled, the engine is selected based on:

- **Backreferences**: Uses BacktrackingJit
- **Lookaround**: Falls back to PikeVm (requires NFA semantics)
- **Non-greedy quantifiers**: Uses TaggedNfa
- **Large Unicode classes**: Uses LazyDfa (avoids state explosion)
- **Alternations without prefilter**: Uses JitShiftOr
- **General patterns**: Uses DFA JIT with SIMD prefiltering

## SIMD Acceleration

### Default Behavior

SIMD acceleration is enabled by default through the `simd` feature. It provides:

- AVX2-accelerated literal search using Teddy algorithm
- Fast multi-pattern matching for 2-8 literals
- Automatic fallback to scalar implementations when SIMD is unavailable

### How It Works

The SIMD prefilter:

1. Extracts required literals from the pattern
2. Uses SIMD instructions to scan for candidates
3. Verifies candidates with the full regex engine
4. Returns matches

Example pattern with effective prefilter:

```rust
let re = Regex::new(r"hello\w+").unwrap();
// SIMD scans for "hello", then engine verifies \w+
```

### Disabling SIMD

Build without SIMD:

```bash
cargo build --no-default-features
```

## Prefix Optimization

### Tokenizer Optimization

For patterns with many literal alternatives (common in tokenizers), prefix optimization merges common prefixes into a trie structure:

```rust
use regexr::RegexBuilder;

let re = RegexBuilder::new(r"(function|for|finally|from)")
    .optimize_prefixes(true)
    .build()
    .unwrap();
```

### How It Works

Without optimization:
```
(function|for|finally|from)
```
Creates separate NFA branches for each alternative.

With optimization:
```
f(unction|or|inally|rom)
```
Merges the common prefix `f`, reducing active NFA threads from O(vocabulary_size) to O(token_length).

### When to Use

Enable prefix optimization when:

- Pattern has many literal alternatives (>10)
- Alternatives share common prefixes
- Used in tokenization or keyword matching

## API Features

### Matching

```rust
let re = Regex::new(r"\d+").unwrap();

// Check if pattern matches
if re.is_match("abc123") {
    println!("Match found");
}

// Find first match
if let Some(m) = re.find("abc123def") {
    println!("Found at {}-{}: {}", m.start(), m.end(), m.as_str());
}

// Find all matches
for m in re.find_iter("123 456 789") {
    println!("{}", m.as_str());
}
```

### Capture Groups

```rust
let re = Regex::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();
let caps = re.captures("2024-01-15").unwrap();

println!("Year: {}", &caps[1]);
println!("Month: {}", &caps[2]);
println!("Day: {}", &caps[3]);

// Iterate over all captures in text
for caps in re.captures_iter("2024-01-15 2024-02-20") {
    println!("{}", &caps[0]);
}
```

### Text Replacement

```rust
let re = Regex::new(r"\d+").unwrap();

// Replace first match
let result = re.replace("Price: 100 dollars", "200");
assert_eq!(result, "Price: 200 dollars");

// Replace all matches
let result = re.replace_all("Price: 100 and 200 dollars", "X");
assert_eq!(result, "Price: X and X dollars");
```

### Pattern Information

```rust
let re = Regex::new(r"(?P<year>\d{4})-(?P<month>\d{2})").unwrap();

// Get original pattern
println!("Pattern: {}", re.as_str());

// Get capture names
for name in re.capture_names() {
    println!("Capture group: {}", name);
}

// Get engine name (for debugging)
println!("Engine: {}", re.engine_name());
```

## Error Handling

All regex compilation returns `Result<Regex, Error>`:

```rust
use regexr::Regex;

match Regex::new(r"(unclosed") {
    Ok(re) => println!("Compiled successfully"),
    Err(e) => eprintln!("Compilation error: {}", e),
}
```

Common errors:
- Unclosed groups
- Invalid escape sequences
- Invalid repetition
- Invalid backreference
- Unsupported features

## Performance Tips

### 1. Enable JIT for Hot Patterns

```rust
let re = RegexBuilder::new(pattern)
    .jit(true)
    .build()
    .unwrap();
```

### 2. Use Prefix Optimization for Tokenizers

```rust
let re = RegexBuilder::new(keyword_pattern)
    .optimize_prefixes(true)
    .build()
    .unwrap();
```

### 3. Anchor Patterns When Possible

```rust
// Better
let re = Regex::new(r"^\d+").unwrap();

// Slower (must search entire string)
let re = Regex::new(r"\d+").unwrap();
```

### 4. Use Character Classes Instead of Alternations

```rust
// Better
let re = Regex::new(r"[abc]").unwrap();

// Slower
let re = Regex::new(r"(a|b|c)").unwrap();
```

### 5. Avoid Unnecessary Captures

```rust
// Better (non-capturing group)
let re = Regex::new(r"(?:cat|dog)+").unwrap();

// Slower (capturing group not needed)
let re = Regex::new(r"(cat|dog)+").unwrap();
```

### 6. Profile Engine Selection

Use `engine_name()` to verify the selected engine:

```rust
let re = Regex::new(pattern).unwrap();
println!("Using engine: {}", re.engine_name());
```

Ensure the engine matches your expectations for the pattern type.

## Limitations

### Current Limitations

1. **JIT**: Only available on x86-64 architecture
2. **Multiline mode**: Currently `.` never matches newline
3. **Backreferences**: Cannot be combined with JIT DFA (uses BacktrackingJit instead)
4. **Variable-width lookbehind**: Limited support (fixed-width lookbehind only)

### Platform Support

- **x86-64**: All features including JIT
- **Other architectures**: Interpreted engines only (no JIT)

### Feature Compatibility

| Feature | PikeVm | ShiftOr | LazyDFA | JIT DFA | BacktrackingJit |
|---------|--------|---------|---------|---------|-----------------|
| Backreferences | ✓ | ✗ | ✗ | ✗ | ✓ |
| Lookaround | ✓ | ✗ | ✗ | ✗ | ✗ |
| Non-greedy | ✓ | ✗ | ✗ | ✗ | ✗ |
| Word boundaries | ✓ | ✗ | ✓ | ✓ | ✓ |
| Anchors | ✓ | ✗ | ✓ | ✓ | ✓ |
| Captures | ✓ | ✓* | ✓* | ✓ | ✓ |

*ShiftOr and LazyDFA fall back to PikeVm for capture extraction.
