"""
AST-based code chunking via tree-sitter.

Design decisions this file implements (see design doc):
- Section 2: chunk at function/class boundaries, not fixed character counts
- Section 9: chunk identity = filepath:qualified_name (not line numbers),
  so reordering functions doesn't cause spurious re-embeds
- Section 9: per-chunk content hash, stored as metadata, so only genuinely
  changed functions get re-embedded
- Decision #39: docs and infra/config files (no AST) are chunked as text —
  whole if small, Markdown by heading, others by line windows
"""

import hashlib
import re
from pathlib import Path
from tree_sitter_language_pack import get_parser

from server.config import (
    LANG_MAP,
    CHUNKABLE_NODE_TYPES,
    IMPORT_NODE_TYPES,
    FILE_FALLBACK_MAX_CHARS,
    FILE_DESCRIPTION_MAX_CHARS,
    TEXT_FILE_TYPES,
    TEXT_FILE_PREFIXES,
    TEXT_WINDOW_LINES,
)

_MD_HEADING = re.compile(r"^#{1,3}\s+(.+)")
_MD_FENCE = re.compile(r"^\s*(```|~~~)")


def text_lang(filename: str) -> str | None:
    """Fence language if this is an indexable non-code text file, else None."""
    for prefix, lang in TEXT_FILE_PREFIXES.items():
        if filename.startswith(prefix):
            return lang
    return TEXT_FILE_TYPES.get(Path(filename).suffix)


def _hash_text(text: str) -> str:
    """Stable content hash for a chunk's source text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _qualified_name(node, source_bytes: bytes, enclosing_class: str | None) -> str:
    """
    Extract the function/class name from a tree-sitter node, and prefix
    with the enclosing class name if there is one (Section 9: handles
    method name collisions across different classes in the same file).
    """
    name_node = node.child_by_field_name("name")
    if name_node is None:
        # Anonymous function/lambda — fall back to a position-based label.
        # Rare in practice; flagged so we can see it happen if it does.
        raw_name = f"anonymous_L{node.start_point[0] + 1}"
    else:
        raw_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")

    if enclosing_class:
        return f"{enclosing_class}.{raw_name}"
    return raw_name


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def _is_leading_doc(node) -> bool:
    """
    A comment, or a Python docstring (a bare string expression). Newer
    tree-sitter-python emits a top-level `string` directly; older versions
    wrap it in an `expression_statement` — handle both.
    """
    if node.type in ("comment", "string"):
        return True
    return (
        node.type == "expression_statement"
        and node.named_child_count > 0
        and node.named_children[0].type == "string"
    )


def _file_chunk(root, source_bytes: bytes, id_prefix: str, function_chunks: list[dict]) -> dict | None:
    """
    One file-level chunk (Section 6, Option B — cheap, no LLM):
    - file has functions -> heuristic summary: path, leading docstring/
      comment, top-level imports, signature line of each function/class
    - file has none -> the file's real content, capped (decision #25)
    """
    source = source_bytes.decode("utf-8")
    if not source.strip():
        return None  # empty file (e.g. __init__.py) — nothing to search for

    if function_chunks:
        kind = "summary"
        lines = [f"File: {id_prefix}"]

        leading = []
        for node in root.children:
            if not _is_leading_doc(node):
                break
            leading.append(_node_text(node, source_bytes))
        if leading:
            lines.append("Description: " + "\n".join(leading)[:FILE_DESCRIPTION_MAX_CHARS])

        imports = [_node_text(n, source_bytes) for n in root.children if n.type in IMPORT_NODE_TYPES]
        if imports:
            lines.append("Imports:")
            lines += [f"  {imp}" for imp in imports]

        lines.append("Defines:")
        lines += [f"  {c['text'].splitlines()[0].strip()}" for c in function_chunks]
        text = "\n".join(lines)
    else:
        kind = "code"
        text = source[:FILE_FALLBACK_MAX_CHARS]

    return {
        "chunk_id": f"{id_prefix}:<file>",  # '<' can't appear in a real qualified name
        "text": text,
        "content_hash": _hash_text(text),
        "start_line": 1,
        "end_line": root.end_point[0] + 1,
        "node_type": root.type,
        "level": "file",
        "kind": kind,
    }


def chunk_file(filepath: str, rel_path: str | None = None) -> list[dict]:
    """
    Parse a single file and return a list of chunk dicts, one per
    function/class definition found via tree-sitter AST walk.

    `filepath` is where the file is read from; `rel_path` (path relative to
    the repo root, Section 8) is what goes into chunk ids. Defaults to
    `filepath` when not given.

    Each chunk dict has:
        chunk_id     - stable id: "<rel_path>:<qualified_name>"
        text         - the exact source text of the chunk
        content_hash - sha256 of `text`, for change detection (Section 9)
        start_line   - 1-indexed start line (metadata only, not identity)
        end_line     - 1-indexed end line (metadata only, not identity)
        node_type    - the tree-sitter node type (function_definition, etc.)
        level        - "function" or "file" (Section 5 layers)
        kind         - "code" (verbatim source) or "summary" (heuristic
                       paraphrase — must be labeled as such, Section 11.3)

    The last chunk (if the file isn't empty) is the file-level chunk.
    """
    path = Path(filepath)
    id_prefix = rel_path if rel_path is not None else filepath
    ext = path.suffix
    language = LANG_MAP.get(ext)
    if language is None:
        if text_lang(path.name):
            return _dedupe_ids(_chunk_text_file(path, id_prefix))
        return []  # unsupported file type — caller should have filtered already

    try:
        parser = get_parser(language)
    except LookupError:
        # tree-sitter-language-pack doesn't have a grammar for this
        # identifier — shouldn't happen if LANG_MAP is correct, but fail
        # soft rather than crashing the whole indexing run over one file.
        return []

    source_bytes = path.read_bytes()
    tree = parser.parse(source_bytes)

    chunks: list[dict] = []

    def walk(node, enclosing_class: str | None):
        # Track whether *this* node is a class, so children know their
        # enclosing class name (Section 9: qualified names for methods).
        current_class = enclosing_class
        if node.type in ("class_definition", "class_declaration"):
            current_class = _qualified_name(node, source_bytes, enclosing_class)

        if node.type in CHUNKABLE_NODE_TYPES:
            qualified_name = _qualified_name(node, source_bytes, enclosing_class)
            text = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
            chunks.append({
                "chunk_id": f"{id_prefix}:{qualified_name}",
                "text": text,
                "content_hash": _hash_text(text),
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "node_type": node.type,
                "level": "function",
                "kind": "code",
            })

        for child in node.children:
            walk(child, current_class)

    walk(tree.root_node, enclosing_class=None)
    _dedupe_ids(chunks)

    file_chunk = _file_chunk(tree.root_node, source_bytes, id_prefix, chunks)
    if file_chunk:
        chunks.append(file_chunk)

    return chunks


def _dedupe_ids(chunks: list[dict]) -> list[dict]:
    """
    Section 9: rare true name collisions (duplicate function names, repeated
    Markdown headings) — append an occurrence index as a last resort so
    every chunk_id is unique.
    """
    seen_ids: dict[str, int] = {}
    for chunk in chunks:
        base_id = chunk["chunk_id"]
        if base_id in seen_ids:
            seen_ids[base_id] += 1
            chunk["chunk_id"] = f"{base_id}:{seen_ids[base_id]}"
        else:
            seen_ids[base_id] = 0
    return chunks


def _text_chunk(chunk_id: str, text: str, start: int, end: int, level: str, kind: str = "code") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "content_hash": _hash_text(text),
        "start_line": start,
        "end_line": end,
        "node_type": "text",
        "level": level,
        "kind": kind,
    }


def _markdown_sections(lines: list[str]) -> list[tuple[str, int, int]]:
    """
    (name, start_idx, end_idx) per #/##/### section. Lines inside fenced
    code blocks never start a section (a `# comment` in a shell example
    isn't a heading). Text before the first heading becomes "#(intro)".
    """
    starts: list[tuple[str, int]] = []
    in_fence = False
    for i, line in enumerate(lines):
        if _MD_FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (m := _MD_HEADING.match(line)):
            starts.append(("#" + m.group(1).strip(), i))

    if not starts or starts[0][1] > 0:
        starts.insert(0, ("#(intro)", 0))
    bounds = [s for _, s in starts[1:]] + [len(lines)]
    return [
        (name, s, e) for (name, s), e in zip(starts, bounds)
        if "".join(lines[s:e]).strip()
    ]


def _chunk_text_file(path: Path, id_prefix: str) -> list[dict]:
    """
    Decision #39:
    - small file -> one file-level chunk with its full text
    - large Markdown -> one detail chunk per section, plus a file-level
      outline (the headings) as the file's summary
    - large other text -> detail chunks of TEXT_WINDOW_LINES lines; ids use
      line ranges, so an insertion re-embeds later windows (cheap locally)
    """
    source = path.read_text(encoding="utf-8")
    if not source.strip():
        return []
    lines = source.splitlines(keepends=True)

    if len(source) <= FILE_FALLBACK_MAX_CHARS:
        return [_text_chunk(f"{id_prefix}:<file>", source, 1, len(lines), "file")]

    if path.suffix == ".md":
        sections = _markdown_sections(lines)
        chunks = [
            _text_chunk(f"{id_prefix}:{name}", "".join(lines[s:e]), s + 1, e, "function")
            for name, s, e in sections
        ]
        outline = [lines[s].rstrip() for name, s, _ in sections if name != "#(intro)"]
        summary = f"File: {id_prefix}\nSections:\n" + "\n".join(f"  {h}" for h in outline)
        chunks.append(_text_chunk(f"{id_prefix}:<file>", summary, 1, len(lines), "file", kind="summary"))
        return chunks

    return [
        _text_chunk(
            f"{id_prefix}:L{s + 1}-{min(s + TEXT_WINDOW_LINES, len(lines))}",
            "".join(lines[s:s + TEXT_WINDOW_LINES]),
            s + 1, min(s + TEXT_WINDOW_LINES, len(lines)), "function",
        )
        for s in range(0, len(lines), TEXT_WINDOW_LINES)
    ]