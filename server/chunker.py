"""
AST-based code chunking via tree-sitter.

Design decisions this file implements (see design doc):
- Section 2: chunk at function/class boundaries, not fixed character counts
- Section 9: chunk identity = filepath:qualified_name (not line numbers),
  so reordering functions doesn't cause spurious re-embeds
- Section 9: per-chunk content hash, stored as metadata, so only genuinely
  changed functions get re-embedded
"""

import hashlib
from pathlib import Path
from tree_sitter_language_pack import get_parser

from server.config import (
    LANG_MAP,
    CHUNKABLE_NODE_TYPES,
    IMPORT_NODE_TYPES,
    FILE_FALLBACK_MAX_CHARS,
    FILE_DESCRIPTION_MAX_CHARS,
)


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

    # Section 9: rare true name collisions (e.g. dynamically generated
    # duplicate names) — append an occurrence index as a last resort so
    # every chunk_id in the returned list is guaranteed unique.
    seen_ids: dict[str, int] = {}
    for chunk in chunks:
        base_id = chunk["chunk_id"]
        if base_id in seen_ids:
            seen_ids[base_id] += 1
            chunk["chunk_id"] = f"{base_id}:{seen_ids[base_id]}"
        else:
            seen_ids[base_id] = 0

    file_chunk = _file_chunk(tree.root_node, source_bytes, id_prefix, chunks)
    if file_chunk:
        chunks.append(file_chunk)

    return chunks