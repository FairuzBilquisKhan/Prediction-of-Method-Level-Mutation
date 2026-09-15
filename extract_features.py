"""
extract_features.py
Extracts per-method features from method definition JSONs, calling the
tree-climber HTTP server to obtain CFGs on demand.

Tree-climber CFG schema (from server's /parse endpoint):
    {
        "nodes": {
            "<node_id>": {
                "node_type": "ENTRY" | "EXIT" | "CONDITION" | "RETURN" | ...,
                "start_index": <byte offset in source>,
                "predecessors": [<node_id>, ...],
                "successors":   [<node_id>, ...],
                ...
            },
            ...
        }
    }

Expected method JSON schema (one file per version, list of method objects):
    [
        {
            "method_id": "Lang_1f_StringUtils_indexOf_42",
            "project": "Lang",
            "bug_id": "1",
            "class_fqn": "org.apache.commons.lang3.StringUtils",
            "method_name": "indexOf",
            "start_line": 42,
            "end_line": 78,
            "body": "public int indexOf(...) { ... }"
        }
    ]

Prerequisites:
    - tree-climber server running:
        cd tree-climber/
        uv run -m uvicorn tree_climber.viz.app:app --reload --host 0.0.0.0 --port 8000

Usage:
    python extract_features.py \
        --methods-dir method_jsons/ \
        --output method_features.csv \
        --server http://localhost:8000
"""

import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd

try:
    import requests
except ImportError:
    print('Error: "requests" not installed. Run: pip install requests', file=sys.stderr)
    sys.exit(1)


DEFAULT_SERVER = "http://localhost:8000"
SKIP_NODE_TYPES = {"ENTRY", "EXIT"}


# ============================================================================
# CODE-LEVEL FEATURES (from raw method body)
# ============================================================================

ARITH_OPS_RE = re.compile(r"(\+|-|\*|/|%)(?!=)")
REL_OPS_RE = re.compile(r"(==|!=|<=|>=|<(?!<)|>(?!>))")
LOGICAL_OPS_RE = re.compile(r"(&&|\|\|)")
METHOD_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
STRING_LIT_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
NUMBER_LIT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def strip_comments(code: str) -> str:
    """Remove // and /* */ comments. Crude but adequate for feature counting."""
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return code


def extract_code_features(body: str) -> Dict[str, Any]:
    """Compute code-level features from a method body string."""
    clean = strip_comments(body)

    loc = sum(
        1 for line in clean.splitlines()
        if line.strip() and line.strip() not in ("{", "}")
    )

    # Parameter count: tokens between the first '(' and matching ')'
    num_params = 0
    paren_open = clean.find("(")
    if paren_open >= 0:
        depth = 0
        end = paren_open
        for i in range(paren_open, len(clean)):
            if clean[i] == "(":
                depth += 1
            elif clean[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        param_str = clean[paren_open + 1:end].strip()
        if param_str:
            depth = 0
            commas = 0
            for ch in param_str:
                if ch == "<":
                    depth += 1
                elif ch == ">":
                    depth -= 1
                elif ch == "," and depth == 0:
                    commas += 1
            num_params = commas + 1

    num_local_vars = len(re.findall(
        r"\b(?:int|long|short|byte|char|boolean|double|float|String|[A-Z][A-Za-z0-9_]*)\s+"
        r"[a-z_][A-Za-z0-9_]*\s*[=;,]",
        clean,
    ))

    no_strings = STRING_LIT_RE.sub('""', clean)
    return {
        "loc": loc,
        "num_params": num_params,
        "num_local_vars": num_local_vars,
        "num_method_calls": len(METHOD_CALL_RE.findall(no_strings)),
        "num_arith_ops": len(ARITH_OPS_RE.findall(no_strings)),
        "num_rel_ops": len(REL_OPS_RE.findall(no_strings)),
        "num_logical_ops": len(LOGICAL_OPS_RE.findall(no_strings)),
        "num_literals": len(STRING_LIT_RE.findall(clean)) + len(NUMBER_LIT_RE.findall(no_strings)),
    }


# ============================================================================
# TREE-CLIMBER HTTP API
# ============================================================================

def fetch_cfg(code: str, server: str) -> Optional[Dict[str, Any]]:
    """POST to /parse and return the cfg dict. Returns None on failure."""
    url = f"{server.rstrip('/')}/parse"
    try:
        response = requests.post(
            url,
            json={"source_code": code, "language": "java"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.ConnectionError:
        print(
            f"Error: could not connect to tree-climber server at {server}.\n"
            "Make sure it's running:\n"
            "  uv run -m uvicorn tree_climber.viz.app:app --reload --host 0.0.0.0 --port 8000",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"  [warn] tree-climber HTTP error: {e}", file=sys.stderr)
        return None

    data = response.json()
    if not data.get("success"):
        print(f"  [warn] tree-climber parse failed: {data.get('error')}", file=sys.stderr)
        return None

    return data.get("cfg") or {}


def wrap_method_for_parsing(body: str) -> Tuple[str, int]:
    """
    Tree-climber needs a full class. Wrap bare methods.
    Returns (wrapped_source, prefix_line_offset).
    """
    if body.strip().startswith("class "):
        return body, 0
    return f"class _Wrapper {{\n{body}\n}}", 1


# ============================================================================
# CFG FEATURE EXTRACTION (real tree-climber schema)
# ============================================================================

def byte_offset_to_line(source: str, offset: int) -> int:
    """Return 1-indexed line number for a byte offset in source."""
    return source[:offset].count("\n") + 1


def _empty_cfg_features() -> Dict[str, Any]:
    """Zero-valued features for methods whose CFG couldn't be built."""
    return {
        "num_nodes": 0,
        "num_edges": 0,
        "cyclomatic_complexity": 0,
        "num_conditions": 0,
        "num_branches": 0,
        "num_joins": 0,
        "num_loops": 0,
        "num_returns": 0,
        "max_nesting_depth": 0,
        "branch_ratio": 0.0,
        "has_loop": 0,
        "has_branching": 0,
        "num_unique_lines": 0,
    }


def _approximate_max_nesting(all_nodes: Dict, real_nodes: Dict) -> int:
    """
    BFS from ENTRY, tracking CONDITION-depth along each path.
    Returns the max nesting depth observed.

    Cycle handling: a node's nesting depth is bounded by the number of
    CONDITION nodes that lexically precede it. We use start_index to detect
    back-edges (predecessor lies after the node in source) and skip them, so
    loops don't artificially inflate depth.
    """
    entry_id = None
    for nid, n in all_nodes.items():
        if n.get("node_type") == "ENTRY":
            entry_id = nid
            break
    if entry_id is None:
        for nid, n in real_nodes.items():
            if not n.get("predecessors"):
                entry_id = nid
                break
    if entry_id is None:
        return 0

    num_conditions = sum(
        1 for n in real_nodes.values() if n.get("node_type") == "CONDITION"
    )
    if num_conditions == 0:
        return 0
    depth_cap = num_conditions  # logical upper bound

    visited_at_depth = {}
    queue = deque([(str(entry_id), 0)])
    max_depth = 0

    while queue:
        nid, depth = queue.popleft()

        if visited_at_depth.get(nid, -1) >= depth:
            continue
        visited_at_depth[nid] = depth

        node = all_nodes.get(nid)
        if node is None:
            continue

        new_depth = depth
        if node.get("node_type") == "CONDITION":
            new_depth = min(depth + 1, depth_cap)
            if new_depth > max_depth:
                max_depth = new_depth

        n_idx = node.get("start_index", -1)
        for succ in node.get("successors", []):
            succ_node = all_nodes.get(str(succ))
            # Skip back-edges (successor lies BEFORE current node in source).
            # This is the loop-tail → loop-head edge.
            if succ_node is not None:
                s_idx = succ_node.get("start_index", -1)
                if s_idx < n_idx and succ_node.get("node_type") not in SKIP_NODE_TYPES:
                    continue
            queue.append((str(succ), new_depth))

    return max_depth


def extract_cfg_features(cfg: Dict[str, Any], source: str, prefix_lines: int) -> Dict[str, Any]:
    """
    Compute CFG features from tree-climber's CFG dict.

    Real CFG schema: nodes is a dict keyed by node_id; each node has
    node_type, start_index, predecessors, successors.
    """
    nodes = cfg.get("nodes", {})
    if not nodes:
        return _empty_cfg_features()

    # Real nodes = exclude synthetic ENTRY/EXIT
    real_nodes = {
        nid: n for nid, n in nodes.items()
        if n.get("node_type", "") not in SKIP_NODE_TYPES
    }

    num_nodes = len(real_nodes)
    if num_nodes == 0:
        return _empty_cfg_features()

    # Edges between real nodes
    num_edges = 0
    for nid, n in real_nodes.items():
        for succ_id in n.get("successors", []):
            if str(succ_id) in real_nodes:
                num_edges += 1

    num_conditions = sum(1 for n in real_nodes.values() if n.get("node_type") == "CONDITION")
    num_returns = sum(1 for n in real_nodes.values() if n.get("node_type") == "RETURN")
    num_branches = sum(1 for n in real_nodes.values() if len(n.get("successors", [])) > 1)
    num_joins = sum(1 for n in real_nodes.values() if len(n.get("predecessors", [])) > 1)

    # Cyclomatic complexity: E - N + 2 for a connected CFG
    cyclomatic = max(1, num_edges - num_nodes + 2)

    # Loops via back-edges: a predecessor whose start_index is *after* the node
    num_loops = 0
    for nid, n in real_nodes.items():
        n_idx = n.get("start_index", -1)
        for pred_id in n.get("predecessors", []):
            pred = nodes.get(str(pred_id))
            if pred and pred.get("start_index", -1) > n_idx:
                num_loops += 1
                break  # one back-edge = one loop header

    max_nesting = _approximate_max_nesting(nodes, real_nodes)
    branch_ratio = num_branches / num_nodes
    has_loop = int(num_loops > 0)
    has_branching = int(num_branches > 0)

    # Unique source lines touched by CFG nodes
    unique_lines = set()
    for n in real_nodes.values():
        idx = n.get("start_index")
        if idx is not None:
            ln = byte_offset_to_line(source, idx) - prefix_lines
            if ln >= 1:
                unique_lines.add(ln)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "cyclomatic_complexity": cyclomatic,
        "num_conditions": num_conditions,
        "num_branches": num_branches,
        "num_joins": num_joins,
        "num_loops": num_loops,
        "num_returns": num_returns,
        "max_nesting_depth": max_nesting,
        "branch_ratio": round(branch_ratio, 4),
        "has_loop": has_loop,
        "has_branching": has_branching,
        "num_unique_lines": len(unique_lines),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def parse_method_signature(method_sig: str) -> Tuple[str, str, str]:
    """
    Parse "org.apache.commons.lang3.SerializationException:<init>(java.lang.String)"
    into (class_fqn, method_name, params).

    The first ':' separates class from method+params.
    Method name is everything before the first '('.
    """
    if ":" not in method_sig:
        return method_sig, "", ""
    class_fqn, rest = method_sig.split(":", 1)
    if "(" in rest:
        paren = rest.index("(")
        method_name = rest[:paren]
        params = rest[paren:]
    else:
        method_name, params = rest, ""
    return class_fqn, method_name, params


def parse_line_range(lines_str: str) -> Tuple[int, int]:
    """Parse '41-43' into (41, 43). Handles single-line '41' as (41, 41)."""
    if "-" in lines_str:
        start, end = lines_str.split("-", 1)
        return int(start), int(end)
    n = int(lines_str)
    return n, n


def parse_version_key(version_key: str) -> Tuple[str, str, str]:
    """
    Parse 'Lang-1f' into ('Lang', '1', 'f').
    'f' = fixed, 'b' = buggy. We use the fixed versions.
    """
    parts = version_key.rsplit("-", 1)
    if len(parts) != 2:
        return version_key, "", ""
    project = parts[0]
    bug_with_suffix = parts[1]
    # Find where the digits end
    i = 0
    while i < len(bug_with_suffix) and bug_with_suffix[i].isdigit():
        i += 1
    bug_id = bug_with_suffix[:i]
    suffix = bug_with_suffix[i:]
    return project, bug_id, suffix


def load_methods(methods_dir: Path) -> pd.DataFrame:
    """
    Load method JSONs in the user's schema:
        { "Lang-1f": [ {"method": "...", "lines": "41-43", "body": "..."}, ... ] }

    One file may contain multiple version keys (one per project version).
    Returns a normalized DataFrame with one row per (version, method),
    deduped on (class_fqn, method_name, start_line, end_line).
    """
    rows = []
    for json_path in sorted(methods_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"  [warn] {json_path.name}: top-level is not a dict, skipping")
            continue
        for version_key, methods_list in data.items():
            project, bug_id, suffix = parse_version_key(version_key)
            for m in methods_list:
                method_sig = m.get("method", "")
                lines_str = m.get("lines", "")
                body = m.get("body", "")

                if not method_sig or not lines_str:
                    continue

                class_fqn, method_name, params = parse_method_signature(method_sig)
                try:
                    start_line, end_line = parse_line_range(lines_str)
                except ValueError:
                    print(f"  [warn] malformed lines '{lines_str}' for {method_sig}")
                    continue

                # Synthesize a stable method_id from the version + class + name + line
                method_id = f"{version_key}_{class_fqn}_{method_name}_{start_line}"

                rows.append({
                    "method_id": method_id,
                    "version_key": version_key,
                    "project": project,
                    "bug_id": bug_id,
                    "version_suffix": suffix,
                    "class_fqn": class_fqn,
                    "method_name": method_name,
                    "params": params,
                    "start_line": start_line,
                    "end_line": end_line,
                    "body": body,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Dedupe: bridge methods (e.g., compareTo(Object) vs compareTo(MutableFloat))
    # occupy the same line range. Keep the first occurrence.
    before = len(df)
    df = df.drop_duplicates(
        subset=["version_key", "class_fqn", "method_name", "start_line", "end_line"],
        keep="first",
    )
    after = len(df)
    if before > after:
        print(f"  [info] deduped {before - after} bridge/duplicate methods "
              f"(same line range)")

    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help="tree-climber HTTP server URL (used if --cfg-dir not given)")
    parser.add_argument("--cfg-dir", type=Path, default=None,
                        help="Optional: directory of pre-built CFG JSONs "
                             "(<method_id>.json), used instead of the server")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: only process first N methods (for testing)")
    args = parser.parse_args()

    print("Loading method definitions ...")
    methods = load_methods(args.methods_dir)
    if args.limit:
        methods = methods.head(args.limit)
    print(f"  {len(methods)} methods to process")

    use_cfg_dir = args.cfg_dir is not None
    if use_cfg_dir:
        print(f"Using pre-built CFGs from {args.cfg_dir}")
    else:
        print(f"Will fetch CFGs from tree-climber server at {args.server}")

    feature_rows = []
    cfg_failures = 0
    for i, row in enumerate(methods.to_dict(orient="records"), 1):
        method_id = row.get("method_id")
        body = row.get("body", "")

        if i % 50 == 0:
            print(f"  [{i}/{len(methods)}] processed ...")

        code_feats = extract_code_features(body)

        wrapped, prefix_lines = wrap_method_for_parsing(body)

        # Get CFG either from disk or by calling the server
        if use_cfg_dir:
            cfg_path = args.cfg_dir / f"{method_id}.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
            else:
                cfg = None
        else:
            cfg = fetch_cfg(wrapped, args.server)

        if cfg is None:
            cfg_failures += 1
            cfg_feats = _empty_cfg_features()
        else:
            cfg_feats = extract_cfg_features(cfg, wrapped, prefix_lines)

        feature_rows.append({
            "method_id": method_id,
            "project": row.get("project"),
            "bug_id": row.get("bug_id"),
            "class_fqn": row.get("class_fqn"),
            "method_name": row.get("method_name"),
            "start_line": row.get("start_line"),
            "end_line": row.get("end_line"),
            **code_feats,
            **cfg_feats,
        })

    if cfg_failures > 0:
        print(f"  [warn] {cfg_failures}/{len(methods)} methods had CFG extraction "
              f"failures; their CFG features are zeroed.")

    df = pd.DataFrame(feature_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
