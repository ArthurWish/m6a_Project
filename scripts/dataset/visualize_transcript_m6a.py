#!/usr/bin/env python3
import argparse
import csv
import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_transcript_rows(csv_path, transcript_id):
    with open(csv_path, "r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV is empty.")
        index = {name: i for i, name in enumerate(header)}
        required = ["transcriptId", "full_sequence", "m6A_position_index", "label"]
        for name in required:
            if name not in index:
                raise ValueError(f"Missing column: {name}")

        rows = []
        for row in reader:
            if not row:
                continue
            if row[index["transcriptId"]] == transcript_id:
                rows.append(row)
        return header, index, rows


def build_label_map(rows, index):
    label_map = {}
    for row in rows:
        try:
            pos = int(row[index["m6A_position_index"]])
        except ValueError:
            continue
        try:
            label = int(row[index["label"]])
        except ValueError:
            label = 0
        if pos in label_map and label_map[pos] != label:
            label_map[pos] = 2
        else:
            label_map[pos] = label
    return label_map


def render_sequence_html(sequence, label_map, line_length):
    lines = []
    seq_len = len(sequence)
    for start in range(0, seq_len, line_length):
        end = min(seq_len, start + line_length)
        prefix = f"{start:>9d} "
        line = [f'<span class="pos">{prefix}</span>']
        for idx in range(start, end):
            base = html.escape(sequence[idx])
            label = label_map.get(idx)
            if label == 1:
                line.append(f'<span class="label-1">{base}</span>')
            elif label == 0:
                line.append(f'<span class="label-0">{base}</span>')
            elif label == 2:
                line.append(f'<span class="label-conflict">{base}</span>')
            else:
                line.append(base)
        lines.append("".join(line))
    return "\n".join(lines)


def write_html(
    out_path,
    transcript_id,
    sequence,
    label_map,
    rows,
    index,
    line_length,
):
    seq_len = len(sequence)
    pos_table = sorted(
        [(int(r[index["m6A_position_index"]]), r[index["label"]]) for r in rows],
        key=lambda item: item[0],
    )
    pos_rows = []
    for pos, label in pos_table:
        pos_rows.append(
            f"<tr><td>{pos}</td><td>{html.escape(str(label))}</td></tr>"
        )
    sequence_html = render_sequence_html(sequence, label_map, line_length)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>m6A YTH Visualization: {html.escape(transcript_id)}</title>
<style>
body {{
  margin: 24px;
  font-family: "Courier New", monospace;
  background: #f5f1e6;
  color: #2b2b2b;
}}
h1 {{
  margin: 0 0 8px 0;
  font-size: 20px;
}}
.meta {{
  margin-bottom: 16px;
  font-size: 13px;
}}
.legend span {{
  padding: 2px 6px;
  border-radius: 3px;
  margin-right: 8px;
  display: inline-block;
}}
.label-1 {{
  background: #f26a6a;
  color: #1b1b1b;
}}
.label-0 {{
  background: #7aa7f7;
  color: #1b1b1b;
}}
.label-conflict {{
  background: #f5c542;
  color: #1b1b1b;
}}
.pos {{
  color: #7b7b7b;
}}
.seq {{
  white-space: pre;
  font-size: 12px;
  line-height: 1.45;
  background: #fffaf2;
  border: 1px solid #e6ddc8;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
}}
table {{
  border-collapse: collapse;
  margin-top: 16px;
  font-size: 12px;
}}
th, td {{
  border: 1px solid #e0d7c2;
  padding: 4px 8px;
}}
th {{
  background: #efe5cc;
}}
</style>
</head>
<body>
  <h1>Transcript {html.escape(transcript_id)}</h1>
  <div class="meta">Length: {seq_len} bases | m6A sites: {len(label_map)}</div>
  <div class="legend">
    <span class="label-1">label=1 (YTH)</span>
    <span class="label-0">label=0 (non-YTH)</span>
    <span class="label-conflict">conflict</span>
  </div>
  <div class="seq">{sequence_html}</div>
  <table>
    <thead>
      <tr><th>m6A position (0-based)</th><th>label</th></tr>
    </thead>
    <tbody>
      {''.join(pos_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    with open(out_path, "w") as out_handle:
        out_handle.write(html_doc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize full sequence and m6A sites for a transcript."
    )
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "data/processed/m6A_YTH_dataset.csv"),
        help="Input dataset CSV with full_sequence and labels.",
    )
    parser.add_argument(
        "--transcript-id",
        required=True,
        help="Transcript ID to visualize (e.g., ENST00000366695).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output HTML path (default: visualize_<transcriptId>.html).",
    )
    parser.add_argument(
        "--line-length",
        type=int,
        default=100,
        help="Number of bases per line in the visualization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    header, index, rows = load_transcript_rows(args.csv, args.transcript_id)
    if not rows:
        raise SystemExit(f"No rows found for transcript {args.transcript_id}.")

    sequences = {row[index["full_sequence"]] for row in rows if row}
    if len(sequences) > 1:
        print(
            f"Warning: multiple full_sequence values for {args.transcript_id}. "
            "Using the longest sequence.",
            file=sys.stderr,
        )
    sequence = max(sequences, key=len)

    label_map = build_label_map(rows, index)
    out_path = args.out or str(
        REPO_ROOT / "outputs/reports" / f"visualize_{args.transcript_id}.html"
    )
    write_html(
        out_path,
        args.transcript_id,
        sequence,
        label_map,
        rows,
        index,
        args.line_length,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
