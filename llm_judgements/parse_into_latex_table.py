#!/usr/bin/env python3
import re
import sys

def parse_block(block):
    """
    Parse a dataset block and return a list of rows.
    Each row is a dict with keys:
      - dataset: the dataset name (from "Dataset:")
      - provider: the provider name (used as Judge)
      - color_tuned: first percentage value (the "Color Tuned" metric)
      - untuned: second percentage value (the "Untuned" metric)
      - diff: difference (without the percent sign)
      - ci_lower: lower bound of the 95% CI
      - ci_upper: upper bound of the 95% CI
      - pvalue: the p-value
    """
    rows = []
    # Extract dataset name; assume it appears once in the block.
    dataset_match = re.search(r"Dataset:\s*(\S+)", block)
    if not dataset_match:
        return rows
    dataset = dataset_match.group(1)
    
    # Split the block on occurrences of "Provider:".
    # The first split part may contain only the dataset header.
    parts = re.split(r"Provider:", block)
    for part in parts[1:]:
        # Skip any block that starts with "Agreement", which is not a provider row.
        if part.strip().startswith("Agreement"):
            continue
        
        # The provider name is the first token in the block.
        prov_match = re.match(r"\s*(\S+)", part)
        if not prov_match:
            continue
        provider = prov_match.group(1)

        # Find the two percentage scores.
        # They follow the model names and are in the format: ": <number>% 1s"
        scores = re.findall(r":\s*([\d.]+)%\s*1s", part)
        if len(scores) < 2:
            continue
        color_tuned, untuned = scores[0], scores[1]

        # Find the Difference field and extract the difference, CI bounds and p-value.
        diff_match = re.search(
            r"Difference:\s*([-\d.]+)%\s*\(95% CI:\s*\[([-\d.]+)%,\s*([-\d.]+)%\]\),\s*p-value:\s*([\d.]+)",
            part
        )
        if not diff_match:
            continue
        diff, ci_lower, ci_upper, pvalue = diff_match.groups()
        
        rows.append({
            'dataset': dataset,
            'provider': provider,
            'color_tuned': color_tuned,
            'untuned': untuned,
            'diff': diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pvalue': pvalue,
        })
    return rows

def generate_latex_table(rows):
    """
    Generate the LaTeX table using the rows data.
    """
    header = r"""\begin{table}[h]
\centering
\caption{Appropriateness of Model Outputs: Binary Yes/No Judgment}
\label{tab:model_performance}
\begin{tabular}{ll S[table-format=3.2] S[table-format=3.2] >{\raggedleft\arraybackslash}p{3cm} S[table-format=1.4]}
\toprule
Dataset & Judge & \multicolumn{1}{c}{\shortstack{Fine Tuned\\(\% Good)}} & \multicolumn{1}{c}{\shortstack{Untuned\\(\% Good)}} & \multicolumn{1}{c}{\shortstack{Diff\\(\% [95\% CI])}} & \multicolumn{1}{c}{p-value} \\
\midrule
"""
    footer = r"""\bottomrule
\end{tabular}
\end{table}"""
    
    body = ""
    for row in rows:
        # Format the difference column as "diff [ci_lower, ci_upper]"
        diff_str = f"{row['diff']} [{row['ci_lower']}, {row['ci_upper']}]"
        # Bold the p-value.
        pvalue_str = f"\\textbf{{{row['pvalue']}}}"
        # Construct the table row.
        body += f"{row['dataset']} & {row['provider']} & {row['color_tuned']} & {row['untuned']} & \\multicolumn{{1}}{{r}}{{{diff_str}}} & {pvalue_str} \\\\\n"
    
    return header + body + footer

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_into_latex_table.py input_file.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split the file content into blocks using a line of dashes as separator.
    blocks = re.split(r"-{10,}", content)
    all_rows = []
    for block in blocks:
        block = block.strip()
        if not block:
            print("Skipping empty block.")
            continue
        rows = parse_block(block)
        all_rows.extend(rows)
    latex_table = generate_latex_table(all_rows)
    print(latex_table)

if __name__ == "__main__":
    main()
