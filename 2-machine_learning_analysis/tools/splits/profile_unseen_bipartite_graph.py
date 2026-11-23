import argparse
from pathlib import Path
from typing import Tuple

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT.parent / "1-data" / "processed" / "graphs" / "model5" / "metadata.csv"
OUT_DIR = ROOT / "cache" / "splits" / "profile"


def build_graph(df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    pep_nodes = "pep::" + df["peptide_group"].astype(str)
    prot_nodes = "prot::" + df["protein_group"].astype(str)
    graph.add_nodes_from(pep_nodes.unique(), bipartite="peptide")
    graph.add_nodes_from(prot_nodes.unique(), bipartite="protein")
    graph.add_edges_from(zip(pep_nodes, prot_nodes))
    return graph


def node_counts(component: set) -> Tuple[int, int]:
    pep = sum(1 for node in component if str(node).startswith("pep::"))
    prot = sum(1 for node in component if str(node).startswith("prot::"))
    return pep, prot


def profile_graph(df: pd.DataFrame, graph_name: str, out_dir: Path) -> dict:
    graph = build_graph(df)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    component_id = {}
    for idx, component in enumerate(components, start=1):
        for node in component:
            component_id[node] = idx

    edge_component = []
    for pep, prot in zip(df["peptide_group"].astype(str), df["protein_group"].astype(str)):
        edge_component.append(component_id[f"pep::{pep}"])
    df_comp = df.copy()
    df_comp["component_id"] = edge_component

    rows = []
    for idx, component in enumerate(components, start=1):
        pep_n, prot_n = node_counts(component)
        sub = df_comp[df_comp["component_id"] == idx]
        rows.append(
            {
                "graph": graph_name,
                "component_id": idx,
                "nodes": len(component),
                "peptide_nodes": pep_n,
                "protein_nodes": prot_n,
                "rows": len(sub),
                "positive_rows": int(sub["label"].sum()),
                "negative_rows": int(len(sub) - sub["label"].sum()),
                "unique_edges": int(sub[["peptide_group", "protein_group"]].drop_duplicates().shape[0]),
            }
        )
    component_df = pd.DataFrame(rows)
    component_df.to_csv(out_dir / f"graph_components_{graph_name}.csv", index=False)

    pep_degrees = [deg for node, deg in graph.degree() if str(node).startswith("pep::")]
    prot_degrees = [deg for node, deg in graph.degree() if str(node).startswith("prot::")]
    largest = component_df.iloc[0] if len(component_df) else None

    def mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "graph": graph_name,
        "rows": int(len(df)),
        "positive_rows": int(df["label"].sum()),
        "negative_rows": int(len(df) - df["label"].sum()),
        "unique_peptide_sequences": int(df["peptide_group"].nunique()),
        "unique_protein_sequences": int(df["protein_group"].nunique()),
        "unique_edges": int(df[["peptide_group", "protein_group"]].drop_duplicates().shape[0]),
        "components": int(len(components)),
        "giant_component_nodes": int(largest["nodes"]) if largest is not None else 0,
        "giant_component_rows": int(largest["rows"]) if largest is not None else 0,
        "giant_component_node_fraction": float(largest["nodes"] / max(1, graph.number_of_nodes())) if largest is not None else 0.0,
        "giant_component_row_fraction": float(largest["rows"] / max(1, len(df))) if largest is not None else 0.0,
        "peptide_degree_max": int(max(pep_degrees)) if pep_degrees else 0,
        "peptide_degree_mean": mean(pep_degrees),
        "protein_degree_max": int(max(prot_degrees)) if prot_degrees else 0,
        "protein_degree_mean": mean(prot_degrees),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile RKI peptide-protein bipartite graph.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--peptide-col", default="clean_pep_seq")
    parser.add_argument("--protein-col", default="clean_prot_seq")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    for col in [args.peptide_col, args.protein_col, "label"]:
        if col not in df.columns:
            raise ValueError(f"Required column not found: {col}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean = df.dropna(subset=[args.peptide_col, args.protein_col, "label"]).copy()
    clean["label"] = pd.to_numeric(clean["label"], errors="raise").astype(int)
    clean["peptide_group"] = clean[args.peptide_col].astype(str)
    clean["protein_group"] = clean[args.protein_col].astype(str)

    summaries = [
        profile_graph(clean[clean["label"] == 1].copy(), "positive_only", args.out_dir),
        profile_graph(clean.copy(), "all_observed", args.out_dir),
    ]
    summary_df = pd.DataFrame(summaries)
    summary_path = args.out_dir / "graph_profile_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {args.out_dir / 'graph_components_positive_only.csv'}")
    print(f"Saved: {args.out_dir / 'graph_components_all_observed.csv'}")


if __name__ == "__main__":
    main()
