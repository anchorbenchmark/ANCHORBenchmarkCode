import math
import os
import json
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm
import tiktoken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def assign_percentile_score(tc, quantiles):
    """Assign a 1-5 conciseness score given token count and 4 quantile boundaries."""
    if tc <= quantiles[0]:
        return 5
    elif tc <= quantiles[1]:
        return 4
    elif tc <= quantiles[2]:
        return 3
    elif tc <= quantiles[3]:
        return 2
    else:
        return 1

def process_file(data_path: str, output_base_dir: str):
    data_name = Path(data_path).stem
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    data_size = len(data)
    
    # We use cl100k_base which is standard for most modern OpenAI models
    enc = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    token_counts = []
    
    for item in tqdm(data, desc=f"Processing {data_name}"):
        output_text = item.get('llm_output', '')
        if not isinstance(output_text, str):
            output_text = str(output_text)
            
        tokens = enc.encode(output_text)
        token_count = len(tokens)
        
        item['token_conciseness_count'] = token_count
        token_counts.append(token_count)
        total_tokens += token_count

    # Within-model percentile scores (not comparable across models)
    if token_counts:
        quantiles = np.percentile(token_counts, [20, 40, 60, 80])
        for item in data:
            item['token_conciseness_score_within_model'] = assign_percentile_score(
                item['token_conciseness_count'], quantiles
            )
    else:
        quantiles = []
        
    stat_info = {
        "Total_Items": data_size,
        "Total_Tokens": total_tokens,
        "Average_Tokens": total_tokens / data_size if data_size > 0 else 0,
        "Max_Tokens": max(token_counts) if token_counts else 0,
        "Min_Tokens": min(token_counts) if token_counts else 0
    }
    
    info = {
        'dataset': data_name,
        'data_num': data_size,
        'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'tiktoken_cl100k_base'
    }
    
    output_subdir = os.path.join(output_base_dir, "token_conciseness")
    os.makedirs(output_subdir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if token_counts:
        sns.set_theme(style="white", context="notebook")
        plt.figure(figsize=(14, 8))
        
        # Use a reasonable number of bins based on data
        num_bins = max(15, min(50, len(set(token_counts))))
        
        # Create a beautiful histogram + KDE plot
        ax = sns.histplot(
            token_counts, bins=num_bins, kde=True, 
            color="#2c3e50", edgecolor="white", alpha=0.5,
            line_kws={'linewidth': 2.5, 'color': '#2c3e50'}
        )
        
        # Aesthetics
        plt.title(f'Output Token Lengths & Conciseness Scoring\n{data_name}', fontsize=18, pad=20, fontweight='bold', color="#2c3e50")
        plt.xlabel('Token Count', fontsize=14, labelpad=15, fontweight='bold', color="#34495e")
        plt.ylabel('Number of Samples', fontsize=14, labelpad=15, fontweight='bold', color="#34495e")
        
        # Define region edges: min=0, 20th, 40th, 60th, 80th, max+padding
        max_val = max(token_counts) * 1.05
        edges = [0] + list(quantiles) + [max_val]
        
        # Clean visually appealing palette for the 5 score regions
        region_colors = sns.color_palette("Set3", 5)
        
        for i in range(5):
            score_val = 5 - i
            if i == 0:
                label = f'Score 5 (Most Concise): ≤ {edges[1]:.0f} tokens'
            elif i == 4:
                label = f'Score 1 (Least Concise): > {edges[4]:.0f} tokens'
            else:
                label = f'Score {score_val}: {edges[i]:.0f} - {edges[i+1]:.0f} tokens'
                
            plt.axvspan(edges[i], edges[i+1], color=region_colors[i], alpha=0.5, label=label)
            
            # Add subtle dashed lines at boundaries
            if i < 4:
                plt.axvline(edges[i+1], color='#7f8c8d', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Add legend
        plt.legend(title='Conciseness Percentile Ranges', title_fontsize='13', fontsize='11', 
                   loc='upper right', framealpha=0.95, edgecolor='#cccccc')
        
        # Clean up axes
        sns.despine(left=True, bottom=True)
        ax.grid(axis='y', linestyle='-', alpha=0.15)
        ax.grid(axis='x', visible=False)
        
        # Adjust layout
        plt.tight_layout()
        
        plot_name = f'Eval_Token_Conciseness_{data_name}_{data_size}_{timestamp}.png'
        plot_path = os.path.join(output_subdir, plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[{data_name}] Saved distribution plot to {plot_path}")

    output_name = f'Eval_Token_Conciseness_{data_name}_{data_size}_{timestamp}.json'
    output_path = os.path.join(output_subdir, output_name)
    
    output_data = {
        'info': info,
        'stat_info': stat_info,
        'results': data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"[{data_name}] Statistics: {stat_info}")
    print(f"[{data_name}] Saved to {output_path}")
    
    # Return stats for the CSV summary
    stat_summary = stat_info.copy()
    stat_summary["file_name"] = data_name
    return stat_summary, token_counts, data, output_path

def main():
    parser = argparse.ArgumentParser(description="Calculate conciseness via token count instead of using LLM judge.")
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        default="./CoT_eval_qwen_omni_only/",
        help="Path to directory containing input JSON files. Default: ./CoT_eval_qwen_omni_only/",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./CoT_eval_results_qwen_omni_only/",
        help="Base output directory for results. Default: ./CoT_eval_results_qwen_omni_only/",
    )
    args = parser.parse_args()

    print(f"Data: {args.data_path} | Output: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.data_path):
        print(f"Data directory {args.data_path} does not exist.")
        return

    files = [f for f in os.listdir(args.data_path) if f.endswith(".json")]
    if not files:
        print(f"No JSON files found in {args.data_path}")
        return
        
    all_stats = []
    all_token_counts = {}
    all_data_by_file = {}   # model_name -> (data list, output_path)
    
    for file in files:
        full_path = os.path.join(args.data_path, file)
        stats, token_counts, data, output_path = process_file(full_path, args.output_dir)
        all_stats.append(stats)
        all_token_counts[stats["file_name"]] = token_counts
        all_data_by_file[stats["file_name"]] = (data, output_path)

    # --- Global percentile scores (comparable across models) ---
    # Pool all token counts from all models to compute shared quintile boundaries
    all_counts_flat = [tc for counts in all_token_counts.values() for tc in counts]
    if all_counts_flat:
        global_quantiles = np.percentile(all_counts_flat, [20, 40, 60, 80])
        print(f"\nGlobal quintile boundaries: {[f'{q:.0f}' for q in global_quantiles]}")

        for model_name, (data, output_path) in all_data_by_file.items():
            for item in data:
                item['token_conciseness_score_global'] = assign_percentile_score(
                    item['token_conciseness_count'], global_quantiles
                )
            # Compute average global conciseness score for this model
            avg_global_score = np.mean([item['token_conciseness_score_global'] for item in data])

            # Re-save the JSON with both within-model and global scores
            with open(output_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            saved['results'] = data
            saved['info']['global_quantile_boundaries'] = list(global_quantiles)
            saved['stat_info']['Avg_Conciseness_Score_Global'] = round(avg_global_score, 4)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(saved, f, ensure_ascii=False, indent=4)
            print(f"[{model_name}] Avg global conciseness score: {avg_global_score:.4f}")
            print(f"[{model_name}] Re-saved with global percentile scores -> {output_path}")

            # Propagate into all_stats so it appears in the CSV summary
            for s in all_stats:
                if s["file_name"] == model_name:
                    s["Avg_Conciseness_Score_Global"] = round(avg_global_score, 4)
                    break
        
    if all_token_counts:
        sns.set_theme(style="whitegrid", context="notebook")
        
        plot_data = []
        for model_name, counts in all_token_counts.items():
            for c in counts:
                plot_data.append({"Model": model_name, "Token Count": c})
        df_plot = pd.DataFrame(plot_data)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_dir = os.path.join(args.output_dir, "token_conciseness")
        os.makedirs(combined_dir, exist_ok=True)
        
        # Combined KDE Plot
        plt.figure(figsize=(16, 10))
        sns.kdeplot(data=df_plot, x="Token Count", hue="Model", fill=True, alpha=0.3, linewidth=2, palette="husl")
        plt.title('Combined Output Token Length Distributions', fontsize=20, pad=20, fontweight='bold', color="#2c3e50")
        plt.xlabel('Token Count', fontsize=16, labelpad=15, fontweight='bold', color="#34495e")
        plt.ylabel('Density', fontsize=16, labelpad=15, fontweight='bold', color="#34495e")
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        combined_kde_path = os.path.join(combined_dir, f"Eval_Token_Conciseness_Combined_KDE_{timestamp}.png")
        plt.savefig(combined_kde_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # --- Publication-ready Ridge Plot (Joyplot) ---
        # Highly recommended for comparing many distributions in research papers
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        
        order = df_plot.groupby("Model")["Token Count"].median().sort_values().index
        palette = sns.color_palette("crest", n_colors=len(order))
        
        # Reduced aspect ratio (5 vs 8) for a more compact, less wide figure
        g = sns.FacetGrid(df_plot, row="Model", hue="Model", aspect=5, height=1.2,
                          palette=palette, row_order=order, hue_order=order)
        
        # Draw the densities: log_scale=True handles the long tails perfectly
        g.map_dataframe(sns.kdeplot, x="Token Count", fill=True, alpha=0.85, linewidth=1.5, log_scale=True)
        g.map_dataframe(sns.kdeplot, x="Token Count", color='white', linewidth=2, log_scale=True)
        
        # Add a solid baseline for each ridge
        g.refline(y=0, linewidth=2, linestyle="-", color="black", clip_on=False)
        
        # Overlap the ridges; left/right margins trimmed to remove excess whitespace
        g.figure.subplots_adjust(hspace=-0.5, left=0.02, right=0.98)
        
        # Clean up axes, ticks, and add model names directly onto the plot
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        
        import matplotlib.ticker as ticker
        
        for ax, model_name in zip(g.axes.flat, order):
            model_name = model_name.replace("_vs_gt", "")
            # Place the model name text on the left side of each ridge
            ax.text(0.01, 0.15, model_name, fontweight="bold", color="#333333", 
                    ha="left", va="center", transform=ax.transAxes, fontsize=14)
            
            # Make the log scale ticks much more fine-grained and readable
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
            
            # Add subtle grid lines for the x-axis
            ax.xaxis.grid(True, linestyle='--', alpha=0.5)
            
        # Set global x-label on the bottom-most axis
        g.axes[-1, 0].set_xlabel('Token Count (Log Scale, Fine-Grained)', fontsize=16, fontweight='bold', labelpad=15)
        
        g.figure.suptitle('Model Conciseness & Token Distributions', fontweight='bold', fontsize=20, y=0.98)
        
        combined_ridge_png = os.path.join(combined_dir, f"Eval_Token_Conciseness_Combined_Ridge_{timestamp}.png")
        combined_ridge_pdf = os.path.join(combined_dir, f"Eval_Token_Conciseness_Combined_Ridge_{timestamp}.pdf")
        
        g.figure.savefig(combined_ridge_png, dpi=400, bbox_inches='tight', facecolor='white')
        g.figure.savefig(combined_ridge_pdf, bbox_inches='tight', facecolor='white')
        plt.close(g.figure)
        
        print(f"\nSaved combined plots to:\n- {combined_kde_path}\n- {combined_ridge_png}\n- {combined_ridge_pdf}")

        # --- Publication-ready Stacked Overlapping KDE Plot ---
        # 3 rows, each row overlays 3 model distributions in different colors
        import matplotlib.ticker as ticker

        sns.set_theme(style="white", context="paper", font_scale=1.2)

        models_per_row = 3
        ordered_models = list(order)
        n_rows = max(1, math.ceil(len(ordered_models) / models_per_row))
        # Use a high-contrast qualitative palette so overlapping colors are distinct
        palette_grid = sns.color_palette("Set2", n_colors=models_per_row)

        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), constrained_layout=True)
        if n_rows == 1:
            axes = [axes]

        for row_idx in range(n_rows):
            ax = axes[row_idx]
            row_models = ordered_models[row_idx * models_per_row : (row_idx + 1) * models_per_row]

            if not row_models:
                ax.set_visible(False)
                continue

            for col_idx, model_name in enumerate(row_models):
                counts = all_token_counts[model_name]
                color = palette_grid[col_idx]
                display_name = model_name.replace("_vs_gt", "")

                sns.kdeplot(
                    counts, ax=ax, log_scale=True,
                    fill=True, alpha=0.35, linewidth=2.0,
                    color=color, label=display_name,
                )
                # Solid line on top for crisp edge
                sns.kdeplot(
                    counts, ax=ax, log_scale=True,
                    fill=False, alpha=0.9, linewidth=2.0,
                    color=color,
                )

            # X axis: fine-grained log ticks
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
            ax.tick_params(axis="x", labelsize=9, rotation=20)

            ax.set_ylabel("Density", fontsize=10, color="#34495e")
            ax.set_xlabel("")
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.grid(axis="y", visible=False)
            sns.despine(ax=ax, left=True, bottom=False)
            ax.set_yticks([])

            # Per-row legend, placed inside top-right
            ax.legend(title=None, fontsize=9, framealpha=0.9,
                      edgecolor="#cccccc", loc="upper right")

        axes[-1].set_xlabel("Token Count (Log Scale)", fontsize=12,
                            fontweight="bold", color="#34495e", labelpad=10)

        # No suptitle — kept minimal for paper embedding

        stacked_png = os.path.join(combined_dir, f"Eval_Token_Conciseness_Stacked_{timestamp}.png")
        stacked_pdf = os.path.join(combined_dir, f"Eval_Token_Conciseness_Stacked_{timestamp}.pdf")
        fig.savefig(stacked_png, dpi=400, bbox_inches="tight", facecolor="white")
        fig.savefig(stacked_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"- {stacked_png}\n- {stacked_pdf}")

    if all_stats:
        # Move file_name to the front of the columns
        df = pd.DataFrame(all_stats)
        cols = ['file_name'] + [c for c in df.columns if c != 'file_name']
        df = df[cols]
        
        csv_path = os.path.join(args.output_dir, "token_conciseness_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nAll results saved to: {csv_path} ({len(all_stats)} entries)")

if __name__ == "__main__":
    main()