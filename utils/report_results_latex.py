import os
import json

def escape_latex(text):
    """Escape special characters for LaTeX."""
    replacements = {
        '_': r'\_',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    for key, val in replacements.items():
        text = text.replace(key, val)
    return text

def highlight_accuracy(accuracy_list):
    """Highlight the highest and second highest accuracy."""
    sorted_accuracies = sorted(accuracy_list, reverse=True)
    highest = sorted_accuracies[0]
    second_highest = sorted_accuracies[1] if len(sorted_accuracies) > 1 else None

    highlighted = []
    for accuracy in accuracy_list:
        if accuracy == highest:
            highlighted.append(f"\\textbf{{{accuracy:.2f}}}")
        elif accuracy == second_highest:
            highlighted.append(f"\\underline{{{accuracy:.2f}}}")
        else:
            highlighted.append(f"{accuracy:.2f}")

    return highlighted

def generate_latex_tables(results_dir="results"):
    models_metrics = []
    detailed_metrics = {}

    # Loop through each model folder
    for model_name in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        # Load the metrics.json from the "test" folder
        metrics_path = os.path.join(model_path, "test", "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"Metrics file not found for model: {model_name}")
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Store overall metrics
        models_metrics.append({
            "model": model_name,
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1_score": metrics.get("f1_score", 0.0)
        })

        # Store detailed metrics for each class
        detailed_metrics[model_name] = metrics.get("classification_report", {})

    # Sort models by accuracy
    models_metrics.sort(key=lambda x: x["accuracy"])

    # Generate overall metrics table
    overall_table = "\\begin{table}[H]\n\\centering\n\\caption{Overall Accuracy for Each Model}\n"
    overall_table += "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
    overall_table += "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1 Score} \\\\ \\hline\n"

    # Extract accuracy for sorting and highlighting
    accuracies = [model['accuracy'] for model in models_metrics]
    highlighted_accuracies = highlight_accuracy(accuracies)

    for i, model in enumerate(models_metrics):
        model_name = escape_latex(model["model"])
        accuracy = highlighted_accuracies[i]
        row = f"{model_name} & {accuracy} & {model['precision']:.2f} & {model['recall']:.2f} & {model['f1_score']:.2f} \\\\ \\hline\n"
        overall_table += row

    overall_table += "\\end{tabular}\n\\end{table}\n"

    # Create separate detailed table for each model
    detailed_tables = ""
    for model_name, class_metrics in detailed_metrics.items():
        model_name_escaped = escape_latex(model_name)

        # Initialize the detailed table for this model
        detailed_table = f"\\begin{{table}}[H]\n\\centering\n\\caption{{Detailed Metrics for {model_name_escaped}}}\n"
        detailed_table += "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
        detailed_table += "\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1 Score} & \\textbf{Support} \\\\ \\hline\n"

        # Filter out non-class entries like 'accuracy', 'macro avg', 'weighted avg'
        class_metrics_sorted = [
            (class_name, metrics)
            for class_name, metrics in class_metrics.items()
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']
        ]

        # Sort class-specific metrics based on 'f1-score'
        class_metrics_sorted = sorted(class_metrics_sorted, key=lambda x: x[1]['f1-score'], reverse=True)

        for i, (class_name, metrics) in enumerate(class_metrics_sorted):
            class_name_escaped = escape_latex(class_name)
            if i == 0:
                row = f"\\textbf{{{class_name_escaped}}} & {metrics['precision']:.2f} & {metrics['recall']:.2f} & {metrics['f1-score']:.2f} & {int(metrics['support'])} \\\\ \\hline\n"
            elif i == 1:
                row = f"\\underline{{{class_name_escaped}}} & {metrics['precision']:.2f} & {metrics['recall']:.2f} & {metrics['f1-score']:.2f} & {int(metrics['support'])} \\\\ \\hline\n"
            else:
                row = f"{class_name_escaped} & {metrics['precision']:.2f} & {metrics['recall']:.2f} & {metrics['f1-score']:.2f} & {int(metrics['support'])} \\\\ \\hline\n"
            detailed_table += row

        detailed_table += "\\end{tabular}\n\\end{table}\n"
        detailed_tables += detailed_table + "\n\n"

    # Write LaTeX tables to file
    output_path = os.path.join(results_dir, "results_tables.tex")
    with open(output_path, "w") as f:
        f.write(overall_table + "\n\n" + detailed_tables)

    print(f"LaTeX tables generated and saved to {output_path}")


if __name__ == "__main__":
    generate_latex_tables()
