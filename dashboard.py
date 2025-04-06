import streamlit as st
import os
import json
import plotly.express as px
import plotly.graph_objects as go

RESULTS_DIR = "results"

def load_all_results():
    """
    Returns a list of result dicts loaded from all .json files in the results/ folder.
    """
    results = []
    if not os.path.exists(RESULTS_DIR):
        return results

    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".json"):
            path = os.path.join(RESULTS_DIR, fname)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    data["filename"] = fname
                    results.append(data)
            except:
                pass
    return results

def main():
    st.title("AI Benchmark Dashboard")

    # Load all benchmark result files
    all_results = load_all_results()
    if not all_results:
        st.warning(f"No JSON files found in {RESULTS_DIR}. Please run a benchmark first.")
        return

    # Let the user choose which result files to visualize
    file_options = [res["filename"] for res in all_results]
    selected_files = st.multiselect("Select result file(s):", file_options, default=file_options[:1])

    if not selected_files:
        st.info("No result file selected.")
        return

    # selected_data is the subset of loaded results that the user wants to explore
    selected_data = [res for res in all_results if res["filename"] in selected_files]

    # --- Display each selected JSON in an expander ---------------------------------
    st.subheader("Selected Runs")
    for data in selected_data:
        with st.expander(f"File: {data['filename']}"):
            st.write({
                "Timestamp": data.get("timestamp", "N/A"),
                "Model": data["model"],
                "Batch Size": data["batch_size"],
                "Device": data["device"],
                "Num Runs": data["num_runs"],
                "Use Half Precision": data.get("use_half_precision", False),
                "Avg Exec Time (s)": data["avg_exec_time_sec"],
                "Avg GPU Mem Diff (MB)": data["avg_gpu_memory_diff_MB"],
                "Avg CPU Mem Diff (MB)": data["avg_cpu_memory_diff_MB"],
            })

            # Create line plots of each run's times
            run_times = data["all_runs"]["times_sec"]
            run_indices = list(range(1, len(run_times) + 1))
            fig = px.line(
                x=run_indices,
                y=run_times,
                markers=True,
                title=f"Execution Time per Run: {data['filename']}",
                labels={"x": "Run #", "y": "Time (s)"}
            )
            st.plotly_chart(fig)

    # --- Combined Chart for All Selected Runs (Optional) ---------------------------
    if len(selected_data) > 1:
        st.subheader("Combined Chart (all selected runs)")

        fig_combined = px.line()
        for data in selected_data:
            run_times = data["all_runs"]["times_sec"]
            run_indices = list(range(1, len(run_times) + 1))
            label = f"{data['filename']} ({data['model']} - {data['device']})"
            fig_combined.add_scatter(x=run_indices, y=run_times, mode='lines+markers', name=label)

        fig_combined.update_layout(
            title="Combined Execution Times",
            xaxis_title="Run #",
            yaxis_title="Time (s)"
        )
        st.plotly_chart(fig_combined)

    # --- Baseline vs. Half-Precision Comparison ------------------------------------
    st.header("Baseline vs. Half-Precision Comparison")

    # We'll look for pairs in 'selected_data' that have the same model/batch/num_runs
    # but differ in the 'use_half_precision' field.
    pairs = []
    for data1 in selected_data:
        for data2 in selected_data:
            if data1 is not data2:
                if (data1["model"] == data2["model"] and
                    data1["batch_size"] == data2["batch_size"] and
                    data1["num_runs"] == data2["num_runs"]):
                    # Found a pair with same config, differs only in "use_half_precision"
                    if data1.get("use_half_precision", False) != data2.get("use_half_precision", False):
                        pairs.append((data1, data2))

    if not pairs:
        st.info("No matching baseline vs. half-precision pairs in the selection.")
    else:
        # For simplicity, just display the first found pair
        baseline, half = pairs[0]

        st.write(f"**Comparing**:\n- Baseline: {baseline['filename']}\n- Half-Precision: {half['filename']}")

        # Show average exec time & memory usage side by side
        st.write({
            "Baseline Exec Time": baseline["avg_exec_time_sec"],
            "Half Exec Time": half["avg_exec_time_sec"],
            "Baseline GPU Mem (MB)": baseline["avg_gpu_memory_diff_MB"],
            "Half GPU Mem (MB)": half["avg_gpu_memory_diff_MB"]
        })

        fig_compare = go.Figure()

        # Bar for Exec Time
        fig_compare.add_trace(go.Bar(
            x=["Baseline", "Half Precision"],
            y=[baseline["avg_exec_time_sec"], half["avg_exec_time_sec"]],
            name="Exec Time (s)"
        ))

        # Bar for GPU Mem
        fig_compare.add_trace(go.Bar(
            x=["Baseline", "Half Precision"],
            y=[baseline["avg_gpu_memory_diff_MB"], half["avg_gpu_memory_diff_MB"]],
            name="GPU Mem Diff (MB)",
            yaxis="y2"
        ))

        fig_compare.update_layout(
            title="Baseline vs. Half-Precision",
            xaxis=dict(title="Mode"),
            yaxis=dict(title="Exec Time (s)", side="left"),
            yaxis2=dict(title="GPU Mem Diff (MB)", overlaying="y", side="right"),
            barmode='group'
        )

        st.plotly_chart(fig_compare)

if __name__ == "__main__":
    main()
