import streamlit as st
import os
import json
import plotly.express as px
import plotly.graph_objects as go

RESULTS_DIR = "results"

def load_all_results():
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

    all_results = load_all_results()
    if not all_results:
        st.warning(f"No JSON files found in {RESULTS_DIR}. Please run a benchmark first.")
        return

    # =========================
    # SELECT FILES
    # =========================
    file_options = [res["filename"] for res in all_results]
    selected_files = st.multiselect("Select result file(s):", file_options, default=file_options[:1])

    if not selected_files:
        st.info("No result file selected.")
        return

    selected_data = [res for res in all_results if res["filename"] in selected_files]

    st.subheader("Selected Runs")
    for data in selected_data:
        with st.expander(f"File: {data['filename']}"):
            # Show basic info
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

            # Plot execution times per run
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

            # =========================
            # NEW: MEMORY USAGE GRAPHS
            # =========================
            # GPU Memory Diff
            gpu_mem = data["all_runs"]["gpu_mem_diff_MB"]
            if any(val != 0 for val in gpu_mem):
                fig_gpu = px.bar(
                    x=run_indices,
                    y=gpu_mem,
                    labels={"x": "Run #", "y": "GPU Memory Diff (MB)"},
                    title=f"GPU Memory Usage per Run: {data['filename']}"
                )
                st.plotly_chart(fig_gpu)
            else:
                st.write("GPU Memory Diff: all zeros (CPU/MPS or no recorded usage).")

            # CPU Memory Diff
            cpu_mem = data["all_runs"]["cpu_mem_diff_MB"]
            fig_cpu = px.bar(
                x=run_indices,
                y=cpu_mem,
                labels={"x": "Run #", "y": "CPU Memory Diff (MB)"},
                title=f"CPU Memory Usage per Run: {data['filename']}"
            )
            st.plotly_chart(fig_cpu)

    # =========================
    # OPTIONAL: COMBINED CHART
    # =========================
    if len(selected_data) > 1:
        st.subheader("Combined Execution Times (all selected)")
        fig_combined = px.line()
        for data in selected_data:
            run_times = data["all_runs"]["times_sec"]
            run_indices = list(range(1, len(run_times) + 1))
            label = f"{data['filename']} ({data['model']} - {data['device']})"
            fig_combined.add_scatter(x=run_indices, y=run_times, mode='lines+markers', name=label)

        fig_combined.update_layout(title="Combined Execution Times", xaxis_title="Run #", yaxis_title="Time (s)")
        st.plotly_chart(fig_combined)

    # =========================
    # BASELINE VS HALF PRECISION
    # =========================
    st.header("Baseline vs. Half-Precision Comparison")

    pairs = []
    for data1 in selected_data:
        for data2 in selected_data:
            if data1 is not data2:
                if (data1["model"] == data2["model"] and
                    data1["batch_size"] == data2["batch_size"] and
                    data1["num_runs"] == data2["num_runs"]):
                    if data1.get("use_half_precision", False) != data2.get("use_half_precision", False):
                        pairs.append((data1, data2))

    if not pairs:
        st.info("No matching baseline vs. half-precision pairs in selection.")
    else:
        baseline, half = pairs[0]
        st.write(f"**Comparing**:\n- Baseline: {baseline['filename']}\n- Half: {half['filename']}")
        st.write({
            "Baseline Exec Time": baseline["avg_exec_time_sec"],
            "Half Exec Time": half["avg_exec_time_sec"],
            "Baseline GPU Mem (MB)": baseline["avg_gpu_memory_diff_MB"],
            "Half GPU Mem (MB)": half["avg_gpu_memory_diff_MB"]
        })

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=["Baseline", "Half Precision"],
            y=[baseline["avg_exec_time_sec"], half["avg_exec_time_sec"]],
            name="Exec Time (s)"
        ))
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

    # =========================
    # COST ESTIMATION SECTION
    # =========================
    st.header("Estimated Cost of Inference")
    # For demonstration, let's assume:
    # GPU cost: $0.50 / hour, CPU cost: $0.05 / hour
    # We'll show a table for each selected file.

    GPU_HOURLY_COST = 0.50
    CPU_HOURLY_COST = 0.05

    st.write("Approx. cost calculation assumes a single GPU or CPU usage for the entire run time. (Demo Only)")

    cost_table = []
    for data in selected_data:
        device = data["device"]
        # We'll take average exec time * num_runs, though your code runs them consecutively
        total_time_s = sum(data["all_runs"]["times_sec"])
        total_time_h = total_time_s / 3600.0

        if device == "cuda":
            cost = total_time_h * GPU_HOURLY_COST
        elif device == "cpu" or device == "mps":
            cost = total_time_h * CPU_HOURLY_COST
        else:
            cost = 0.0  # fallback

        cost_table.append({
            "filename": data["filename"],
            "device": device,
            "model": data["model"],
            "batch_size": data["batch_size"],
            "num_runs": data["num_runs"],
            "total_exec_time_s": total_time_s,
            "estimated_cost_$": round(cost, 4)
        })
    st.write(cost_table)

if __name__ == "__main__":
    main()

