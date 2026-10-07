#!/usr/bin/env python3
"""Analyze validated CSVs using the actual V1 and vendor baselines."""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def find_csv():
    for p in (ROOT/"results/target_results.csv", ROOT/"results/benchmark_results.csv",
              ROOT/"build/results/target_results.csv", ROOT/"build/results/benchmark_results.csv"):
        if p.is_file(): return p
    raise FileNotFoundError("Run run_all.bat or supply --csv PATH.")
def load_results(path):
    rows=[]
    with Path(path).open(newline="",encoding="utf-8-sig") as stream:
        for number,r in enumerate(csv.DictReader(stream),2):
            row={k:r[k] for k in ("operator","shape","kernel")}
            for k in ("avg_ms","min_ms","med_ms","tflops"):
                row[k]=float(r.get(k,"0"))
                if not math.isfinite(row[k]) or row[k]<0: raise ValueError(f"Row {number}: invalid {k}")
            if row["avg_ms"]<=0 or row["med_ms"]<=0: raise ValueError(f"Row {number}: nonpositive time")
            if r.get("vendor_speedup_min_trial"): row["vendor_speedup_min_trial"]=float(r["vendor_speedup_min_trial"])
            row["trial_count"]=int(r.get("trial_count") or "0")
            rows.append(row)
    if not rows: raise ValueError("No benchmark records.")
    return rows
def is_vendor(r): return r["kernel"].lower() in ("cublas","cudnn")
def is_naive(r):
    n=r["kernel"].lower()
    return "(v1)" in n or "v1-" in n or n.endswith("-v1") or n=="naive"
def grouped(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r["operator"],r["shape"])].append(r)
    return groups
def compute_speedups(rows,metric="med_ms"):
    result=[]
    for group in grouped(rows).values():
        base=next((r for r in group if is_naive(r)),None)
        vendor=next((r for r in group if is_vendor(r)),None)
        for r in group: result.append({**r,"speedup":base[metric]/r[metric] if base else None,
                                       "vendor_speedup":vendor[metric]/r[metric] if vendor else None})
    return result
def comparisons(rows,metric):
    for (op,shape),group in sorted(grouped(rows).items()):
        custom=[r for r in group if not is_vendor(r)]
        if not custom: continue
        yield op,shape,next((r for r in group if is_naive(r)),None),min(custom,key=lambda r:r[metric]),next((r for r in group if is_vendor(r)),None)
def print_table(rows):
    print("\nOperator     Shape                 Kernel                   Avg ms     Median ms")
    for r in sorted(rows,key=lambda r:(r["operator"],r["shape"],r["med_ms"])):
        print(f'{r["operator"]:12} {r["shape"]:21} {r["kernel"]:24} {r["avg_ms"]:10.6f} {r["med_ms"]:10.6f}')
def print_speedup_summary(rows,metric="med_ms"):
    print(f"\nSpeedups based on {metric}; baseline is actual V1.")
    for op,shape,base,best,vendor in comparisons(rows,metric):
        naive=f'{base[metric]/best[metric]:.3f}x vs V1' if base else "V1 missing"
        vend=f'{vendor[metric]/best[metric]:.3f}x vs vendor' if vendor else "vendor missing"
        print(f'{op} {shape}: {best["kernel"]}, {naive}, {vend}')

def print_final_comparison(rows,metric="med_ms"):
    records=[]
    counts={"CUSTOM WIN":0,"CUSTOM LEADS":0,"VENDOR LEADS":0,"TIE":0,"NO VENDOR":0}
    order={"MatMul":0,"LayerNorm":1,"Softmax":2}
    comparisons_=sorted(comparisons(rows,metric),key=lambda item:(order.get(item[0],3),item[1]))
    for op,shape,base,best,vendor in comparisons_:
        ratio=vendor[metric]/best[metric] if vendor else None
        if vendor is None:
            result="NO VENDOR"
        elif ratio==1:
            result="TIE"
        elif (metric=="med_ms" and ratio>=1.05
              and best.get("trial_count",0)>=3
              and best.get("vendor_speedup_min_trial",0)>=1.05):
            result="CUSTOM WIN"
        elif ratio>1:
            result="CUSTOM LEADS"
        else:
            result="VENDOR LEADS"
        counts[result]+=1
        records.append([
            op,shape,best["kernel"],f'{best[metric]:.6f}',
            vendor["kernel"] if vendor else "N/A",
            f'{vendor[metric]:.6f}' if vendor else "N/A",
            f"{ratio:.3f}x" if ratio is not None else "N/A",result,
        ])
    headers=["Operator","Shape","Best custom","Custom ms","Vendor","Vendor ms","Speedup","Result"]
    widths=[max(minimum,len(header),*(len(row[i]) for row in records))
            for i,(header,minimum) in enumerate(zip(headers,[9,16,16,10,6,10,8,19]))]
    border="+"+"+".join("-"*(width+2) for width in widths)+"+"
    def line(cells):
        return "| "+" | ".join(str(cell).ljust(width) for cell,width in zip(cells,widths))+" |"
    metric_label="median" if metric=="med_ms" else "average"
    print(f"\nFINAL RESULT COMPARISON ({metric_label} GPU latency in milliseconds)")
    print(border)
    print(line(headers))
    print(border)
    previous=None
    for record in records:
        if previous is not None and previous!=record[0]:
            print(border)
        print(line(record))
        previous=record[0]
    print(border)
    print("Speedup = vendor time / custom time. Higher than 1 means custom is faster.")
    print("CUSTOM WIN: >=5% faster in every trial, with at least 3 trials (median metric).")
    print("CUSTOM LEADS: lower aggregate latency; repeatability threshold not met.")
    print("VENDOR LEADS: vendor has lower latency. NO VENDOR: comparison unavailable.")
    print(f"Confirmed custom wins: {counts['CUSTOM WIN']} / {len(records)} target shapes.")
    print("Scope: FP32, cached-buffer GPU execution; allocation and transfers excluded.")

def write_markdown(rows,out_path=ROOT/"results/summary.md",metric="med_ms"):
    path=Path(out_path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as out:
        out.write("# Benchmark Summary\n\n")
        out.write(f"Latency: {metric}. Ratios above 1 mean custom is faster. V1 means the actual V1 row. "
                  "A vendor win requires at least 5% improvement in every measured trial.\n\n")
        out.write("| Operator | Shape | V1 ms | Best custom | Custom ms | Vendor | Vendor ms | vs V1 | vs vendor | Evidence |\n")
        out.write("|---|---|---:|---|---:|---|---:|---:|---:|---|\n")
        for op,shape,base,best,vendor in comparisons(rows,metric):
            bt=f'{base[metric]:.6f}' if base else "N/A"
            bs=f'{base[metric]/best[metric]:.3f}x' if base else "N/A"
            vn=vendor["kernel"] if vendor else "missing"
            vt=f'{vendor[metric]:.6f}' if vendor else "N/A"
            vs=f'{vendor[metric]/best[metric]:.3f}x' if vendor else "N/A"
            lower=best.get("vendor_speedup_min_trial",0)
            evidence="comparison incomplete" if not vendor else ">=5% faster in every trial" if lower>=1.05 and best["trial_count"]>=3 and metric=="med_ms" else "consistency threshold not met" if vendor[metric]>best[metric] else "vendor faster"
            out.write(f"| {op} | {shape} | {bt} | {best['kernel']} | {best[metric]:.6f} | {vn} | {vt} | {bs} | {vs} | {evidence} |\n")
    print(f"Markdown: {path}")
def plot_results(rows,do_plot=False,out_path=ROOT/"results/benchmark_chart.png",metric="med_ms"):
    if not do_plot:return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping optional charts.");return
    ops=sorted({r["operator"] for r in rows})
    fig,axes=plt.subplots(len(ops),1,figsize=(11,4*len(ops)),squeeze=False)
    for i,op in enumerate(ops):
        records=[r for r in comparisons(rows,metric) if r[0]==op];ax=axes[i,0]
        for shift,label,pos in ((-.25,"V1",2),(0,"Best custom",3),(.25,"Vendor",4)):
            ax.bar([j+shift for j in range(len(records))],
                   [r[pos][metric] if r[pos] else float("nan") for r in records],width=.25,label=label)
        ax.set_xticks(range(len(records)),[r[1] for r in records]);ax.set_title(op);ax.set_ylabel("Latency (ms)");ax.legend()
    fig.tight_layout();p=Path(out_path);p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=160);plt.close(fig);print(f"Chart: {p}")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",type=Path);p.add_argument("--plot",action="store_true")
    p.add_argument("--metric",choices=("avg_ms","med_ms"),default="med_ms")
    p.add_argument("--out-dir",type=Path,default=ROOT/"results")
    a=p.parse_args()
    try:
        path=a.csv if a.csv is not None else find_csv();rows=load_results(path)
        print(f"CSV: {path} ({len(rows)} records)")
        enhanced=compute_speedups(rows,a.metric);print_table(enhanced);print_speedup_summary(enhanced,a.metric)
        write_markdown(enhanced,a.out_dir/"summary.md",a.metric)
        plot_results(enhanced,a.plot,a.out_dir/"benchmark_chart.png",a.metric)
        print_final_comparison(enhanced,a.metric)
        return 0
    except (OSError,ValueError,KeyError) as e:print(f"[ERROR] {e}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())
