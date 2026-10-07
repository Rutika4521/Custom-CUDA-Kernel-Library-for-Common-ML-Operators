import csv
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/analyze_results.py"
A=runpy.run_path(str(SCRIPT),run_name="analysis_tests")
def row(name,ms,**extra):
    return dict(operator="MatMul",shape="8x8x8",kernel=name,avg_ms=ms,min_ms=ms,med_ms=ms,tflops=0,trial_count=0,**extra)
class AnalysisTests(unittest.TestCase):
    def test_actual_v1_baseline(self):
        data=[row("Naive(V1)",10),row("Unrolled32(V4)",20),row("Shape",2),row("cuBLAS",1)]
        result=A["compute_speedups"](data)
        self.assertEqual(result[0]["speedup"],1)
        self.assertEqual(result[2]["speedup"],5)
        self.assertEqual(result[2]["vendor_speedup"],.5)
    def test_missing_baselines(self):
        result=A["compute_speedups"]([row("Shape",2)])
        self.assertIsNone(result[0]["speedup"])
        self.assertIsNone(result[0]["vendor_speedup"])
    def test_summary_requires_repetition(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"sub"/"summary.md"
            A["write_markdown"]([row("Shape",1,vendor_speedup_min_trial=4),row("cuBLAS",4)],path)
            self.assertNotIn(">=5% faster in every trial",path.read_text())
    def test_cli_and_bad_values(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/"input.csv"
            data=[row("Naive(V1)",10),row("Shape",2),row("cuBLAS",1)]
            with path.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
            p=subprocess.run([sys.executable,str(SCRIPT),"--csv",str(path),"--out-dir",str(root/"output")],cwd=root,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertTrue((root/"output/summary.md").is_file())
            p=subprocess.run([sys.executable,str(SCRIPT),"--csv",str(root/"missing.csv")],cwd=root,capture_output=True,text=True)
            self.assertEqual(p.returncode,1)
            text=path.read_text().replace(",10,10,10,",",nan,10,10,")
            path.write_text(text)
            with self.assertRaises(ValueError):A["load_results"](path)
    def test_detection_without_cwd_dependency(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"build/results").mkdir(parents=True)
            expected=root/"build/results/benchmark_results.csv";expected.write_text("placeholder")
            globals_=A["find_csv"].__globals__;old=globals_["ROOT"];globals_["ROOT"]=root
            try:self.assertEqual(A["find_csv"](),expected)
            finally:globals_["ROOT"]=old
if __name__=="__main__":unittest.main()
