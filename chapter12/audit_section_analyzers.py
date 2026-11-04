"""Check that real-data examples fail closed and published metrics plot."""
from pathlib import Path
import ast, subprocess, sys
ROOT=Path(__file__).resolve().parent
paths=[
"section_12_distributed_training/analyze_published_megatron.py",
"section_12_manufacturing/analyze_real_secom.py",
"section_12_telecommunications/analyze_real_network_measurements.py",
"section_12_pharmaceutical/analyze_real_bioactivity.py",
"section_12_clinical_imaging/analyze_real_external_validation.py",
"section_12_geospatial_fleet/analyze_real_tlc_trips.py",
]
for path in paths:
    script=ROOT/path
    ast.parse(script.read_text(), filename=str(script))
    if "distributed_training" in path:
        run=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=40)
        assert run.returncode==0, run.stderr
        print(path,"published figures reproduced")
    else:
        output=ROOT/"_audit_no_data"/script.parent.name
        assert not output.exists()
        run=subprocess.run([sys.executable,str(script),"--input",str(ROOT/"NONEXISTENT_REAL_DATA.csv"),"--output",str(output)],capture_output=True,text=True,timeout=40)
        assert run.returncode!=0 and not output.exists(),(path,run.stdout,run.stderr)
        print(path,"missing real data correctly rejected")
fleet=(ROOT/"section_12_geospatial_fleet/analyze_real_tlc_trips.py").read_text()
assert 'features=["PULocationID","DOLocationID","hour","weekday"]' in fleet
print("All section-level data-provenance guards passed")
