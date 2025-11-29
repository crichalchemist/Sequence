import importlib.util
import sys
from pathlib import Path
import zipfile


def test_parse_zip_fixture(tmp_path: Path):
    parser_module = _load_parser_module()
    parser = parser_module.GDELTParser()
    sample_row = "\t".join(
        [
            "0",
            "20240101000000",
            "",
            "",
            "",
            "STATENAME#1;NUMARTS#2",
            "THEME1;THEME2;",
            "ENHANCED1#1;ENHANCED2#2;",
            "1#New York#NY#US#40.71#-74.0",
            "Person1;Person2;",
            "Org1;Org2;",
            "1,2,3,4,5",
            "",
            "",
            "",
            "c1:0.5,c2:1.5",
        ]
    )
    sample_row += "\n"

    zip_path = tmp_path / "gdelt_sample.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sample.gkgv2.csv", sample_row)

    records = list(parser.parse_file(zip_path))

    assert len(records) == 1
    record = records[0]
    assert record.datetime.year == 2024
    assert record.themes == ["ENHANCED1", "ENHANCED2", "THEME1", "THEME2"]
    assert {count.type for count in record.counts} == {"STATENAME", "NUMARTS"}
    assert record.tone.polarity == 4.0
    assert record.locations[0].name == "New York"
    assert record.persons == ["Person1", "Person2"]
    assert record.orgs == ["Org1", "Org2"]
    assert record.gcam == {"c1": 0.5, "c2": 1.5}


def _load_parser_module():
    module_path = Path(__file__).resolve().parents[1] / "gdelt" / "parser.py"
    spec = importlib.util.spec_from_file_location("gdelt_parser", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
