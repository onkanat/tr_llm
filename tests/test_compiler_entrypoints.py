"""tests/test_compiler_entrypoints.py

T-0039 / T-0040 — Derleme giriş noktalarını ve donmuş yol muhafızını doğrulama testleri.
Spec gereksinimleri:
1) Düzeltilmiş giriş noktası ile (vocab_path=data/rebuild/vocab_base_32852.json, literal_entity_mode=True)
   $TMPDIR'a .bin derle -> okunan id dizisinde max(id) >= 32850 VE 32850 (') dizide VAR.
2) AYNI girdiyle literal_entity_mode=False -> max(id) < 32816 VE 32850 dizide YOK.
   (Ayırt edici eksen, vakum kontrolü).
3) allow_frozen_write=False ile hedef "data/train_balanced_sft.bin" -> RuntimeError fırlatılmalı
   VE dosyanın sha256'sı çağrı öncesi/sonrası AYNI olmalı.
4) prepare_dataset(save_vocab_to="data/vocab.json") -> RuntimeError fırlatılmalı (boş girdi dahil).
5) D3 (T-0040): Eski kod testi /tmp önbelleği yerine HER ÇALIŞTIRMADA tmp_path altına git HEAD'den çıkarılarak taze sınanır.
6) D4 (T-0040): is_frozen_path gerçek dosyayı okur; fail-closed testi (olmayan/bozuk dosyada RuntimeError).
"""

import os
import sys
import json
import hashlib
import tempfile
import subprocess
import importlib.util
import ast
import numpy as np
import pytest

from scripts.run_goal_pipeline import compile_jsonl_to_bin, check_frozen_save_path
from train_dpo import check_frozen_save_path as check_frozen_save_path_dpo
from train import check_frozen_save_path as check_frozen_save_path_train
from src.llm.prepare import prepare_dataset
from src.llm.frozen_guard import is_frozen_path, load_frozen_patterns, check_frozen_save_path as check_frozen_save_path_canonical

VOCAB_32852 = "data/rebuild/vocab_base_32852.json"
FROZEN_BIN = "data/train_balanced_sft.bin"
FROZEN_VOCAB = "data/vocab.json"
FROZEN_MODEL = "data/kristal_model.pt"
FROZEN_CARPENTER_MODEL = "data/kristal_carpenter_model.pt"


def _sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture
def sample_jsonl(tmp_path):
    jsonl_file = tmp_path / "sample.jsonl"
    record = {
        "instruction": "Tarihsel nüfus kaydını oku.",
        "input": "Zumrutlukent'in nufusu 1973'te 14:30'da kaydedildi.",
        "output": "Kayıt başarıyla işlendi."
    }
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(jsonl_file)


def test_compile_entrypoint_mode_true(sample_jsonl, tmp_path):
    out_bin = str(tmp_path / "test_true.bin")
    token_count = compile_jsonl_to_bin(
        jsonl_paths=[sample_jsonl],
        output_bin=out_bin,
        vocab_path=VOCAB_32852,
        literal_entity_mode=True,
        block_size=64
    )
    assert token_count > 0
    assert os.path.exists(out_bin)
    
    tokens = np.fromfile(out_bin, dtype=np.uint16)
    assert int(np.max(tokens)) >= 32850
    assert 32850 in tokens


def test_compile_entrypoint_mode_false_vacuum_check(sample_jsonl, tmp_path):
    out_bin = str(tmp_path / "test_false.bin")
    token_count = compile_jsonl_to_bin(
        jsonl_paths=[sample_jsonl],
        output_bin=out_bin,
        vocab_path=VOCAB_32852,
        literal_entity_mode=False,
        block_size=64
    )
    assert token_count > 0
    assert os.path.exists(out_bin)
    
    tokens = np.fromfile(out_bin, dtype=np.uint16)
    # literal_entity_mode=False iken entity ve kesme tokenleri (id >= 32816, ' için 32850) dizide asla bulunamaz
    assert int(np.max(tokens)) < 32816
    assert 32850 not in tokens


def test_compile_entrypoint_frozen_write_protection():
    sha_before = _sha256(FROZEN_BIN)
    with pytest.raises(RuntimeError) as exc_info:
        compile_jsonl_to_bin(
            jsonl_paths=["data/future_train_vector.jsonl"],
            output_bin=FROZEN_BIN,
            vocab_path=VOCAB_32852,
            literal_entity_mode=True,
            allow_frozen_write=False
        )
    assert "Donmuş yola yazma engellendi" in str(exc_info.value)
    sha_after = _sha256(FROZEN_BIN)
    assert sha_before == sha_after


def test_prepare_dataset_frozen_vocab_write_protection(tmp_path):
    dummy_input = tmp_path / "dummy.txt"
    dummy_input.write_text("Örnek girdi metni.\n\nİkinci paragraf metni.", encoding="utf-8")
    dummy_output = tmp_path / "dummy.bin"
    
    sha_before = _sha256(FROZEN_VOCAB)
    with pytest.raises(RuntimeError) as exc_info:
        prepare_dataset(
            input_filepath=str(dummy_input),
            output_filepath=str(dummy_output),
            vocab_path=VOCAB_32852,
            literal_entity_mode=True,
            save_vocab_to=FROZEN_VOCAB
        )
    assert "Donmuş sözlük yoluna geri yazma engellendi" in str(exc_info.value)
    sha_after = _sha256(FROZEN_VOCAB)
    assert sha_before == sha_after


def test_prepare_dataset_empty_input_frozen_vocab_fails_early(tmp_path):
    # T-0039 doğrulama bulgusu V3: Boş girdi verildiğinde de save_vocab_to donmuşsa hemen RuntimeError fırlatılmalı
    empty_input = tmp_path / "empty.txt"
    empty_input.write_text("", encoding="utf-8")
    dummy_output = tmp_path / "empty.bin"

    sha_before = _sha256(FROZEN_VOCAB)
    with pytest.raises(RuntimeError) as exc_info:
        prepare_dataset(
            input_filepath=str(empty_input),
            output_filepath=str(dummy_output),
            vocab_path=VOCAB_32852,
            literal_entity_mode=True,
            save_vocab_to=FROZEN_VOCAB
        )
    assert "Donmuş sözlük yoluna geri yazma engellendi" in str(exc_info.value)
    assert not os.path.exists(dummy_output)
    sha_after = _sha256(FROZEN_VOCAB)
    assert sha_before == sha_after


def test_probe_passes_on_new_code_fails_on_old_code(sample_jsonl, tmp_path):
    # 1. Yeni kod testi
    new_bin = str(tmp_path / "probe_new.bin")
    compile_jsonl_to_bin(
        jsonl_paths=[sample_jsonl],
        output_bin=new_bin,
        vocab_path=VOCAB_32852,
        literal_entity_mode=True,
        block_size=64
    )
    tokens_new = np.fromfile(new_bin, dtype=np.uint16)
    assert 32850 in tokens_new

    # 2. Eski kod testi (T-0040 D3: /tmp önbelleği yerine HER ÇALIŞTIRMADA tmp_path altına çıkarılır)
    # NOT (16 Eyl 2026): "eski kod" HEAD ile ALINMAZ — HEAD ilerledikçe (ör. 4251c53) bu öncül bozulur
    # ve test yanlışlıkla kırılır. Eski davranışı temsil eden commit AÇIKÇA sabitlenir.
    ESKI_KOD_COMMIT = "9e1c1df"   # F2 onarımlarından ÖNCEKİ run_goal_pipeline.py
    eski_path = str(tmp_path / "eski_rgp.py")
    res = subprocess.run(
        ["git", "show", f"{ESKI_KOD_COMMIT}:scripts/run_goal_pipeline.py"],
        capture_output=True,
        text=True,
        check=True
    )
    with open(eski_path, "w", encoding="utf-8") as f:
        f.write(res.stdout)
    
    spec = importlib.util.spec_from_file_location("eski_rgp_fresh", eski_path)
    eski_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eski_mod)
    
    old_bin = str(tmp_path / "probe_old.bin")
    # Eski compile_jsonl_to_bin(jsonl_paths, output_bin, block_size, oversample_factor)
    eski_mod.compile_jsonl_to_bin([sample_jsonl], old_bin)
    tokens_old = np.fromfile(old_bin, dtype=np.uint16)
    # Eski kod data/vocab.json (31357 token) ve mode=False kullandığı için 32850 asla içeremez
    assert 32850 not in tokens_old


def test_frozen_guard_authority_and_fail_closed(tmp_path):
    # T-0040 D4: Dinamik dosya okuma ve fail-closed testi
    custom_frozen_file = tmp_path / "custom_frozen.json"
    custom_patterns = [
        "data/eval/**",
        "custom/frozen/*.bin"
    ]
    with open(custom_frozen_file, "w", encoding="utf-8") as f:
        json.dump({"patterns": custom_patterns}, f)

    # 1. Özel desen dosyası ile kontrol: data/eval/x donmuş olmalı
    assert is_frozen_path("data/eval/test_report.json", frozen_file=str(custom_frozen_file)) is True
    # Varsayılan dosyada data/eval/ donmuş DEĞİLDİR:
    assert is_frozen_path("data/eval/test_report.json") is False

    # 2. Fail-closed: Olmayan dosya
    with pytest.raises(RuntimeError) as exc1:
        is_frozen_path("data/test.bin", frozen_file=str(tmp_path / "nonexistent.json"))
    assert "[FAIL-CLOSED]" in str(exc1.value)

    # 3. Fail-closed: Bozuk JSON
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{invalid_json: true", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc2:
        is_frozen_path("data/test.bin", frozen_file=str(corrupt_file))
    assert "[FAIL-CLOSED]" in str(exc2.value)

    # 4. Fail-closed: 'patterns' listesi olmayan dosya
    empty_patterns_file = tmp_path / "empty_patterns.json"
    empty_patterns_file.write_text(json.dumps({"patterns": []}), encoding="utf-8")
    with pytest.raises(RuntimeError) as exc3:
        is_frozen_path("data/test.bin", frozen_file=str(empty_patterns_file))
    assert "[FAIL-CLOSED]" in str(exc3.value)


def test_check_frozen_save_path_gate():
    """KAPI a: check_frozen_save_path üç durumlu fonksiyon testi.
    1. Donmuş yol + allow_frozen_write=False -> RuntimeError fırlatmalı
    2. Donmuş yol + allow_frozen_write=True -> RuntimeError fırlatmamalı (geçmeli)
    3. Donmamış yol + allow_frozen_write=False -> RuntimeError fırlatmamalı (geçmeli)
    """
    for fn in (check_frozen_save_path, check_frozen_save_path_dpo, check_frozen_save_path_train, check_frozen_save_path_canonical):
        # 1. Donmuş yollar: data/kristal_model.pt ve data/kristal_carpenter_model.pt
        with pytest.raises(RuntimeError) as exc1:
            fn(FROZEN_MODEL, allow_frozen_write=False)
        assert "Donmuş yola yazma engellendi" in str(exc1.value)

        with pytest.raises(RuntimeError) as exc2:
            fn(FROZEN_CARPENTER_MODEL, allow_frozen_write=False)
        assert "Donmuş yola yazma engellendi" in str(exc2.value)

        # 2. Donmuş yol + allow_frozen_write=True -> Hata yok
        fn(FROZEN_MODEL, allow_frozen_write=True)
        fn(FROZEN_CARPENTER_MODEL, allow_frozen_write=True)

        # 3. Donmamış yol + allow_frozen_write=False -> Hata yok
        fn("scratch/test_model.pt", allow_frozen_write=False)
        fn("tests/fixtures/test_model.pt", allow_frozen_write=False)


def test_ast_run_goal_pipeline_has_frozen_guards():
    """KAPI b & c & T-0048 ekseni (i) & (ii): AST testi.
    scripts/run_goal_pipeline.py main() fonksiyonunda:
    (i) Her eğitim run_cmd çağrısından önce check_frozen_save_path çağrısı bulunmalı (sıra ve sayı tam).
    (ii) shutil.copyfile çağrısından önce hedefi ('data/kristal_model_sft.pt') denetleyen check_frozen_save_path çağrısı bulunmalı.
    """
    pipeline_path = "scripts/run_goal_pipeline.py"
    with open(pipeline_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=pipeline_path)

    main_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
            break
    assert main_fn is not None, "main() fonksiyonu bulunamadı"

    # main() gövdesindeki Expr çağrılarını sırayla incele
    calls = []
    for stmt in main_fn.body:
        for subnode in ast.walk(stmt):
            if isinstance(subnode, ast.Call):
                func_name = None
                if isinstance(subnode.func, ast.Name):
                    func_name = subnode.func.id
                elif isinstance(subnode.func, ast.Attribute):
                    val = getattr(subnode.func.value, 'id', '')
                    func_name = f"{val}.{subnode.func.attr}" if val else subnode.func.attr
                if func_name in ("check_frozen_save_path", "run_cmd", "shutil.copyfile"):
                    args_repr = []
                    for a in subnode.args:
                        if isinstance(a, ast.Constant):
                            args_repr.append(a.value)
                        elif isinstance(a, ast.Name):
                            args_repr.append(a.id)
                    calls.append((func_name, getattr(subnode, 'lineno', 0), args_repr))

    # (i) run_cmd öncesi guard denetimi (5 adet eğitim çağrısı)
    guard_count = sum(1 for name, _, _ in calls if name == "check_frozen_save_path")
    assert guard_count == 6, f"Beklenen 6 check_frozen_save_path çağrısı (5 run_cmd + 1 copyfile), bulunan: {guard_count}"

    for i, (name, lineno, _) in enumerate(calls):
        if name == "run_cmd":
            prev_name, prev_lineno, _ = calls[i - 1]
            assert prev_name == "check_frozen_save_path", (
                f"run_cmd (satır {lineno}) öncesinde check_frozen_save_path bulunamadı! "
                f"Önceki çağrı: {prev_name} (satır {prev_lineno})"
            )
            assert prev_lineno < lineno, f"check_frozen_save_path satırı ({prev_lineno}) >= run_cmd ({lineno})"

    # (ii) copyfile öncesi guard denetimi
    copyfile_indices = [i for i, (name, _, _) in enumerate(calls) if name == "shutil.copyfile"]
    assert len(copyfile_indices) == 1, f"Beklenen 1 shutil.copyfile çağrısı, bulunan: {len(copyfile_indices)}"
    cp_idx = copyfile_indices[0]
    assert cp_idx > 0, "shutil.copyfile öncesinde hiçbir çağrı yok"
    prev_name, prev_lineno, prev_args = calls[cp_idx - 1]
    assert prev_name == "check_frozen_save_path", (
        f"shutil.copyfile öncesinde check_frozen_save_path yok! Önceki çağrı: {prev_name}"
    )
    assert len(prev_args) >= 1 and prev_args[0] == "data/kristal_model_sft.pt", (
        f"shutil.copyfile öncesindeki guard hedefi 'data/kristal_model_sft.pt' olmalı, bulunan: {prev_args}"
    )


def test_ast_train_dpo_has_frozen_guards():
    """train_dpo.py AST testi: torch.save çağrısından önce check_frozen_save_path çağrısı bulunmalı."""
    dpo_path = "train_dpo.py"
    with open(dpo_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=dpo_path)

    main_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
            break
    assert main_fn is not None, "train_dpo.py içinde main() bulunamadı"

    calls = []
    for stmt in main_fn.body:
        for subnode in ast.walk(stmt):
            if isinstance(subnode, ast.Call):
                func_name = None
                if isinstance(subnode.func, ast.Name):
                    func_name = subnode.func.id
                elif isinstance(subnode.func, ast.Attribute):
                    func_name = f"{getattr(subnode.func.value, 'id', '')}.{subnode.func.attr}"
                if func_name in ("check_frozen_save_path", "torch.save"):
                    calls.append((func_name, getattr(subnode, 'lineno', 0)))

    save_indices = [i for i, (name, _) in enumerate(calls) if name == "torch.save"]
    assert len(save_indices) == 1, f"Beklenen 1 torch.save çağrısı, bulunan: {len(save_indices)}"
    save_idx = save_indices[0]
    assert save_idx > 0, "torch.save'den önce çağrı yok"
    prev_name, prev_lineno = calls[save_idx - 1]
    assert prev_name == "check_frozen_save_path", (
        f"torch.save öncesinde check_frozen_save_path yok! Önceki: {prev_name}"
    )


def test_ast_train_py_has_frozen_guards():
    """T-0048 ekseni (iii): train.py AST testi (train.py IMPORT EDİLMEZ, main() ÇAĞRILMAZ).
    - Argüman ayrıştırmasından hemen sonra erken check_frozen_save_path çağrısı bulunmalı.
    - torch.save çağrısından önce guard yer almalı.
    """
    train_path = "train.py"
    with open(train_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=train_path)

    main_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
            break
    assert main_fn is not None, "train.py içinde main() bulunamadı"

    calls = []
    for stmt in main_fn.body:
        for subnode in ast.walk(stmt):
            if isinstance(subnode, ast.Call):
                func_name = None
                if isinstance(subnode.func, ast.Name):
                    func_name = subnode.func.id
                elif isinstance(subnode.func, ast.Attribute):
                    val = getattr(subnode.func.value, 'id', '')
                    func_name = f"{val}.{subnode.func.attr}" if val else subnode.func.attr
                if func_name in ("check_frozen_save_path", "torch.save", "torch.load"):
                    calls.append((func_name, getattr(subnode, 'lineno', 0)))

    # Erken guard denetimi: torch.load/model resume işleminden ÖNCE check_frozen_save_path çağrılmalı
    guard_indices = [i for i, (name, _) in enumerate(calls) if name == "check_frozen_save_path"]
    assert len(guard_indices) >= 1, "train.py içinde check_frozen_save_path çağrısı bulunamadı"
    first_guard_idx = guard_indices[0]

    # torch.save denetimi: torch.save çağrısından önce guard bulunmalı
    save_indices = [i for i, (name, _) in enumerate(calls) if name == "torch.save"]
    assert len(save_indices) == 1, f"Beklenen 1 torch.save çağrısı, bulunan: {len(save_indices)}"
    save_idx = save_indices[0]
    assert first_guard_idx < save_idx, (
        f"Erken guard satırı ({calls[first_guard_idx][1]}) >= torch.save satırı ({calls[save_idx][1]})"
    )

    # Sıralama: İlk guard çağrısı model yükleme/eğitim döngüsünden önce gelmelidir
    load_indices = [i for i, (name, _) in enumerate(calls) if name == "torch.load"]
    if load_indices:
        assert first_guard_idx < load_indices[0], (
            f"Erken guard satırı ({calls[first_guard_idx][1]}) >= torch.load satırı ({calls[load_indices[0]][1]})"
        )


def test_ast_retrain_clean_models_has_frozen_guards():
    """T-0049: scripts/retrain_clean_models.py AST testi (script ÇALIŞTIRILMAZ, main() ÇAĞRILMAZ).
    (a) İki copyfile çağrısından (:59 pre_clean, :106 sft) HEMEN ÖNCE guard var ve hedefler doğru.
    (b) 5 eğitim çağrısından (run_cmd) HEMEN ÖNCE guard var ve SIRA doğru.
    (c) Her eğitim komut listesinde koşullu '--allow-frozen-write' eklenişi var.
    """
    script_path = "scripts/retrain_clean_models.py"
    with open(script_path, "r", encoding="utf-8") as f:
        source_code = f.read()
        tree = ast.parse(source_code, filename=script_path)

    main_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
            break
    assert main_fn is not None, "retrain_clean_models.py içinde main() bulunamadı"

    calls = []
    for stmt in main_fn.body:
        for subnode in ast.walk(stmt):
            if isinstance(subnode, ast.Call):
                func_name = None
                if isinstance(subnode.func, ast.Name):
                    func_name = subnode.func.id
                elif isinstance(subnode.func, ast.Attribute):
                    val = getattr(subnode.func.value, 'id', '')
                    func_name = f"{val}.{subnode.func.attr}" if val else subnode.func.attr
                if func_name in ("check_frozen_save_path", "run_cmd", "shutil.copyfile"):
                    args_repr = []
                    for a in subnode.args:
                        if isinstance(a, ast.Constant):
                            args_repr.append(a.value)
                        elif isinstance(a, ast.Name):
                            args_repr.append(a.id)
                    calls.append((func_name, getattr(subnode, 'lineno', 0), args_repr))

    # (a) İki copyfile öncesi guard denetimi
    copyfile_indices = [i for i, (name, _, _) in enumerate(calls) if name == "shutil.copyfile"]
    assert len(copyfile_indices) == 2, f"Beklenen 2 copyfile çağrısı, bulunan: {len(copyfile_indices)}"

    # Copyfile 1: data/kristal_model_pre_clean.pt
    cp1_idx = copyfile_indices[0]
    assert cp1_idx > 0, "İlk copyfile öncesinde çağrı yok"
    prev1_name, prev1_line, prev1_args = calls[cp1_idx - 1]
    assert prev1_name == "check_frozen_save_path", f"Copyfile 1 öncesi guard yok! Önceki: {prev1_name}"
    assert len(prev1_args) >= 1 and prev1_args[0] == "data/kristal_model_pre_clean.pt", (
        f"Copyfile 1 guard hedefi 'data/kristal_model_pre_clean.pt' olmalı, bulunan: {prev1_args}"
    )

    # Copyfile 2: data/kristal_model_sft.pt
    cp2_idx = copyfile_indices[1]
    assert cp2_idx > 0, "İkinci copyfile öncesinde çağrı yok"
    prev2_name, prev2_line, prev2_args = calls[cp2_idx - 1]
    assert prev2_name == "check_frozen_save_path", f"Copyfile 2 öncesi guard yok! Önceki: {prev2_name}"
    assert len(prev2_args) >= 1 and prev2_args[0] == "data/kristal_model_sft.pt", (
        f"Copyfile 2 guard hedefi 'data/kristal_model_sft.pt' olmalı, bulunan: {prev2_args}"
    )

    # (b) 5 eğitim run_cmd çağrısı ve öncesindeki guard'lar
    run_cmd_indices = [i for i, (name, _, _) in enumerate(calls) if name == "run_cmd"]
    assert len(run_cmd_indices) == 5, f"Beklenen 5 run_cmd çağrısı, bulunan: {len(run_cmd_indices)}"

    for idx in run_cmd_indices:
        rc_name, rc_line, rc_args = calls[idx]
        prev_name, prev_line, prev_args = calls[idx - 1]
        assert prev_name == "check_frozen_save_path", (
            f"run_cmd (satır {rc_line}) öncesinde check_frozen_save_path yok! Önceki: {prev_name}"
        )
        assert prev_line < rc_line, f"guard satırı ({prev_line}) >= run_cmd ({rc_line})"

    # Toplam guard sayısı: 2 copyfile + 5 run_cmd = 7 adet
    guard_count = sum(1 for name, _, _ in calls if name == "check_frozen_save_path")
    assert guard_count == 7, f"Beklenen 7 check_frozen_save_path çağrısı, bulunan: {guard_count}"

    # (c) Her eğitim komutunda koşullu '--allow-frozen-write' eklenişi varlığı
    # AST'de if allow_frozen_write kontrolü ve .append('--allow-frozen-write') araması
    append_count = 0
    for node in ast.walk(main_fn):
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "allow_frozen_write":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "append":
                        if sub.args and isinstance(sub.args[0], ast.Constant) and sub.args[0].value == "--allow-frozen-write":
                            append_count += 1
    assert append_count == 5, f"Beklenen 5 adet '--allow-frozen-write' ekleme koşulu, bulunan: {append_count}"



