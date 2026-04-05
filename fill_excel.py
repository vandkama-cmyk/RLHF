"""
Fill empty columns in CogSci Special Issue 2026 - График 1.xlsx
with actual training results from JSON/CSV files.

Run this script after training completes to populate the Excel file.
Usage: C:/condam/envs/rlhfenv/python.exe fill_excel.py
"""
import json
import csv
import re
import os
from pathlib import Path
import openpyxl

BASE = Path("c:/Users/Полина/Desktop/Работа/huawei/rlhf")
EXCEL = BASE / "CogSci Special Issue 2026 - График 1.xlsx"
BACKUP = BASE / "CogSci Special Issue 2026 - График 1_backup.xlsx"

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def find_header_row(ws):
    """Find the row number (1-indexed) where headers are (contains 'epoch')."""
    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if row and row[0] == 'epoch':
            return i
    return None

def get_epoch_rows(ws, header_row):
    """Return dict {epoch_int: row_index_1based} for all data rows."""
    result = {}
    for i, row in enumerate(ws.iter_rows(min_row=header_row + 1, values_only=True), start=header_row + 1):
        if row and row[0] is not None:
            try:
                result[int(row[0])] = i
            except (TypeError, ValueError):
                pass
    return result

def r(v, decimals=4):
    """Round float to decimals, pass through None."""
    if v is None:
        return None
    try:
        return round(float(v), decimals)
    except (TypeError, ValueError):
        return v


def set_num(cell, value):
    """Write a numeric value and force General format to prevent date corruption.

    Excel cells pre-formatted as Date will display any float as a serial date
    (e.g. train_loss=7.82 → '1900-01-07'). Explicitly resetting number_format
    to 'General' prevents this.
    """
    try:
        cell.value = value
        if value is not None:
            cell.number_format = 'General'
    except AttributeError:
        # MergedCell is read-only; skip safely.
        return


def clear_cell(cell):
    """Clear cell safely; skip merged read-only cells."""
    try:
        cell.value = None
    except AttributeError:
        return


# ─────────────────────────────────────────────────────────────────────
# Stage 2A — fill training_loss (col 8) and val_accuracy (col 9)
# NOTE: Stage 2A val_loss is average BCE across 3 reward-classification
# heads (consistent/correct/useful) on 31 val samples, evaluated using
# EMA weights (decay=0.999) with OneCycleLR scheduling.  Oscillation
# after epoch 4 is expected: it reflects EMA lag during the high-LR
# phase of OneCycleLR — not a data bug.
# ─────────────────────────────────────────────────────────────────────

def fill_stage2a(ws):
    json_path = BASE / "stage2/stage2A/outputs/stage2a_results.json"
    csv_path  = BASE / "stage2/stage2A/outputs/stage2a_metrics.csv"

    if not json_path.exists() or not csv_path.exists():
        print("  Stage 2A: source files not found, skipping")
        return

    with open(json_path) as f:
        data = json.load(f)
    train_loss = {e["epoch"]: e["loss"] for e in data.get("training_history", [])}

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    val_acc     = {int(ro["epoch"]): float(ro["reward_acc_mean"]) for ro in rows}
    val_loss_map = {int(ro["epoch"]): float(ro["val_loss"]) for ro in rows if ro.get("val_loss")}

    header_row = find_header_row(ws)
    if header_row is None:
        print("  Stage 2A: header row not found")
        return

    # Column layout (0-based): epoch=0, bertscore=1..codebleu=5, val_loss=6,
    # training_loss=7, val_Accuracy=8
    updated = 0
    for row in ws.iter_rows(min_row=header_row + 1):
        epoch_cell = row[0].value
        if epoch_cell is None:
            continue
        try:
            epoch = int(epoch_cell)
        except (TypeError, ValueError):
            continue

        # Fill val_loss (col 7 = index 6)
        if epoch in val_loss_map:
            set_num(row[6], r(val_loss_map[epoch]))
            updated += 1

        # Fill training loss (col 8 = index 7)
        if epoch in train_loss:
            set_num(row[7], r(train_loss[epoch]))
            updated += 1

        # Fill val accuracy (col 9 = index 8)
        if epoch in val_acc:
            set_num(row[8], r(val_acc[epoch]))
            updated += 1

    # Append rows for epochs not yet in the sheet
    existing_epochs = get_epoch_rows(ws, header_row)
    all_data_epochs = set(train_loss.keys()) | set(val_acc.keys())
    added = 0
    for epoch in sorted(all_data_epochs):
        if epoch not in existing_epochs:
            vl = r(val_loss_map.get(epoch))
            tl = r(train_loss.get(epoch))
            va = r(val_acc.get(epoch))
            # epoch + 5 Nones (indices 1-5) + val_loss(6) + train_loss(7) + val_acc(8)
            ws.append([epoch, None, None, None, None, None, vl, tl, va])
            new_row = ws.max_row
            for col_idx in [6, 7, 8]:
                cell = ws.cell(row=new_row, column=col_idx + 1)
                if cell.value is not None:
                    cell.number_format = 'General'
            added += 1

    print(f"  Stage 2A: updated {updated} cells, added {added} new epoch rows")


# ─────────────────────────────────────────────────────────────────────
# Stage 2B — add epochs from log file
# NOTE: Stage 2B text metrics (bertscore/rouge/bleu/ruby/codebleu) in the
# 30-epoch log are INVALID — they were produced by a self-comparison bug
# (val_answers compared to val_answers, not model-generated text).
# Only reward accuracy and loss metrics are valid.
# Text metric columns are set to None (blank) here.
# ─────────────────────────────────────────────────────────────────────

def fill_stage2b(ws):
    log_path = BASE / "stage2/stage2B/outputs/run_log_30epochs.txt"
    if not log_path.exists():
        print("  Stage 2B: log not found, skipping")
        return

    with open(log_path) as f:
        content = f.read()

    # Parse full metrics dict per epoch (multi-line safe via JSON-like format)
    pattern = r"Epoch (\d+) Metrics: (\{[^}]+\})"
    matches = re.findall(pattern, content)
    if not matches:
        print("  Stage 2B: no epoch metrics found in log")
        return

    stage2b = {}
    for epoch_str, metrics_str in matches:
        epoch = int(epoch_str)
        try:
            m = eval(metrics_str)
        except Exception:
            continue
        stage2b[epoch] = {
            # Text metrics are INVALID (self-comparison bug) — store None
            "bertscore": None,
            "rouge":     None,
            "bleu":      None,
            "ruby":      None,
            "codebleu":  None,
            # These reward/accuracy metrics are valid
            "train_loss":      m.get("train_loss"),
            "val_acc_mean":    m.get("val_accuracy_mean"),
            "train_acc_mean":  m.get("train_accuracy_mean"),
        }

    header_row = find_header_row(ws)
    if header_row is None:
        print("  Stage 2B: header row not found")
        return

    existing_epochs = get_epoch_rows(ws, header_row)

    # Also clear any existing invalid text metric cells in rows that are already present
    cleared = 0
    for row in ws.iter_rows(min_row=header_row + 1):
        epoch_cell = row[0].value
        if epoch_cell is None:
            continue
        try:
            epoch = int(epoch_cell)
        except (TypeError, ValueError):
            continue
        if epoch in stage2b:
            # Columns 1-5 (0-indexed) = bertscore, rouge, bleu, ruby, codebleu
            for col_idx in range(1, 6):
                if col_idx < len(row) and row[col_idx].value is not None:
                    row[col_idx].value = None
                    cleared += 1
    if cleared:
        print(f"  Stage 2B: cleared {cleared} invalid text metric cells from existing rows")

    added = 0
    for epoch in sorted(stage2b.keys()):
        if epoch not in existing_epochs:
            ws.append([epoch, None, None, None, None, None])
            added += 1

    print(f"  Stage 2B: added {added} new epoch rows (text metrics blanked as invalid)")


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Case 1 and Case 2
# Fill from history JSON only if BERTScore > 0 (i.e., new correct run)
# ─────────────────────────────────────────────────────────────────────

def fill_stage3_case(ws, json_path, case_label):
    if not json_path.exists():
        print(f"  Stage 3 {case_label}: JSON not found, skipping")
        return

    with open(json_path) as f:
        data = json.load(f)

    if not data:
        print(f"  Stage 3 {case_label}: empty history, skipping")
        return

    # Check if BERTScore is meaningful (not the broken 0.0 run)
    first_bs = data[0].get("bertscore", 0)
    if first_bs < 0.1:
        print(f"  Stage 3 {case_label}: BERTScore=0.0 (broken run), skipping")
        return

    by_epoch = {e["epoch"]: e for e in data}

    header_row = find_header_row(ws)
    if header_row is None:
        print(f"  Stage 3 {case_label}: header row not found")
        return

    # Column mapping (0-based): epoch=0, bertscore=1, rouge=2, bleu=3, ruby=4,
    # codebleu=5, train_loss=6, train_acc=7, val_loss=8, val_acc=9, reward=10, gradient_norm=11
    existing_epochs = get_epoch_rows(ws, header_row)
    updated = 0
    added = 0

    # Update existing rows
    for row in ws.iter_rows(min_row=header_row + 1):
        epoch_cell = row[0].value
        if epoch_cell is None:
            continue
        try:
            epoch = int(epoch_cell)
        except (TypeError, ValueError):
            continue
        if epoch not in by_epoch:
            continue

        e = by_epoch[epoch]
        # Overwrite all metric columns (old data might be broken).
        # Use set_num() to reset cell number_format so Excel does not
        # render floats as serial dates (e.g. train_loss=7.82 → "1900-01-07").
        set_num(row[1], r(e.get("bertscore")))
        set_num(row[2], r(e.get("rouge")))
        set_num(row[3], r(e.get("bleu")))
        set_num(row[4], r(e.get("ruby")))
        set_num(row[5], r(e.get("codebleu")))
        set_num(row[6], r(e.get("train_loss")))
        # train_acc (index 7): not in SFT history — always clear stale "?" placeholders
        row[7].value = None
        # val_loss: always write (None clears stale 0.0000 placeholders from old manual entry)
        val_loss = e.get("val_loss")
        set_num(row[8], r(val_loss) if isinstance(val_loss, (int, float)) else None)
        # val_acc (index 9): not in SFT history — always clear
        row[9].value = None
        set_num(row[10], r(e.get("reward")))
        set_num(row[11], r(e.get("gradient_norm")))
        if len(row) > 12:
            set_num(row[12], r(e.get("entropy")))
        if len(row) > 13:
            # Field renamed from kl_divergence → kl_from_uniform.
            # Value = log(V) - H(π), i.e. KL(π||uniform), NOT KL(π||π_ref).
            # Grows monotonically as SFT sharpens the distribution (expected).
            kl_val = e.get("kl_from_uniform", e.get("kl_divergence"))
            set_num(row[13], r(kl_val))
        updated += 1

    # Add rows for epochs not already present
    for epoch in sorted(by_epoch.keys()):
        if epoch not in existing_epochs:
            e = by_epoch[epoch]
            val_loss = e.get("val_loss")
            val_loss_val = r(val_loss) if isinstance(val_loss, (int, float)) else None
            ws.append([
                epoch,
                r(e.get("bertscore")),
                r(e.get("rouge")),
                r(e.get("bleu")),
                r(e.get("ruby")),
                r(e.get("codebleu")),
                r(e.get("train_loss")),
                None,   # train_acc — not in SFT history
                val_loss_val,
                None,   # val_acc — not in SFT history
                r(e.get("reward")),
                r(e.get("gradient_norm")),
                r(e.get("entropy")),
                r(e.get("kl_from_uniform", e.get("kl_divergence"))),
            ])
            # Reset number_format on numeric columns of the new row to prevent date corruption
            new_row = ws.max_row
            for col_idx in [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13]:  # 0-based → +1 for openpyxl
                cell = ws.cell(row=new_row, column=col_idx + 1)
                if cell.value is not None:
                    cell.number_format = 'General'
            added += 1

    print(f"  Stage 3 {case_label}: updated {updated} rows, added {added} new rows")


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — fill epochs 11–30 (only empty cells)
# ─────────────────────────────────────────────────────────────────────

def fill_stage1(ws):
    """Fill Stage 1 sheet from stage1_metrics.csv and run_log_30epochs.txt."""
    csv_path = BASE / "stage1/output/stage1_metrics.csv"
    log_path = BASE / "stage1/output/run_log_30epochs.txt"

    by_epoch = {}

    # Load CSV if it has >10 epochs (from new run)
    if csv_path.exists():
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if len(rows) > 10:
            for ro in rows:
                try:
                    ep = int(ro.get("epoch", ro.get("Epoch", -1)))
                    by_epoch[ep] = {
                        "bertscore": ro.get("BERTScore", ro.get("bertscore", 0)),
                        "rouge":     ro.get("ROUGE",     ro.get("rouge", 0)),
                        "bleu":      ro.get("BLEU",      ro.get("bleu", 0)),
                        "ruby":      ro.get("Ruby",      ro.get("ruby", 0)),
                        "codebleu":  ro.get("CodeBLEU",  ro.get("codebleu", 0)),
                    }
                except (TypeError, ValueError):
                    pass

    # Also try to parse from log file (epoch summary lines)
    if log_path.exists():
        with open(log_path) as f:
            content = f.read()

        # Parse lines like: "Epoch 11: BERTScore=0.81, ..."
        epoch_pattern = r"Epoch (\d+).*?BERTScore[:\s=]+([0-9.]+)"
        for m in re.finditer(epoch_pattern, content, re.IGNORECASE):
            ep = int(m.group(1))
            if ep not in by_epoch:
                by_epoch[ep] = {"bertscore": float(m.group(2))}

    if not by_epoch:
        print("  Stage 1: no data available yet")
        return

    header_row = find_header_row(ws)
    if header_row is None:
        print("  Stage 1: header row not found")
        return

    existing = get_epoch_rows(ws, header_row)
    updated = 0
    added = 0

    for row in ws.iter_rows(min_row=header_row + 1):
        epoch_cell = row[0].value
        if epoch_cell is None:
            continue
        try:
            epoch = int(epoch_cell)
        except (TypeError, ValueError):
            continue
        if epoch not in by_epoch:
            continue

        e = by_epoch[epoch]
        # Always overwrite — source data from completed 30-epoch run takes priority
        if "bertscore" in e:
            set_num(row[1], r(e["bertscore"]))
            updated += 1
        if "rouge" in e:
            set_num(row[2], r(e["rouge"]))
            updated += 1
        if "bleu" in e:
            set_num(row[3], r(e["bleu"]))
            updated += 1
        if "ruby" in e:
            set_num(row[4], r(e["ruby"]))
            updated += 1
        if "codebleu" in e:
            set_num(row[5], r(e["codebleu"]))
            updated += 1

    # Add entirely new epoch rows
    for epoch in sorted(by_epoch.keys()):
        if epoch not in existing:
            e = by_epoch[epoch]
            ws.append([
                epoch,
                r(e.get("bertscore")),
                r(e.get("rouge")),
                r(e.get("bleu")),
                r(e.get("ruby")),
                r(e.get("codebleu")),
                "",  # notes
            ])
            added += 1

    print(f"  Stage 1: updated {updated} cells, added {added} new rows")


# ─────────────────────────────────────────────────────────────────────
# Stage 4A — three classifier sections (Consistency / Agreement / Usefulness)
#
# Stage 4A/B/C train MLP classifiers, not text generators. The embedding
# metrics are now computed as confidence-weighted quality of validation
# answers vs references and therefore can vary by epoch.
#
# Sections are processed in REVERSE order (Usefulness → Agreement →
# Consistency) so that the row-insertion for Consistency epochs 11-15
# happens last and does not corrupt the already-written rows of the two
# lower sections.
# ─────────────────────────────────────────────────────────────────────

def fill_stage4a(ws):
    SECTIONS = [
        # Processed in reverse order — see note above.
        {
            'name':        'Usefulness',
            'json':        BASE / 'stage4/stage4C_useful/artifacts/training_history_useful.json',
            'train_x_key': 'train_useful_acc',
            'val_x_key':   'val_useful_acc',
            'data_start':  33,   # 1-based row of first data row
        },
        {
            'name':        'Agreement',
            'json':        BASE / 'stage4/stage4B_corct/artifacts/training_history_correct.json',
            'train_x_key': 'train_correct_acc',
            'val_x_key':   'val_correct_acc',
            'data_start':  18,
        },
        {
            'name':        'Consistency',
            'json':        BASE / 'stage4/stage4A_consist/artifacts/training_history_consistent.json',
            'train_x_key': 'train_consistent_acc',
            'val_x_key':   'val_consistent_acc',
            'data_start':  3,
        },
    ]

    for sec in SECTIONS:
        if not sec['json'].exists():
            print(f"  Stage 4A {sec['name']}: JSON not found, skipping")
            continue

        with open(sec['json']) as f:
            data = json.load(f)
        by_epoch = {e['epoch']: e for e in data}

        # Scan existing epoch rows (stop at first blank epoch cell).
        existing = {}   # epoch_int → 1-based row number
        row_idx = sec['data_start']
        while True:
            cell_val = ws.cell(row=row_idx, column=1).value
            if cell_val is None:
                break
            try:
                existing[int(cell_val)] = row_idx
            except (TypeError, ValueError):
                break
            row_idx += 1

        # ── overwrite / clear existing rows ──
        updated = 0
        for epoch, row_num in existing.items():
            if epoch in by_epoch:
                e = by_epoch[epoch]
                set_num(ws.cell(row=row_num, column=2), r(e['train_loss']))
                set_num(ws.cell(row=row_num, column=3), r(e['val_loss']))
                set_num(ws.cell(row=row_num, column=4), r(e['train_acc']))
                set_num(ws.cell(row=row_num, column=5), r(e['val_acc']))
                set_num(ws.cell(row=row_num, column=6), r(e[sec['train_x_key']]))
                set_num(ws.cell(row=row_num, column=7), r(e[sec['val_x_key']]))
                set_num(ws.cell(row=row_num, column=8), r(e.get('val_bertscore')))
                set_num(ws.cell(row=row_num, column=9), r(e.get('val_rouge')))
                set_num(ws.cell(row=row_num, column=10), r(e.get('val_bleu')))
                set_num(ws.cell(row=row_num, column=11), r(e.get('val_ruby')))
                set_num(ws.cell(row=row_num, column=12), r(e.get('val_codebleu')))
            else:
                # Epoch exists in sheet but not in JSON (e.g. Usefulness ep 10).
                for col in range(2, 13):
                    clear_cell(ws.cell(row=row_num, column=col))
            updated += 1

        # ── insert new epoch rows not yet in sheet ──
        new_epochs = sorted(ep for ep in by_epoch if ep not in existing)
        added = 0
        if new_epochs:
            # Insert blank rows immediately after the last existing data row
            # so the new epochs stay visually inside their section.
            insert_at = (max(existing.values()) + 1) if existing else sec['data_start']
            ws.insert_rows(insert_at, len(new_epochs))
            for offset, epoch in enumerate(new_epochs):
                e = by_epoch[epoch]
                wr = insert_at + offset   # write row
                ws.cell(row=wr, column=1).value = epoch
                set_num(ws.cell(row=wr, column=2), r(e['train_loss']))
                set_num(ws.cell(row=wr, column=3), r(e['val_loss']))
                set_num(ws.cell(row=wr, column=4), r(e['train_acc']))
                set_num(ws.cell(row=wr, column=5), r(e['val_acc']))
                set_num(ws.cell(row=wr, column=6), r(e[sec['train_x_key']]))
                set_num(ws.cell(row=wr, column=7), r(e[sec['val_x_key']]))
                set_num(ws.cell(row=wr, column=8), r(e.get('val_bertscore')))
                set_num(ws.cell(row=wr, column=9), r(e.get('val_rouge')))
                set_num(ws.cell(row=wr, column=10), r(e.get('val_bleu')))
                set_num(ws.cell(row=wr, column=11), r(e.get('val_ruby')))
                set_num(ws.cell(row=wr, column=12), r(e.get('val_codebleu')))
                ws.cell(row=wr, column=1).number_format = 'General'
                added += 1

        print(f"  Stage 4A {sec['name']}: updated {updated} rows, added {added} new rows")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading: {EXCEL}")
    wb = openpyxl.load_workbook(EXCEL)

    # Backup before modifying
    import shutil
    shutil.copy2(EXCEL, BACKUP)
    print(f"Backup saved: {BACKUP}")

    print("\nFilling Stage 1...")
    fill_stage1(wb["Stage 1"])

    print("\nFilling Stage 2A...")
    fill_stage2a(wb["Stage 2A"])

    print("\nFilling Stage 2B...")
    fill_stage2b(wb["Stage 2B"])

    print("\nFilling Stage 3 - Case 1...")
    fill_stage3_case(
        wb["Stage 3 - Case 1"],
        BASE / "stage3/outputs/case1_11samples_history.json",
        "Case 1"
    )

    print("\nFilling Stage 3 - Case 2...")
    fill_stage3_case(
        wb["Stage 3 - Case 2"],
        BASE / "stage3/outputs/case2_1247samples_history.json",
        "Case 2"
    )

    print("\nFilling Stage 4A...")
    fill_stage4a(wb["Stage 4A"])

    print(f"\nSaving to: {EXCEL}")
    wb.save(EXCEL)
    print("Done!")


if __name__ == "__main__":
    main()
