# Next Task: Windows Side

**Created:** 2026-01-28
**From:** Linux dev machine
**Status:** PENDING

## Context

The Parquet export has a bug - regular channels (RPM, temps, pressures) are all NaN while GPS channels work fine. Likely cause: time base mismatch (regular channels may be in milliseconds like GPS, but weren't being converted).

## Fix Applied

Modified `src/extract/data_loader.py` to auto-detect millisecond timestamps:
- If a regular channel's max time > 1000, assume it's in milliseconds and divide by 1000
- GPS channels already had this conversion

## Your Tasks

### 1. Run Diagnostic (Optional but helpful)
```bash
cd <project_root>
python scripts/diagnose_extraction.py "data/uploads/Andy McDermid_24_AS_Road America_Race_a_0037.xrk"
```
This will show raw time values for all channels. Look for:
- Regular channels: Are times in ms (large values like 977900) or seconds (977.9)?
- GPS channels: Should be in ms before conversion

Save output to `.memory/diagnostic_output.txt` if useful.

### 2. Run Test Extraction
```bash
python scripts/test_extraction.py "data/uploads/Andy McDermid_24_AS_Road America_Race_a_0037.xrk"
```
This will:
- Extract all channels with the fix
- Report data quality per channel
- Save new Parquet to `data/exports/`

### 3. Report Results

Create `.memory/RESULT.md` with:
- Did regular channels (RPM, OIL PRESSURE, WATER TEMP, etc.) get data?
- How many channels have data vs empty?
- Any errors?

Then:
```bash
git add .memory/RESULT.md data/exports/*.parquet
git commit -m "Windows test results: extraction fix"
git push
```

## Expected Outcome

**Success:** Most/all channels have data (not just GPS)
**Failure:** Still only GPS channels have data → need more investigation

## Files Changed

- `src/extract/data_loader.py` - time base fix
- `scripts/diagnose_extraction.py` - new diagnostic tool
- `scripts/test_extraction.py` - new test tool
