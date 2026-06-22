# Camillion data layout

All real 1-minute CSVs live in ONE place on your Google Drive:

    /content/drive/MyDrive/Camillion_data/

**One file per symbol**, named `<SYMBOL>_1m.csv`:

    Camillion_data/EURUSD_1m.csv
    Camillion_data/US30_1m.csv
    Camillion_data/XAUUSD_1m.csv

## CSV format (flexible — the loader auto-detects columns)
- a datetime/timestamp column (or separate `date` + `time`)  -> bar time
- `open`, `high`, `low`, `close`                              -> OHLC
- `volume` (optional; defaults to 1)

`src/data/cache_builder.load_ohlcv_csv` reads this, and `build_aligned_indicators`
resamples the 1m bars into 5m / 30m / 4h / 1d candles (leak-safe). You never build
higher timeframes yourself.

## In the notebooks you only set:
    SYMBOL     = "EURUSD"
    DRIVE_ROOT = "/content/drive/MyDrive/Camillion_data"
`CSV_PATH` is derived as `DRIVE_ROOT/<SYMBOL>_1m.csv`. Everything else is defaulted.
