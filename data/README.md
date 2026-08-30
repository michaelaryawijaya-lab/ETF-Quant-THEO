# Data

Raw provider downloads are cached in `data/raw/` and intentionally ignored by Git. Processed, aligned panels belong in `data/processed/`, also ignored by Git. The loader uses an inner join, drops incomplete rows, and never forward-fills quotes.
