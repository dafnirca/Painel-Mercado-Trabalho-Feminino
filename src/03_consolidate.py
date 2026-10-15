"""
03_consolidate.py — Etapa de CONSOLIDAÇÃO do pipeline.

Junta os arquivos tratados de data/processed/ (rais_agregado_*.parquet)
num único dataset final (rais_consolidado.parquet), o nome fixo que
app.py espera encontrar. Sempre rode este script por último, mesmo que
01_ingest.py tenha sido chamado uma única vez para o Brasil inteiro.

Como rodar:
    python src/03_consolidate.py
"""

from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "rais_consolidado.parquet"


def main():
    arquivos = [
        f for f in PROCESSED_DIR.glob("rais_agregado_*.parquet")
        if f.name != OUTPUT_PATH.name
    ]

    if not arquivos:
        print(f"Nenhum arquivo tratado encontrado em {PROCESSED_DIR}. "
              "Rode 01_ingest.py e 02_transform.py antes.")
        return

    print(f"Consolidando {len(arquivos)} arquivo(s):")
    for f in arquivos:
        print(f"  - {f.name}")

    df = pd.concat([pd.read_parquet(f) for f in arquivos], ignore_index=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n{len(df):,} linhas consolidadas em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
