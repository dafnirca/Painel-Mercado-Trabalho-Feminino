"""
02_transform.py — Etapa de TRANSFORMAÇÃO do pipeline.

Lê o arquivo já agregado gerado por 01_ingest.py em data/raw/ e aplica
uma checagem final de qualidade antes de salvar em data/processed/.

A remoção de nulos e remunerações inválidas já acontece dentro da query
SQL de 01_ingest.py. Esta etapa cuida do que só faz sentido tratar
depois de baixado: duplicatas exatas, checagem de tipos, e o
enriquecimento com a coluna setor_amigavel.

SOBRE setor_amigavel: as 87 divisões oficiais do CNAE (setor_descricao)
usam nomenclatura técnica de governo, pouco reconhecível para quem não
trabalha com classificação econômica. data/mappings/setor_amigavel.csv
é uma tabela de mapeamento (dimensão conformada) que agrupa as 87
divisões em 24 categorias de leitura imediata (ex.: "Tecnologia da
Informação e Telecomunicações", "Saúde", "Comércio e Varejo"). A coluna
setor_descricao (CNAE oficial) é mantida intacta — setor_amigavel é uma
coluna adicional, não uma substituição, para permitir ao painel oferecer
o filtro amplo como padrão e o detalhe técnico como opção secundária.

Como rodar:
    python src/02_transform.py --input data/raw/rais_agregado_BR_2020_2024.parquet
"""

import argparse
from pathlib import Path
import pandas as pd

MAPEAMENTO_SETOR_PATH = Path("data/mappings/setor_amigavel.csv")


def aplicar_setor_amigavel(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    mapeamento = pd.read_csv(MAPEAMENTO_SETOR_PATH)
    df = df.merge(mapeamento, on="setor_descricao", how="left")
    nao_mapeados = sorted(df.loc[df["setor_amigavel"].isna(), "setor_descricao"].dropna().unique().tolist())
    df["setor_amigavel"] = df["setor_amigavel"].fillna("Outros / não classificado")
    return df, nao_mapeados


def tratar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    relatorio = {"linhas_entrada": len(df)}

    antes = len(df)
    df = df.drop_duplicates()
    relatorio["removidas_duplicadas"] = antes - len(df)

    df["remuneracao_media"] = df["remuneracao_media"].astype(float)
    df["qtd_vinculos"] = df["qtd_vinculos"].astype(int)

    df, nao_mapeados = aplicar_setor_amigavel(df)
    relatorio["setores_sem_categoria_amigavel"] = nao_mapeados

    relatorio["linhas_saida"] = len(df)
    relatorio["total_vinculos_representados"] = int(df["qtd_vinculos"].sum())
    return df, relatorio


def main():
    parser = argparse.ArgumentParser(description="Trata o dataset agregado da RAIS.")
    parser.add_argument("--input", required=True, help="Caminho do parquet agregado (data/raw/...)")
    parser.add_argument("--output", default=None, help="Caminho de saída (default: data/processed/<nome>.parquet)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df_tratado, relatorio = tratar(df)

    out_path = args.output or args.input.replace("data/raw", "data/processed")
    df_tratado.to_parquet(out_path, index=False)

    print("Relatório de qualidade do tratamento:")
    for chave, valor in relatorio.items():
        if chave == "setores_sem_categoria_amigavel":
            if valor:
                print(f"  ⚠ {len(valor)} setor(es) SEM categoria amigável definida (revisar data/mappings/setor_amigavel.csv):")
                for s in valor:
                    print(f"      - {s}")
            else:
                print("  setores_sem_categoria_amigavel: nenhum — todos os setores do recorte estão mapeados")
        else:
            print(f"  {chave}: {valor:,}")
    print(f"\nDataset tratado salvo em {out_path}")


if __name__ == "__main__":
    main()
