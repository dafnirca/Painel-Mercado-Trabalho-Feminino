"""
00_check_schema.py

Confirma os nomes exatos das colunas disponíveis na tabela de microdados
da RAIS na Base dos Dados. Útil para checar se a estrutura mudou em
versões futuras da tabela.

Como rodar:
    python src/00_check_schema.py
"""

import basedosdados as bd

BILLING_PROJECT_ID = "painel-mercado-feminino"

QUERY_SCHEMA = """
SELECT column_name, data_type
FROM `basedosdados.br_me_rais.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'microdados_vinculos'
"""

if __name__ == "__main__":
    df_schema = bd.read_sql(QUERY_SCHEMA, billing_project_id=BILLING_PROJECT_ID)
    print(df_schema.to_string())
    df_schema.to_csv("data/raw/_schema_microdados_vinculos.csv", index=False)
