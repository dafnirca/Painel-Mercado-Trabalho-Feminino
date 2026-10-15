"""
01_ingest.py — Etapa de INGESTÃO do pipeline.

Extrai os microdados da RAIS da plataforma Base dos Dados (BigQuery) e
salva o resultado em data/raw/, JÁ AGREGADO por combinação de
(ano, estado, sexo, faixa etária, raça/cor, setor econômico, faixa
salarial), com a média de remuneração e a contagem de vínculos por grupo.

Usamos setor_descricao no nível de DIVISÃO do CNAE 2.0 (~87 categorias,
ex.: "Atividades dos serviços de tecnologia da informação"), não no
nível de SEÇÃO (~21 categorias amplas, ex.: "Informação e comunicação").
A seção é mantida como coluna auxiliar (setor_secao) para agrupamentos
mais amplos, se necessário no futuro.

POR QUE AGREGAR NA QUERY, E NÃO DEPOIS EM PANDAS:
O painel só precisa de médias e contagens por grupo — nenhuma
visualização usa o dado no nível de vínculo individual. Buscar os
microdados linha a linha (potencialmente milhões de registros) e agregar
depois localmente desperdiça banda e tempo à toa. Colocar o GROUP BY
dentro da própria query do BigQuery reduz o resultado baixado de milhões
de linhas para, tipicamente, alguns milhares — permitindo trazer o
Brasil inteiro numa única chamada.

Como rodar (Brasil inteiro, período 2020-2024):
    python src/01_ingest.py --ano-inicio 2020 --ano-fim 2024

Como rodar (só alguns estados, útil para testes rápidos):
    python src/01_ingest.py --ufs RS,SP,BA --ano-inicio 2020 --ano-fim 2024
"""

import argparse
import basedosdados as bd

BILLING_PROJECT_ID = "painel-mercado-feminino"


def build_query(ufs: list[str] | None, ano_inicio: int, ano_fim: int) -> str:
    filtro_uf = ""
    if ufs:
        lista_ufs = ", ".join(f"'{uf.strip().upper()}'" for uf in ufs)
        filtro_uf = f"AND dados.sigla_uf IN ({lista_ufs})"

    return f"""
    WITH
    dicionario_sexo AS (
        SELECT chave AS chave_sexo, valor AS descricao_sexo
        FROM `basedosdados.br_me_rais.dicionario`
        WHERE nome_coluna = 'sexo' AND id_tabela = 'microdados_vinculos'
    ),
    dicionario_faixa_etaria AS (
        SELECT chave AS chave_faixa_etaria, valor AS descricao_faixa_etaria
        FROM `basedosdados.br_me_rais.dicionario`
        WHERE nome_coluna = 'faixa_etaria' AND id_tabela = 'microdados_vinculos'
    ),
    dicionario_raca_cor AS (
        SELECT chave AS chave_raca_cor, valor AS descricao_raca_cor
        FROM `basedosdados.br_me_rais.dicionario`
        WHERE nome_coluna = 'raca_cor' AND id_tabela = 'microdados_vinculos'
    ),
    dicionario_faixa_remuneracao_media_sm AS (
        SELECT chave AS chave_faixa_remuneracao_media_sm, valor AS descricao_faixa_remuneracao_media_sm
        FROM `basedosdados.br_me_rais.dicionario`
        WHERE nome_coluna = 'faixa_remuneracao_media_sm' AND id_tabela = 'microdados_vinculos'
    ),
    base AS (
        SELECT
            dados.ano AS ano,
            dados.sigla_uf AS sigla_uf,
            diretorio_sigla_uf.nome AS sigla_uf_nome,
            descricao_sexo AS sexo,
            descricao_faixa_etaria AS faixa_etaria,
            descricao_raca_cor AS raca_cor,
            diretorio_cnae_2.descricao_secao AS setor_secao,
            diretorio_cnae_2.descricao_divisao AS setor_descricao,
            descricao_faixa_remuneracao_media_sm AS faixa_remuneracao_media_sm,
            dados.valor_remuneracao_media AS valor_remuneracao_media
        FROM `basedosdados.br_me_rais.microdados_vinculos` AS dados
        LEFT JOIN (
            SELECT DISTINCT sigla, nome FROM `basedosdados.br_bd_diretorios_brasil.uf`
        ) AS diretorio_sigla_uf
            ON dados.sigla_uf = diretorio_sigla_uf.sigla
        LEFT JOIN `dicionario_sexo`
            ON dados.sexo = chave_sexo
        LEFT JOIN `dicionario_faixa_etaria`
            ON dados.faixa_etaria = chave_faixa_etaria
        LEFT JOIN `dicionario_raca_cor`
            ON dados.raca_cor = chave_raca_cor
        LEFT JOIN `dicionario_faixa_remuneracao_media_sm`
            ON dados.faixa_remuneracao_media_sm = chave_faixa_remuneracao_media_sm
        LEFT JOIN (
            SELECT DISTINCT subclasse, descricao_secao, descricao_divisao
            FROM `basedosdados.br_bd_diretorios_brasil.cnae_2`
        ) AS diretorio_cnae_2
            ON dados.cnae_2_subclasse = diretorio_cnae_2.subclasse
        WHERE dados.ano BETWEEN {ano_inicio} AND {ano_fim}
          AND dados.valor_remuneracao_media > 0
          AND dados.sexo IS NOT NULL
          {filtro_uf}
    )
    SELECT
        ano,
        sigla_uf,
        sigla_uf_nome,
        sexo,
        faixa_etaria,
        raca_cor,
        setor_secao,
        setor_descricao,
        faixa_remuneracao_media_sm,
        AVG(valor_remuneracao_media) AS remuneracao_media,
        COUNT(*) AS qtd_vinculos
    FROM base
    GROUP BY ano, sigla_uf, sigla_uf_nome, sexo, faixa_etaria, raca_cor, setor_secao, setor_descricao, faixa_remuneracao_media_sm
    """


def main():
    parser = argparse.ArgumentParser(description="Extrai (já agregados) os microdados da RAIS.")
    parser.add_argument("--ufs", default=None, help="Lista de UFs separadas por vírgula (ex.: RS,SP,BA). Omitir = Brasil inteiro.")
    parser.add_argument("--ano-inicio", type=int, required=True)
    parser.add_argument("--ano-fim", type=int, required=True)
    args = parser.parse_args()

    ufs = args.ufs.split(",") if args.ufs else None
    query = build_query(ufs, args.ano_inicio, args.ano_fim)
    print("Executando query agregada:\n", query)

    df = bd.read_sql(query, billing_project_id=BILLING_PROJECT_ID)

    sufixo_uf = "_".join(ufs) if ufs else "BR"
    out_path = f"data/raw/rais_agregado_{sufixo_uf}_{args.ano_inicio}_{args.ano_fim}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"{len(df):,} linhas (já agregadas) salvas em {out_path}")


if __name__ == "__main__":
    main()
