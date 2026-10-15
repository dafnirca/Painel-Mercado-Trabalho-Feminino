# Pipeline de Dados — Painel Mercado de Trabalho Feminino no Brasil

Pipeline de extração e tratamento dos microdados da RAIS, estruturado em
etapas isoladas (ingestão já agregada → transformação → consolidação →
aplicação Streamlit).

## Estrutura

```
data/raw/         # saída bruta (já agregada) da extração (01_ingest.py)
data/processed/   # dataset tratado e consolidado (rais_consolidado.parquet)
data/geo/         # geojson dos estados do Brasil, usado nos mapas
data/mappings/    # setor_amigavel.csv — 87 divisões CNAE → 24 categorias amigáveis
src/
  00_check_schema.py   # confirma nomes de colunas na Base dos Dados
  01_ingest.py         # extrai e agrega os microdados da RAIS via BigQuery
  02_transform.py      # qualidade dos dados + aplica o mapeamento de setor_amigavel
  03_consolidate.py    # junta os arquivos tratados no dataset final
app.py             # aplicação Streamlit
```

## Como rodar

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python src\01_ingest.py --ano-inicio 2020 --ano-fim 2024
python src\02_transform.py --input data\raw\rais_agregado_BR_2020_2024.parquet
python src\03_consolidate.py
streamlit run app.py
```

Note que 03_consolidate.py é sempre o último passo, mesmo quando
01_ingest.py foi rodado uma única vez para o Brasil inteiro — é ele quem
gera o arquivo com o nome fixo que app.py espera encontrar.

## Fonte de dados

RAIS (Relação Anual de Informações Sociais), Ministério do Trabalho e
Emprego, acessada via [Base dos Dados](https://basedosdados.org)
(`basedosdados.br_me_rais.microdados_vinculos`).

## Design do painel

Narrativa em 5 telas + Início (navegação por botões no topo), com filtros fixos
na barra lateral válidos para todas as telas:

| Tela | Pergunta | Principais visuais |
|---|---|---|
| Início | O que é, como usar, limitações | Passos, fonte, setores (87 → 24), avisos |
| 1 · Panorama | Como está o mercado hoje? | Mapa clicável (3 métricas), evolução da participação, regiões, top setores |
| 2 · Onde estão | Em que setores e estados? | Presença e concentração por setor, estados, dispersão presença × salário |
| 3 · Quanto ganham | Onde o gap é maior e muda? | Salário F×M e gap por setor, evolução, gap por estado, faixas salariais |
| 4 · Perfil | Idade e raça/cor mudam o quadro? | Faixa etária, raça/cor, mapa de calor raça × idade |
| 5 · Comparar setores | Como se comparam lado a lado? | Tabela ordenável, busca, download CSV |

**Clique no mapa:** clicar em um estado (Panorama) define o "Estado em foco" e filtra
o painel inteiro; o mesmo filtro está na barra lateral. Requer Streamlit >= 1.40
(sem isso, o filtro da barra lateral continua funcionando).

**Dois conceitos de tempo:** "Ano de referência" (mapa, setores, perfil) e
"Período da evolução" (só gráficos de linha). Evita misturar R$ de anos diferentes.

Paleta: Bubblegum Pink (#FF5367, série Feminino), Vintage Berry (#81475D,
série Masculino), Bright Amber (#FDD22B, estado em foco/destaques), Sweet Salmon
(#FF9885, extremo baixo das escalas). Cada cor tem um único papel.

## Status

- [x] Arquitetura do pipeline definida e codificada
- [x] Query validada contra a interface oficial da Base dos Dados
- [x] Setor econômico mapeado: 87 divisões do CNAE → 24 categorias amigáveis
- [x] Extração real concluída — Brasil, 2020-2025 (~4,1 milhões de linhas agregadas, representando ~439 milhões de vínculos)
- [x] Aplicação Streamlit com narrativa em 5 telas + Início, filtros na barra lateral e clique no mapa
- [x] Correção de tema: `backgroundColor` do app (antes branco puro, agora cinza-claro `#F2F2F2`, igual à `secondaryBackgroundColor`)
- [x] Refinamento de design e usabilidade (tipografia maior, cartões com conclusão, tela Início)
- [ ] Validar o design com dados reais e com as usuárias (avaliação de outubro)
- [ ] Nova extração para incluir escolaridade (avaliação de custo/benefício pendente)
