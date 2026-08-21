# Painel de Dados Interativo sobre o Mercado de Trabalho Feminino no Brasil

> **Projeto em desenvolvimento.** Este é o README inicial, criado antes do início da implementação. As seções de arquitetura, instalação e uso serão atualizadas conforme o painel for construído (cronograma abaixo).

> Trabalho de Conclusão de Curso — Tecnologia em Análise e Desenvolvimento de Sistemas (UNISINOS), dez/2026.
> Painel interativo que vai consolidar dados públicos (RAIS) sobre salário, presença e evolução das mulheres no mercado de trabalho brasileiro por setor, estado e período — hoje fragmentados em múltiplas fontes.

---

## O problema

Apesar de existirem iniciativas governamentais de visualização de dados sobre desigualdade de gênero no Brasil (ex. DataMulheres), nenhuma delas é voltada especificamente para mulheres como usuárias finais em processo de decisão de carreira — a informação está espalhada entre relatórios, plataformas e formatos diferentes. Este projeto vai consolidar esses dados em um único painel navegável, com base nos requisitos já levantados diretamente com o público-alvo (mulheres em decisão de carreira, professoras/pesquisadoras e recrutadoras).

## O que o painel vai entregar

- Salário médio por setor econômico e diferença salarial entre mulheres e homens
- Percentual e evolução da presença feminina por setor e por estado
- Filtros por setor, estado, período, faixa etária e faixa salarial
- Visualizações em barras, mapa por estado e linhas de evolução temporal

*(Escopo validado pelo questionário de levantamento de requisitos aplicado no Projeto Final I.)*

## Stack planejada

| Camada | Tecnologia |
|---|---|
| Linguagem | Python |
| Manipulação de dados | Pandas |
| Visualização | Plotly, Folium |
| Interface / app | Streamlit |
| Fonte de dados | RAIS (Relação Anual de Informações Sociais), via [Base dos Dados](https://basedosdados.org) |
| Publicação | Streamlit Community Cloud |

## Status do projeto

| Fase | Período | Status |
|---|---|---|
| Extração e tratamento dos microdados da RAIS | Ago/2026 | Não iniciado |
| Desenvolvimento do painel interativo | Set/2026 | Não iniciado |
| Publicação e aplicação do questionário de avaliação | Out/2026 | Não iniciado |
| Análise dos resultados e ajustes no artefato | Nov/2026 | Não iniciado |
| Redação final e defesa | Dez/2026 | Não iniciado |

## Estrutura planejada do repositório

```
├── data/           # raw (não versionado) e processed
├── src/            # ingestão, transformação, pipeline
├── app/            # aplicação Streamlit
├── notebooks/       # exploração pontual
├── tests/           # testes de validação de dados/transformações
├── docs/            # decisões técnicas, modelo de dados, requisitos
└── requirements.txt
```

## Contexto acadêmico

Este projeto segue a metodologia **Design Science Research (DSR)**, seguindo o ciclo regulador de Wieringa (investigar → projetar → validar → implementar → avaliar). É o artefato central do Projeto Final I e II do curso, com validação em duas etapas: levantamento de requisitos (Projeto Final I, concluído) e avaliação de usabilidade com o público-alvo após a publicação (Projeto Final II, out/2026).

-  Síntese dos requisitos levantados: [`docs/requisitos.md`](docs/requisitos.md) *(a adicionar)*

## Autora

**Dafni Rosa** — Estagiária de MIS (Alura) · Tecnologia em Análise e Desenvolvimento de Sistemas

[LinkedIn](https://www.linkedin.com/in/dafni-rosa-76466a23a/)
