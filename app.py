"""
app.py — Painel interativo do mercado de trabalho feminino no Brasil.

Consome exclusivamente o dataset já agregado e tratado em
data/processed/rais_consolidado.parquet (uma linha por combinação de
ano, estado, sexo, faixa etária, raça/cor, setor e faixa salarial, com a
remuneração média do grupo e a quantidade de vínculos que ela
representa). Toda média recalculada aqui é PONDERADA por qtd_vinculos —
nunca uma média simples das médias dos grupos.

NARRATIVA (Início → contexto → visão macro → exploração → comparação)
    Início              O que é, como usar, de onde vêm os dados, limitações
    1 · Panorama        Como está o mercado de trabalho feminino hoje?
    2 · Onde estão      Em que setores e estados as mulheres estão?
    3 · Quanto ganham   Onde o gap salarial é maior e ele está mudando?
    4 · Perfil          Idade e raça/cor mudam o quadro?
    5 · Comparar        Como os setores se comparam lado a lado?

Cada tela abre com uma pergunta e uma frase-resposta gerada a partir dos
dados filtrados; cada gráfico fica num cartão com título, contexto e uma
conclusão — assim os gráficos conversam entre si em vez de ficarem soltos.

FILTROS (barra lateral fixa) valem para TODAS as telas. O clique em um
estado no mapa do Panorama define o "Estado em foco" e filtra o painel
inteiro (equivale a escolher o estado na barra lateral).

DOIS CONCEITOS DE TEMPO — decisão para não misturar R$ de anos diferentes:
    Ano de referência     gráficos "de retrato" (mapa, setores, perfil)
    Período da evolução   apenas os gráficos de linha (evolução no tempo)

Identidade visual — paleta definida pela autora; cada cor tem um papel:
    Bubblegum Pink  #FF5367 — série "Feminino" (protagonista)
    Vintage Berry   #81475D — série "Masculino" / comparação
    Bright Amber    #FDD22B — único acento interativo (estado em foco, destaques)
    Sweet Salmon    #FF9885 — extremo "baixo" das escalas de cor

Como rodar:
    streamlit run app.py
"""

import json
import re
import textwrap
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

DATA_PATH = Path("data/processed/rais_consolidado.parquet")
GEOJSON_PATH = Path("data/geo/brazil_states.geojson")

COR_FEMININO = "#FF5367"    # Bubblegum Pink
COR_MASCULINO = "#81475D"   # Vintage Berry
COR_ACENTO = "#FDD22B"      # Bright Amber
COR_SECUNDARIA = "#FF9885"  # Sweet Salmon
COR_TEXTO = "#262626"
COR_MUTED = "#6B6B6B"
COR_GRADE = "#ECECEC"
COR_FUNDO = "#F2F2F2"

ESCALA_ROSA = [[0.0, "#FFF1F2"], [1.0, COR_FEMININO]]
ESCALA_GAP = [[0.0, "#FFE1D9"], [0.5, COR_SECUNDARIA], [1.0, COR_MASCULINO]]

FONTE_CORPO = "Work Sans, sans-serif"
FONTE_TITULO = "Sora, sans-serif"

SEXO_F, SEXO_M = "Feminino", "Masculino"
BRASIL = "Brasil inteiro"
MIN_SHARE_SETOR = 0.003      # setores com < 0,3% dos vínculos não entram em rankings (evita ruído de setor minúsculo)
MIN_VINCULOS_CELULA = 200    # células do mapa de calor com menos vínculos que isso ficam em branco

REGIAO_DA_UF = {
    **dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "Norte"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
}

PAGINAS = [
    "Início",
    "1 · Panorama",
    "2 · Onde estão",
    "3 · Quanto ganham",
    "4 · Perfil",
    "5 · Comparar setores",
]

st.set_page_config(
    page_title="Mercado de Trabalho Feminino no Brasil",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Compatibilidade entre versões do Streamlit
# ─────────────────────────────────────────────────────────────────────────────


def _versao_streamlit() -> tuple:
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


# A partir da 1.51, "use_container_width" está depreciado em favor de width="stretch".
LARGURA = {"width": "stretch"} if _versao_streamlit() >= (1, 51) else {"use_container_width": True}

_contador_cartao = {"n": 0}
# "localized" (separador de milhar do navegador) existe nas versões mais novas; antes disso, inteiro simples.
FORMATO_INTEIRO = "localized" if _versao_streamlit() >= (1, 42) else "%d"

# ─────────────────────────────────────────────────────────────────────────────
# Formatação (padrão brasileiro)
# ─────────────────────────────────────────────────────────────────────────────


def _vazio(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)


def brl(v) -> str:
    return "—" if _vazio(v) else "R$ " + f"{v:,.0f}".replace(",", ".")


def pct(v, casas: int = 1) -> str:
    return "—" if _vazio(v) else f"{v:.{casas}f}%".replace(".", ",")


def num(v) -> str:
    return "—" if _vazio(v) else f"{int(round(v)):,}".replace(",", ".")


def pp(v) -> str:
    return "—" if _vazio(v) else f"{v:+.1f}".replace(".", ",") + " p.p."


def quebrar(txt, n: int = 26) -> str:
    """Quebra rótulos longos em várias linhas (<br>) para não espremer o gráfico."""
    return "<br>".join(textwrap.wrap(str(txt), n)) or str(txt)


def chave_ordem(rotulo: str) -> float:
    """Ordena faixas como '18 a 24', 'Até 0,50', '0,51 a 1,00', 'Mais de 20,00' pelo primeiro número."""
    r = str(rotulo)
    if r.lower().startswith("até"):
        return -1.0
    m = re.search(r"\d+(?:[.,]\d+)?", r)
    return float(m.group().replace(",", ".")) if m else 1e9


def ordenar_faixas(valores) -> list:
    return sorted([v for v in valores if pd.notna(v)], key=chave_ordem)


def limite_superior(faixa: str) -> float:
    """Limite superior (em SM) de uma faixa salarial; 'Mais de X' → infinito."""
    f = str(faixa)
    if f.lower().startswith("mais"):
        return float("inf")
    numeros = re.findall(r"\d+(?:[.,]\d+)?", f)
    return float(numeros[-1].replace(",", ".")) if numeros else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Dados
# ─────────────────────────────────────────────────────────────────────────────

COLUNAS_CATEGORICAS = [
    "sigla_uf", "sigla_uf_nome", "sexo", "faixa_etaria", "raca_cor",
    "setor_secao", "setor_descricao", "setor_amigavel", "faixa_remuneracao_media_sm",
]


@st.cache_data(show_spinner="Carregando os dados…")
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["soma_pond"] = df["remuneracao_media"].astype(float) * df["qtd_vinculos"].astype(float)
    df["regiao"] = df["sigla_uf"].astype(str).map(REGIAO_DA_UF).fillna("Outra")
    for c in COLUNAS_CATEGORICAS + ["regiao"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


@st.cache_data
def load_catalogo(path: Path) -> dict:
    """Listas pequenas usadas pelos filtros e pela tela Início (lidas só das colunas necessárias)."""
    cols = ["ano", "sigla_uf", "sigla_uf_nome", "setor_amigavel", "setor_descricao",
            "faixa_etaria", "faixa_remuneracao_media_sm", "qtd_vinculos"]
    d = pd.read_parquet(path, columns=cols)
    ufs = d[["sigla_uf", "sigla_uf_nome"]].drop_duplicates().astype(str)
    setores = d[["setor_amigavel", "setor_descricao"]].drop_duplicates().astype(str)
    return {
        "anos": sorted(int(a) for a in d["ano"].unique()),
        "uf_nomes": dict(zip(ufs["sigla_uf"], ufs["sigla_uf_nome"])),
        "setores": setores.sort_values(["setor_amigavel", "setor_descricao"]).reset_index(drop=True),
        "faixas_etarias": ordenar_faixas(d["faixa_etaria"].dropna().astype(str).unique()),
        "faixas_salariais": ordenar_faixas(d["faixa_remuneracao_media_sm"].dropna().astype(str).unique()),
        "linhas": int(len(d)),
        "vinculos_total": int(d["qtd_vinculos"].sum()),
    }


@st.cache_data
def load_geojson(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def agregar(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Soma vínculos e recalcula a remuneração média PONDERADA por vínculos."""
    g = df.groupby(cols, observed=True, as_index=False).agg(
        qtd_vinculos=("qtd_vinculos", "sum"), soma=("soma_pond", "sum")
    )
    g["remuneracao_media"] = g["soma"] / g["qtd_vinculos"]
    return g.drop(columns="soma")


def painel_sexo(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Uma linha por grupo com: qtd_f, qtd_m, rem_f, rem_m, total, part (% feminina), gap (%)."""
    a = agregar(df, cols + ["sexo"])
    f = a[a["sexo"] == SEXO_F].drop(columns="sexo").rename(columns={"qtd_vinculos": "qtd_f", "remuneracao_media": "rem_f"})
    m = a[a["sexo"] == SEXO_M].drop(columns="sexo").rename(columns={"qtd_vinculos": "qtd_m", "remuneracao_media": "rem_m"})
    out = f.merge(m, on=cols, how="outer")
    for c in ("qtd_f", "qtd_m"):
        out[c] = out[c].fillna(0)
    out["total"] = out["qtd_f"] + out["qtd_m"]
    out["part"] = np.where(out["total"] > 0, out["qtd_f"] / out["total"] * 100, np.nan)
    out["gap"] = (out["rem_m"] - out["rem_f"]) / out["rem_m"] * 100
    return out


def totais(df: pd.DataFrame) -> dict:
    f = df[df["sexo"] == SEXO_F]
    m = df[df["sexo"] == SEXO_M]
    qf, qm = float(f["qtd_vinculos"].sum()), float(m["qtd_vinculos"].sum())
    rf = f["soma_pond"].sum() / qf if qf else np.nan
    rm = m["soma_pond"].sum() / qm if qm else np.nan
    tot = qf + qm
    return {
        "qtd_f": qf, "qtd_m": qm, "total": tot, "rem_f": rf, "rem_m": rm,
        "part": qf / tot * 100 if tot else np.nan,
        "gap": (rm - rf) / rm * 100 if (qf and qm and rm) else np.nan,
    }


def relevantes(pan: pd.DataFrame, fracao: float = MIN_SHARE_SETOR) -> pd.DataFrame:
    """Remove grupos minúsculos (ruído) dos rankings."""
    if pan.empty:
        return pan
    return pan[pan["total"] >= fracao * pan["total"].sum()]


# ─────────────────────────────────────────────────────────────────────────────
# Estilo
# ─────────────────────────────────────────────────────────────────────────────


def aplicar_estilo():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Work+Sans:wght@400;500;600&display=swap');

        html {{ font-size: 110%; }}
        html, body, .stApp, [data-testid="stMarkdownContainer"], .stMarkdown p, label, li {{
            font-family: 'Work Sans', sans-serif; color: {COR_TEXTO};
        }}
        .stApp {{ background: {COR_FUNDO}; }}
        .block-container {{ max-width: 1500px; padding-top: 2.2rem; padding-bottom: 4rem; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        .stDeployButton, [data-testid="stAppDeployButton"] {{ display: none; }}

        /* Barra lateral */
        [data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {COR_GRADE}; }}
        [data-testid="stSidebar"] label p {{ font-weight: 600; font-size: 0.95rem; }}
        .side-marca {{ font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.15rem; line-height: 1.25; color: {COR_TEXTO}; }}
        .side-marca span {{ color: {COR_FEMININO}; }}
        .side-grupo {{ font-size: 0.78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
                      color: {COR_MUTED}; margin: 1.1rem 0 .1rem; }}
        .side-foco {{ background: {COR_ACENTO}; border-radius: 12px; padding: .55rem .8rem; font-weight: 600;
                     font-size: .95rem; margin: .3rem 0 .5rem; }}

        /* Tipografia das telas */
        .kicker {{ font-size: .85rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: {COR_FEMININO}; margin-bottom: .2rem; }}
        .pg-titulo {{ font-family: 'Sora', sans-serif; font-weight: 700; font-size: 2.15rem; line-height: 1.2; margin: 0 0 .6rem; }}
        .manchete {{ font-size: 1.3rem; line-height: 1.5; background: #FFFFFF; border-left: 6px solid {COR_FEMININO};
                    border-radius: 14px; padding: 1rem 1.3rem; margin: .4rem 0 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,.05); }}
        .manchete b {{ color: {COR_FEMININO}; }}

        /* Chips de recorte */
        .chips {{ margin: .2rem 0 1rem; }}
        .chip {{ display: inline-block; background: #FFFFFF; border: 1px solid {COR_GRADE}; border-radius: 999px;
                padding: .3rem .9rem; font-size: .9rem; font-weight: 600; color: #4A4A4A; margin: 0 .5rem .4rem 0; }}
        .chip-foco {{ background: {COR_ACENTO}; border-color: {COR_ACENTO}; color: {COR_TEXTO}; }}

        /* KPIs */
        .kpi {{ background: #FFFFFF; border: 1px solid {COR_GRADE}; border-radius: 20px; padding: 1.1rem 1.3rem; min-height: 172px; }}
        .kpi-destaque {{ background: {COR_FEMININO}; border: none; box-shadow: 0 6px 18px rgba(255,83,103,.30); }}
        .kpi-destaque .kpi-r, .kpi-destaque .kpi-v, .kpi-destaque .kpi-s {{ color: #FFFFFF; }}
        .kpi-r {{ font-size: .95rem; font-weight: 600; color: {COR_MUTED}; }}
        .kpi-v {{ font-family: 'Sora', sans-serif; font-weight: 700; font-size: 2.3rem; line-height: 1.15; margin: .15rem 0; }}
        .kpi-s {{ font-size: .88rem; color: {COR_MUTED}; }}

        /* Cartões de gráfico */
        div[class*="st-key-card_"] {{ background: #FFFFFF; border: 1px solid {COR_GRADE} !important; border-radius: 20px !important;
                                     padding: 1.2rem 1.4rem 1.1rem; box-shadow: 0 1px 3px rgba(0,0,0,.04); }}
        .cartao-titulo {{ font-family: 'Sora', sans-serif; font-weight: 600; font-size: 1.15rem; line-height: 1.3; margin-bottom: .15rem; }}
        .cartao-ctx {{ font-size: .95rem; color: {COR_MUTED}; margin-bottom: .6rem; }}
        .conclusao {{ font-size: 1rem; line-height: 1.45; border-left: 4px solid {COR_ACENTO}; background: #FFFBEA;
                     border-radius: 0 10px 10px 0; padding: .55rem .9rem; margin-top: .5rem; }}

        /* Destaques da tela Comparar */
        .destaque {{ background: #FFFFFF; border: 1px solid {COR_GRADE}; border-radius: 18px; padding: 1rem 1.2rem; min-height: 150px; }}
        .destaque-r {{ font-size: .85rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: {COR_MUTED}; }}
        .destaque-v {{ font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.5rem; color: {COR_FEMININO}; margin: .15rem 0; }}
        .destaque-s {{ font-size: .95rem; line-height: 1.35; }}

        /* Tela Início */
        .passo {{ background: #FFFFFF; border: 1px solid {COR_GRADE}; border-radius: 18px; padding: 1rem 1.1rem; min-height: 245px; }}
        .passo-curto {{ min-height: 190px; }}
        .passo-n {{ display: inline-block; width: 2rem; height: 2rem; line-height: 2rem; text-align: center; border-radius: 50%;
                   background: {COR_FEMININO}; color: #FFFFFF; font-weight: 700; margin-bottom: .4rem; }}
        .passo-t {{ font-family: 'Sora', sans-serif; font-weight: 600; font-size: 1.05rem; margin-bottom: .2rem; }}
        .passo-q {{ font-size: .95rem; color: {COR_MUTED}; line-height: 1.4; }}
        .aviso {{ background: #FFFBEA; border: 1px solid {COR_ACENTO}; border-radius: 16px; padding: 1rem 1.3rem; }}
        .secao {{ font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.45rem; margin: 2rem 0 .6rem; }}

        /* Navegação e seletores em formato de pílula (HTML novo e antigo do Streamlit) */
        div[data-testid="stRadio"] div[role="radiogroup"] {{ gap: .5rem; flex-wrap: wrap; }}
        div[data-testid="stRadio"] label[data-testid="stRadioOption"],
        div[data-testid="stRadio"] label[data-baseweb="radio"] {{
            background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 999px; padding: .4rem 1.05rem; margin: 0; cursor: pointer; }}
        div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]),
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{ display: none; }}
        div[data-testid="stRadio"] label:has(input:checked) {{ background: {COR_FEMININO}; border-color: {COR_FEMININO}; }}
        div[data-testid="stRadio"] label:has(input:checked) p {{ color: #FFFFFF; }}
        div[data-testid="stRadio"] label p {{ font-weight: 600; font-size: 1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def estilizar(fig, altura: int = 330, legenda: bool = False, horizontal: bool = False):
    """Acabamento único para todo gráfico Plotly: fonte grande, fundo transparente,
    grade discreta, legenda no topo, separadores no padrão brasileiro."""
    fig.update_layout(
        font=dict(family=FONTE_CORPO, size=14, color=COR_TEXTO),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=altura,
        margin=dict(t=44 if legenda else 8, b=8, l=8, r=8),
        showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None, font_size=14),
        hoverlabel=dict(bgcolor="white", font_size=14, font_family=FONTE_CORPO, bordercolor="#E0E0E0"),
        separators=",.",
        bargap=0.28,
    )
    if horizontal:
        fig.update_xaxes(showgrid=True, gridcolor=COR_GRADE, zeroline=False, showline=False, title=None, tickfont_size=13, tickformat=",.0f")
        fig.update_yaxes(showgrid=False, showline=False, title=None, tickfont_size=13, automargin=True)
    else:
        fig.update_xaxes(showgrid=False, showline=True, linecolor="#DADADA", title=None, tickfont_size=13, automargin=True)
        fig.update_yaxes(showgrid=True, gridcolor=COR_GRADE, zeroline=False, showline=False, title=None, tickfont_size=13, tickformat=",.0f")
    try:
        fig.update_traces(marker_cornerradius=6, selector=dict(type="bar"))
    except Exception:
        pass
    return fig


def mostrar(fig, chave: str | None = None):
    kwargs = dict(config={"displayModeBar": False}, **LARGURA)
    if chave:
        kwargs["key"] = chave
    st.plotly_chart(fig, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Componentes de interface
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def cartao(titulo: str, contexto: str | None = None):
    """Cartão branco com título, contexto e (dentro do bloco) o gráfico e a conclusão."""
    _contador_cartao["n"] += 1
    try:
        caixa = st.container(border=True, key=f"card_{_contador_cartao['n']}")
    except TypeError:  # Streamlit antigo, sem 'key' em container
        caixa = st.container(border=True)
    with caixa:
        html = f'<div class="cartao-titulo">{titulo}</div>'
        if contexto:
            html += f'<div class="cartao-ctx">{contexto}</div>'
        st.markdown(html, unsafe_allow_html=True)
        yield


def conclusao(texto: str | None):
    if texto:
        st.markdown(f'<div class="conclusao">{texto}</div>', unsafe_allow_html=True)


def manchete(texto: str):
    st.markdown(f'<div class="manchete">{texto}</div>', unsafe_allow_html=True)


def cabecalho_pagina(kicker: str, pergunta: str, ctx=None):
    st.markdown(f'<div class="kicker">{kicker}</div><div class="pg-titulo">{pergunta}</div>', unsafe_allow_html=True)
    if ctx is not None:
        cabecalho_analitico(ctx)


def kpi_html(rotulo: str, valor: str, sub: str, destaque: bool = False, ajuda: str = "") -> str:
    cls = "kpi kpi-destaque" if destaque else "kpi"
    return (f'<div class="{cls}" title="{ajuda}"><div class="kpi-r">{rotulo}</div>'
            f'<div class="kpi-v">{valor}</div><div class="kpi-s">{sub}</div></div>')


def tendencia(v0, v1, a0, a1, fmt=pct) -> str:
    """Frase curta sobre a mudança de uma métrica em % entre dois anos."""
    if _vazio(v0) or _vazio(v1) or a0 == a1:
        return ""
    d = v1 - v0
    if abs(d) < 0.3:
        return f"ficou estável, em torno de {fmt(v1)} entre {a0} e {a1}"
    verbo = "subiu" if d > 0 else "caiu"
    return f"{verbo} {fmt(abs(d)).replace('%', '')} p.p. (de {fmt(v0)} em {a0} para {fmt(v1)} em {a1})"


def ir_para(pagina: str):
    st.session_state["pagina"] = pagina


def rodape_navegacao(proxima: str, texto: str):
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.button(f"Continuar: {proxima} →", key=f"prox_{proxima}", on_click=ir_para, args=(proxima,),
              help=texto, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# Gráficos reutilizáveis
# ─────────────────────────────────────────────────────────────────────────────


def cores_na_escala(valores, escala) -> list:
    v = np.asarray(valores, dtype=float)
    if len(v) == 0:
        return []
    lo, hi = np.nanmin(v), np.nanmax(v)
    t = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return sample_colorscale(escala, list(np.nan_to_num(t)))


def barras_h(dados, col_valor, col_rotulo, nome_hover, sufixo="%", prefixo="", cor=COR_FEMININO,
             cores=None, destaque=None, linha_ref=None, fmt=".1f", largura_rotulo=26, altura=None, eixo_max=None):
    d = dados.reset_index(drop=True).copy()
    rotulos = d[col_rotulo].astype(str)
    if cores is None:
        cores = [COR_ACENTO if (destaque is not None and r == destaque) else cor for r in rotulos]
    fig = go.Figure(go.Bar(
        x=d[col_valor], y=rotulos.map(lambda s: quebrar(s, largura_rotulo)), orientation="h",
        marker=dict(color=cores), customdata=rotulos,
        text=d[col_valor], texttemplate=f"{prefixo}%{{x:{fmt}}}{sufixo}", textposition="outside",
        cliponaxis=False, textfont=dict(size=13),
        hovertemplate=f"<b>%{{customdata}}</b><br>{nome_hover}: {prefixo}%{{x:{fmt}}}{sufixo}<extra></extra>",
    ))
    n = len(d)
    estilizar(fig, altura or max(240, 36 * n + 40), horizontal=True)
    fig.update_layout(margin=dict(t=30 if linha_ref is not None else 8, r=60, b=8, l=8))
    fig.update_yaxes(autorange="reversed")
    teto = eixo_max or (float(np.nanmax(d[col_valor])) * 1.2 if n else 1)
    fig.update_xaxes(range=[0, teto])
    if linha_ref is not None:
        fig.add_vline(x=linha_ref, line_dash="dot", line_color="#8A8A8A", line_width=1.5,
                      annotation_text="paridade", annotation_position="top", annotation_font_size=12,
                      annotation_font_color="#6B6B6B")
    return fig


def colunas_estados(dados, col_valor, nome_hover, destaque_sigla=None, cores=None, sufixo="%", fmt=".1f",
                    altura=380, linha_ref=None, teto=None):
    """Ranking de estados em colunas (27 barras ficam legíveis com siglas; o nome completo aparece no hover)."""
    d = dados.reset_index(drop=True)
    siglas = d["sigla_uf"].astype(str)
    if cores is None:
        cores = [COR_ACENTO if (destaque_sigla and s == destaque_sigla) else COR_FEMININO for s in siglas]
    fig = go.Figure(go.Bar(
        x=siglas, y=d[col_valor], marker=dict(color=cores), customdata=d["sigla_uf_nome"].astype(str),
        hovertemplate=f"<b>%{{customdata}}</b><br>{nome_hover}: %{{y:{fmt}}}{sufixo}<extra></extra>",
    ))
    estilizar(fig, altura)
    fig.update_layout(margin=dict(t=30 if linha_ref is not None else 8, r=8, b=8, l=8), bargap=0.2)
    fig.update_xaxes(tickfont_size=12, tickangle=0)
    fig.update_yaxes(range=[0, teto or float(d[col_valor].max()) * 1.15], ticksuffix=sufixo)
    if linha_ref is not None:
        fig.add_hline(y=linha_ref, line_dash="dot", line_color="#8A8A8A", line_width=1.5, annotation_text="paridade",
                      annotation_position="top right", annotation_font_size=12, annotation_font_color="#6B6B6B")
    return fig


def barras_h_grupo(dados, col_rotulo, col_f="rem_f", col_m="rem_m", nome_hover="Remuneração média",
                   prefixo="R$ ", fmt=",.0f", largura_rotulo=26, altura=None):
    d = dados.reset_index(drop=True).copy()
    rotulos = d[col_rotulo].astype(str)
    y = rotulos.map(lambda s: quebrar(s, largura_rotulo))
    fig = go.Figure()
    for nome, col, cor in ((SEXO_F, col_f, COR_FEMININO), (SEXO_M, col_m, COR_MASCULINO)):
        fig.add_trace(go.Bar(
            x=d[col], y=y, name=nome, orientation="h", marker=dict(color=cor), customdata=rotulos,
            hovertemplate=f"<b>%{{customdata}}</b><br>{nome}: {prefixo}%{{x:{fmt}}}<extra></extra>",
        ))
    estilizar(fig, altura or max(300, 62 * len(d) + 60), legenda=True, horizontal=True)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_xaxes(tickprefix=prefixo.strip() + " " if prefixo else "")
    return fig


def barras_v_grupo(dados, col_x, series, prefixo="R$ ", fmt=",.0f", altura=330, com_texto=False, legenda=True):
    d = dados.reset_index(drop=True)
    fig = go.Figure()
    for nome, col, cor in series:
        fig.add_trace(go.Bar(
            x=d[col_x].astype(str), y=d[col], name=nome, marker=dict(color=cor),
            text=d[col] if com_texto else None, texttemplate=f"{prefixo}%{{y:{fmt}}}" if com_texto else None,
            textposition="outside", cliponaxis=False, textfont=dict(size=12),
            hovertemplate=f"<b>%{{x}}</b><br>{nome}: {prefixo}%{{y:{fmt}}}<extra></extra>",
        ))
    estilizar(fig, altura, legenda=legenda)
    fig.update_layout(barmode="group", bargap=0.3)
    return fig


def linhas(dados, col_x, series, fmt, sufixo="", prefixo="", altura=330, legenda=True, referencia=None, y_range=None):
    """series: lista de (nome, coluna, cor, tracejada). O último ponto de cada série recebe rótulo direto."""
    d = dados.sort_values(col_x)
    fig = go.Figure()
    for nome, col, cor, tracejada in series:
        valores = d[col]
        n = len(d)
        rotulos = [""] * max(n - 1, 0) + [f"{prefixo}{format(valores.iloc[-1], fmt)}{sufixo}".replace(",", "X").replace(".", ",").replace("X", ".")] if n else []
        fig.add_trace(go.Scatter(
            x=d[col_x], y=valores, name=nome, mode="lines+markers+text" if not tracejada else "lines",
            line=dict(color=cor, width=3 if not tracejada else 2, dash="dash" if tracejada else "solid"),
            marker=dict(size=8, color=cor, line=dict(color="white", width=2)),
            text=rotulos if not tracejada else None, textposition="top center", textfont=dict(size=13, color=COR_TEXTO),
            hovertemplate=f"<b>%{{x}}</b><br>{nome}: {prefixo}%{{y:{fmt}}}{sufixo}<extra></extra>",
        ))
    estilizar(fig, altura, legenda=legenda)
    fig.update_xaxes(dtick=1, tickformat="d")
    if y_range:
        fig.update_yaxes(range=y_range)
    if referencia is not None:
        fig.add_hline(y=referencia, line_dash="dot", line_color="#8A8A8A", line_width=1.5,
                      annotation_text="paridade", annotation_position="top left", annotation_font_size=12,
                      annotation_font_color="#6B6B6B")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Mapa clicável (clique → "Estado em foco")
# ─────────────────────────────────────────────────────────────────────────────


def sigla_do_ponto(ponto: dict) -> str | None:
    """Extrai a sigla do estado de um ponto selecionado no mapa (formatos variam por versão)."""
    loc = ponto.get("location")
    if loc:
        return str(loc)
    cd = ponto.get("customdata")
    if cd:
        return str(cd[0] if isinstance(cd, (list, tuple)) else cd)
    return None


def ao_clicar_mapa(chave: str):
    try:
        pontos = st.session_state[chave]["selection"]["points"]
    except Exception:
        return
    if not pontos:
        return
    sigla = sigla_do_ponto(pontos[0])
    nome = st.session_state.get("_uf_nomes", {}).get(sigla)
    if nome:
        st.session_state["foco_uf"] = nome


def desenhar_mapa(fig, chave: str):
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, key=chave, on_select=partial(ao_clicar_mapa, chave),
                        selection_mode="points", config=cfg, **LARGURA)
    except TypeError:  # versão sem seleção em gráficos: o filtro continua disponível na barra lateral
        st.plotly_chart(fig, config=cfg, **LARGURA)


METRICAS_MAPA = {
    "Participação feminina": dict(campo="part", escala=ESCALA_ROSA, sufixo="%", prefixo="", fmt=".1f"),
    "Gap salarial": dict(campo="gap", escala=ESCALA_GAP, sufixo="%", prefixo="", fmt=".1f"),
    "Remuneração feminina": dict(campo="rem_f", escala=ESCALA_ROSA, sufixo="", prefixo="R$ ", fmt=",.0f"),
}


def figura_mapa(est: pd.DataFrame, geojson: dict, metrica: str, foco_sigla: str | None):
    cfg = METRICAS_MAPA[metrica]
    est = est.dropna(subset=[cfg["campo"]]).reset_index(drop=True)
    fig = px.choropleth(
        est, geojson=geojson, locations="sigla_uf", featureidkey="properties.sigla",
        color=cfg["campo"], color_continuous_scale=cfg["escala"],
        custom_data=["sigla_uf", "sigla_uf_nome", "part", "gap", "rem_f"],
    )
    fig.update_traces(
        marker_line_color="white", marker_line_width=1.2,
        hovertemplate=("<b>%{customdata[1]}</b><br>Participação feminina: %{customdata[2]:.1f}%"
                       "<br>Gap salarial: %{customdata[3]:.1f}%<br>Remuneração feminina: R$ %{customdata[4]:,.0f}"
                       "<br><i>Clique para focar neste estado</i><extra></extra>"),
    )
    if foco_sigla and foco_sigla in set(est["sigla_uf"].astype(str)):
        idx = int(est.index[est["sigla_uf"].astype(str) == foco_sigla][0])
        fig.update_traces(selectedpoints=[idx], selected=dict(marker=dict(opacity=1)),
                          unselected=dict(marker=dict(opacity=0.4)))
    fig.update_geos(visible=False, bgcolor="rgba(0,0,0,0)", projection_type="mercator",
                    lonaxis_range=[-74.5, -33.0], lataxis_range=[-34.5, 5.8])
    estilizar(fig, altura=720)
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        coloraxis_colorbar=dict(orientation="h", y=-0.02, yanchor="top", thickness=12, len=0.55,
                                title=None, tickfont_size=13, ticksuffix=cfg["sufixo"],
                                tickprefix=cfg["prefixo"], tickformat=",.0f" if cfg["campo"] == "rem_f" else ".0f"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Barra lateral (filtros) e contexto
# ─────────────────────────────────────────────────────────────────────────────


def barra_lateral(cat: dict) -> dict:
    sb = st.sidebar
    sb.markdown('<div class="side-marca">Mercado de Trabalho<br><span>Feminino</span> no Brasil</div>', unsafe_allow_html=True)

    anos = cat["anos"]
    sb.markdown('<div class="side-grupo">Quando</div>', unsafe_allow_html=True)
    ano_ref = sb.selectbox("Ano de referência", anos[::-1], index=0, key="ano_ref",
                           help="Vale para os gráficos de retrato: mapa, setores, perfil e comparativo.")
    if len(anos) > 1:
        periodo = sb.slider("Período da evolução", anos[0], anos[-1], (anos[0], anos[-1]), key="periodo",
                            help="Vale apenas para os gráficos de linha (evolução ao longo do tempo).")
    else:
        periodo = (anos[0], anos[0])

    sb.markdown('<div class="side-grupo">Onde</div>', unsafe_allow_html=True)
    nomes_uf = sorted(cat["uf_nomes"].values())
    foco_nome = sb.selectbox("Estado em foco", [BRASIL] + nomes_uf, key="foco_uf",
                             help="Também é definido ao clicar em um estado no mapa do Panorama.")
    if foco_nome != BRASIL:
        sb.button("Limpar foco (voltar ao Brasil)", key="limpar_foco", on_click=lambda: st.session_state.update(foco_uf=BRASIL))

    sb.markdown('<div class="side-grupo">Em quê</div>', unsafe_allow_html=True)
    todos_setores = sorted(cat["setores"]["setor_amigavel"].unique())
    setores = sb.multiselect("Setor", todos_setores, key="setores", placeholder="Todos os setores")
    pool = cat["setores"]
    if setores:
        pool = pool[pool["setor_amigavel"].isin(setores)]
    especificos = sb.multiselect("Setor específico (CNAE)", sorted(pool["setor_descricao"].unique()), key="cnae",
                                 placeholder="Todos", help="As divisões oficiais do CNAE dentro dos setores escolhidos.")

    sb.markdown('<div class="side-grupo">Quem</div>', unsafe_allow_html=True)
    idades = sb.multiselect("Faixa etária", cat["faixas_etarias"], key="idades", placeholder="Todas")
    salarios = sb.multiselect("Faixa salarial (salários mínimos)", cat["faixas_salariais"], key="salarios", placeholder="Todas")

    sb.caption("Deixe um filtro vazio para não restringir.")
    return dict(ano_ref=int(ano_ref), periodo=tuple(int(p) for p in periodo), foco_nome=foco_nome,
                setores=setores, especificos=especificos, idades=idades, salarios=salarios)


def montar_contexto(df: pd.DataFrame, cat: dict, f: dict, geojson) -> SimpleNamespace:
    mask = np.ones(len(df), dtype=bool)
    if f["setores"]:
        mask &= df["setor_amigavel"].isin(f["setores"]).to_numpy()
    if f["especificos"]:
        mask &= df["setor_descricao"].isin(f["especificos"]).to_numpy()
    if f["idades"]:
        mask &= df["faixa_etaria"].isin(f["idades"]).to_numpy()
    if f["salarios"]:
        mask &= df["faixa_remuneracao_media_sm"].isin(f["salarios"]).to_numpy()
    base = df[mask]

    foco_sigla = None
    if f["foco_nome"] != BRASIL:
        foco_sigla = next((s for s, n in cat["uf_nomes"].items() if n == f["foco_nome"]), None)
    foco_df = base[base["sigla_uf"] == foco_sigla] if foco_sigla else base

    a0, a1 = f["periodo"]
    em_periodo = lambda d: d[d["ano"].between(a0, a1)]
    return SimpleNamespace(
        df=df, cat=cat, base=base, geojson=geojson, f=f,
        ano_ref=f["ano_ref"], periodo=f["periodo"],
        foco_sigla=foco_sigla, foco_nome=f["foco_nome"],
        local=f["foco_nome"] if foco_sigla else "Brasil",
        snap=foco_df[foco_df["ano"] == f["ano_ref"]],
        snap_br=base[base["ano"] == f["ano_ref"]],
        snap_ant=foco_df[foco_df["ano"] == f["ano_ref"] - 1],
        evol=em_periodo(foco_df), evol_br=em_periodo(base),
    )


def cabecalho_analitico(ctx):
    a0, a1 = ctx.periodo
    f = ctx.f
    chips = []
    if ctx.foco_sigla:
        chips.append(f'<span class="chip chip-foco">Foco: {ctx.foco_nome}</span>')
    else:
        chips.append('<span class="chip">Brasil inteiro</span>')
    chips.append(f'<span class="chip">Ano de referência: {ctx.ano_ref}</span>')
    chips.append(f'<span class="chip">Evolução: {a0}–{a1}</span>')
    if f["setores"] or f["especificos"]:
        n = len(f["especificos"]) or len(f["setores"])
        chips.append(f'<span class="chip">{n} setor(es) filtrado(s)</span>')
    if f["idades"]:
        chips.append(f'<span class="chip">{len(f["idades"])} faixa(s) etária(s)</span>')
    if f["salarios"]:
        chips.append(f'<span class="chip">{len(f["salarios"])} faixa(s) salarial(is)</span>')
    st.markdown(f'<div class="chips">{"".join(chips)}</div>', unsafe_allow_html=True)


def kpis(ctx):
    t = totais(ctx.snap)
    ant = totais(ctx.snap_ant) if not ctx.snap_ant.empty else None
    ano, ano_ant = ctx.ano_ref, ctx.ano_ref - 1

    def delta_pp(atual, anterior):
        return f"{pp(atual - anterior)} vs {ano_ant}" if ant and not _vazio(anterior) and not _vazio(atual) else f"em {ano}"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_html("Remuneração feminina", brl(t["rem_f"]), f"média em {ano}, em R$ correntes", True,
                         "Média ponderada pelos vínculos femininos do recorte."), unsafe_allow_html=True)
    c2.markdown(kpi_html("Participação feminina", pct(t["part"]), delta_pp(t["part"], ant["part"] if ant else None),
                         ajuda="Mulheres como % dos vínculos formais do recorte."), unsafe_allow_html=True)
    c3.markdown(kpi_html("Gap salarial de gênero", pct(t["gap"]), delta_pp(t["gap"], ant["gap"] if ant else None),
                         ajuda="Quanto a menos, em média, as mulheres recebem em relação aos homens."), unsafe_allow_html=True)
    c4.markdown(kpi_html("Vínculos femininos", num(t["qtd_f"]), f"de {num(t['total'])} vínculos em {ano}",
                         ajuda="Vínculos formais registrados na RAIS no ano. Não é o número de mulheres."), unsafe_allow_html=True)
    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tela: Início
# ─────────────────────────────────────────────────────────────────────────────


def pagina_inicio(cat: dict):
    anos = cat["anos"]
    st.markdown('<div class="kicker">Painel de apoio à tomada de decisão</div>'
                '<div class="pg-titulo" style="font-size:2.7rem">Onde estão as oportunidades de trabalho para as mulheres no Brasil?</div>',
                unsafe_allow_html=True)
    manchete("Um painel interativo para <b>explorar o mercado de trabalho formal</b> pela perspectiva das mulheres: "
             "quantas são, em que setores e estados estão, quanto ganham e como isso se compara aos homens — "
             "para apoiar escolhas de carreira com dados públicos.")

    st.markdown('<div class="secao" style="margin-top:.6rem">Uma história em cinco passos</div>', unsafe_allow_html=True)
    passos = [
        ("1 · Panorama", "Como está o mercado hoje?", "Mapa do Brasil, participação e salário por região. Clique em um estado para focar nele."),
        ("2 · Onde estão", "Em que setores e estados?", "Quais setores têm mais mulheres e onde elas se concentram."),
        ("3 · Quanto ganham", "E quanto a menos que os homens?", "O gap salarial por setor, estado e ao longo do tempo."),
        ("4 · Perfil", "Idade e raça/cor mudam o quadro?", "Desigualdades dentro do próprio grupo de trabalhadoras."),
        ("5 · Comparar setores", "Como se comparam lado a lado?", "Tabela completa, ordenável e para baixar."),
    ]
    cols = st.columns(5)
    for i, (col, (nome, pergunta, descricao)) in enumerate(zip(cols, passos), start=1):
        with col:
            st.markdown(f'<div class="passo"><div class="passo-n">{i}</div><div class="passo-t">{nome.split(" · ")[1]}</div>'
                        f'<div class="passo-q"><b>{pergunta}</b><br>{descricao}</div></div>', unsafe_allow_html=True)
            st.button("Abrir →", key=f"abrir_{i}", on_click=ir_para, args=(nome,))

    st.markdown('<div class="secao">Como usar</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    usos = [
        ("Escolha uma perspectiva", "Navegue pelos botões no topo, na ordem sugerida ou direto para o que importa para você."),
        ("Filtre pela barra lateral", "Ano, estado, setor, faixa etária e faixa salarial valem para todas as telas."),
        ("Clique e passe o mouse", "Clique em um estado no mapa para focar; passe o mouse nos gráficos para ver os valores."),
        ("Compare", "Mude os filtros e observe como setores, regiões e grupos se comportam."),
    ]
    for i, (col, (t, d)) in enumerate(zip((c1, c2, c3, c4), usos), start=1):
        col.markdown(f'<div class="passo passo-curto"><div class="passo-n">{i}</div><div class="passo-t">{t}</div>'
                     f'<div class="passo-q">{d}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="secao">Sobre os dados</div>', unsafe_allow_html=True)
    try:
        gerado = pd.Timestamp(DATA_PATH.stat().st_mtime, unit="s").strftime("%d/%m/%Y")
    except Exception:
        gerado = "—"
    k1, k2, k3, k4 = st.columns(4)
    for col, (rotulo, valor, sub_) in zip((k1, k2, k3, k4), [
        ("Período", f"{anos[0]}–{anos[-1]}", f"{len(anos)} anos de RAIS"),
        ("Grupos agregados", num(cat["linhas"]), "ano × estado × setor × sexo × idade × raça/cor × faixa salarial"),
        ("Vínculos somados", num(cat["vinculos_total"]), f"nos {len(anos)} anos do período"),
        ("Dataset gerado em", gerado, "a partir da Base dos Dados"),
    ]):
        col.markdown(f'<div class="destaque"><div class="destaque-r">{rotulo}</div><div class="destaque-v">{valor}</div>'
                     f'<div class="destaque-s">{sub_}</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.markdown("**Fonte:** RAIS — Relação Anual de Informações Sociais, do Ministério do Trabalho e Emprego, "
                "acessada pela plataforma [Base dos Dados](https://basedosdados.org). Os microdados foram agregados "
                "antes de chegar ao painel, o que mantém a navegação rápida.")

    st.markdown('<div class="secao">Como organizamos os setores</div>', unsafe_allow_html=True)
    n_div = cat["setores"]["setor_descricao"].nunique()
    n_cat = cat["setores"]["setor_amigavel"].nunique()
    st.markdown(
        f"A classificação oficial de atividades (CNAE) tem **{n_div} divisões** com nomes técnicos, como "
        f"“Atividades de atenção à saúde humana integradas com assistência social, prestadas em residências coletivas e particulares”. "
        f"Para facilitar a leitura, agrupamos essas divisões em **{n_cat} categorias** de reconhecimento imediato "
        f"(ex.: Saúde, Educação, Tecnologia da Informação e Telecomunicações). Quem quiser o detalhe oficial pode "
        f"refinar pelo filtro **Setor específico (CNAE)** na barra lateral."
    )
    with st.expander("Ver quais divisões compõem cada categoria"):
        tab = (cat["setores"].groupby("setor_amigavel")["setor_descricao"]
               .agg(divisoes="count", exemplos=lambda s: "; ".join(list(s)[:3]) + ("…" if len(s) > 3 else ""))
               .reset_index().rename(columns={"setor_amigavel": "Categoria", "divisoes": "Nº de divisões CNAE", "exemplos": "Exemplos"}))
        st.dataframe(tab, hide_index=True, **LARGURA)

    st.markdown('<div class="secao">Leia com cuidado</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aviso"><ul style="margin:0;padding-left:1.2rem;line-height:1.7">'
        "<li><b>Vínculos não são pessoas.</b> Os números contam vínculos formais registrados na RAIS ao longo do ano; "
        "uma mesma mulher pode ter mais de um vínculo, e quem saiu do emprego durante o ano também aparece.</li>"
        "<li><b>Só o mercado formal.</b> Trabalho informal, autônomo e por conta própria não estão na RAIS — "
        "o painel não representa todas as mulheres que trabalham.</li>"
        "<li><b>Valores em reais de cada ano.</b> A remuneração não foi corrigida pela inflação; compare anos "
        "pela participação e pelo gap (%), não pelo valor em R$.</li>"
        "<li><b>Gap não é “mesma função, salário diferente”.</b> É a diferença entre as médias de mulheres e homens "
        "no recorte, sem controlar por cargo, escolaridade, jornada ou tempo de empresa.</li>"
        "<li><b>Recortes pequenos oscilam.</b> Com muitos filtros ao mesmo tempo, poucos vínculos sustentam a média; "
        "desconfie de valores extremos.</li></ul></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.button("Começar pelo Panorama →", on_click=ir_para, args=("1 · Panorama",), type="primary", key="comecar")
    st.caption("Projeto Final II — Análise e Desenvolvimento de Sistemas, UNISINOS · Dafni Rosa Chananeco de Almeida.")


# ─────────────────────────────────────────────────────────────────────────────
# Tela 1: Panorama
# ─────────────────────────────────────────────────────────────────────────────


def pagina_panorama(ctx):
    cabecalho_pagina("1 · Panorama", "Como está o mercado de trabalho feminino hoje?", ctx)
    t = totais(ctx.snap)
    manchete(f"<b>{ctx.local}</b>, {ctx.ano_ref}: as mulheres ocupavam <b>{pct(t['part'])}</b> dos vínculos formais e recebiam, "
             f"em média, <b>{brl(t['rem_f'])}</b> — <b>{pct(t['gap'])}</b> a menos que os homens.")
    kpis(ctx)

    est = painel_sexo(ctx.snap_br, ["sigla_uf", "sigla_uf_nome"])
    # Regiões comparam sempre o Brasil inteiro; com um estado em foco, a região dele fica em amarelo.
    reg = painel_sexo(ctx.snap_br, ["regiao"])
    reg = reg[reg["regiao"].astype(str) != "Outra"]
    regiao_foco = REGIAO_DA_UF.get(ctx.foco_sigla) if ctx.foco_sigla else None
    aviso_regiao = f" · {regiao_foco}, região do foco, em amarelo" if regiao_foco else ""
    ev = painel_sexo(ctx.evol, ["ano"]).sort_values("ano")

    col_mapa, col_dir = st.columns([3, 2])

    with col_mapa:
        with cartao("Mapa do Brasil", "Clique em um estado para filtrar o painel inteiro. Escolha a métrica abaixo."):
            metrica = st.radio("Métrica", list(METRICAS_MAPA), horizontal=True, label_visibility="collapsed", key="metrica_mapa")
            if ctx.geojson is None:
                st.warning(f"Arquivo de geojson não encontrado em {GEOJSON_PATH}.")
            elif est.empty:
                st.info("Sem dados para este recorte.")
            else:
                desenhar_mapa(figura_mapa(est, ctx.geojson, metrica, ctx.foco_sigla), f"mapa_{ctx.foco_sigla}")
                campo = METRICAS_MAPA[metrica]["campo"]
                e = est.dropna(subset=[campo]).sort_values(campo, ascending=False)
                fmt = {"part": pct, "gap": pct, "rem_f": brl}[campo]
                if len(e) >= 2:
                    conclusao(f"Maior: <b>{e.iloc[0]['sigla_uf_nome']}</b> ({fmt(e.iloc[0][campo])}). "
                              f"Menor: <b>{e.iloc[-1]['sigla_uf_nome']}</b> ({fmt(e.iloc[-1][campo])}).")

    with col_dir:
        with cartao("Participação feminina ao longo do tempo",
                    f"% de mulheres nos vínculos · {ctx.local}" + (" (tracejada: Brasil)" if ctx.foco_sigla else "")):
            if len(ev) >= 1:
                series = [("Participação feminina", "part", COR_FEMININO, False)]
                dados = ev
                if ctx.foco_sigla:
                    br = painel_sexo(ctx.evol_br, ["ano"]).sort_values("ano").rename(columns={"part": "part_br"})
                    dados = ev.merge(br[["ano", "part_br"]], on="ano", how="left")
                    series.append(("Brasil", "part_br", "#8A8A8A", True))
                lo, hi = float(dados["part"].min()), float(dados["part"].max())
                mostrar(linhas(dados, "ano", series, ".1f", sufixo="%", altura=300, legenda=ctx.foco_sigla is not None,
                               referencia=50, y_range=[max(0, min(lo, 50) - 6), max(hi, 50) + 6]))
                if len(ev) >= 2:
                    conclusao(f"A participação feminina {tendencia(ev.iloc[0]['part'], ev.iloc[-1]['part'], int(ev.iloc[0]['ano']), int(ev.iloc[-1]['ano']))}.")
            else:
                st.info("Sem dados no período escolhido.")

        with cartao("Participação feminina por região", f"% de mulheres nos vínculos · Brasil{aviso_regiao}"):
            if not reg.empty:
                r = reg.sort_values("part", ascending=False)
                mostrar(barras_h(r, "part", "regiao", "Participação", linha_ref=50, destaque=regiao_foco, eixo_max=max(60, r["part"].max() * 1.2), altura=270))
                conclusao(f"Maior presença feminina: <b>{r.iloc[0]['regiao']}</b> ({pct(r.iloc[0]['part'])}); menor: "
                          f"<b>{r.iloc[-1]['regiao']}</b> ({pct(r.iloc[-1]['part'])}).")
            else:
                st.info("Sem dados para este recorte.")

    c1, c2 = st.columns(2)
    with c1:
        with cartao("Remuneração média por região", "Mulheres e homens, em R$ · Brasil"):
            if not reg.empty:
                r = reg.sort_values("rem_f", ascending=False)
                mostrar(barras_v_grupo(r, "regiao", [(SEXO_F, "rem_f", COR_FEMININO), (SEXO_M, "rem_m", COR_MASCULINO)], altura=340))
                conclusao(f"A maior remuneração feminina está no <b>{r.iloc[0]['regiao']}</b> ({brl(r.iloc[0]['rem_f'])}); "
                          f"a menor, no <b>{r.iloc[-1]['regiao']}</b> ({brl(r.iloc[-1]['rem_f'])}).")
            else:
                st.info("Sem dados para este recorte.")
    with c2:
        setores = relevantes(painel_sexo(ctx.snap, ["setor_amigavel"])).dropna(subset=["part"])
        with cartao("Setores com maior presença feminina", "Top 8 · % de mulheres nos vínculos do setor"):
            if not setores.empty:
                top = setores.sort_values("part", ascending=False).head(8)
                mostrar(barras_h(top, "part", "setor_amigavel", "Participação", linha_ref=50, largura_rotulo=30,
                                 altura=340, eixo_max=max(60, top["part"].max() * 1.2)))
                conclusao(f"<b>{top.iloc[0]['setor_amigavel']}</b> lidera, com {pct(top.iloc[0]['part'])} de mulheres.")
            else:
                st.info("Sem dados para este recorte.")

    rodape_navegacao("2 · Onde estão", "Veja em quais setores e estados as mulheres estão.")


# ─────────────────────────────────────────────────────────────────────────────
# Tela 2: Onde estão
# ─────────────────────────────────────────────────────────────────────────────


def pagina_onde_estao(ctx):
    cabecalho_pagina("2 · Onde estão", "Em que setores e estados as mulheres estão?", ctx)
    pan = relevantes(painel_sexo(ctx.snap, ["setor_amigavel"])).dropna(subset=["part"])
    if pan.empty:
        st.info("Sem dados para este recorte.")
        return
    mais = pan.sort_values("part", ascending=False)
    maior_vol = pan.sort_values("qtd_f", ascending=False)
    manchete(f"Em <b>{ctx.local}</b>, a presença feminina vai de <b>{pct(mais.iloc[-1]['part'], 0)}</b> em {mais.iloc[-1]['setor_amigavel']} "
             f"a <b>{pct(mais.iloc[0]['part'], 0)}</b> em {mais.iloc[0]['setor_amigavel']}. "
             f"Mas o setor que mais emprega mulheres em volume é <b>{maior_vol.iloc[0]['setor_amigavel']}</b>.")
    kpis(ctx)

    c1, c2 = st.columns([3, 2])
    with c1:
        with cartao("Presença: onde as mulheres são maioria",
                    f"% de mulheres nos vínculos de cada setor · {ctx.ano_ref}"):
            mostrar(barras_h(mais, "part", "setor_amigavel", "Participação", linha_ref=50, largura_rotulo=36,
                             eixo_max=max(60, mais["part"].max() * 1.15), altura=max(300, 38 * len(mais) + 40)))
            acima = int((mais["part"] >= 50).sum())
            conclusao(f"Em <b>{acima} de {len(mais)}</b> setores as mulheres são metade ou mais dos vínculos.")
    with c2:
        with cartao("Concentração: onde elas trabalham",
                    "% de todos os vínculos femininos que estão em cada setor (top 10)"):
            conc = pan.assign(conc=pan["qtd_f"] / pan["qtd_f"].sum() * 100).sort_values("conc", ascending=False).head(10)
            mostrar(barras_h(conc, "conc", "setor_amigavel", "Dos vínculos femininos", largura_rotulo=28, altura=420))
            top5 = conc.head(5)["conc"].sum()
            conclusao(f"Só 5 setores reúnem <b>{pct(top5, 0)}</b> dos vínculos femininos.")
        uf = painel_sexo(ctx.snap_br, ["sigla_uf", "sigla_uf_nome"]).dropna(subset=["part"]).sort_values("part", ascending=False)
        with cartao("Presença por estado", "% de mulheres nos vínculos do estado" + (" · estado em foco em amarelo" if ctx.foco_sigla else "")):
            if not uf.empty:
                mostrar(colunas_estados(uf, "part", "Participação", destaque_sigla=ctx.foco_sigla, linha_ref=50,
                                        teto=max(60, uf["part"].max() * 1.12), altura=330))
                conclusao(f"Maior: <b>{uf.iloc[0]['sigla_uf_nome']}</b> ({pct(uf.iloc[0]['part'])}). Menor: "
                          f"<b>{uf.iloc[-1]['sigla_uf_nome']}</b> ({pct(uf.iloc[-1]['part'])}).")

    with cartao("Presença × remuneração",
                "Cada bolha é um setor; o tamanho mostra o volume de vínculos. Passe o mouse para ver o nome."):
        d = pan.dropna(subset=["rem_f"]).copy()
        fig = go.Figure(go.Scatter(
            x=d["part"], y=d["rem_f"], mode="markers", customdata=np.stack([d["setor_amigavel"].astype(str), d["gap"]], axis=-1),
            marker=dict(size=np.sqrt(d["total"]) / np.sqrt(d["total"].max()) * 46 + 8, color=COR_FEMININO, opacity=.75,
                        line=dict(color="white", width=1.5)),
            hovertemplate="<b>%{customdata[0]}</b><br>Participação feminina: %{x:.1f}%<br>Remuneração feminina: R$ %{y:,.0f}"
                          "<br>Gap: %{customdata[1]:.1f}%<extra></extra>",
        ))
        # rótulos só para os 6 maiores setores (rotulagem seletiva)
        for _, r in d.nlargest(6, "total").iterrows():
            fig.add_annotation(x=r["part"], y=r["rem_f"], text=quebrar(r["setor_amigavel"], 26), showarrow=False,
                               yshift=30, font=dict(size=12, color=COR_TEXTO), bgcolor="rgba(255,255,255,.8)")
        fig.add_vline(x=50, line_dash="dot", line_color="#8A8A8A", line_width=1.5, annotation_text="paridade",
                      annotation_position="top", annotation_font_size=12, annotation_font_color="#6B6B6B")
        estilizar(fig, 560)
        fig.update_layout(margin=dict(t=30, b=8, l=8, r=16))
        fig.update_xaxes(title_text="Participação feminina no setor (%)", ticksuffix="%", showgrid=True, gridcolor=COR_GRADE, title_font_size=13)
        fig.update_yaxes(title_text="Remuneração média feminina (R$)", tickprefix="R$ ", title_font_size=13)
        mostrar(fig)
        if len(d) >= 5 and d["part"].std() > 0 and d["rem_f"].std() > 0:
            r_ = float(np.corrcoef(d["part"], d["rem_f"])[0, 1])
            if r_ <= -0.3:
                frase = "setores com <b>mais mulheres tendem a pagar menos</b> a elas"
            elif r_ >= 0.3:
                frase = "setores com <b>mais mulheres tendem a pagar mais</b> a elas"
            else:
                frase = "<b>não há relação clara</b> entre a presença feminina e o salário feminino"
            conclusao(f"Neste recorte, {frase} (correlação {str(round(r_, 2)).replace('.', ',')}). "
                      "É uma associação entre setores, não prova de causa.")

    rodape_navegacao("3 · Quanto ganham", "Agora, quanto elas ganham em relação aos homens.")


# ─────────────────────────────────────────────────────────────────────────────
# Tela 3: Quanto ganham
# ─────────────────────────────────────────────────────────────────────────────


def pagina_quanto_ganham(ctx):
    cabecalho_pagina("3 · Quanto ganham", "Quanto as mulheres ganham — e quanto a menos que os homens?", ctx)
    t = totais(ctx.snap)
    pan = relevantes(painel_sexo(ctx.snap, ["setor_amigavel"])).dropna(subset=["gap"])
    if pan.empty:
        st.info("Sem dados para este recorte (é preciso haver mulheres e homens).")
        return
    por_gap = pan.sort_values("gap", ascending=False)
    manchete(f"Em <b>{ctx.local}</b>, {ctx.ano_ref}, as mulheres ganharam em média <b>{pct(t['gap'])} a menos</b> que os homens. "
             f"O gap é maior em <b>{por_gap.iloc[0]['setor_amigavel']}</b> ({pct(por_gap.iloc[0]['gap'])}) e menor em "
             f"<b>{por_gap.iloc[-1]['setor_amigavel']}</b> ({pct(por_gap.iloc[-1]['gap'])}).")
    kpis(ctx)

    altura_setores = max(560, 38 * len(por_gap) + 40)
    c1, c2 = st.columns(2)
    with c1:
        top = pan.sort_values("total", ascending=False).head(10).sort_values("rem_f", ascending=False)
        with cartao("Remuneração por setor: mulheres × homens", f"10 setores com mais vínculos · R$ · {ctx.ano_ref}"):
            mostrar(barras_h_grupo(top, "setor_amigavel", largura_rotulo=34, altura=altura_setores))
            conclusao(f"Em <b>{int((top['rem_f'] < top['rem_m']).sum())} dos {len(top)}</b> maiores setores, a média feminina fica abaixo da masculina.")
    with c2:
        with cartao("Gap salarial por setor", "% a menos que as mulheres recebem em relação aos homens · todos os setores"):
            cores = cores_na_escala(por_gap["gap"], ESCALA_GAP)
            mostrar(barras_h(por_gap, "gap", "setor_amigavel", "Gap", cores=cores, largura_rotulo=36, altura=altura_setores,
                             eixo_max=max(10, por_gap["gap"].max() * 1.2)))
            conclusao(f"O gap médio entre setores é de <b>{pct(por_gap['gap'].mean())}</b>; "
                      f"em <b>{int((por_gap['gap'] > 0).sum())} de {len(por_gap)}</b> setores as mulheres ganham menos.")

    ev = painel_sexo(ctx.evol, ["ano"]).sort_values("ano")
    c3, c4 = st.columns(2)
    with c3:
        with cartao("Evolução da remuneração média", f"Mulheres e homens · {ctx.local} · R$ correntes de cada ano"):
            if len(ev) >= 1:
                mostrar(linhas(ev, "ano", [(SEXO_F, "rem_f", COR_FEMININO, False), (SEXO_M, "rem_m", COR_MASCULINO, False)],
                               ",.0f", prefixo="R$ ", altura=340))
                conclusao("Valores nominais: parte da alta vem da inflação. Para saber se a desigualdade mudou, veja o gráfico ao lado.")
    with c4:
        with cartao("A desigualdade está diminuindo?", f"Gap salarial (%) por ano · {ctx.local}" + (" (tracejada: Brasil)" if ctx.foco_sigla else "")):
            if len(ev) >= 1:
                series = [("Gap salarial", "gap", COR_TEXTO, False)]
                dados = ev
                if ctx.foco_sigla:
                    br = painel_sexo(ctx.evol_br, ["ano"]).sort_values("ano").rename(columns={"gap": "gap_br"})
                    dados = ev.merge(br[["ano", "gap_br"]], on="ano", how="left")
                    series.append(("Brasil", "gap_br", "#8A8A8A", True))
                teto = float(np.nanmax(dados[["gap"] + (["gap_br"] if ctx.foco_sigla else [])].to_numpy())) * 1.3
                mostrar(linhas(dados, "ano", series, ".1f", sufixo="%", altura=340, legenda=ctx.foco_sigla is not None,
                               y_range=[0, max(teto, 5)]))
                if len(ev) >= 2:
                    conclusao(f"O gap salarial {tendencia(ev.iloc[0]['gap'], ev.iloc[-1]['gap'], int(ev.iloc[0]['ano']), int(ev.iloc[-1]['ano']))}. "
                              "O eixo começa em zero (zero = sem desigualdade).")

    c5, c6 = st.columns([3, 2])
    with c5:
        uf = painel_sexo(ctx.snap_br, ["sigla_uf", "sigla_uf_nome"]).dropna(subset=["gap"]).sort_values("gap", ascending=False)
        with cartao("Gap salarial por estado", "% a menos" + (" · estado em foco em amarelo" if ctx.foco_sigla else " · passe o mouse para ver o nome do estado")):
            if not uf.empty:
                cores_uf = None
                if not ctx.foco_sigla:
                    cores_uf = cores_na_escala(uf["gap"], ESCALA_GAP)
                else:
                    cores_uf = [COR_ACENTO if s == ctx.foco_sigla else COR_SECUNDARIA for s in uf["sigla_uf"].astype(str)]
                mostrar(colunas_estados(uf, "gap", "Gap", cores=cores_uf, altura=380, teto=max(10, uf["gap"].max() * 1.15)))
                conclusao(f"Maior gap: <b>{uf.iloc[0]['sigla_uf_nome']}</b> ({pct(uf.iloc[0]['gap'])}). "
                          f"Menor: <b>{uf.iloc[-1]['sigla_uf_nome']}</b> ({pct(uf.iloc[-1]['gap'])}).")
    with c6:
        dist = agregar(ctx.snap, ["faixa_remuneracao_media_sm", "sexo"])
        with cartao("Distribuição na escala salarial", "% dos vínculos de cada sexo em cada faixa, em salários mínimos (SM)"):
            if not dist.empty:
                dist["pct"] = dist["qtd_vinculos"] / dist.groupby("sexo", observed=True)["qtd_vinculos"].transform("sum") * 100
                faixas = ordenar_faixas(dist["faixa_remuneracao_media_sm"].astype(str).unique())
                dist["faixa"] = dist["faixa_remuneracao_media_sm"].astype(str)
                pf = dist[dist["sexo"] == SEXO_F].set_index("faixa")["pct"].reindex(faixas).fillna(0)
                pm = dist[dist["sexo"] == SEXO_M].set_index("faixa")["pct"].reindex(faixas).fillna(0)
                dd = pd.DataFrame({"faixa": faixas, "f": pf.values, "m": pm.values})
                mostrar(barras_v_grupo(dd, "faixa", [(SEXO_F, "f", COR_FEMININO), (SEXO_M, "m", COR_MASCULINO)],
                                       prefixo="", fmt=".1f", altura=380))
                baixos = [fx for fx in faixas if limite_superior(fx) <= 2.0]
                if baixos:
                    conclusao(f"<b>{pct(pf[baixos].sum(), 0)}</b> dos vínculos femininos pagam até 2 SM, contra "
                              f"<b>{pct(pm[baixos].sum(), 0)}</b> dos masculinos.")

    rodape_navegacao("4 · Perfil", "Veja como idade e raça/cor mudam o quadro.")


# ─────────────────────────────────────────────────────────────────────────────
# Tela 4: Perfil
# ─────────────────────────────────────────────────────────────────────────────


def pagina_perfil(ctx):
    cabecalho_pagina("4 · Perfil", "Quem são as trabalhadoras — idade e raça/cor mudam o quadro?", ctx)
    idade = painel_sexo(ctx.snap, ["faixa_etaria"]).dropna(subset=["rem_f"])
    raca = painel_sexo(ctx.snap, ["raca_cor"]).dropna(subset=["rem_f"])
    if idade.empty or raca.empty:
        st.info("Sem dados para este recorte.")
        return
    idade["_o"] = idade["faixa_etaria"].astype(str).map(chave_ordem)
    idade = idade.sort_values("_o")
    r_top = raca[raca["total"] >= 0.01 * raca["total"].sum()].sort_values("rem_f", ascending=False)
    i_gap = idade.dropna(subset=["gap"])
    frase_idade = (f"o gap é maior na faixa de <b>{i_gap.sort_values('gap').iloc[-1]['faixa_etaria']}</b> "
                   f"({pct(i_gap.sort_values('gap').iloc[-1]['gap'])})" if not i_gap.empty else "")
    frase_raca = (f" Entre as mulheres, a maior remuneração é de <b>{r_top.iloc[0]['raca_cor']}</b> ({brl(r_top.iloc[0]['rem_f'])}) e a menor de "
                  f"<b>{r_top.iloc[-1]['raca_cor']}</b> ({brl(r_top.iloc[-1]['rem_f'])})." if len(r_top) >= 2 else "")
    manchete(f"Em <b>{ctx.local}</b>, {ctx.ano_ref}: {frase_idade}.{frase_raca}")
    kpis(ctx)

    c1, c2 = st.columns(2)
    with c1:
        with cartao("Remuneração por faixa etária", "Mulheres e homens · R$"):
            mostrar(barras_v_grupo(idade, "faixa_etaria", [(SEXO_F, "rem_f", COR_FEMININO), (SEXO_M, "rem_m", COR_MASCULINO)], altura=360))
            pico = idade.sort_values("rem_f").iloc[-1]
            conclusao(f"A remuneração feminina é maior na faixa de <b>{pico['faixa_etaria']}</b> ({brl(pico['rem_f'])}).")
    with c2:
        with cartao("O gap cresce com a idade?", "% a menos que as mulheres recebem, por faixa etária"):
            if not i_gap.empty:
                fig = go.Figure(go.Bar(
                    x=i_gap["faixa_etaria"].astype(str), y=i_gap["gap"], marker=dict(color=cores_na_escala(i_gap["gap"], ESCALA_GAP)),
                    text=i_gap["gap"], texttemplate="%{y:.1f}%", textposition="outside", cliponaxis=False, textfont=dict(size=13),
                    hovertemplate="<b>%{x}</b><br>Gap: %{y:.1f}%<extra></extra>"))
                estilizar(fig, 360)
                fig.update_yaxes(range=[0, max(10, i_gap["gap"].max() * 1.2)], ticksuffix="%")
                mostrar(fig)
                primeiro, ultimo = i_gap.iloc[0], i_gap.iloc[-1]
                conclusao(f"O gap vai de <b>{pct(primeiro['gap'])}</b> ({primeiro['faixa_etaria']}) a <b>{pct(ultimo['gap'])}</b> ({ultimo['faixa_etaria']}).")

    c3, c4 = st.columns(2)
    with c3:
        with cartao("Remuneração feminina por raça/cor", "R$ · mulheres"):
            r = raca.sort_values("rem_f", ascending=False)
            mostrar(barras_h(r, "rem_f", "raca_cor", "Remuneração feminina", sufixo="", prefixo="R$ ", fmt=",.0f", altura=max(260, 46 * len(r) + 40)))
            if len(r_top) >= 2:
                dif = (r_top.iloc[0]["rem_f"] - r_top.iloc[-1]["rem_f"]) / r_top.iloc[0]["rem_f"] * 100
                conclusao(f"Entre as maiores categorias, a distância entre a maior e a menor remuneração feminina é de <b>{pct(dif)}</b>.")
    with c4:
        with cartao("Composição das trabalhadoras por raça/cor", "% dos vínculos femininos"):
            r2 = raca.assign(comp=raca["qtd_f"] / raca["qtd_f"].sum() * 100).sort_values("comp", ascending=False)
            mostrar(barras_h(r2, "comp", "raca_cor", "Dos vínculos femininos", altura=max(260, 46 * len(r2) + 40)))
            conclusao("“Não identificado” indica raça/cor não informada na RAIS — pode ser grande em alguns anos e estados.")

    with cartao("Interseção: raça/cor × idade", f"Remuneração média feminina (R$) · células com menos de {MIN_VINCULOS_CELULA} vínculos ficam em branco"):
        h = painel_sexo(ctx.snap, ["raca_cor", "faixa_etaria"])
        h["raca_cor"] = h["raca_cor"].astype(str)
        h["faixa_etaria"] = h["faixa_etaria"].astype(str)
        h.loc[h["qtd_f"] < MIN_VINCULOS_CELULA, "rem_f"] = np.nan
        ordem_raca = (h.groupby("raca_cor")["qtd_f"].sum().sort_values(ascending=False).index.tolist())
        ordem_idade = ordenar_faixas(h["faixa_etaria"].unique())
        z = h.pivot(index="raca_cor", columns="faixa_etaria", values="rem_f").reindex(index=ordem_raca, columns=ordem_idade)
        texto = [["" if pd.isna(v) else f"R$ {v:,.0f}".replace(",", ".") for v in linha] for linha in z.values]
        fig = go.Figure(go.Heatmap(
            z=z.values, x=z.columns.tolist(), y=z.index.tolist(), colorscale=ESCALA_ROSA, xgap=4, ygap=4,
            text=texto, texttemplate="%{text}", textfont=dict(size=13, color=COR_TEXTO), showscale=False, hoverongaps=False,
            hovertemplate="<b>%{y}</b> · %{x}<br>Remuneração feminina: R$ %{z:,.0f}<extra></extra>"))
        estilizar(fig, max(300, 62 * len(ordem_raca) + 50))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showline=False, side="top")
        mostrar(fig)
        validos = z.stack().dropna()
        if len(validos) >= 2:
            (rmax, imax), (rmin, imin) = validos.idxmax(), validos.idxmin()
            conclusao(f"Maior: <b>{rmax}</b>, {imax} ({brl(validos.max())}). Menor: <b>{rmin}</b>, {imin} ({brl(validos.min())}).")

    rodape_navegacao("5 · Comparar setores", "Compare todos os setores lado a lado.")


# ─────────────────────────────────────────────────────────────────────────────
# Tela 5: Comparar setores
# ─────────────────────────────────────────────────────────────────────────────


def pagina_comparar(ctx):
    cabecalho_pagina("5 · Comparar setores", "Como os setores se comparam lado a lado?", ctx)
    manchete("Use a tabela para <b>ordenar pelo que importa na sua decisão</b>: onde há mais mulheres, onde elas ganham mais "
             "ou onde a diferença para os homens é menor.")

    nivel = st.radio("Nível de detalhe", ["Categorias (setores amigáveis)", "Divisões CNAE (oficial)"], horizontal=True, key="nivel_cmp")
    coluna = "setor_amigavel" if nivel.startswith("Categorias") else "setor_descricao"
    pan = painel_sexo(ctx.snap, [coluna])
    pan = pan[pan["total"] >= 30].dropna(subset=["part"]).copy()
    if pan.empty:
        st.info("Sem dados para este recorte.")
        return
    pan[coluna] = pan[coluna].astype(str)
    rel = relevantes(pan, 0.001)

    def destaque(rotulo, linha, valor):
        return (f'<div class="destaque"><div class="destaque-r">{rotulo}</div><div class="destaque-v">{valor}</div>'
                f'<div class="destaque-s">{linha}</div></div>')

    d1, d2, d3, d4 = st.columns(4)
    if not rel.empty:
        a = rel.sort_values("part", ascending=False).iloc[0]
        b = rel.dropna(subset=["rem_f"]).sort_values("rem_f", ascending=False).iloc[0]
        c = rel.dropna(subset=["gap"]).sort_values("gap").iloc[0]
        d = rel.sort_values("qtd_f", ascending=False).iloc[0]
        d1.markdown(destaque("Maior presença feminina", a[coluna], pct(a["part"])), unsafe_allow_html=True)
        d2.markdown(destaque("Maior remuneração feminina", b[coluna], brl(b["rem_f"])), unsafe_allow_html=True)
        d3.markdown(destaque("Menor gap salarial", c[coluna], pct(c["gap"])), unsafe_allow_html=True)
        d4.markdown(destaque("Mais mulheres empregadas (vínculos)", d[coluna], num(d["qtd_f"])), unsafe_allow_html=True)
    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)

    with cartao(f"Comparativo por setor · {ctx.local} · {ctx.ano_ref}", "Clique no título de uma coluna para ordenar."):
        b1, b2 = st.columns([2, 1])
        busca = b1.text_input("Buscar setor", placeholder="Digite parte do nome…", key="busca_setor")
        ordem = b2.selectbox("Ordenar por", ["% de mulheres", "Rem. feminina (R$)", "Gap salarial", "Vínculos fem."], key="ordem_cmp")
        tab = pan.rename(columns={coluna: "Setor", "part": "% de mulheres", "rem_f": "Rem. feminina (R$)",
                                  "rem_m": "Rem. masculina (R$)", "gap": "Gap salarial", "qtd_f": "Vínculos fem."})
        if busca:
            tab = tab[tab["Setor"].str.contains(busca, case=False, na=False)]
        asc = ordem == "Gap salarial"
        tab = tab.sort_values(ordem, ascending=asc)[["Setor", "% de mulheres", "Rem. feminina (R$)", "Rem. masculina (R$)", "Gap salarial", "Vínculos fem."]]
        st.dataframe(
            tab, hide_index=True, height=min(720, 38 * (len(tab) + 1) + 6), **LARGURA,
            column_config={
                "Setor": st.column_config.TextColumn("Setor", width="medium"),
                "% de mulheres": st.column_config.ProgressColumn("% de mulheres", min_value=0, max_value=100, format="%.1f%%"),
                "Rem. feminina (R$)": st.column_config.NumberColumn("Rem. feminina (R$)", format=FORMATO_INTEIRO),
                "Rem. masculina (R$)": st.column_config.NumberColumn("Rem. masculina (R$)", format=FORMATO_INTEIRO),
                "Gap salarial": st.column_config.NumberColumn("Gap salarial", format="%.1f%%"),
                "Vínculos fem.": st.column_config.NumberColumn("Vínculos fem.", format=FORMATO_INTEIRO),
            },
        )
        csv = tab.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
        st.download_button("Baixar esta tabela (CSV)", csv, file_name="comparativo_setores.csv", mime="text/csv", key="baixar_csv")
        conclusao("Setores com menos de 30 vínculos no recorte não aparecem. Médias de setores pequenos oscilam bastante.")

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    st.button("← Voltar ao início", on_click=ir_para, args=("Início",), key="voltar_inicio")


# ─────────────────────────────────────────────────────────────────────────────
# Principal
# ─────────────────────────────────────────────────────────────────────────────

TELAS = {
    "1 · Panorama": pagina_panorama,
    "2 · Onde estão": pagina_onde_estao,
    "3 · Quanto ganham": pagina_quanto_ganham,
    "4 · Perfil": pagina_perfil,
    "5 · Comparar setores": pagina_comparar,
}


def main():
    _contador_cartao["n"] = 0
    aplicar_estilo()

    if not DATA_PATH.exists():
        st.warning(f"Dataset tratado não encontrado em {DATA_PATH}. Rode antes: 01_ingest.py, 02_transform.py e 03_consolidate.py.")
        return

    df = load_data(DATA_PATH)
    faltando = [c for c in ["setor_amigavel", "qtd_vinculos", "remuneracao_media"] if c not in df.columns]
    if faltando:
        st.error(f"O dataset não tem as colunas {faltando}. Rode 02_transform.py e 03_consolidate.py novamente.")
        return
    cat = load_catalogo(DATA_PATH)
    st.session_state["_uf_nomes"] = cat["uf_nomes"]
    geojson = load_geojson(GEOJSON_PATH) if GEOJSON_PATH.exists() else None

    filtros = barra_lateral(cat)

    if "pagina" not in st.session_state:
        st.session_state["pagina"] = PAGINAS[0]
    st.radio("Navegação", PAGINAS, horizontal=True, label_visibility="collapsed", key="pagina")
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    pagina = st.session_state["pagina"]
    if pagina == "Início":
        pagina_inicio(cat)
        return

    ctx = montar_contexto(df, cat, filtros, geojson)
    if ctx.snap.empty:
        cabecalho_analitico(ctx)
        st.info(f"Nenhum registro para {ctx.local} em {ctx.ano_ref} com estes filtros. Ajuste a barra lateral.")
        return
    TELAS[pagina](ctx)


if __name__ == "__main__":
    main()
