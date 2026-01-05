"""
Sistema DIMP - Análise de Meios de Pagamento (CNPJ vs CPF de Sócios)
Receita Estadual de Santa Catarina
Versão 1.0 - Dashboard Streamlit Completo
Auditor Fiscal: Tiago Severo
"""

# =============================================================================
# 1. IMPORTS E CONFIGURAÇÕES INICIAIS
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import warnings
import ssl
import hashlib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Configuração SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="DIMP - Análise de Meios de Pagamento",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. SISTEMA DE AUTENTICAÇÃO
# =============================================================================

SENHA = "tsevero963"  # Troque conforme necessário

def check_password():
    """Sistema de autenticação."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("<div style='text-align: center; padding: 50px;'><h1>🔐 Acesso Restrito - Sistema DIMP</h1></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            senha_input = st.text_input("Digite a senha:", type="password", key="pwd_input")
            if st.button("Entrar", use_container_width=True):
                if senha_input == SENHA:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Senha incorreta")
        st.stop()

check_password()

# =============================================================================
# 3. ESTILOS CSS CUSTOMIZADOS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1565c0;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

        /* ESTILO DOS KPIs - BORDA PRETA */
    div[data-testid="stMetric"] {
        background-color: #ffffff;        /* Fundo branco */
        border: 2px solid #2c3e50;        /* Borda: 2px de largura, sólida, cor cinza-escuro */
        border-radius: 10px;              /* Cantos arredondados (10 pixels de raio) */
        padding: 15px;                    /* Espaçamento interno (15px em todos os lados) */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);  /* Sombra: horizontal=0, vertical=2px, blur=4px, cor preta 10% opacidade */
    }
    
    /* Título do métrica */
    div[data-testid="stMetric"] > label {
        font-weight: 600;                 /* Negrito médio */
        color: #2c3e50;                   /* Cor do texto */
    }
    
    /* Valor do métrica */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;                /* Tamanho da fonte do valor */
        font-weight: bold;                /* Negrito */
        color: #1f77b4;                   /* Cor azul */
    }
    
    /* Delta (variação) */
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem;                /* Tamanho menor para delta */
        
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-critico {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-alto {
        background-color: #fff3e0;
        border-left: 5px solid #ef6c00;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-positivo {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 4. FUNÇÕES DE CONEXÃO E CARREGAMENTO
# =============================================================================

IMPALA_HOST = 'bdaworkernode02.sef.sc.gov.br'
IMPALA_PORT = 21050
DATABASE = 'teste'

IMPALA_USER = st.secrets.get("impala_credentials", {}).get("user", "tsevero")
IMPALA_PASSWORD = st.secrets.get("impala_credentials", {}).get("password", "")

@st.cache_resource
def get_impala_engine():
    """Cria engine de conexão Impala."""
    try:
        engine = create_engine(
            f'impala://{IMPALA_HOST}:{IMPALA_PORT}/{DATABASE}',
            connect_args={
                'user': IMPALA_USER,
                'password': IMPALA_PASSWORD,
                'auth_mechanism': 'LDAP',
                'use_ssl': True
            }
        )
        return engine
    except Exception as e:
        st.sidebar.error(f"Erro na conexão: {str(e)[:100]}")
        return None

@st.cache_data(ttl=7200)
def verificar_colunas_tabela(_engine, tabela):
    """Verifica quais colunas existem na tabela."""
    try:
        query = f"DESCRIBE {tabela}"
        df_desc = pd.read_sql(query, _engine)
        colunas = df_desc.iloc[:, 0].tolist()  # Primeira coluna tem os nomes
        return colunas
    except Exception as e:
        st.sidebar.warning(f"Não foi possível verificar colunas: {str(e)[:50]}")
        return []

@st.cache_data(ttl=3600)
def carregar_resumo_geral(_engine):
    """Carrega dados agregados iniciais (rápido)."""
    if _engine is None:
        return {}
    
    # Verificar colunas disponíveis
    colunas_disponiveis = verificar_colunas_tabela(_engine, 'teste.dimp_score_final')
    
    resumo = {}
    resumo['colunas_disponiveis'] = colunas_disponiveis
    
    try:
        # Panorama Geral
        query_panorama = """
        SELECT 
            COUNT(DISTINCT cnpj) AS total_empresas,
            COUNT(DISTINCT CASE WHEN classificacao_risco = 'ALTO' THEN cnpj END) AS empresas_alto_risco,
            COUNT(DISTINCT CASE WHEN classificacao_risco = 'MÉDIO-ALTO' THEN cnpj END) AS empresas_medio_alto,
            CAST(SUM(total_geral) AS DOUBLE) AS volume_total,
            CAST(SUM(total_recebido_cpf) AS DOUBLE) AS volume_cpf,
            CAST(SUM(total_recebido_cnpj) AS DOUBLE) AS volume_cnpj,
            CAST(AVG(perc_recebido_cpf) AS DOUBLE) AS media_perc_cpf,
            CAST(AVG(score_risco_final) AS DOUBLE) AS media_score,
            COUNT(DISTINCT CASE WHEN perc_recebido_cpf >= 80 THEN cnpj END) AS empresas_80pct_cpf
        FROM teste.dimp_score_final
        """
        
        df_panorama = pd.read_sql(query_panorama, _engine)
        resumo['panorama'] = df_panorama.to_dict('records')[0] if not df_panorama.empty else {}
        
        # Distribuição por Risco
        query_dist_risco = """
        SELECT 
            classificacao_risco,
            COUNT(*) AS qtd_empresas,
            CAST(SUM(total_recebido_cpf) AS DOUBLE) AS volume_cpf,
            CAST(AVG(score_risco_final) AS DOUBLE) AS score_medio
        FROM teste.dimp_score_final
        GROUP BY classificacao_risco
        """
        
        resumo['dist_risco'] = pd.read_sql(query_dist_risco, _engine)
        
        # Top 20 Municípios
        query_municipios = """
        SELECT 
            municipio,
            uf,
            COUNT(DISTINCT cnpj) AS qtd_empresas,
            CAST(SUM(total_recebido_cpf) AS DOUBLE) AS volume_cpf
        FROM teste.dimp_score_final
        WHERE municipio IS NOT NULL
        GROUP BY municipio, uf
        ORDER BY volume_cpf DESC
        LIMIT 20
        """
        
        resumo['top_municipios'] = pd.read_sql(query_municipios, _engine)
        
        # Distribuição por UF (substitui GERFE que não existe)
        query_uf = """
        SELECT 
            uf,
            COUNT(*) AS qtd_empresas,
            CAST(SUM(total_recebido_cpf) AS DOUBLE) AS volume_cpf,
            CAST(AVG(score_risco_final) AS DOUBLE) AS score_medio
        FROM teste.dimp_score_final
        WHERE uf IS NOT NULL
        GROUP BY uf
        ORDER BY volume_cpf DESC
        """
        
        resumo['por_uf'] = pd.read_sql(query_uf, _engine)
        
        st.sidebar.success("✅ Resumo geral carregado!")
        
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar resumo: {str(e)[:100]}")
    
    return resumo

@st.cache_data(ttl=3600)
def carregar_lista_empresas(_engine):
    """Carrega apenas lista de empresas para seleção."""
    query = """
    SELECT 
        cnpj,
        nm_razao_social,
        regime_tributario,
        classificacao_risco,
        CAST(score_risco_final AS DOUBLE) AS score_risco_final,
        CAST(total_recebido_cpf AS DOUBLE) AS total_recebido_cpf,
        municipio,
        uf
    FROM teste.dimp_score_final
    ORDER BY score_risco_final DESC
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar lista: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
@st.cache_data(ttl=600)
def carregar_detalhes_empresa(_engine, cnpj):
    """Carrega detalhes completos de uma empresa específica (sob demanda)."""
    detalhes = {}
    
    try:
        # Dados principais
        query_principal = f"""
        SELECT *
        FROM teste.dimp_score_final
        WHERE cnpj = '{cnpj}'
        """
        detalhes['principal'] = pd.read_sql(query_principal, _engine)
        
        # Sócios que recebem
        query_socios = f"""
        SELECT 
            cpf_socio,
            nome_socio,
            nm_qualificacao,
            CAST(perc_participacao AS DOUBLE) AS perc_participacao,
            CAST(SUM(vl_total) AS DOUBLE) AS total_recebido,
            COUNT(DISTINCT referencia) AS meses_recebeu
        FROM teste.dimp_pagamentos_cpf
        WHERE cnpj = '{cnpj}'
        GROUP BY cpf_socio, nome_socio, nm_qualificacao, perc_participacao
        ORDER BY total_recebido DESC
        """
        detalhes['socios'] = pd.read_sql(query_socios, _engine)
        
        # Evolução mensal
        query_evolucao = f"""
        WITH cnpj_pagtos AS (
            SELECT referencia, CAST(SUM(vl_total) AS DOUBLE) AS vl_cnpj
            FROM teste.dimp_pagamentos_cnpj
            WHERE cnpj = '{cnpj}'
            GROUP BY referencia
        ),
        cpf_pagtos AS (
            SELECT referencia, CAST(SUM(vl_total) AS DOUBLE) AS vl_cpf
            FROM teste.dimp_pagamentos_cpf
            WHERE cnpj = '{cnpj}'
            GROUP BY referencia
        )
        SELECT 
            COALESCE(c.referencia, p.referencia) AS referencia,
            COALESCE(c.vl_cnpj, 0) AS vl_cnpj,
            COALESCE(p.vl_cpf, 0) AS vl_cpf
        FROM cnpj_pagtos c
        FULL OUTER JOIN cpf_pagtos p ON c.referencia = p.referencia
        ORDER BY referencia
        """
        detalhes['evolucao'] = pd.read_sql(query_evolucao, _engine)
        
        # Operações suspeitas - CORRIGIDO
        query_operacoes = f"""
        SELECT 
            referencia,
            identificador,
            tipo_identificador,
            nome_socio,
            nm_qualificacao,
            CAST(vl_credito AS DOUBLE) AS vl_credito,
            CAST(vl_debito AS DOUBLE) AS vl_debito,
            CAST(vl_pix AS DOUBLE) AS vl_pix,
            CAST(vl_boleto AS DOUBLE) AS vl_boleto,
            CAST(vl_transferencia AS DOUBLE) AS vl_transferencia,
            CAST(vl_dinheiro AS DOUBLE) AS vl_dinheiro,
            CAST(vl_total AS DOUBLE) AS vl_total
        FROM teste.dimp_operacoes_suspeitas
        WHERE cnpj = '{cnpj}'
        ORDER BY referencia DESC, vl_total DESC
        LIMIT 100
        """
        detalhes['operacoes'] = pd.read_sql(query_operacoes, _engine)
        
    except Exception as e:
        st.error(f"Erro ao carregar detalhes: {str(e)}")
    
    return detalhes

@st.cache_data(ttl=3600)
def carregar_dados_ml(_engine):
    """Carrega dados para Machine Learning."""
    query = """
    SELECT 
        cnpj,
        nm_razao_social,
        CAST(total_recebido_cnpj AS DOUBLE) AS feat_total_cnpj,
        CAST(total_recebido_cpf AS DOUBLE) AS feat_total_cpf,
        CAST(perc_recebido_cpf AS DOUBLE) AS feat_perc_cpf,
        CAST(qtd_socios_recebendo AS DOUBLE) AS feat_qtd_socios,
        CAST(meses_com_pagto_cpf AS DOUBLE) AS feat_meses_cpf,
        CAST(score_proporcao AS DOUBLE) AS score_proporcao,
        CAST(score_volume_cpf AS DOUBLE) AS score_volume,
        CAST(score_qtd_socios AS DOUBLE) AS score_socios,
        CAST(score_consistencia AS DOUBLE) AS score_consistencia,
        CAST(score_risco_final AS DOUBLE) AS score_final,
        CASE WHEN classificacao_risco IN ('ALTO', 'MÉDIO-ALTO') THEN 1 ELSE 0 END AS target_suspeito,
        classificacao_risco,
        regime_tributario
    FROM teste.dimp_score_final
    WHERE score_risco_final IS NOT NULL
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados ML: {str(e)}")
        return pd.DataFrame()

# =============================================================================
# 5. FUNÇÕES DE PROCESSAMENTO E ANÁLISE
# =============================================================================

def calcular_kpis_resumo(resumo):
    """Calcula KPIs a partir do resumo."""
    if not resumo or 'panorama' not in resumo or not resumo['panorama']:
        return {k: 0 for k in ['total_empresas', 'volume_total', 'volume_cpf', 
                                'media_perc_cpf', 'empresas_alto_risco', 'empresas_80pct']}
    
    p = resumo['panorama']
    
    return {
        'total_empresas': int(p.get('total_empresas', 0)),
        'volume_total': float(p.get('volume_total', 0)),
        'volume_cpf': float(p.get('volume_cpf', 0)),
        'volume_cnpj': float(p.get('volume_cnpj', 0)),
        'media_perc_cpf': float(p.get('media_perc_cpf', 0)),
        'media_score': float(p.get('media_score', 0)),
        'empresas_alto_risco': int(p.get('empresas_alto_risco', 0)),
        'empresas_medio_alto': int(p.get('empresas_medio_alto', 0)),
        'empresas_80pct': int(p.get('empresas_80pct_cpf', 0))
    }

def treinar_modelo_ml(df_ml):
    """Treina modelo de Machine Learning."""
    if df_ml.empty:
        return None, None, None
    
    # Preparar features
    features = ['feat_perc_cpf', 'feat_total_cpf', 'feat_qtd_socios',
                'feat_meses_cpf', 'score_proporcao', 'score_volume',
                'score_socios', 'score_consistencia']
    
    X = df_ml[features].fillna(0)
    y = df_ml['target_suspeito']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Treinar Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predições
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Métricas
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    resultados = {
        'model': rf_model,
        'report': report,
        'confusion_matrix': cm,
        'feature_importance': importance_df,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    return resultados, X_test.index, y_proba

def detectar_anomalias(df_ml):
    """Detecta anomalias usando Isolation Forest."""
    if df_ml.empty:
        return None
    
    features = ['feat_perc_cpf', 'feat_total_cpf', 'feat_qtd_socios', 'feat_meses_cpf']
    X = df_ml[features].fillna(0)
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    anomalias = iso_forest.fit_predict(X_scaled)
    scores = iso_forest.score_samples(X_scaled)
    
    df_ml['anomalia'] = anomalias
    df_ml['anomalia_score'] = scores
    
    return df_ml

# =============================================================================
# 6. FUNÇÕES DE FILTROS
# =============================================================================

def criar_filtros_sidebar():
    """Cria painel de filtros na sidebar."""
    filtros = {}
    
    with st.sidebar.expander("🎯 Filtros Globais", expanded=True):
        
        filtros['classificacoes'] = st.multiselect(
            "Classificações de Risco",
            ['ALTO', 'MÉDIO-ALTO', 'MÉDIO', 'BAIXO'],
            default=['ALTO', 'MÉDIO-ALTO']
        )
        
        filtros['perc_cpf_min'] = st.slider(
            "% CPF Mínimo",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )
        
        filtros['score_min'] = st.slider(
            "Score Mínimo",
            min_value=0,
            max_value=100,
            value=60,
            step=5
        )
        
        filtros['valor_min'] = st.number_input(
            "Valor CPF Mínimo (R$)",
            min_value=0,
            max_value=10000000,
            value=10000,
            step=10000,
            format="%d"
        )
    
    with st.sidebar.expander("📊 Visualização", expanded=False):
        filtros['tema'] = st.selectbox(
            "Tema dos Gráficos",
            ["plotly", "plotly_white", "plotly_dark"],
            index=1
        )
        
        filtros['mostrar_valores'] = st.checkbox("Mostrar valores nos gráficos", value=True)
    
    return filtros

# =============================================================================
# 7. PÁGINAS DO DASHBOARD
# =============================================================================

def pagina_dashboard_executivo(resumo, filtros):
    """Dashboard executivo principal."""
    st.markdown("<h1 class='main-header'>💳 Dashboard Executivo DIMP</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Sistema DIMP:</b> Análise de meios de pagamento, identificando empresas que recebem 
    valores significativos via CPF de sócios, em vez do CNPJ da empresa.
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs principais
    kpis = calcular_kpis_resumo(resumo)
    
    st.subheader("📊 Indicadores Principais")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Empresas Analisadas", f"{kpis['total_empresas']:,}")
    
    with col2:
        st.metric("Volume Total", f"R$ {kpis['volume_total']/1e6:.1f}M")
    
    with col3:
        st.metric("Volume via CPF", f"R$ {kpis['volume_cpf']/1e6:.1f}M")
    
    with col4:
        perc_total_cpf = (kpis['volume_cpf'] / kpis['volume_total'] * 100) if kpis['volume_total'] > 0 else 0
        st.metric("% Total via CPF", f"{perc_total_cpf:.1f}%")
    
    with col5:
        st.metric("Score Médio", f"{kpis['media_score']:.1f}")
    
    # Segunda linha
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Alto Risco", f"{kpis['empresas_alto_risco']:,}")
    
    with col2:
        st.metric("Médio-Alto Risco", f"{kpis['empresas_medio_alto']:,}")
    
    with col3:
        st.metric("80%+ via CPF", f"{kpis['empresas_80pct']:,}")
    
    with col4:
        perc_80 = (kpis['empresas_80pct'] / kpis['total_empresas'] * 100) if kpis['total_empresas'] > 0 else 0
        st.metric("% 80%+ CPF", f"{perc_80:.1f}%")
    
    st.divider()
    
    # Gráficos
    if 'dist_risco' in resumo and not resumo['dist_risco'].empty:
        st.subheader("📈 Análises Visuais")
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_dist = resumo['dist_risco']
            
            fig = px.pie(
                df_dist,
                values='qtd_empresas',
                names='classificacao_risco',
                title='Distribuição por Classificação de Risco',
                template=filtros['tema'],
                color='classificacao_risco',
                color_discrete_map={
                    'ALTO': '#c62828',
                    'MÉDIO-ALTO': '#ef6c00',
                    'MÉDIO': '#fbc02d',
                    'BAIXO': '#388e3c'
                },
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df_dist,
                x='classificacao_risco',
                y='volume_cpf',
                title='Volume CPF por Classificação',
                template=filtros['tema'],
                color='classificacao_risco',
                color_discrete_map={
                    'ALTO': '#c62828',
                    'MÉDIO-ALTO': '#ef6c00',
                    'MÉDIO': '#fbc02d',
                    'BAIXO': '#388e3c'
                }
            )
            fig.update_yaxes(title_text="Volume (R$)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Top Municípios
    if 'top_municipios' in resumo and not resumo['top_municipios'].empty:
        st.subheader("🗺️ Top 20 Municípios por Volume")
        
        df_mun = resumo['top_municipios']
        
        fig = px.bar(
            df_mun,
            x='volume_cpf',
            y='municipio',
            orientation='h',
            title='Volume CPF por Município',
            template=filtros['tema'],
            color='volume_cpf',
            color_continuous_scale='Reds',
            hover_data=['uf', 'qtd_empresas']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribuição por UF
    if 'por_uf' in resumo and not resumo['por_uf'].empty:
        st.subheader("🗺️ Análise por UF")
        
        df_uf = resumo['por_uf']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_uf.head(10),
                x='qtd_empresas',
                y='uf',
                orientation='h',
                title='Top 10 UFs - Quantidade',
                template=filtros['tema'],
                color='qtd_empresas',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df_uf.head(10),
                x='volume_cpf',
                y='uf',
                orientation='h',
                title='Top 10 UFs - Volume',
                template=filtros['tema'],
                color='volume_cpf',
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig, use_container_width=True)

def pagina_ranking_empresas(engine, filtros):
    """Ranking de empresas com drill-down."""
    st.markdown("<h1 class='main-header'>🎯 Ranking de Empresas</h1>", unsafe_allow_html=True)
    
    # Carregar lista
    with st.spinner('Carregando lista de empresas...'):
        df_lista = carregar_lista_empresas(engine)
    
    if df_lista.empty:
        st.error("Nenhuma empresa encontrada.")
        return
    
    # Aplicar filtros
    df_filtrado = df_lista[
        (df_lista['classificacao_risco'].isin(filtros['classificacoes'])) &
        (df_lista['score_risco_final'] >= filtros['score_min'])
    ].copy()
    
    st.info(f"📊 {len(df_filtrado):,} empresas após filtros")
    
    # Configurações do ranking
    col1, col2, col3 = st.columns(3)
    
    with col1:
        criterio = st.selectbox(
            "Ordenar por",
            ['Score de Risco', 'Valor CPF', 'Razão Social'],
            index=0
        )
    
    with col2:
        top_n = st.slider("Top N empresas", 10, 100, 50, 10)
    
    with col3:
        ordem = st.radio("Ordem", ['Decrescente', 'Crescente'], index=0)
    
    # Mapear critério
    mapa_criterio = {
        'Score de Risco': 'score_risco_final',
        'Valor CPF': 'total_recebido_cpf',
        'Razão Social': 'nm_razao_social'
    }
    
    col_ordenacao = mapa_criterio[criterio]
    ascending = (ordem == 'Crescente')
    
    ranking = df_filtrado.sort_values(col_ordenacao, ascending=ascending).head(top_n)
    
    # Exibir ranking
    st.subheader(f"📋 Top {top_n} Empresas - {criterio}")
    
    ranking_display = ranking.copy()
    ranking_display.insert(0, 'Posição', range(1, len(ranking_display) + 1))
    
    # Exibir tabela sem formatação problemática
    st.dataframe(
        ranking_display,
        use_container_width=True,
        height=600
    )
    
    st.divider()
    
    # Seleção para drill-down
    st.subheader("🔍 Drill-Down: Selecione uma Empresa")
    
    empresa_selecionada = st.selectbox(
        "Empresa:",
        ranking['cnpj'].tolist(),
        format_func=lambda x: f"{ranking[ranking['cnpj']==x]['nm_razao_social'].iloc[0]} - {x}",
        key="ranking_empresa_select"
    )
    
    if st.button("📊 Analisar Empresa Selecionada", type="primary"):
        st.session_state['empresa_drill_down'] = empresa_selecionada
        st.session_state['pagina_atual'] = "🔍 Drill-Down Empresa"
        # Forçar atualização do radio
        st.rerun()

def pagina_drill_down_empresa(engine, filtros):
    """Análise detalhada de empresa específica."""
    st.markdown("<h1 class='main-header'>🔍 Drill-Down por Empresa</h1>", unsafe_allow_html=True)
    
    # Seleção da empresa
    if 'empresa_drill_down' not in st.session_state:
        st.session_state['empresa_drill_down'] = None
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Busca por CNPJ
        cnpj_input = st.text_input(
            "Digite o CNPJ (apenas números):",
            value=st.session_state.get('empresa_drill_down', ''),
            max_chars=14,
            help="Digite apenas os 14 números do CNPJ"
        )
        
        if cnpj_input:
            cnpj_limpo = ''.join(filter(str.isdigit, cnpj_input))
            
            if len(cnpj_limpo) != 14:
                st.warning(f"⚠️ CNPJ deve ter 14 dígitos. Você digitou {len(cnpj_limpo)} dígitos.")
                return
            
            empresa_selecionada = cnpj_limpo.zfill(14)
        else:
            st.info("Digite um CNPJ para análise.")
            return
    
    # Carregar detalhes
    with st.spinner(f'Carregando detalhes de {empresa_selecionada}...'):
        detalhes = carregar_detalhes_empresa(engine, empresa_selecionada)
    
    if not detalhes or detalhes['principal'].empty:
        st.error(f"❌ CNPJ {empresa_selecionada} não encontrado na base de dados.")
        return
    
    empresa_info = detalhes['principal'].iloc[0]
    
    # Header da empresa
    st.markdown(f"### {empresa_info['nm_razao_social']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.caption(f"**CNPJ:** {empresa_info['cnpj']}")
        st.caption(f"**Regime:** {empresa_info.get('regime_tributario', 'N/A')}")
    
    with col2:
        st.caption(f"**Município:** {empresa_info.get('municipio', 'N/A')}")
        st.caption(f"**UF:** {empresa_info.get('uf', 'N/A')}")
    
    with col3:
        st.caption(f"**CNAE:** {empresa_info.get('cd_cnae1', 'N/A')} - {empresa_info.get('nm_cnae1', 'N/A')[:30]}")
    
    with col4:
        st.caption(f"**UF:** {empresa_info.get('uf', 'N/A')}")
    
    st.divider()
    
    # Indicadores
    st.subheader("📊 Indicadores da Empresa")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Score Risco", f"{empresa_info['score_risco_final']:.1f}")
    
    with col2:
        st.metric("Classificação", empresa_info['classificacao_risco'])
    
    with col3:
        st.metric("Total CNPJ", f"R$ {empresa_info['total_recebido_cnpj']/1e3:.1f}K")
    
    with col4:
        st.metric("Total CPF", f"R$ {empresa_info['total_recebido_cpf']/1e3:.1f}K")
    
    with col5:
        st.metric("% CPF", f"{empresa_info['perc_recebido_cpf']:.1f}%")
    
    # Alertas
    if empresa_info['classificacao_risco'] == 'ALTO':
        st.markdown(
            f"<div class='alert-critico'>"
            f"<b>⚠️ ALERTA CRÍTICO:</b> Empresa classificada como ALTO RISCO<br>"
            f"Score: {empresa_info['score_risco_final']:.1f} | "
            f"% CPF: {empresa_info['perc_recebido_cpf']:.1f}%"
            f"</div>",
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Tabs de análise
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Evolução", "👥 Sócios", "💳 Operações", "📊 Detalhes"])
    
    with tab1:
        if not detalhes['evolucao'].empty:
            df_evol = detalhes['evolucao']
            
            # Converter referência para data
            df_evol['data'] = pd.to_datetime(df_evol['referencia'].astype(str), format='%Y%m')
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_evol['data'],
                y=df_evol['vl_cnpj'],
                name='CNPJ',
                marker_color='#1f77b4'
            ))
            
            fig.add_trace(go.Bar(
                x=df_evol['data'],
                y=df_evol['vl_cpf'],
                name='CPF Sócios',
                marker_color='#ff7f0e'
            ))
            
            fig.update_layout(
                title='Evolução Mensal de Recebimentos',
                xaxis_title='Mês',
                yaxis_title='Valor (R$)',
                barmode='group',
                template=filtros['tema'],
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados de evolução.")
    
    with tab2:
        if not detalhes['socios'].empty:
            st.subheader("👥 Sócios que Recebem Pagamentos")
            
            df_socios = detalhes['socios']
            
            # Exibir sem formatação para evitar erros com None
            st.dataframe(
                df_socios,
                use_container_width=True,
                height=400
            )
            
            # Gráfico
            fig = px.bar(
                df_socios.head(10),
                x='total_recebido',
                y='nome_socio',
                orientation='h',
                title='Top 10 Sócios por Valor Recebido',
                template=filtros['tema'],
                color='total_recebido',
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhum sócio recebe pagamentos via CPF.")
    
    with tab3:
        if not detalhes['operacoes'].empty:
            st.subheader("💳 Operações Suspeitas (Últimas 100)")
            
            df_ops = detalhes['operacoes']
            
            # Converter referencia para data legível
            df_ops['mes_ano'] = pd.to_datetime(df_ops['referencia'].astype(str), format='%Y%m').dt.strftime('%m/%Y')
            
            # Resumo
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Operações", len(df_ops))
            with col2:
                st.metric("Valor Total", f"R$ {df_ops['vl_total'].sum():,.2f}")
            with col3:
                st.metric("Valor Médio", f"R$ {df_ops['vl_total'].mean():,.2f}")
            with col4:
                meses_distintos = df_ops['referencia'].nunique()
                st.metric("Meses", meses_distintos)
            
            # Gráfico por tipo de operação
            st.markdown("#### Distribuição por Tipo de Operação")
            
            valores_por_tipo = pd.DataFrame({
                'Tipo': ['PIX', 'Boleto', 'Transferência', 'Dinheiro'],
                'Valor': [
                    df_ops['vl_pix'].sum(),
                    df_ops['vl_boleto'].sum(),
                    df_ops['vl_transferencia'].sum(),
                    df_ops['vl_dinheiro'].sum()
                ]
            })
            
            fig = px.bar(
                valores_por_tipo,
                x='Tipo',
                y='Valor',
                title='Volume por Tipo de Operação',
                template=filtros['tema'],
                color='Valor',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela detalhada
            st.markdown("#### Detalhamento das Operações")
            
            # Preparar DataFrame para exibição
            df_ops_display = df_ops[[
                'mes_ano', 'identificador', 'tipo_identificador', 'nome_socio', 'nm_qualificacao',
                'vl_credito', 'vl_debito', 'vl_pix', 'vl_boleto', 
                'vl_transferencia', 'vl_dinheiro', 'vl_total'
            ]].copy()
            
            st.dataframe(
                df_ops_display.style.format({
                    'vl_credito': 'R$ {:,.2f}',
                    'vl_debito': 'R$ {:,.2f}',
                    'vl_pix': 'R$ {:,.2f}',
                    'vl_boleto': 'R$ {:,.2f}',
                    'vl_transferencia': 'R$ {:,.2f}',
                    'vl_dinheiro': 'R$ {:,.2f}',
                    'vl_total': 'R$ {:,.2f}'
                }).background_gradient(
                    subset=['vl_total'],
                    cmap='Reds'
                ),
                use_container_width=True,
                height=500
            )
            
            # Botão de exportação
            csv = df_ops.to_csv(index=False, encoding='utf-8-sig', sep=';')
            st.download_button(
                "📥 Exportar Operações (CSV)",
                csv.encode('utf-8-sig'),
                f"operacoes_{empresa_selecionada}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            st.info("Sem operações suspeitas registradas para esta empresa.")
    
    with tab4:
        st.subheader("📊 Detalhes Completos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Scores Componentes:**")
            st.metric("Score Proporção", f"{empresa_info.get('score_proporcao', 0):.0f}")
            st.metric("Score Volume", f"{empresa_info.get('score_volume_cpf', 0):.0f}")
            st.metric("Score Sócios", f"{empresa_info.get('score_qtd_socios', 0):.0f}")
        
        with col2:
            st.markdown("**Outros Indicadores:**")
            st.metric("Score Desvio Regime", f"{empresa_info.get('score_desvio_regime', 0):.0f}")
            st.metric("Score Consistência", f"{empresa_info.get('score_consistencia', 0):.0f}")
            st.metric("Sócios Recebendo", f"{int(empresa_info.get('qtd_socios_recebendo', 0))}")

def pagina_machine_learning(engine, filtros):
    """Sistema de Machine Learning para priorização."""
    st.markdown("<h1 class='main-header'>🤖 Sistema de Machine Learning</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Treinar modelo de ML para identificar automaticamente empresas 
    com maior probabilidade de irregularidades, priorizando fiscalização.
    </div>
    """, unsafe_allow_html=True)
    
    # Carregar dados
    with st.spinner('Carregando dados para ML...'):
        df_ml = carregar_dados_ml(engine)
    
    if df_ml.empty:
        st.error("Dados não carregados.")
        return
    
    st.success(f"✅ {len(df_ml):,} registros carregados para análise")
    
    # Estatísticas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Empresas", f"{len(df_ml):,}")
    
    with col2:
        suspeitas = df_ml[df_ml['target_suspeito'] == 1]
        st.metric("Suspeitas (Alto/Médio-Alto)", f"{len(suspeitas):,}")
    
    with col3:
        perc_susp = len(suspeitas) / len(df_ml) * 100
        st.metric("% Suspeitas", f"{perc_susp:.1f}%")
    
    with col4:
        st.metric("Features", "8")
    
    st.divider()
    
    # Configuração do modelo
    st.subheader("⚙️ Configuração do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        executar_treinamento = st.checkbox("Executar Treinamento", value=False)
        executar_anomalias = st.checkbox("Detectar Anomalias", value=False)
    
    with col2:
        exportar_resultados = st.checkbox("Habilitar Exportação", value=False)
    
    # Treinamento
    if executar_treinamento:
        if st.button("🚀 Treinar Modelo", type="primary"):
            with st.spinner('Treinando Random Forest...'):
                resultados, indices_test, probabilidades = treinar_modelo_ml(df_ml)
            
            if resultados:
                st.success("✅ Modelo treinado com sucesso!")
                
                # Métricas
                st.subheader("📊 Métricas do Modelo")
                
                report = resultados['report']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Acurácia", f"{report['accuracy']:.3f}")
                
                with col2:
                    st.metric("Precisão (Classe 1)", f"{report['1']['precision']:.3f}")
                
                with col3:
                    st.metric("Recall (Classe 1)", f"{report['1']['recall']:.3f}")
                
                with col4:
                    st.metric("F1-Score (Classe 1)", f"{report['1']['f1-score']:.3f}")
                
                # Matriz de confusão
                col1, col2 = st.columns(2)
                
                with col1:
                    cm = resultados['confusion_matrix']
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Não Suspeito', 'Suspeito'],
                        y=['Não Suspeito', 'Suspeito'],
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 16},
                        colorscale='Blues'
                    ))
                    
                    fig.update_layout(
                        title='Matriz de Confusão',
                        xaxis_title='Predição',
                        yaxis_title='Real',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    importance_df = resultados['feature_importance']
                    
                    fig = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Importância das Features',
                        template=filtros['tema'],
                        color='importance',
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Top empresas por probabilidade
                st.subheader("🎯 Top 50 Empresas por Probabilidade ML")
                
                df_ml['ml_probability'] = 0.0
                df_ml.loc[indices_test, 'ml_probability'] = probabilidades
                
                df_top_ml = df_ml.nlargest(50, 'ml_probability')
                
                df_display = df_top_ml[[
                    'cnpj', 'nm_razao_social', 'classificacao_risco',
                    'feat_perc_cpf', 'feat_total_cpf', 'score_final', 'ml_probability'
                ]].copy()
                
                df_display.insert(0, 'Rank', range(1, len(df_display) + 1))
                
                st.dataframe(
                    df_display.style.format({
                        'feat_perc_cpf': '{:.1f}%',
                        'feat_total_cpf': 'R$ {:,.2f}',
                        'score_final': '{:.1f}',
                        'ml_probability': '{:.3f}'
                    }),
                    use_container_width=True,
                    height=600
                )
                
                if exportar_resultados:
                    csv = df_top_ml.to_csv(index=False, encoding='utf-8-sig', sep=';')
                    st.download_button(
                        label="📥 Baixar Resultados (CSV)",
                        data=csv.encode('utf-8-sig'),
                        file_name=f"ml_resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
    
    # Detecção de anomalias
    if executar_anomalias:
        if st.button("🔍 Detectar Anomalias", type="primary"):
            with st.spinner('Executando Isolation Forest...'):
                df_anomalias = detectar_anomalias(df_ml.copy())
            
            if df_anomalias is not None:
                st.success("✅ Detecção de anomalias concluída!")
                
                anomalias_detectadas = df_anomalias[df_anomalias['anomalia'] == -1]
                
                st.subheader(f"⚠️ {len(anomalias_detectadas):,} Anomalias Detectadas")
                
                df_anom_display = anomalias_detectadas.nlargest(50, 'score_final')[[
                    'cnpj', 'nm_razao_social', 'classificacao_risco',
                    'feat_perc_cpf', 'feat_total_cpf', 'score_final', 'anomalia_score'
                ]].copy()
                
                df_anom_display.insert(0, 'Rank', range(1, len(df_anom_display) + 1))
                
                st.dataframe(
                    df_anom_display.style.format({
                        'feat_perc_cpf': '{:.1f}%',
                        'feat_total_cpf': 'R$ {:,.2f}',
                        'score_final': '{:.1f}',
                        'anomalia_score': '{:.4f}'
                    }),
                    use_container_width=True,
                    height=600
                )

def pagina_analise_setorial(engine, filtros):
    """Análise por setor (CNAE)."""
    st.markdown("<h1 class='main-header'>🏭 Análise Setorial</h1>", unsafe_allow_html=True)
    
    # Query para agregação setorial - CAST cd_cnae1 para STRING
    query = """
    SELECT 
        SUBSTR(CAST(cd_cnae1 AS STRING), 1, 2) AS setor_cnae,
        nm_cnae1,
        COUNT(DISTINCT cnpj) AS qtd_empresas,
        CAST(SUM(total_recebido_cpf) AS DOUBLE) AS volume_cpf,
        CAST(AVG(perc_recebido_cpf) AS DOUBLE) AS media_perc_cpf,
        CAST(AVG(score_risco_final) AS DOUBLE) AS score_medio,
        COUNT(DISTINCT CASE WHEN classificacao_risco = 'ALTO' THEN cnpj END) AS qtd_alto_risco
    FROM teste.dimp_score_final
    WHERE cd_cnae1 IS NOT NULL
    GROUP BY SUBSTR(CAST(cd_cnae1 AS STRING), 1, 2), nm_cnae1
    HAVING COUNT(DISTINCT cnpj) >= 5
    ORDER BY volume_cpf DESC
    LIMIT 50
    """
    
    with st.spinner('Carregando dados setoriais...'):
        try:
            df_setores = pd.read_sql(query, engine)
        except Exception as e:
            st.error(f"Erro: {str(e)}")
            return
    
    if df_setores.empty:
        st.warning("Nenhum dado setorial encontrado.")
        return
    
    st.success(f"✅ {len(df_setores)} setores carregados")
    
    # Seleção de setor
    st.subheader("🎯 Selecione um Setor para Drill-Down")
    
    setor_selecionado = st.selectbox(
        "Setor (CNAE 2 dígitos):",
        df_setores['setor_cnae'].tolist(),
        format_func=lambda x: f"{x} - {df_setores[df_setores['setor_cnae']==x]['nm_cnae1'].iloc[0][:50]}"
    )
    
    setor_info = df_setores[df_setores['setor_cnae'] == setor_selecionado].iloc[0]
    
    # Indicadores do setor
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Empresas", f"{int(setor_info['qtd_empresas']):,}")
    
    with col2:
        st.metric("Volume CPF", f"R$ {setor_info['volume_cpf']/1e6:.1f}M")
    
    with col3:
        st.metric("% CPF Médio", f"{setor_info['media_perc_cpf']:.1f}%")
    
    with col4:
        st.metric("Alto Risco", f"{int(setor_info['qtd_alto_risco']):,}")
    
    st.divider()
    
    # Empresas do setor
    query_empresas_setor = f"""
    SELECT 
        cnpj,
        nm_razao_social,
        municipio,
        uf,
        CAST(total_recebido_cpf AS DOUBLE) AS total_cpf,
        CAST(perc_recebido_cpf AS DOUBLE) AS perc_cpf,
        CAST(score_risco_final AS DOUBLE) AS score_final,
        classificacao_risco
    FROM teste.dimp_score_final
    WHERE SUBSTR(CAST(cd_cnae1 AS STRING), 1, 2) = '{setor_selecionado}'
    ORDER BY score_final DESC
    LIMIT 100
    """
    
    with st.spinner('Carregando empresas do setor...'):
        df_empresas_setor = pd.read_sql(query_empresas_setor, engine)
    
    st.subheader(f"📋 Top 100 Empresas do Setor {setor_selecionado}")
    
    # Exibir sem formatação para evitar erros com None
    st.dataframe(
        df_empresas_setor,
        use_container_width=True,
        height=600
    )
    
    # Gráficos do setor
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df_empresas_setor,
            x='perc_cpf',
            nbins=20,
            title='Distribuição % CPF no Setor',
            template=filtros['tema'],
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df_empresas_setor,
            y='score_final',
            color='classificacao_risco',
            title='Distribuição de Scores no Setor',
            template=filtros['tema']
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# FUNÇÕES PARA ANÁLISE DE FUNCIONÁRIOS - VERSÃO COM TABELA AGREGADA
# =============================================================================

@st.cache_data(ttl=3600)
def carregar_resumo_funcionarios_agregado(_engine):
    """Carrega estatísticas gerais das novas tabelas (SUPER RÁPIDO)."""
    query = """
    SELECT 
        COUNT(DISTINCT cnpj) AS total_empresas,
        COUNT(DISTINCT cpf_funcionario) AS total_funcionarios,
        COUNT(DISTINCT CASE WHEN classificacao_risco = 'ALTO' THEN cpf_funcionario END) AS func_alto_risco,
        COUNT(DISTINCT CASE WHEN classificacao_risco = 'MÉDIO-ALTO' THEN cpf_funcionario END) AS func_medio_alto,
        CAST(SUM(dimp_total_funcionario) AS DOUBLE) AS volume_total,
        CAST(SUM(valor_cpf) AS DOUBLE) AS volume_cpf,
        CAST(SUM(valor_cpf_pix) AS DOUBLE) AS volume_pix,
        CAST(AVG(multiplicador_salario) AS DOUBLE) AS media_multiplicador,
        CAST(AVG(score_risco_final) AS DOUBLE) AS media_score,
        SUM(CASE WHEN multiplicador_salario >= 3 THEN 1 ELSE 0 END) AS func_mult_3x,
        SUM(CASE WHEN multiplicador_salario >= 5 THEN 1 ELSE 0 END) AS func_mult_5x,
        SUM(CASE WHEN multiplicador_salario >= 10 THEN 1 ELSE 0 END) AS func_mult_10x
    FROM teste.dimp_func_score_final
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar resumo: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def carregar_empresas_dimp_agregado(_engine, limite=500, filtrar_cnae=True):
    """Carrega empresas usando a tabela agregada (MUITO RÁPIDO)."""
    
    # Filtro de CNAE (opcional)
    filtro_cnae = """
      AND (LEFT(CAST(CAD.cd_CNAE AS STRING), 2) = '47'
           OR CAD.cd_CNAE IN (4530703, 4530705, 4541206, 5611201, 5611203, 5611204, 5611205))
    """ if filtrar_cnae else ""
    
    query = f"""
    SELECT 
        CAD.nu_cnpj,
        CAD.nm_razao_social AS razao_social,
        CAD.nm_fantasia AS nome_fantasia,
        CAD.nm_reg_apuracao AS regime,
        CAD.nm_munic AS municipio,
        CAD.cd_uf AS uf,
        CAD.cd_cnae AS cnae_principal,
        CAD.de_cnae AS descricao_cnae,
        
        CAST(AGG.valor_nivel1 AS DOUBLE) AS valor_nivel1,
        CAST(AGG.valor_pix AS DOUBLE) AS valor_pix,
        CAST((AGG.valor_nivel1 + AGG.valor_pix) AS DOUBLE) AS total_recebido,
        AGG.qtd_operacoes
        
    FROM teste.dimp_funcionarios_agregado AS AGG
    JOIN usr_sat_ods.vw_ods_contrib AS CAD 
        ON AGG.cnpj_cpf = CAD.nu_cnpj
    WHERE CAD.cd_sit_cadastral = 1
      AND LENGTH(AGG.cnpj_cpf) = 14
      AND (AGG.valor_nivel1 > 0 OR AGG.valor_pix > 0)
      {filtro_cnae}
    ORDER BY (AGG.valor_nivel1 + AGG.valor_pix) DESC
    LIMIT {limite}
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar empresas: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def carregar_empresas_funcionarios(_engine, limite=500, filtrar_cnae=False):
    """Carrega empresas com funcionários recebendo via CPF."""
    
    filtro_cnae = """
      AND (SUBSTR(CAST(cnae_principal AS STRING), 1, 2) = '47'
           OR cnae_principal IN (4530703, 4530705, 4541206, 5611201, 5611203, 5611204, 5611205))
    """ if filtrar_cnae else ""
    
    query = f"""
    SELECT 
        cnpj,
        nm_razao_social AS razao_social,
        nm_fantasia AS nome_fantasia,
        regime_tributario AS regime,
        municipio,
        cnae_principal,
        descricao_cnae,
        
        COUNT(DISTINCT cpf_funcionario) AS qtd_funcionarios_recebendo,
        CAST(SUM(dimp_total_funcionario) AS DOUBLE) AS total_dimp,
        CAST(SUM(valor_cpf) AS DOUBLE) AS total_cpf,
        CAST(SUM(valor_cpf_pix) AS DOUBLE) AS total_pix,
        CAST(SUM(salario_contratual) AS DOUBLE) AS folha_total,
        CAST(AVG(multiplicador_salario) AS DOUBLE) AS multiplicador_medio,
        CAST(AVG(diferenca_dimp_faturamento) AS DOUBLE) AS diferenca_media,
        CAST(AVG(score_risco_final) AS DOUBLE) AS score_medio,
        
        SUM(CASE WHEN classificacao_risco = 'ALTO' THEN 1 ELSE 0 END) AS qtd_alto_risco,
        SUM(CASE WHEN multiplicador_salario >= 3 THEN 1 ELSE 0 END) AS qtd_mult_3x,
        SUM(CASE WHEN multiplicador_salario >= 5 THEN 1 ELSE 0 END) AS qtd_mult_5x
        
    FROM teste.dimp_func_score_final
    WHERE 1=1
    {filtro_cnae}
    GROUP BY cnpj, nm_razao_social, nm_fantasia, regime_tributario, 
             municipio, cnae_principal, descricao_cnae
    ORDER BY total_dimp DESC
    LIMIT {limite}
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar empresas: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def carregar_funcionarios_empresa(_engine, cnpj):
    """Carrega funcionários de uma empresa específica."""
    query = f"""
    SELECT 
        cpf_funcionario,
        ocupacao,
        tamanho_estabelecimento,
        CAST(salario_contratual AS DOUBLE) AS salario_contratual,
        CAST(valor_cpf AS DOUBLE) AS valor_cpf,
        CAST(valor_cpf_pix AS DOUBLE) AS valor_pix,
        CAST(dimp_total_funcionario AS DOUBLE) AS total_recebido,
        CAST(multiplicador_salario AS DOUBLE) AS multiplicador_salario,
        qtd_cnpjs_do_cpf AS qtd_empresas_funcionario,
        CAST(diferenca_dimp_faturamento AS DOUBLE) AS diferenca_dimp_fat,
        CAST(score_risco_final AS DOUBLE) AS score_risco,
        classificacao_risco,
        flag_recente,
        CAST(valor_cpf_ultimo_mes AS DOUBLE) AS valor_ultimo_mes
    FROM teste.dimp_func_score_final
    WHERE cnpj = '{cnpj}'
    ORDER BY multiplicador_salario DESC
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar funcionários: {str(e)}")
        return pd.DataFrame()
        
@st.cache_data(ttl=1800)
def carregar_funcionarios_multiplos(_engine, limite=200):
    """Carrega funcionários em múltiplas empresas."""
    query = f"""
    SELECT 
        cpf_funcionario,
        qtd_empresas_vinculadas,
        CAST(total_recebido_todas_empresas AS DOUBLE) AS total_recebido,
        CAST(total_salarios AS DOUBLE) AS total_salarios,
        CAST(media_multiplicador AS DOUBLE) AS multiplicador_medio,
        nivel_dispersao,
        cnpjs_vinculados
    FROM teste.dimp_func_rede_multiplas
    ORDER BY qtd_empresas_vinculadas DESC, total_recebido DESC
    LIMIT {limite}
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar múltiplos: {str(e)}")
        return pd.DataFrame()
        
@st.cache_data(ttl=3600)
def carregar_top_suspeitos(_engine, limite=100):
    """Carrega top funcionários suspeitos."""
    query = f"""
    SELECT 
        cnpj,
        nm_razao_social AS razao_social,
        cpf_funcionario,
        ocupacao,
        CAST(salario_contratual AS DOUBLE) AS salario_contratual,
        CAST(dimp_total_funcionario AS DOUBLE) AS total_recebido,
        CAST(multiplicador_salario AS DOUBLE) AS multiplicador_salario,
        CAST(diferenca_dimp_faturamento AS DOUBLE) AS diferenca_dimp_fat,
        qtd_cnpjs_do_cpf,
        CAST(score_risco_final AS DOUBLE) AS score_risco,
        classificacao_risco,
        regime_tributario,
        municipio,
        cnae_principal,
        descricao_cnae
    FROM teste.dimp_func_top_suspeitos
    LIMIT {limite}
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar suspeitos: {str(e)}")
        return pd.DataFrame()

def pagina_analise_funcionarios(engine, filtros):
    """Análise de funcionários - VERSÃO NOVA COM TABELAS OTIMIZADAS."""
    st.markdown("<h1 class='main-header'>👔 Análise de Funcionários (RAIS/CAGED + DIMP)</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Identificar funcionários registrados (RAIS/CAGED) que recebem pagamentos 
    via CPF acima do salário formal. Sistema com scoring de risco e análise de rede.
    <br><b>⚡ Nova versão:</b> Usando tabelas pré-calculadas com scores de risco.
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar se tabelas existem
    try:
        test_query = "SELECT COUNT(*) as cnt FROM teste.dimp_func_score_final LIMIT 1"
        result = pd.read_sql(test_query, engine)
        tabela_existe = True
        total_registros = result['cnt'].iloc[0]
    except:
        tabela_existe = False
        st.error("❌ Tabelas não encontradas! Execute o script SQL de criação primeiro.")
        
        with st.expander("📜 Instruções", expanded=True):
            st.markdown("""
            **Execute o script SQL fornecido no Big Data Impala Hue.**
            
            O script cria as seguintes tabelas:
            - `teste.dimp_func_score_final` - Dados consolidados com scores
            - `teste.dimp_func_rede_multiplas` - Funcionários em múltiplas empresas
            - `teste.dimp_func_top_suspeitos` - View com casos prioritários
            
            Após executar, recarregue esta página.
            """)
        return
    
    st.success(f"✅ Tabelas carregadas: {total_registros:,} registros de funcionários")
    
    # Carregar estatísticas gerais (RÁPIDO)
    with st.spinner('📊 Carregando estatísticas...'):
        df_stats = carregar_resumo_funcionarios_agregado(engine)
    
    if df_stats.empty:
        st.warning("Sem dados disponíveis")
        return
    
    stats = df_stats.iloc[0]
    
    # KPIs Gerais
    st.subheader("📊 Panorama Geral")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Empresas", f"{int(stats['total_empresas']):,}")
    
    with col2:
        st.metric("Funcionários", f"{int(stats['total_funcionarios']):,}")
    
    with col3:
        st.metric("Volume Total", f"R$ {stats['volume_total']/1e6:.1f}M")
    
    with col4:
        st.metric("Mult. Médio", f"{stats['media_multiplicador']:.2f}x")
    
    with col5:
        st.metric("Score Médio", f"{stats['media_score']:.1f}")
    
    # Segunda linha de KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Alto Risco", f"{int(stats['func_alto_risco']):,}", 
                 delta=f"{int(stats['func_medio_alto']):,} médio-alto")
    
    with col2:
        st.metric("Mult. ≥3x", f"{int(stats['func_mult_3x']):,}")
    
    with col3:
        st.metric("Mult. ≥5x", f"{int(stats['func_mult_5x']):,}")
    
    with col4:
        st.metric("Mult. ≥10x", f"{int(stats['func_mult_10x']):,}")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏢 Por Empresa", 
        "🔍 Detalhes Empresa",
        "⚠️ Top Suspeitos",
        "👥 Múltiplas Empresas",
        "📊 Análises"
    ])
    
    with tab1:
        st.subheader("🏢 Análise por Empresa")
        
        # Controles
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            limite = st.selectbox("Quantidade:", [50, 100, 200, 500], index=2)
        
        with col2:
            min_funcionarios = st.number_input("Mín. Funcionários:", 1, 50, 2)
        
        with col3:
            filtrar_cnae = st.checkbox("Apenas Comércio/Alimentação", value=False)
        
        with col4:
            if st.button("🔄 Atualizar", type="primary"):
                st.cache_data.clear()
        
        # Carregar
        with st.spinner('Carregando empresas...'):
            df_empresas = carregar_empresas_funcionarios(engine, limite, filtrar_cnae)
        
        if df_empresas.empty:
            st.warning("Nenhuma empresa encontrada")
            return
        
        # Filtrar
        df_emp_filt = df_empresas[
            df_empresas['qtd_funcionarios_recebendo'] >= min_funcionarios
        ].copy()
        
        st.success(f"✅ {len(df_emp_filt):,} empresas carregadas")
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Empresas", f"{len(df_emp_filt):,}")
        
        with col2:
            st.metric("Volume Total", f"R$ {df_emp_filt['total_dimp'].sum()/1e6:.1f}M")
        
        with col3:
            st.metric("Folha Total", f"R$ {df_emp_filt['folha_total'].sum()/1e6:.1f}M")
        
        with col4:
            total_alto_risco = df_emp_filt['qtd_alto_risco'].sum()
            st.metric("Funcs. Alto Risco", f"{int(total_alto_risco):,}")
        
        # Distribuição por regime
        st.markdown("### 📊 Distribuição por Regime")
        
        df_regime = df_emp_filt.groupby('regime').agg({
            'cnpj': 'count',
            'qtd_funcionarios_recebendo': 'sum',
            'total_dimp': 'sum',
            'qtd_alto_risco': 'sum'
        }).reset_index()
        df_regime.columns = ['Regime', 'Qtd_Empresas', 'Total_Funcs', 'Volume', 'Alto_Risco']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.dataframe(
                df_regime.style.format({
                    'Volume': 'R$ {:,.2f}'
                }),
                use_container_width=True
            )
        
        with col2:
            fig = px.pie(df_regime, values='Qtd_Empresas', names='Regime',
                        title='Por Quantidade', template=filtros['tema'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.pie(df_regime, values='Volume', names='Regime',
                        title='Por Volume', template=filtros['tema'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_emp_filt.head(20),
                x='total_dimp',
                y='razao_social',
                orientation='h',
                title='Top 20 - Volume DIMP',
                template=filtros['tema'],
                color='multiplicador_medio',
                color_continuous_scale='Reds',
                hover_data=['qtd_funcionarios_recebendo', 'qtd_mult_3x']
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df_emp_filt,
                x='qtd_funcionarios_recebendo',
                y='total_dimp',
                size='diferenca_media',
                color='multiplicador_medio',
                title='Funcionários vs Volume',
                template=filtros['tema'],
                hover_data=['razao_social', 'regime'],
                log_y=True
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabela
        st.markdown("### 📋 Lista Completa")
        
        st.dataframe(
            df_emp_filt[[
                'cnpj', 'razao_social', 'regime', 'municipio',
                'qtd_funcionarios_recebendo', 'folha_total', 'total_dimp',
                'multiplicador_medio', 'qtd_alto_risco', 'qtd_mult_3x', 'score_medio'
            ]].style.format({
                'folha_total': 'R$ {:,.2f}',
                'total_dimp': 'R$ {:,.2f}',
                'multiplicador_medio': '{:.2f}x',
                'score_medio': '{:.1f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Exportar
        if st.button("📥 Exportar (CSV)"):
            csv = df_emp_filt.to_csv(index=False, encoding='utf-8-sig', sep=';')
            st.download_button(
                "⬇️ Download",
                csv.encode('utf-8-sig'),
                f"empresas_funcionarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    with tab2:
        st.subheader("🔍 Detalhes por Empresa")
        
        if 'df_empresas' not in locals():
            df_empresas = carregar_empresas_funcionarios(engine, 500, False)
        
        empresa_sel = st.selectbox(
            "Selecione:",
            df_empresas['cnpj'].tolist(),
            format_func=lambda x: f"{df_empresas[df_empresas['cnpj']==x]['razao_social'].iloc[0]} - {x}"
        )
        
        if st.button("📥 Carregar Funcionários", type="primary"):
            with st.spinner('Carregando...'):
                df_funcs = carregar_funcionarios_empresa(engine, empresa_sel)
            
            if df_funcs.empty:
                st.warning("Nenhum funcionário encontrado")
            else:
                info = df_empresas[df_empresas['cnpj'] == empresa_sel].iloc[0]
                
                st.markdown(f"### {info['razao_social']}")
                st.caption(f"**CNPJ:** {empresa_sel} | **Regime:** {info['regime']}")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Funcionários", len(df_funcs))
                
                with col2:
                    st.metric("Volume", f"R$ {df_funcs['total_recebido'].sum()/1e3:.1f}K")
                
                with col3:
                    st.metric("Mult. Médio", f"{df_funcs['multiplicador_salario'].mean():.2f}x")
                
                with col4:
                    alto = len(df_funcs[df_funcs['classificacao_risco'] == 'ALTO'])
                    st.metric("Alto Risco", alto)
                
                with col5:
                    mult3 = len(df_funcs[df_funcs['multiplicador_salario'] >= 3])
                    st.metric("Mult. ≥3x", mult3)
                
                # Gráficos
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        df_funcs[df_funcs['multiplicador_salario'] <= 15],
                        x='multiplicador_salario',
                        nbins=30,
                        title='Distribuição Multiplicador',
                        template=filtros['tema'],
                        color_discrete_sequence=['#e74c3c']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        df_funcs,
                        x='salario_contratual',
                        y='total_recebido',
                        size='multiplicador_salario',
                        color='classificacao_risco',
                        title='Salário vs Recebido',
                        template=filtros['tema']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tabela
                st.dataframe(
                    df_funcs[[
                        'cpf_funcionario', 'ocupacao', 'salario_contratual', 'total_recebido',
                        'multiplicador_salario', 'qtd_empresas_funcionario',
                        'score_risco', 'classificacao_risco'
                    ]].style.format({
                        'salario_contratual': 'R$ {:,.2f}',
                        'total_recebido': 'R$ {:,.2f}',
                        'multiplicador_salario': '{:.2f}x',
                        'score_risco': '{:.1f}'
                    }).background_gradient(
                        subset=['multiplicador_salario'],
                        cmap='Reds',
                        vmin=0,
                        vmax=10
                    ),
                    use_container_width=True,
                    height=500
                )
    
    with tab3:
        st.subheader("⚠️ Top Funcionários Suspeitos")
        
        st.markdown("""
        <div class='alert-critico'>
        <b>🔴 Casos Prioritários:</b> Funcionários com alto risco e multiplicador ≥3x
        </div>
        """, unsafe_allow_html=True)
        
        limite_susp = st.slider("Limite:", 20, 200, 100, 20)
        
        if st.button("🔍 Carregar Top Suspeitos", type="primary"):
            with st.spinner('Carregando...'):
                df_susp = carregar_top_suspeitos(engine, limite_susp)
            
            if not df_susp.empty:
                st.success(f"✅ {len(df_susp)} casos encontrados")
                
                # KPIs
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total", len(df_susp))
                
                with col2:
                    st.metric("Volume", f"R$ {df_susp['total_recebido'].sum()/1e6:.1f}M")
                
                with col3:
                    st.metric("Mult. Médio", f"{df_susp['multiplicador_salario'].mean():.2f}x")
                
                with col4:
                    st.metric("Score Médio", f"{df_susp['score_risco'].mean():.1f}")
                
                # Gráfico
                fig = px.scatter(
                    df_susp.head(50),
                    x='multiplicador_salario',
                    y='total_recebido',
                    size='score_risco',
                    color='classificacao_risco',
                    title='Top 50 - Multiplicador vs Volume',
                    template=filtros['tema'],
                    hover_data=['razao_social', 'municipio']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela
                st.dataframe(
                    df_susp[[
                        'cnpj', 'razao_social', 'cpf_funcionario', 'salario_contratual',
                        'total_recebido', 'multiplicador_salario', 'qtd_cnpjs_do_cpf',
                        'score_risco', 'classificacao_risco', 'regime_tributario', 'municipio'
                    ]].style.format({
                        'salario_contratual': 'R$ {:,.2f}',
                        'total_recebido': 'R$ {:,.2f}',
                        'multiplicador_salario': '{:.2f}x',
                        'score_risco': '{:.1f}'
                    }),
                    use_container_width=True,
                    height=500
                )
                
                # Exportar
                csv = df_susp.to_csv(index=False, encoding='utf-8-sig', sep=';')
                st.download_button(
                    "📥 Exportar Suspeitos",
                    csv.encode('utf-8-sig'),
                    f"top_suspeitos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
    
    with tab4:
        st.subheader("👥 Funcionários em Múltiplas Empresas")
        
        limite_mult = st.slider("Limite:", 50, 500, 200, 50)
        
        if st.button("🔍 Carregar", type="primary"):
            with st.spinner('Carregando...'):
                df_mult = carregar_funcionarios_multiplos(engine, limite_mult)
            
            if not df_mult.empty:
                st.success(f"✅ {len(df_mult)} funcionários")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total", len(df_mult))
                
                with col2:
                    st.metric("Volume", f"R$ {df_mult['total_recebido'].sum()/1e6:.1f}M")
                
                with col3:
                    st.metric("Máx. Empresas", int(df_mult['qtd_empresas_vinculadas'].max()))
                
                with col4:
                    st.metric("Mult. Médio", f"{df_mult['multiplicador_medio'].mean():.2f}x")
                
                # Gráfico
                fig = px.scatter(
                    df_mult,
                    x='qtd_empresas_vinculadas',
                    y='total_recebido',
                    size='total_recebido',
                    color='nivel_dispersao',
                    title='Empresas vs Volume',
                    template=filtros['tema']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela
                st.dataframe(
                    df_mult[[
                        'cpf_funcionario', 'qtd_empresas_vinculadas', 'total_salarios',
                        'total_recebido', 'multiplicador_medio', 'nivel_dispersao'
                    ]].style.format({
                        'total_salarios': 'R$ {:,.2f}',
                        'total_recebido': 'R$ {:,.2f}',
                        'multiplicador_medio': '{:.2f}x'
                    }),
                    use_container_width=True,
                    height=500
                )
    
    with tab5:
        st.subheader("📊 Análises Estatísticas")
        st.info("Em desenvolvimento - análises adicionais")
        
def pagina_analise_socios_multiplos(engine, filtros):
    """Análise de sócios que recebem em múltiplas empresas."""
    st.markdown("<h1 class='main-header'>👥 Análise de Sócios em Múltiplas Empresas</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Identificar sócios que recebem pagamentos em múltiplas empresas,
    padrão que pode indicar estruturas de dispersão de receita.
    </div>
    """, unsafe_allow_html=True)
    
    # Query para sócios em múltiplas empresas
    query = """
    SELECT 
        cpf_socio,
        nome_socio,
        qtd_empresas,
        nivel_dispersao,
        CAST(total_recebido AS DOUBLE) AS total_recebido,
        cnpjs_relacionados
    FROM teste.dimp_socios_multiplas_empresas
    ORDER BY qtd_empresas DESC, total_recebido DESC
    LIMIT 100
    """
    
    with st.spinner('Carregando dados de sócios...'):
        try:
            df_socios = pd.read_sql(query, engine)
        except Exception as e:
            st.error(f"Erro: {str(e)}")
            return
    
    if df_socios.empty:
        st.warning("Nenhum sócio em múltiplas empresas encontrado.")
        return
    
    st.success(f"✅ {len(df_socios)} sócios em múltiplas empresas")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sócios", f"{len(df_socios):,}")
    
    with col2:
        st.metric("Volume Total", f"R$ {df_socios['total_recebido'].sum()/1e6:.1f}M")
    
    with col3:
        media_not = df_socios['qtd_empresas'].mean() if len(df_socios) > 0 else 0
        st.metric("Média Empresas/Sócio", f"{media_not:.1f}")
    
    with col4:
        st.metric("Máx. Empresas", f"{df_socios['qtd_empresas'].max()}")
    
    st.divider()
    
    # Distribuição por nível de dispersão
    st.subheader("📊 Distribuição por Nível de Dispersão")
    
    dist_dispersao = df_socios['nivel_dispersao'].value_counts().reset_index()
    dist_dispersao.columns = ['Nível', 'Quantidade']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            dist_dispersao,
            x='Nível',
            y='Quantidade',
            title='Sócios por Nível de Dispersão',
            template=filtros['tema'],
            color='Quantidade',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df_socios,
            x='qtd_empresas',
            y='total_recebido',
            title='Empresas vs Volume Recebido',
            template=filtros['tema'],
            color='nivel_dispersao',
            size='total_recebido',
            hover_data=['nome_socio']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de sócios
    st.subheader("📋 Top 100 Sócios com Maior Dispersão")
    
    df_display = df_socios.copy()
    df_display.insert(0, 'Rank', range(1, len(df_display) + 1))
    
    # Limitar CNPJs relacionados
    df_display['empresas_resumo'] = df_display['cnpjs_relacionados'].apply(
        lambda x: str(x)[:100] + '...' if len(str(x)) > 100 else str(x)
    )
    
    # Exibir sem formatação
    st.dataframe(
        df_display[['Rank', 'cpf_socio', 'nome_socio', 'qtd_empresas', 
                    'nivel_dispersao', 'total_recebido', 'empresas_resumo']],
        use_container_width=True,
        height=600
    )

def pagina_analise_temporal(engine, filtros):
    """Análise da evolução temporal dos pagamentos."""
    st.markdown("<h1 class='main-header'>📈 Análise Temporal</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Analisar a evolução dos pagamentos via CPF e CNPJ ao longo do tempo,
    identificando tendências e padrões sazonais.
    </div>
    """, unsafe_allow_html=True)
    
    # Query para evolução temporal
    query_cnpj = """
    SELECT 
        referencia,
        COUNT(DISTINCT cnpj) AS qtd_empresas,
        CAST(SUM(vl_total) AS DOUBLE) AS volume_total
    FROM teste.dimp_pagamentos_cnpj
    GROUP BY referencia
    ORDER BY referencia
    """
    
    query_cpf = """
    SELECT 
        referencia,
        COUNT(DISTINCT cnpj) AS qtd_empresas,
        CAST(SUM(vl_total) AS DOUBLE) AS volume_total
    FROM teste.dimp_pagamentos_cpf
    GROUP BY referencia
    ORDER BY referencia
    """
    
    with st.spinner('Carregando dados temporais...'):
        try:
            df_cnpj = pd.read_sql(query_cnpj, engine)
            df_cpf = pd.read_sql(query_cpf, engine)
        except Exception as e:
            st.error(f"Erro: {str(e)}")
            return
    
    if df_cnpj.empty or df_cpf.empty:
        st.warning("Dados temporais não disponíveis.")
        return
    
    # Converter referência para data
    df_cnpj['data'] = pd.to_datetime(df_cnpj['referencia'].astype(str), format='%Y%m')
    df_cpf['data'] = pd.to_datetime(df_cpf['referencia'].astype(str), format='%Y%m')
    
    # Merge
    df_temporal = pd.merge(
        df_cnpj[['data', 'referencia', 'qtd_empresas', 'volume_total']],
        df_cpf[['data', 'qtd_empresas', 'volume_total']],
        on='data',
        suffixes=('_cnpj', '_cpf'),
        how='outer'
    ).fillna(0)
    
    df_temporal = df_temporal.sort_values('data')
    
    st.success(f"✅ {len(df_temporal)} meses analisados")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Período", f"{df_temporal['data'].min().strftime('%m/%Y')} - {df_temporal['data'].max().strftime('%m/%Y')}")
    
    with col2:
        st.metric("Volume Total CNPJ", f"R$ {df_temporal['volume_total_cnpj'].sum()/1e6:.1f}M")
    
    with col3:
        st.metric("Volume Total CPF", f"R$ {df_temporal['volume_total_cpf'].sum()/1e6:.1f}M")
    
    with col4:
        perc_cpf_total = (df_temporal['volume_total_cpf'].sum() / 
                         (df_temporal['volume_total_cnpj'].sum() + df_temporal['volume_total_cpf'].sum()) * 100)
        st.metric("% CPF do Total", f"{perc_cpf_total:.1f}%")
    
    st.divider()
    
    # Gráfico de evolução
    st.subheader("📊 Evolução Mensal dos Volumes")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_temporal['data'],
        y=df_temporal['volume_total_cnpj'] / 1e6,
        mode='lines+markers',
        name='CNPJ',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_temporal['data'],
        y=df_temporal['volume_total_cpf'] / 1e6,
        mode='lines+markers',
        name='CPF',
        line=dict(color='#ff7f0e', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title='Evolução do Volume de Pagamentos (Milhões R$)',
        xaxis_title='Mês',
        yaxis_title='Volume (Milhões R$)',
        template=filtros['tema'],
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de empresas
    st.subheader("📈 Evolução da Quantidade de Empresas")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_temporal['data'],
        y=df_temporal['qtd_empresas_cnpj'],
        name='Recebem no CNPJ',
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        x=df_temporal['data'],
        y=df_temporal['qtd_empresas_cpf'],
        name='Sócios Recebem CPF',
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title='Quantidade de Empresas por Mês',
        xaxis_title='Mês',
        yaxis_title='Quantidade de Empresas',
        template=filtros['tema'],
        height=500,
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas de crescimento
    st.subheader("📊 Estatísticas de Crescimento")
    
    if len(df_temporal) >= 2:
        # Crescimento CPF
        crescimento_cpf = ((df_temporal['volume_total_cpf'].iloc[-1] / 
                           df_temporal['volume_total_cpf'].iloc[0]) - 1) * 100 if df_temporal['volume_total_cpf'].iloc[0] > 0 else 0
        
        # Crescimento CNPJ
        crescimento_cnpj = ((df_temporal['volume_total_cnpj'].iloc[-1] / 
                            df_temporal['volume_total_cnpj'].iloc[0]) - 1) * 100 if df_temporal['volume_total_cnpj'].iloc[0] > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Crescimento CPF", f"{crescimento_cpf:+.1f}%")
        
        with col2:
            st.metric("Crescimento CNPJ", f"{crescimento_cnpj:+.1f}%")
        
        with col3:
            media_mensal_cpf = df_temporal['volume_total_cpf'].mean()
            st.metric("Média Mensal CPF", f"R$ {media_mensal_cpf/1e6:.1f}M")
        
        with col4:
            desvio_cpf = df_temporal['volume_total_cpf'].std()
            st.metric("Desvio Padrão CPF", f"R$ {desvio_cpf/1e6:.1f}M")

def pagina_padroes_suspeitos(engine, filtros):
    """Análise de padrões suspeitos específicos."""
    st.markdown("<h1 class='main-header'>🚨 Padrões Suspeitos</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Identificar padrões específicos de comportamento que indicam
    possíveis irregularidades fiscais ou estruturas de planejamento tributário abusivo.
    </div>
    """, unsafe_allow_html=True)
    
    # Padrão 1: 100% CPF
    st.subheader("🔴 Padrão 1: Empresas que SÓ recebem via CPF (100%)")
    
    query_100 = """
    SELECT 
        cnpj,
        nm_razao_social,
        regime_tributario,
        municipio,
        uf,
        cd_cnae1,
        nm_cnae1,
        CAST(total_recebido_cpf AS DOUBLE) AS total_cpf,
        qtd_socios_recebendo,
        meses_com_pagto_cpf,
        CAST(score_risco_final AS DOUBLE) AS score_final
    FROM teste.dimp_score_final
    WHERE perc_recebido_cpf = 100
        AND total_recebido_cpf >= 10000
    ORDER BY total_cpf DESC
    LIMIT 50
    """
    
    with st.spinner('Carregando padrão 100% CPF...'):
        df_100 = pd.read_sql(query_100, engine)
    
    if not df_100.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Empresas 100% CPF", f"{len(df_100):,}")
        
        with col2:
            st.metric("Volume Total", f"R$ {df_100['total_cpf'].sum()/1e6:.1f}M")
        
        with col3:
            st.metric("Score Médio", f"{df_100['score_final'].mean():.1f}")
        
        st.markdown("""
        <div class='alert-critico'>
        <b>⚠️ ATENÇÃO:</b> Essas empresas NUNCA recebem no CNPJ, apenas nos CPFs dos sócios.
        Padrão altamente suspeito que pode indicar estruturas fantasmas ou subfaturamento.
        </div>
        """, unsafe_allow_html=True)
        
        # Exibir sem formatação
        st.dataframe(
            df_100,
            use_container_width=True,
            height=400
        )
    
    st.divider()
    
    # Padrão 2: Alto valor + Alta proporção
    st.subheader("🟠 Padrão 2: Alto Valor (>100K) + Alta Proporção (>80%)")
    
    query_alto = """
    SELECT 
        cnpj,
        nm_razao_social,
        regime_tributario,
        municipio,
        CAST(total_recebido_cpf AS DOUBLE) AS total_cpf,
        CAST(perc_recebido_cpf AS DOUBLE) AS perc_cpf,
        qtd_socios_recebendo,
        CAST(score_risco_final AS DOUBLE) AS score_final
    FROM teste.dimp_score_final
    WHERE total_recebido_cpf >= 100000
        AND perc_recebido_cpf >= 80
    ORDER BY total_cpf DESC
    LIMIT 50
    """
    
    with st.spinner('Carregando padrão alto valor...'):
        df_alto = pd.read_sql(query_alto, engine)
    
    if not df_alto.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Empresas Alto Valor", f"{len(df_alto):,}")
        
        with col2:
            st.metric("Volume Total", f"R$ {df_alto['total_cpf'].sum()/1e6:.1f}M")
        
        with col3:
            st.metric("Impacto Médio", f"R$ {df_alto['total_cpf'].mean()/1e3:.1f}K")
        
        st.markdown("""
        <div class='alert-alto'>
        <b>⚠️ ALERTA ALTO:</b> Empresas com volume significativo e alta proporção em CPF.
        Maior potencial de impacto fiscal em caso de irregularidade.
        </div>
        """, unsafe_allow_html=True)
        
        # Exibir sem formatação
        st.dataframe(
            df_alto,
            use_container_width=True,
            height=400
        )
    
    st.divider()
    
    # Padrão 3: Múltiplos sócios recebendo
    st.subheader("🟡 Padrão 3: Múltiplos Sócios Recebendo (5+)")
    
    query_mult = """
    SELECT 
        cnpj,
        nm_razao_social,
        regime_tributario,
        CAST(total_recebido_cpf AS DOUBLE) AS total_cpf,
        qtd_socios_recebendo,
        CAST(total_recebido_cpf / qtd_socios_recebendo AS DOUBLE) AS media_por_socio,
        CAST(score_risco_final AS DOUBLE) AS score_final
    FROM teste.dimp_score_final
    WHERE qtd_socios_recebendo >= 5
    ORDER BY qtd_socios_recebendo DESC, total_cpf DESC
    LIMIT 50
    """
    
    with st.spinner('Carregando padrão múltiplos sócios...'):
        df_mult = pd.read_sql(query_mult, engine)
    
    if not df_mult.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Empresas 5+ Sócios", f"{len(df_mult):,}")
        
        with col2:
            st.metric("Máx. Sócios", f"{df_mult['qtd_socios_recebendo'].max()}")
        
        with col3:
            st.metric("Volume Total", f"R$ {df_mult['total_cpf'].sum()/1e6:.1f}M")
        
        st.markdown("""
        <div class='alert-alto'>
        <b>⚠️ ATENÇÃO:</b> Estruturas com muitos sócios recebendo podem indicar
        dispersão intencional de receita para dificultar fiscalização.
        </div>
        """, unsafe_allow_html=True)
        
        # Exibir sem formatação
        st.dataframe(
            df_mult,
            use_container_width=True,
            height=400
        )

def pagina_diagnostico(engine, resumo):
    """Página de diagnóstico do sistema."""
    st.markdown("<h1 class='main-header'>🔧 Diagnóstico do Sistema</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Verificar a estrutura das tabelas e disponibilidade dos dados.
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar colunas disponíveis
    if 'colunas_disponiveis' in resumo:
        st.subheader("📋 Colunas Disponíveis na Tabela dimp_score_final")
        
        colunas = resumo['colunas_disponiveis']
        
        # Separar por tipo
        colunas_texto = [c for c in colunas if 'nm_' in c or 'de_' in c or 'cd_' in c]
        colunas_valores = [c for c in colunas if 'vl_' in c or 'total_' in c or 'perc_' in c]
        colunas_scores = [c for c in colunas if 'score_' in c or 'classificacao' in c]
        colunas_outras = [c for c in colunas if c not in colunas_texto + colunas_valores + colunas_scores]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Colunas de Texto/Identificação:**")
            for col in sorted(colunas_texto):
                st.code(col, language="")
            
            st.markdown("**Colunas de Valores:**")
            for col in sorted(colunas_valores):
                st.code(col, language="")
        
        with col2:
            st.markdown("**Colunas de Score/Classificação:**")
            for col in sorted(colunas_scores):
                st.code(col, language="")
            
            st.markdown("**Outras Colunas:**")
            for col in sorted(colunas_outras):
                st.code(col, language="")
    
    st.divider()
    
    # Testar queries
    st.subheader("🧪 Teste de Queries")
    
    if st.button("Testar Query Básica"):
        try:
            query_teste = """
            SELECT *
            FROM teste.dimp_score_final
            LIMIT 5
            """
            df_teste = pd.read_sql(query_teste, engine)
            st.success(f"✅ Query executada com sucesso! {len(df_teste)} registros retornados.")
            st.dataframe(df_teste, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
    
    st.divider()
    
    # Estatísticas das tabelas
    st.subheader("📊 Estatísticas das Tabelas")
    
    tabelas = [
        'teste.dimp_cnpj_base',
        'teste.dimp_socios',
        'teste.dimp_pagamentos_cnpj',
        'teste.dimp_pagamentos_cpf',
        'teste.dimp_comparacao_cnpj_cpf',
        'teste.dimp_score_final',
        'teste.dimp_operacoes_suspeitas',
        'teste.dimp_socios_multiplas_empresas'
    ]
    
    for tabela in tabelas:
        try:
            query_count = f"SELECT COUNT(*) as cnt FROM {tabela}"
            result = pd.read_sql(query_count, engine)
            count = result['cnt'].iloc[0]
            
            if count > 0:
                st.success(f"✅ {tabela}: {count:,} registros")
            else:
                st.warning(f"⚠️ {tabela}: VAZIA")
        except Exception as e:
            st.error(f"❌ {tabela}: Erro - {str(e)[:100]}")

def pagina_sobre():
    """Página sobre o sistema."""
    st.markdown("<h1 class='main-header'>ℹ️ Sobre o Sistema DIMP</h1>", unsafe_allow_html=True)
    
    texto_sobre = """
    ## Sistema de Análise de Meios de Pagamento
    
    ### Descrição
    
    O Sistema DIMP é uma ferramenta desenvolvida pela Receita Estadual de Santa Catarina para 
    identificar e analisar empresas que recebem valores significativos via CPF de sócios, 
    em vez do CNPJ da empresa, padrão que pode indicar irregularidades fiscais.
    
    ### Funcionalidades
    
    - **Dashboard Executivo**: Visão geral com KPIs principais
    - **Ranking de Empresas**: Listagem priorizada com drill-down
    - **Análise Detalhada**: Drill-down completo por empresa
    - **Machine Learning**: Modelo preditivo para priorização
    - **Análise Setorial**: Comparação por setor econômico
    - **Análise de Sócios**: Sócios em múltiplas empresas
    - **Análise Temporal**: Evolução dos padrões ao longo do tempo
    - **Padrões Suspeitos**: Identificação de comportamentos anômalos
    - **Filtros Avançados**: Sistema de filtros dinâmicos
    - **Exportação**: Download de dados para análise offline
    
    ### Metodologia
    
    O sistema utiliza um modelo de scoring baseado em múltiplos indicadores:
    
    1. **Proporção CPF vs CNPJ** (peso 30%): Percentual recebido em CPF
    2. **Volume Absoluto** (peso 25%): Valor total em CPF
    3. **Quantidade de Sócios** (peso 15%): Número de CPFs recebendo
    4. **Desvio vs Regime** (peso 20%): Comparação com média do regime
    5. **Consistência Temporal** (peso 10%): Recebimento contínuo
    
    ### Classificações de Risco
    
    - **ALTO**: Score ≥ 80 - Prioridade máxima para fiscalização
    - **MÉDIO-ALTO**: Score 60-79 - Alta prioridade
    - **MÉDIO**: Score 40-59 - Monitoramento
    - **BAIXO**: Score < 40 - Padrão normal
    
    ### Padrões Suspeitos Identificados
    
    1. **100% CPF**: Empresas que nunca recebem no CNPJ
    2. **Alto Valor + Alta Proporção**: >R$ 100K e >80% via CPF
    3. **Múltiplos Sócios**: 5 ou mais sócios recebendo simultaneamente
    4. **Dispersão em Rede**: Sócios em múltiplas empresas
    5. **Crescimento Anormal**: Aumento súbito de recebimentos
    
    ### Tecnologias Utilizadas
    
    - **Python**: Linguagem principal
    - **Streamlit**: Framework de dashboard
    - **Impala**: Banco de dados (Big Data)
    - **Plotly**: Visualizações interativas
    - **Scikit-learn**: Machine Learning
    - **Pandas**: Manipulação de dados
    - **SQLAlchemy**: Conexão com banco
    
    ### Arquitetura de Dados
    
    ```
    Fonte de Dados → Impala (Big Data) → Cache Streamlit → Visualização
         │                                      │
         │                                      └→ Drill-Down sob demanda
         └→ Tabelas:
            • dimp_cnpj_base
            • dimp_pagamentos_cnpj
            • dimp_pagamentos_cpf
            • dimp_score_final
            • dimp_socios_multiplas_empresas
            • dimp_operacoes_suspeitas
    ```
    
    ### Performance
    
    - **Carga Inicial**: Apenas agregados (~2-5 segundos)
    - **Drill-Down**: Sob demanda (~1-3 segundos)
    - **Cache TTL**: 1 hora (ajustável)
    - **Empresas Analisadas**: 10.000+
    - **Período**: 2024-2025 (configurável)
    
    ### Desenvolvimento
    
    **Auditor Fiscal:** Tiago Severo  
    **Órgão:** Receita Estadual de Santa Catarina  
    **Versão:** 1.0  
    **Data:** Outubro 2025  
    **Ambiente:** Produção
    
    ### Próximas Funcionalidades
    
    - [ ] Exportação automática de relatórios
    - [ ] Alertas por e-mail
    - [ ] Integração com outros sistemas
    - [ ] Análise preditiva avançada
    - [ ] Dashboard mobile
    
    ### Suporte
    
    Para dúvidas, sugestões ou reportar problemas, contate:
    - **E-mail**: tsevero@sef.sc.gov.br
    """
    
    st.markdown(texto_sobre)
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Versão", "1.0")
    
    with col2:
        st.metric("Última Atualização", datetime.now().strftime('%d/%m/%Y'))
    
    with col3:
        st.metric("Ambiente", "PRODUÇÃO")
    
    with col4:
        st.metric("Uptime", "99.9%")

# =============================================================================
# 8. FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal do dashboard."""
    
    # Sidebar - Menu
    st.sidebar.title("💳 Sistema DIMP")
    st.sidebar.caption("Análise de Meios de Pagamento")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Menu de Navegação")
    
    paginas = [
        "Dashboard Executivo",
        "🎯 Ranking de Empresas",
        "🔍 Drill-Down Empresa",
        "🤖 Machine Learning",
        "🏭 Análise Setorial",
        "👥 Sócios Múltiplas Empresas",
        "👔 Funcionários (RAIS/CAGED)",
        "📈 Análise Temporal",
        "🚨 Padrões Suspeitos",
        "🔧 Diagnóstico",
        "ℹ️ Sobre o Sistema"
    ]
    
    # Controle de página
    if 'pagina_atual' not in st.session_state:
        st.session_state['pagina_atual'] = "Dashboard Executivo"
    
    # Radio button com key única
    pagina_selecionada = st.sidebar.radio(
        "Selecione:",
        paginas,
        index=paginas.index(st.session_state['pagina_atual']) if st.session_state['pagina_atual'] in paginas else 0,
        label_visibility="collapsed",
        key="menu_radio"
    )
    
    # Atualizar estado apenas se mudou
    if pagina_selecionada != st.session_state['pagina_atual']:
        st.session_state['pagina_atual'] = pagina_selecionada
        st.rerun()
    
    # Conexão
    engine = get_impala_engine()
    
    if engine is None:
        st.error("❌ Não foi possível conectar ao banco de dados.")
        return
    
    # Carregar resumo inicial (apenas na primeira carga)
    if 'resumo_geral' not in st.session_state:
        with st.spinner('Carregando resumo geral...'):
            st.session_state['resumo_geral'] = carregar_resumo_geral(engine)
    
    resumo = st.session_state['resumo_geral']
    
    # Indicador de dados carregados
    if resumo and 'panorama' in resumo:
        kpis = calcular_kpis_resumo(resumo)
        st.sidebar.success(f"✅ {kpis['total_empresas']:,} empresas")
        st.sidebar.info(f"R$ {kpis['volume_cpf']/1e6:.1f}M via CPF")
    
    # Filtros
    filtros = criar_filtros_sidebar()
    
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("ℹ️ Informações"):
        st.caption(f"**Versão:** 1.0")
        st.caption(f"**Atualização:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        st.caption(f"**Dev:** Tiago Severo - AFRE")
    
    # Roteamento de páginas
    try:
        if pagina_selecionada == "Dashboard Executivo":
            pagina_dashboard_executivo(resumo, filtros)
        
        elif pagina_selecionada == "🎯 Ranking de Empresas":
            pagina_ranking_empresas(engine, filtros)
        
        elif pagina_selecionada == "🔍 Drill-Down Empresa":
            pagina_drill_down_empresa(engine, filtros)
        
        elif pagina_selecionada == "🤖 Machine Learning":
            pagina_machine_learning(engine, filtros)
        
        elif pagina_selecionada == "🏭 Análise Setorial":
            pagina_analise_setorial(engine, filtros)
        
        elif pagina_selecionada == "👥 Sócios Múltiplas Empresas":
            pagina_analise_socios_multiplos(engine, filtros)
        
        elif pagina_selecionada == "👔 Funcionários (RAIS/CAGED)":
            pagina_analise_funcionarios(engine, filtros)
        
        elif pagina_selecionada == "📈 Análise Temporal":
            pagina_analise_temporal(engine, filtros)
        
        elif pagina_selecionada == "🚨 Padrões Suspeitos":
            pagina_padroes_suspeitos(engine, filtros)
        
        elif pagina_selecionada == "🔧 Diagnóstico":
            pagina_diagnostico(engine, resumo)
        
        elif pagina_selecionada == "ℹ️ Sobre o Sistema":
            pagina_sobre()
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar a página: {str(e)}")
        st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"Sistema DIMP v1.0 | SEFAZ/SC | "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')}"
        f"</div>",
        unsafe_allow_html=True
    )

# =============================================================================
# 9. EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    main()