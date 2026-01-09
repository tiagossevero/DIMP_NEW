# DIMP - Sistema de Análise de Meios de Pagamento

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-Governo%20SC-green.svg)
![Status](https://img.shields.io/badge/Status-Produção-brightgreen.svg)

**Sistema de inteligência fiscal para análise de pagamentos via CNPJ vs CPF de sócios**

*Receita Estadual de Santa Catarina - SEFAZ/SC*

</div>

---

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Metodologia de Scoring](#metodologia-de-scoring)
- [Estrutura do Banco de Dados](#estrutura-do-banco-de-dados)
- [Cache e Performance](#cache-e-performance)
- [Segurança](#segurança)
- [Contribuição](#contribuição)
- [Autor](#autor)
- [Licença](#licença)

---

## Sobre o Projeto

O **DIMP** (Declaração de Informações de Meios de Pagamento) é um sistema analítico avançado desenvolvido pela Receita Estadual de Santa Catarina para identificar e analisar empresas que recebem valores significativos via CPF de sócios ao invés do CNPJ da empresa.

Este padrão de comportamento frequentemente indica potenciais irregularidades fiscais, como:
- Omissão de receitas
- Sonegação de impostos
- Estruturas empresariais suspeitas
- Fraudes em meios de pagamento

O sistema combina **dashboards interativos**, **machine learning** e **análise de redes** para fornecer aos auditores fiscais uma ferramenta completa de investigação.

### Principais Objetivos

- Identificar empresas com alto percentual de recebimentos via CPF
- Calcular scores de risco baseados em múltiplos fatores
- Detectar padrões suspeitos de pagamento
- Analisar redes de sócios em múltiplas empresas
- Comparar recebimentos de funcionários com salários formais
- Fornecer análises setoriais e temporais

---

## Funcionalidades

### 1. Dashboard Executivo
- KPIs principais com métricas consolidadas
- Indicadores de risco (Alto, Médio-Alto, Médio, Baixo)
- Visão geral de empresas analisadas
- Tooltips explicativos em todos os indicadores

### 2. Ranking de Empresas
- Ordenação por múltiplos critérios (Score, Valor, Nome)
- Seleção customizável de Top N (10-100 empresas)
- Drill-down interativo para análise detalhada
- Filtros por regime tributário e setor

### 3. Drill-Down por Empresa
- Busca por CNPJ com validação
- Breakdown completo do score de risco
- Composição de receitas (CNPJ vs CPF)
- Distribuição por método de pagamento (Pix, Cartão, Boleto, etc.)
- Informações de sócios e quadro societário
- Análise temporal de transações

### 4. Machine Learning
- Treinamento de modelo Random Forest para classificação de risco
- Detecção de anomalias com Isolation Forest
- Análise de importância de features
- Métricas de classificação (Precision, Recall, F1-Score)
- Matriz de confusão interativa
- Exportação de modelos treinados

### 5. Análise Setorial
- Análise de risco por CNAE (código de atividade econômica)
- Métricas por setor:
  - Quantidade de empresas
  - Volume total e distribuição
  - Score médio de risco
  - Empresas de alto risco por setor

### 6. Análise de Funcionários (RAIS/CAGED)
- Cruzamento entre funcionários formais e pagamentos DIMP
- Identificação de funcionários recebendo acima do salário
- Score de risco por funcionário
- Análise de rede de funcionários em múltiplas empresas
- Scores pré-calculados para performance

### 7. Sócios em Múltiplas Empresas
- Identificação de sócios recebendo em várias empresas
- Classificação de nível de dispersão
- Análise de relacionamento em rede
- Identificação de estruturas suspeitas

### 8. Análise Temporal
- Séries temporais de pagamentos
- Comparação de tendências CNPJ vs CPF
- Evolução mensal de volumes
- Identificação de padrões sazonais

### 9. Padrões Suspeitos
Detecção automática de múltiplos padrões:
- **100% CPF**: Empresas que recebem apenas via CPF
- **Alto Valor + Alta Proporção**: >R$100K e >80% via CPF
- **Múltiplos Sócios**: 5+ sócios recebendo simultaneamente
- **Dispersão em Rede**: Sócios em muitas empresas
- **Crescimento Anormal**: Aumentos súbitos de recebimentos

### 10. Diagnóstico do Sistema
- Verificação de esquemas de tabelas
- Checagem de disponibilidade de colunas
- Interface de teste de queries
- Estatísticas de tabelas
- Validação de estrutura de dados

### 11. Sobre o Sistema
- Documentação completa
- Explicação da metodologia
- Definições de classificação de risco
- Stack tecnológico
- Métricas de performance

---

## Tecnologias Utilizadas

### Frontend & Framework
| Tecnologia | Descrição |
|------------|-----------|
| **Streamlit** | Framework principal do dashboard |
| **Plotly** | Visualizações interativas |
| **HTML/CSS** | Estilos customizados |

### Backend & Dados
| Tecnologia | Descrição |
|------------|-----------|
| **Python 3.8+** | Linguagem principal |
| **SQLAlchemy** | ORM e abstração de banco |
| **Apache Impala** | Engine de queries Big Data |
| **Pandas** | Manipulação de dados |
| **NumPy** | Computação numérica |

### Machine Learning
| Tecnologia | Descrição |
|------------|-----------|
| **Scikit-learn** | Algoritmos de ML |
| **RandomForestClassifier** | Classificação de risco |
| **IsolationForest** | Detecção de anomalias |
| **StandardScaler** | Normalização de features |

### Segurança
| Tecnologia | Descrição |
|------------|-----------|
| **SSL/TLS** | Conexões seguras |
| **LDAP** | Autenticação corporativa |
| **Streamlit Secrets** | Gerenciamento de credenciais |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                        DIMP - Arquitetura                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Frontend   │    │   Backend    │    │  Big Data    │      │
│  │  Streamlit   │───▶│   Python     │───▶│   Impala     │      │
│  │   Plotly     │    │  SQLAlchemy  │    │   Cluster    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Cache     │    │     ML       │    │   Tabelas    │      │
│  │  Streamlit   │    │  Scikit-     │    │    DIMP      │      │
│  │   Session    │    │   learn      │    │   (13+)      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Estrutura de Arquivos

```
DIMP_NEW/
├── DIMP (3).py          # Aplicação principal (~3.400 linhas)
├── DIMP.json            # Configurações e queries exportadas
├── README.md            # Este arquivo
└── .git/                # Controle de versão
```

---

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- Acesso à rede SEFAZ/SC
- Credenciais LDAP válidas

### Passo a Passo

1. **Clone o repositório**
```bash
git clone https://github.com/tiagossevero/DIMP_NEW.git
cd DIMP_NEW
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**
```bash
pip install streamlit pandas numpy plotly sqlalchemy scikit-learn impyla
```

### Dependências Completas

```txt
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
plotly>=5.0.0
sqlalchemy>=1.4.0
scikit-learn>=1.0.0
impyla>=0.18.0
```

---

## Configuração

### 1. Configuração de Secrets do Streamlit

Crie o arquivo `.streamlit/secrets.toml`:

```toml
[database]
host = "bdaworkernode02.sef.sc.gov.br"
port = 21050
database = "teste"
username = "seu_usuario_ldap"
password = "sua_senha_ldap"

[app]
password = "senha_de_acesso_ao_sistema"
```

### 2. Configuração de SSL (Opcional)

Para ambientes com certificados auto-assinados, o sistema automaticamente configura o contexto SSL.

### 3. Variáveis de Ambiente

```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

---

## Uso

### Executando o Sistema

```bash
streamlit run "DIMP (3).py"
```

O sistema estará disponível em: `http://localhost:8501`

### Autenticação

1. Acesse a URL do sistema
2. Digite a senha configurada
3. Navegue pelo menu lateral

### Navegação

O menu lateral oferece acesso às seguintes páginas:

| Página | Descrição |
|--------|-----------|
| Dashboard Executivo | Visão geral e KPIs |
| Ranking de Empresas | Lista ordenada por risco |
| Drill-Down por Empresa | Análise detalhada |
| Machine Learning | Modelos preditivos |
| Análise Setorial | Visão por CNAE |
| Análise de Funcionários | Cruzamento RAIS/CAGED |
| Sócios Múltiplas Empresas | Análise de dispersão |
| Análise Temporal | Tendências e evolução |
| Padrões Suspeitos | Detecção automática |
| Diagnóstico | Saúde do sistema |
| Sobre o Sistema | Documentação |

---

## Metodologia de Scoring

### Modelo de 5 Fatores

O score de risco é calculado com base em 5 fatores ponderados:

| Fator | Peso | Descrição |
|-------|------|-----------|
| **Proporção CPF** | 30% | Percentual de recebimentos via CPF |
| **Volume CPF** | 25% | Valor absoluto recebido via CPF |
| **Qtd. Sócios** | 15% | Número de CPFs recebendo |
| **Desvio do Regime** | 20% | Comparação com média do regime |
| **Consistência Temporal** | 10% | Regularidade dos pagamentos |

### Classificação de Risco

| Classificação | Score | Prioridade |
|---------------|-------|------------|
| **ALTO** | ≥ 80 | Máxima - Investigação prioritária |
| **MÉDIO-ALTO** | 60-79 | Alta - Monitoramento intensivo |
| **MÉDIO** | 40-59 | Moderada - Acompanhamento |
| **BAIXO** | < 40 | Normal - Padrão esperado |

### Fórmula de Cálculo

```
Score = (Proporção_CPF × 0.30) +
        (Volume_CPF_Norm × 0.25) +
        (Qtd_Socios_Norm × 0.15) +
        (Desvio_Regime × 0.20) +
        (Consistencia × 0.10)
```

---

## Estrutura do Banco de Dados

### Tabelas Principais

| Tabela | Descrição |
|--------|-----------|
| `dimp_cnpj_base` | Base de empresas ativas |
| `dimp_socios` | Sócios e administradores |
| `dimp_pagamentos_cnpj` | Recebimentos via CNPJ |
| `dimp_pagamentos_cpf` | Recebimentos via CPF de sócios |
| `dimp_score_final` | Scores finais calculados |
| `dimp_socios_multiplas_empresas` | Sócios em múltiplas empresas |
| `dimp_operacoes_suspeitas` | Transações suspeitas |
| `dimp_comparacao_cnpj_cpf` | Análise comparativa |
| `dimp_metricas_risco` | Métricas de risco calculadas |
| `dimp_funcionarios_agregado` | Dados agregados de funcionários |
| `dimp_func_score_final` | Scores de funcionários |
| `dimp_func_rede_multiplas` | Funcionários em múltiplas empresas |
| `dimp_func_top_suspeitos` | View de casos mais suspeitos |

### Diagrama ER Simplificado

```
┌─────────────────┐     ┌─────────────────┐
│  dimp_cnpj_base │────▶│   dimp_socios   │
└─────────────────┘     └─────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐     ┌─────────────────┐
│ dimp_pagamentos │     │ dimp_pagamentos │
│     _cnpj       │     │     _cpf        │
└─────────────────┘     └─────────────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
            ┌─────────────────┐
            │ dimp_score_final│
            └─────────────────┘
```

---

## Cache e Performance

### Estratégia de Cache

O sistema utiliza múltiplos níveis de cache para otimizar a performance:

| Nível | TTL | Uso |
|-------|-----|-----|
| `@st.cache_resource` | Persistente | Engine de banco de dados |
| `@st.cache_data(ttl=7200)` | 2 horas | Queries gerais |
| `@st.cache_data(ttl=3600)` | 1 hora | Dados sumarizados |
| `@st.cache_data(ttl=1800)` | 30 minutos | Dados de funcionários |
| `@st.cache_data(ttl=600)` | 10 minutos | Queries de detalhe |

### Métricas de Performance

| Operação | Tempo Esperado |
|----------|----------------|
| Carregamento inicial | 2-5 segundos |
| Drill-down de empresa | 1-3 segundos |
| Queries de ranking | 1-2 segundos |
| Treinamento ML | 5-15 segundos |

### Otimizações Implementadas

- Consultas agregadas no banco de dados
- Pré-cálculo de scores em tabelas
- Lazy loading de dados detalhados
- Cache de sessão para filtros
- Queries parametrizadas

---

## Segurança

### Medidas Implementadas

1. **Autenticação**
   - Senha de acesso ao sistema
   - Integração LDAP com banco de dados
   - Session state para controle de sessão

2. **Comunicação**
   - SSL/TLS em todas as conexões
   - Certificados corporativos
   - Rede interna SEFAZ

3. **Dados**
   - Acesso restrito por credenciais
   - Logs de auditoria
   - Timeout de sessão

### Boas Práticas

- Nunca compartilhe credenciais
- Use senhas fortes
- Mantenha o sistema atualizado
- Reporte vulnerabilidades

---

## Contribuição

### Como Contribuir

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona NovaFeature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

### Padrões de Código

- Siga PEP 8 para Python
- Docstrings em todas as funções
- Comentários em português
- Testes para novas funcionalidades

---

## Autor

**Tiago Severo**
- Cargo: Auditor Fiscal da Receita Estadual (AFRE)
- Organização: Receita Estadual de Santa Catarina (SEFAZ/SC)
- Versão: 1.0

---

## Licença

Este projeto é de uso exclusivo da **Secretaria de Estado da Fazenda de Santa Catarina (SEFAZ/SC)**.

Todos os direitos reservados. O código fonte, documentação e dados são confidenciais e destinados apenas ao uso interno da administração tributária estadual.

---

## Suporte

Para suporte técnico ou dúvidas sobre o sistema:

1. Consulte a página "Sobre o Sistema" no aplicativo
2. Verifique a documentação no sidebar
3. Entre em contato com a equipe de desenvolvimento

---

<div align="center">

**DIMP - Sistema de Análise de Meios de Pagamento**

*Receita Estadual de Santa Catarina*

*Desenvolvido com Python e Streamlit*

</div>
