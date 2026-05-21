# Sentinela Rondon: Monitoramento Inteligente e Telemetria Urbana

Prova de Conceito (PoC) de um sistema de Visão Computacional e Engenharia de Dados para prevenção de desastres urbanos. O Sentinela analisa o fluxo de veículos em tempo real a partir de imagens aéreas top-down, cruza os dados com telemetria meteorológica e emite alertas autônomos de risco de enchentes.

---

## Sumário

1. [O Problema e a Solução](#o-problema-e-a-solução)
2. [Como Funciona](#como-funciona)
3. [Desafios de Engenharia](#desafios-de-engenharia)
4. [Tecnologias](#tecnologias)
5. [Estrutura do Repositório](#estrutura-do-repositório)
6. [Como Executar](#como-executar)

---

## O Problema e a Solução

Cidades com vias construídas sobre rios canalizados — como a Avenida Rondon Pacheco em Uberlândia — são vulneráveis a enchentes repentinas. O risco crítico não é apenas o volume de água, mas o congestionamento que prende veículos na via durante a tempestade, antes que qualquer alerta seja emitido.

O Sentinela preenche essa lacuna de monitoramento atuando como um orquestrador de dados em três etapas:

1. Analisa o feed de vídeo aéreo e rastreia cada veículo individualmente via ByteTrack.
2. Consome a API do Open-Meteo para verificar o volume de chuva (mm) em tempo real.
3. Cruza as variáveis e altera o status do painel de controle conforme o nível de risco detectado.

---

## Como Funciona

```
Feed de vídeo aéreo (top-down)
          │
          ▼
[ YOLOv8 — Detecção de veículos ]
          │  bounding boxes + classe + confiança
          ▼
[ Filtros Morfológicos — aspect ratio + área em pixels ]
          │  descarta falsos positivos por regra geométrica
          ▼
[ ByteTrack — Rastreamento ]
          │  ID único por veículo entre frames
          ├─────────────────────────────────────┐
          ▼                                     ▼
[ API Open-Meteo ]                    [ Persistência SQLite ]
  volume de chuva (mm)                  data/hora · classe
  em tempo real                         confiança · clima
          │                                     │
          └──────────────┬──────────────────────┘
                         ▼
              [ Motor de Alertas ]
              Cruzamento de variáveis:
              Chuva > 10mm + Alta densidade
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Verde          Amarelo      Alerta Vermelho
      (normal)       (atenção)       (crítico)
                         │
                         ▼
              [ Painel de Controle ]
              Pronto para Power BI / Metabase
```

---

## Desafios de Engenharia

### Abordagem data-centric contra overfitting

Modelos padrão treinados para detectar veículos em perspectiva frontal falham em visão aérea. A IA base confundia pedestres nas calçadas com bicicletas na ciclovia.

A solução não foi ajuste de código, mas qualidade de dados: foi desenvolvido um script de extração estratégica de frames com intervalos calibrados para garantir diversidade de cenário, seguido de anotação manual rigorosa e Data Augmentation com ruído, variação de brilho e inversão horizontal. O Fine-Tuning resultante elevou a precisão do modelo para a faixa dos 90%.

### Filtros morfológicos como camada de validação geométrica

Para blindar o sistema contra falsos positivos remanescentes, foi implementada uma camada de cálculo geométrico em Python que avalia o aspect ratio (proporção largura/altura) e a área em pixels de cada bounding box. Se a IA classificar um objeto como veículo pesado, mas as dimensões não corresponderem às regras matemáticas de um caminhão visto de cima, a detecção é descartada antes de chegar ao banco.

### De vídeo para produto de dados

Visão computacional sem persistência é apenas um vídeo passando. O motor do Sentinela foi arquitetado para gravar no SQLite, usando o ID único do ByteTrack como chave de deduplicação — cada veículo é registrado exatamente uma vez, com data/hora, classe, nível de confiança e condições climáticas do momento. A camada analítica está pronta para conexão direta a ferramentas de BI.

---

## Tecnologias

| Categoria | Tecnologia |
|-----------|-----------|
| Detecção e visão computacional | YOLOv8 (Ultralytics), OpenCV |
| Rastreamento | ByteTrack |
| Treinamento e curadoria | Roboflow, Transfer Learning (VisDrone), Data Augmentation |
| Dados meteorológicos | API REST Open-Meteo |
| Persistência analítica | SQLite |
| Linguagem | Python |

---

## Estrutura do Repositório

```
/
├── sentinela.py               # Motor principal: inferência, alertas e gravação no banco
├── extrair_frames.py          # Script de curadoria estratégica do dataset
├── treinamento_modelo.ipynb   # Pipeline de Transfer Learning documentado (Google Colab)
├── best.pt                    # Pesos do modelo YOLOv8 treinado
└── sentinela_dados.db         # Banco de dados analítico (gerado na primeira execução)
```

> `sentinela_dados.db` não é versionado. O arquivo é criado automaticamente ao executar `sentinela.py` pela primeira vez.

---

## Como Executar

**Pré-requisitos:** Python 3.9+, `best.pt` disponível na raiz do projeto.

```bash
# 1. Clone o repositório
git clone https://github.com/<usuario>/sentinela-rondon.git
cd sentinela-rondon

# 2. Instale as dependências
pip install ultralytics opencv-python requests

# 3. Execute o painel principal
python sentinela.py
```

O banco de dados `sentinela_dados.db` será criado automaticamente na raiz do projeto. Para re-treinar ou inspecionar o pipeline de treinamento, abra `treinamento_modelo.ipynb` no Google Colab.
