# Sentinela Rondon: Monitoramento Inteligente e Telemetria Urbana

Um sistema de Visão Computacional e Engenharia de Dados desenvolvido como uma Prova de Conceito (PoC) para prevenção de desastres urbanos. O Sentinela analisa o fluxo de veículos em tempo real (via imagens aéreas *top-down*) e cruza essas informações com dados meteorológicos para emitir alertas autônomos de risco de enchentes.

## 🛠️ Linguagens e Tecnologias Utilizadas

* **Visão Computacional & IA:** YOLOv8 (Ultralytics), OpenCV.
* **Rastreamento:** Algoritmo ByteTrack (atribuição de ID único por veículo).
* **Engenharia de Dados:** SQLite (persistência de logs analíticos para BI).
* **Treinamento e Curadoria:** Roboflow, Transfer Learning (VisDrone), Data Augmentation.
* **Linguagem & Integração:** Python, consumo de API REST (Open-Meteo).

---

## 🚀 O Problema e a Solução

Cidades com vias de escoamento rápido construídas sobre rios canalizados (como a Avenida Rondon Pacheco em Uberlândia) sofrem com enchentes repentinas. O maior risco não é apenas a água, mas o **engarrafamento** que prende veículos na via durante a tempestade.

O **Sentinela** resolve essa lacuna de monitoramento atuando como um orquestrador de dados:
1. Analisa o feed de vídeo aéreo e rastreia os veículos.
2. Consome a API do Open-Meteo para verificar o volume de chuva (mm) em tempo real.
3. Cruza as variáveis (Ex: *Chuva > 10mm* **E** *Alta densidade de veículos*) para alterar o status do painel de controle (Verde, Amarelo ou Alerta Vermelho Crítico).

## 🧠 Desafios de Engenharia e Soluções (Highlights do Projeto)

Este projeto foi construído superando desafios que não costumam aparecer em tutoriais básicos de IA:

### 1. Abordagem Data-Centric contra o Overfitting
Modelos padrões treinados para ver carros "de frente" falham em visão aérea (*top-down*). A IA base confundia pedestres nas calçadas com bicicletas na ciclovia.
* **A Solução:** Em vez de tentar consertar com código, o foco foi para a qualidade dos dados. Foi desenvolvido um script de extração estratégica de frames (pulando intervalos de segundos para garantir diversidade de cenário), seguido de anotação manual rigorosa e *Data Augmentation* (ruído, variação de brilho e inversão horizontal). O resultado do *Fine-Tuning* elevou a precisão do modelo para a faixa dos 90%.

### 2. Filtros Morfológicos (Regras Espaciais)
Para blindar a IA contra falsos positivos remanescentes, foi implementada uma camada de cálculo geométrico no Python. O sistema avalia o *aspect ratio* (proporção largura/altura) e a área em pixels das caixas delimitadoras (*bounding boxes*). Se a IA classificar algo como "veículo pesado", mas as dimensões não baterem com a regra matemática de um caminhão visto de cima, a detecção é descartada.

### 3. De "Vídeo" para "Produto de Dados" (Camada Analítica)
Visão computacional sem armazenamento é apenas um vídeo passando. O motor do Sentinela foi arquitetado para se conectar nativamente a um banco de dados **SQLite**.
* Para cada veículo detectado, o sistema usa o ID único do ByteTrack para gravar no banco de dados (apenas uma vez) a exata data/hora, classe do veículo, nível de confiança da rede neural e as condições climáticas do momento. O sistema está pronto para ser plugado em ferramentas de BI (como Power BI ou Metabase) para geração de *dashboards* de telemetria urbana.

## 📁 Estrutura do Repositório

* `sentinela.py`: O motor principal do painel, inferência e gravação no banco de dados.
* `extrair_frames.py`: Script de engenharia de dados para curadoria do dataset.
* `treinamento_modelo.ipynb`: Pipeline no Google Colab documentando o *Transfer Learning* e geração dos pesos da rede.
* `best.pt`: Os pesos do modelo YOLOv8 treinado.
(O sistema irá gerar automaticamente o banco de dados sentinela_dados.db após a primeira execução).

---

Projeto desenvolvido como estudo prático de integração entre Inteligência Artificial, visão top-down e arquitetura de dados.
