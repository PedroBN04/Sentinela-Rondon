import cv2
import requests
import os
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
import sqlite3

# Garante que o diretório de execução seja o mesmo do script
diretorio_script = os.path.dirname(os.path.abspath(__file__))
os.chdir(diretorio_script)

CONFIG = {
    "fonte_video": "transito.mp4",
    "modelo_pt": "best.pt",    
    "confianca": 0.40,         
    "pular_frames": 2,         
    "largura": 1280,
    "altura":  720,
    "limiar_chuva_mm": 10.0,
    "limiar_veiculos": 15,
    "lat": -18.9186,
    "lon": -48.2772,
    "cidade": "Uberlandia",
}

PERFIL_CLASSES = {
    "leve":      {"cor": (50,  200, 255), "area_min": 800,  "area_max": 18000, "ratio_min": 1.0, "ratio_max": 3.5},
    "pesado":    {"cor": (0,   140, 255), "area_min": 2000, "area_max": 60000, "ratio_min": 1.5, "ratio_max": 6.0},
    "moto":      {"cor": (255, 100, 220), "area_min": 100,  "area_max": 2500,  "ratio_min": 1.2, "ratio_max": 4.0},
    "bicicleta": {"cor": (0,   255, 200), "area_min": 150,  "area_max": 1500,  "ratio_min": 1.8, "ratio_max": 5.0},
}

def filtrar_deteccao(box, nome_classe):
    """Aplica filtros morfológicos (área e proporção) para validar a detecção."""
    perfil = PERFIL_CLASSES.get(nome_classe)
    if not perfil: return False 

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    w, h = x2 - x1, y2 - y1
    
    if min(w, h) == 0: return False
    area = w * h
    aspect_ratio = max(w, h) / min(w, h)

    if area < perfil["area_min"] or area > perfil["area_max"]:
        return False
    if aspect_ratio < perfil["ratio_min"] or aspect_ratio > perfil["ratio_max"]:
        return False
    return True

def obter_clima(lat, lon, cidade):
    """Consulta a API Open-Meteo para obter dados climáticos em tempo real."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,precipitation,weather_code"
    try:
        r = requests.get(url, timeout=5).json()
        d = r["current"]
        c = d["weather_code"]
        if   c == 0: status = "Ceu Limpo"
        elif c < 50: status = "Nublado"
        elif c < 80: status = "Chuva"
        else:        status = "Tempestade"
        return {"chuva_mm": d["precipitation"], "temp_c": d["temperature_2m"], "status": status}
    except:
        return {"chuva_mm": 0.0, "temp_c": 22.0, "status": "Sem Dados"}

def iniciar_banco_dados():
    """Inicializa a conexão com o SQLite e cria a tabela de registros se necessário."""
    conexao = sqlite3.connect('sentinela_dados.db')
    cursor = conexao.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS registro_trafego (
            id_evento INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            modo_execucao TEXT,
            id_rastreamento INTEGER,
            classe TEXT,
            confianca_pct REAL,
            clima_status TEXT,
            chuva_mm REAL
        )
    ''')
    conexao.commit()
    return conexao, cursor

# Configurações de interface gráfica
TEMA = {
    "fundo":         (12,  14,  18),
    "painel":        (18,  22,  30),
    "borda":         (40,  50,  65),
    "texto_titulo":  (220, 230, 245),
    "texto_detalhe": (140, 155, 175),
    "verde":         (60,  220, 130),
    "amarelo":       (0,   200, 255), 
    "vermelho":      (60,   60, 235),
    "acento":        (180, 100,  50),
}

def alpha_blend(img, x1, y1, x2, y2, cor, alpha=0.72):
    if y1 < 0 or x1 < 0 or y2 > img.shape[0] or x2 > img.shape[1]: return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, cor)
    blended = cv2.addWeighted(overlay, 1 - alpha, roi, alpha, 0)
    img[y1:y2, x1:x2] = blended

def texto(img, txt, x, y, escala=0.5, cor=None, espessura=1):
    cor = cor or TEMA["texto_titulo"]
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_DUPLEX, escala, cor, espessura, cv2.LINE_AA)

def desenhar_ui_geral(img, status_txt, cor_status, qtd_total, media_conf, fps, clima, risco_chuva, modo):
    W, H = CONFIG["largura"], CONFIG["altura"]
    
    alpha_blend(img, 0, 0, W, 52, TEMA["painel"], alpha=0.3)
    cv2.line(img, (0, 52), (W, 52), TEMA["borda"], 1)
    cv2.rectangle(img, (0, 0), (5, 52), cor_status, -1)
    
    texto(img, "SENTINELA", 18, 20, 0.45, TEMA["acento"], 1)
    texto(img, "RONDON",    18, 40, 0.55, TEMA["texto_titulo"], 2)
    cv2.line(img, (105, 8), (105, 44), TEMA["borda"], 1)

    (tw, _), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
    texto(img, status_txt, (W - tw) // 2, 34, 0.65, cor_status, 2)
    texto(img, f"FPS: {fps:.0f}", W - 90, 32, 0.42, TEMA["texto_detalhe"])

    x0_clima = W - 230 - 14
    alpha_blend(img, x0_clima, 62, x0_clima + 230, 167, TEMA["painel"], alpha=0.25)
    cv2.rectangle(img, (x0_clima, 62), (x0_clima + 230, 167), TEMA["vermelho"] if risco_chuva else TEMA["borda"], 1)
    alpha_blend(img, x0_clima, 62, x0_clima + 230, 84, TEMA["borda"], alpha=0.55)
    texto(img, "CLIMA  AO  VIVO", x0_clima + 10, 77, 0.38, TEMA["texto_detalhe"])
    
    yi = 100
    for label, val, cor_val in [("Condicao", clima["status"], TEMA["texto_titulo"]), 
                                ("Temp", f"{clima['temp_c']} C", TEMA["texto_titulo"]), 
                                ("Chuva 1h", f"{clima['chuva_mm']} mm", TEMA["vermelho"] if risco_chuva else TEMA["verde"])]:
        texto(img, label, x0_clima + 10, yi, 0.38, TEMA["texto_detalhe"])
        texto(img, val,   x0_clima + 105, yi, 0.40, cor_val, 1)
        yi += 22

    ch = 80
    y0_leg = H - ch - 30 
    alpha_blend(img, 14, y0_leg, 260, y0_leg + ch, TEMA["painel"], alpha=0.25)
    cv2.rectangle(img, (14, y0_leg), (260, y0_leg + ch), TEMA["borda"], 1)
    alpha_blend(img, 14, y0_leg, 260, y0_leg + 20, TEMA["borda"], alpha=0.55)
    texto(img, "METRICAS DE RASTREAMENTO", 24, y0_leg + 14, 0.38, TEMA["texto_detalhe"])
    
    texto(img, "Veiculos Roteados:", 24, y0_leg + 45, 0.45, TEMA["texto_detalhe"])
    texto(img, str(qtd_total), 180, y0_leg + 45, 0.55, cor_status, 1)
    
    texto(img, "Confiabilidade:", 24, y0_leg + 68, 0.45, TEMA["texto_detalhe"])
    cor_conf = TEMA["verde"] if media_conf > 50.0 else TEMA["amarelo"]
    texto(img, f"{media_conf:.1f}%", 180, y0_leg + 68, 0.50, cor_conf, 1)

    alpha_blend(img, 0, H - 22, W, H, TEMA["painel"], alpha=0.35)
    texto(img, f"MODO: {modo}", 10, H - 6, 0.35, TEMA["amarelo"] if modo == "SIMULACAO" else TEMA["verde"])
    texto(img, datetime.now().strftime("%d/%m/%Y  %H:%M:%S"), W - 150, H - 6, 0.35, TEMA["texto_detalhe"])

def desenhar_deteccoes(img, caixas):
    for (x1, y1, x2, y2, nome_classe, cor, conf, track_id) in caixas:
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 1, cv2.LINE_AA)
        tam = min(8, (x2 - x1) // 3, (y2 - y1) // 3)
        for (px, py), (dx, dy) in zip([(x1, y1), (x2, y1), (x1, y2), (x2, y2)], [(1, 1), (-1, 1), (1, -1), (-1, -1)]):
            cv2.line(img, (px, py), (px + dx * tam, py), cor, 2, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * tam), cor, 2, cv2.LINE_AA)
        
        texto_label = f"#{track_id} {nome_classe.capitalize()}" if track_id else nome_classe.capitalize()
        (tw, th), _ = cv2.getTextSize(texto_label, cv2.FONT_HERSHEY_DUPLEX, 0.38, 1)
        by1 = max(0, y1 - th - 6)
        alpha_blend(img, x1, by1, x1 + tw + 6, by1 + th + 6, TEMA["fundo"], alpha=0.2)
        texto(img, texto_label, x1 + 3, by1 + th + 3, 0.38, cor)

def executar_monitoramento(modo_simulacao=False):
    """Executa o loop principal de captura de vídeo, rastreamento e registro de dados."""
    modo = "SIMULACAO" if modo_simulacao else "AO VIVO"
    clima = {"chuva_mm": 20.0, "temp_c": 18.0, "status": "Tempestade"} if modo_simulacao else obter_clima(CONFIG["lat"], CONFIG["lon"], CONFIG["cidade"])
    
    conexao_sql, cursor_sql = iniciar_banco_dados()
    ids_registrados = set() # Evita duplicidade de registros
    
    modelo = YOLO(CONFIG["modelo_pt"])
    video = cv2.VideoCapture(CONFIG["fonte_video"])
    
    nome_arquivo = "Sentinela_Simulacao.mp4" if modo_simulacao else "Sentinela_AoVivo.mp4"
    fps_video = video.get(cv2.CAP_PROP_FPS)
    if fps_video == 0 or np.isnan(fps_video): fps_video = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    gravador = cv2.VideoWriter(nome_arquivo, fourcc, fps_video, (CONFIG["largura"], CONFIG["altura"]))
    
    frame_count = 0
    ultimo_caixas = []
    qtd_veiculos_total = 0
    media_confianca = 0.0
    t_prev = time.time()
    fps_display = 30.0

    while video.isOpened():
        sucesso, frame = video.read()
        if not sucesso: break
        
        frame = cv2.resize(frame, (CONFIG["largura"], CONFIG["altura"]))

        if frame_count % CONFIG["pular_frames"] == 0:
            resultados = modelo.track(frame, conf=CONFIG["confianca"], persist=True, tracker="bytetrack.yaml", verbose=False)
            
            novas_caixas = []
            soma_confianca = 0.0

            if resultados[0].boxes.id is not None:
                boxes = resultados[0].boxes.xyxy.cpu().numpy()
                clss = resultados[0].boxes.cls.cpu().numpy()
                confs = resultados[0].boxes.conf.cpu().numpy()
                track_ids = resultados[0].boxes.id.int().cpu().tolist()

                for box, cls, conf, track_id in zip(boxes, clss, confs, track_ids):
                    cls_id = int(cls)
                    nome_classe = modelo.names[cls_id].lower() 

                    if not filtrar_deteccao(resultados[0].boxes[track_ids.index(track_id)], nome_classe): 
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    perfil = PERFIL_CLASSES[nome_classe]
                    
                    novas_caixas.append((x1, y1, x2, y2, nome_classe, perfil["cor"], conf, track_id))
                    soma_confianca += conf

                    # Registra a telemetria do veículo no banco de dados
                    if track_id not in ids_registrados:
                        ids_registrados.add(track_id)
                        hora_exata = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        confianca_formatada = round(float(conf) * 100, 1)
                        
                        cursor_sql.execute('''
                            INSERT INTO registro_trafego 
                            (data_hora, modo_execucao, id_rastreamento, classe, confianca_pct, clima_status, chuva_mm)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (hora_exata, modo, track_id, nome_classe.capitalize(), confianca_formatada, clima["status"], clima["chuva_mm"]))
                        
                        conexao_sql.commit()

            ultimo_caixas = novas_caixas
            qtd_veiculos_total = len(novas_caixas) 
            media_confianca = (soma_confianca / qtd_veiculos_total * 100) if qtd_veiculos_total > 0 else 0.0

            risco_chuva = clima["chuva_mm"] >= CONFIG["limiar_chuva_mm"]
            risco_transito = qtd_veiculos_total >= CONFIG["limiar_veiculos"]

            if risco_chuva and risco_transito:
                status_txt, cor_status = "ALERTA CRITICO - RISCO DE ENCHENTE", TEMA["vermelho"]
            elif risco_chuva:
                status_txt, cor_status = "ATENCAO - PISTA MOLHADA (FLUXO LIVRE)", TEMA["amarelo"]
            else:
                status_txt, cor_status = "MONITORAMENTO - VIA SEGURA", TEMA["verde"]

        desenhar_deteccoes(frame, ultimo_caixas)
        desenhar_ui_geral(frame, status_txt, cor_status, qtd_veiculos_total, media_confianca, fps_display, clima, risco_chuva, modo)

        t_now = time.time()
        fps_display = 0.85 * fps_display + 0.15 * (1.0 / max(t_now - t_prev, 1e-5))
        t_prev = t_now

        gravador.write(frame)
        cv2.imshow("Sentinela Rondon - Dashboard Defesa Civil", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    gravador.release()
    video.release()
    cv2.destroyAllWindows()
    conexao_sql.close() 

if __name__ == "__main__":
    executar_monitoramento(modo_simulacao=False)
    executar_monitoramento(modo_simulacao=True)
    print("\nExecução finalizada. Banco de dados SQL 'sentinela_dados.db' atualizado.")