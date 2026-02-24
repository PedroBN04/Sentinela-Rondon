import cv2
import os

video_nome = 'transito.mp4'  
output_folder = r'C:\Sentinela_Dataset' 
intervalo_segundos = 2 

if not os.path.exists(output_folder):
    try:
        os.makedirs(output_folder)
        print(f"Pasta criada em: {output_folder}")
    except Exception as e:
        print(f"Não foi possível criar a pasta no C:. Erro: {e}")
        output_folder = 'dataset_sentinela'
        if not os.path.exists(output_folder): os.makedirs(output_folder)

cap = cv2.VideoCapture(video_nome)

if not cap.isOpened():
    print(f"Erro: Não encontrei o vídeo '{video_nome}'.")
    print("Verifique se o vídeo está na mesma pasta deste script.")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    intervalo_frames = int(fps * intervalo_segundos)
    count = 0
    saved_count = 0

    print(f"Extraindo frames...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        if count % intervalo_frames == 0:
            filename = f"rondon_t{saved_count:03d}.jpg"
            path_final = os.path.join(output_folder, filename)
            sucesso = cv2.imwrite(path_final, frame)
            
            if sucesso:
                print(f"[{saved_count}] SALVO COM SUCESSO em: {path_final}")
                saved_count += 1
            else:
                print(f"FALHA ao gravar o arquivo: {filename}")

        count += 1

    cap.release()
    print(f"\n FIM! Total de {saved_count} imagens.")
    print(f"Vá em: Este Computador > Disco Local (C:) > Sentinela_Dataset")