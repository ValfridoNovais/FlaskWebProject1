# -*- coding: utf-8 -*-

from flask import Flask, render_template, Response
import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import openpyxl

app = Flask(__name__)

# Caminhos de arquivos e planilhas
arquivo_xls = 'dados_acesso.xlsx'
planilha_permitidos = 'PERMITIDOS'
planilha_registro = 'REGISTRO ENTRADA'

def carregar_dados_autorizados():
    df = pd.read_excel(arquivo_xls, sheet_name=planilha_permitidos)
    rostos_conhecidos = []
    dados_conhecidos = []

    for index, row in df.iterrows():
        imagem = face_recognition.load_image_file(row['Caminho da Foto'])
        encoding = face_recognition.face_encodings(imagem)[0]
        rostos_conhecidos.append(encoding)
        dados_conhecidos.append((row['Nome'], row['NR PM']))

    return rostos_conhecidos, dados_conhecidos

def registrar_entrada(nome, nr_pm):
    df = pd.read_excel(arquivo_xls, sheet_name=planilha_registro)
    data_hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    novo_registro = {'Nome': nome, 'NR PM': nr_pm, 'Data': data_hora.split()[0], 'Hora': data_hora.split()[1]}
    df = df.append(novo_registro, ignore_index=True)

    with pd.ExcelWriter(arquivo_xls, mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=planilha_registro, index=False)

def gen_frames():
    rostos_conhecidos, dados_conhecidos = carregar_dados_autorizados()
    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            rgb_frame = frame[:, :, ::-1]
            localizacoes_rostos = face_recognition.face_locations(rgb_frame)
            codificacoes_rostos = face_recognition.face_encodings(rgb_frame, localizacoes_rostos)

            for (top, right, bottom, left), face_encoding in zip(localizacoes_rostos, codificacoes_rostos):
                matches = face_recognition.compare_faces(rostos_conhecidos, face_encoding)
                nome, nr_pm = "Desconhecido", None

                if True in matches:
                    match_index = matches.index(True)
                    nome, nr_pm = dados_conhecidos[match_index]
                    registrar_entrada(nome, nr_pm)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, nome, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Pagina inicial."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream de video."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
