import random
import subprocess
import os
import gradio as gr
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))

def convert(segment_length, video, audio, progress=gr.Progress()):
    if segment_length is None:
        segment_length = 0
    print(video, audio)

    if segment_length != 0:
        video_segments = cut_video_segments(video, segment_length)
        audio_segments = cut_audio_segments(audio, segment_length)
    else:
        # Verifique e crie o diretório temp/video
        video_dir = 'temp/video'
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, os.path.basename(video))

        # Verifique se o arquivo de vídeo existe antes de movê-lo
        if not os.path.isfile(video):
            raise FileNotFoundError(f"O arquivo de vídeo '{video}' não foi encontrado.")
        
        shutil.move(video, video_path)
        video_segments = [video_path]

        # Verifique e crie o diretório temp/audio
        audio_dir = 'temp/audio'
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, os.path.basename(audio))

        # Verifique se o arquivo de áudio existe antes de movê-lo
        if not os.path.isfile(audio):
            raise FileNotFoundError(f"O arquivo de áudio '{audio}' não foi encontrado.")
        
        shutil.move(audio, audio_path)
        audio_segments = [audio_path]

    processed_segments = []
    for i, (video_seg, audio_seg) in progress.tqdm(enumerate(zip(video_segments, audio_segments))):
        processed_output = process_segment(video_seg, audio_seg, i)
        processed_segments.append(processed_output)

    output_file = f"results/output_{random.randint(0,1000)}.mp4"
    concatenate_videos(processed_segments, output_file)

    # Remove arquivos temporários
    cleanup_temp_files(video_segments + audio_segments)

    # Retorna o arquivo de vídeo concatenado
    return output_file

def cleanup_temp_files(file_list):
    for file_path in file_list:
        if os.path.isfile(file_path):
            os.remove(file_path)

def cut_video_segments(video_file, segment_length):
    temp_directory = 'temp/audio'
    shutil.rmtree(temp_directory, ignore_errors=True)
    os.makedirs(temp_directory, exist_ok=True)
    segment_template = f"{temp_directory}/{random.randint(0,1000)}_%03d.mp4"
    command = ["ffmpeg", "-i", video_file, "-c", "copy", "-f", "segment", "-segment_time", str(segment_length), segment_template]
    subprocess.run(command, check=True)

    video_segments = [segment_template % i for i in range(len(os.listdir(temp_directory)))]
    return video_segments

def cut_audio_segments(audio_file, segment_length):
    temp_directory = 'temp/video'
    shutil.rmtree(temp_directory, ignore_errors=True)
    os.makedirs(temp_directory, exist_ok=True)
    segment_template = f"{temp_directory}/{random.randint(0,1000)}_%03d.mp3"
    command = ["ffmpeg", "-i", audio_file, "-f", "segment", "-segment_time", str(segment_length), segment_template]
    subprocess.run(command, check=True)

    audio_segments = [segment_template % i for i in range(len(os.listdir(temp_directory)))]
    return audio_segments

def process_segment(video_seg, audio_seg, i):
    output_file = f"results/{random.randint(10,100000)}_{i}.mp4"
    command = ["python", "inference.py", "--face", video_seg, "--audio", audio_seg, "--outfile", output_file]
    subprocess.run(command, check=True)

    return output_file

def concatenate_videos(video_segments, output_file):
    with open("segments.txt", "w") as file:
        for segment in video_segments:
            file.write(f"file '{segment}'\n")
    command = ["ffmpeg", "-f", "concat", "-i", "segments.txt", "-c", "copy", output_file]
    subprocess.run(command, check=True)

with gr.Blocks(
    title="Audio-based Lip Synchronization",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as demo:
    with gr.Row():
        gr.Markdown("# Audio-based Lip Synchronization")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                seg = gr.Number(label="Segment length (Segundo), 0 para não segmentar")
            with gr.Row():
                with gr.Column():
                    v = gr.Video(label='Vídeo de Origem')

                with gr.Column():
                    a = gr.Audio(type='filepath', label='Áudio Alvo')

            with gr.Row():
                btn = gr.Button(value="Sincronizar", variant="primary")
            with gr.Row():
                gr.Examples(
                    label="Exemplos de Vídeo",
                    examples=[
                        os.path.join(os.path.dirname(__file__), "examples/face/1.mp4"),
                        os.path.join(os.path.dirname(__file__), "examples/face/2.mp4"),
                        os.path.join(os.path.dirname(__file__), "examples/face/3.mp4"),
                        os.path.join(os.path.dirname(__file__), "examples/face/4.mp4"),
                        os.path.join(os.path.dirname(__file__), "examples/face/5.mp4"),
                    ],
                    inputs=[v],
                    fn=convert,
                )
            with gr.Row():
                gr.Examples(
                    label="Exemplos de Áudio",
                    examples=[
                        os.path.join(os.path.dirname(__file__), "examples/audio/1.wav"),
                        os.path.join(os.path.dirname(__file__), "examples/audio/2.wav"),
                    ],
                    inputs=[a],
                    fn=convert,
                )

        with gr.Column():
            o = gr.Video(label="Vídeo Resultante")

    btn.click(fn=convert, inputs=[seg, v, a], outputs=[o])

demo.queue().launch()