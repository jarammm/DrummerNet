import torch
import librosa
import numpy as np
import soundfile as sf
import os
import pretty_midi
from globals import *  # SR, DEVICE, DRUM_NAMES 등
from drummer_net import DrummerNet
from inst_src_set import get_instset_drum
from inst_dataset import load_drum_srcs
import argparser
from midi2audio import FluidSynth


def save_midi_from_activation(activation, output_mid_path, sr=SR):
    """
    activation: np.array of shape (n_channels, time)
    output_mid_path: 저장할 .mid 파일 경로
    sr: 샘플레이트
    """
    # 드럼 채널별 MIDI 노트 매핑 (필요에 따라 수정)
    drum_note_mapping = {0: 36, 1: 39, 2: 42, 3: 49}
    # MIDI 오브젝트 생성
    midi_object = pretty_midi.PrettyMIDI()
    drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
    
    # 각 채널에 대해 피크 픽킹 수행 후 note 이벤트 생성
    for ch in range(activation.shape[0]):
        # 피크 픽킹 파라미터 (evaluation.py의 pickpeak_fix 참고)
        pre_max = sr // 20
        post_max = sr // 20
        pre_avg = sr // 10
        post_avg = sr // 10
        delta = 1.0 / 4
        wait = sr // 16
        # 정규화
        act = activation[ch].copy()
        if act.max() > 0:
            act = act / act.max()
        else:
            act = act
        peak_idxs = librosa.util.peak_pick(
            act,
            pre_max=pre_max,
            post_max=post_max,
            pre_avg=pre_avg,
            post_avg=post_avg,
            delta=delta,
            wait=wait
        )
        
        # 각 피크를 note 이벤트로 추가 (노트 길이는 0.1초, velocity는 100)
        for idx in peak_idxs:
            start_time = librosa.samples_to_time(idx, sr=sr)
            end_time = start_time + 0.1
            note_number = drum_note_mapping.get(ch, 35)  # 기본값 35
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=end_time)
            drum_instrument.notes.append(note)
    
    midi_object.instruments.append(drum_instrument)
    midi_object.write(output_mid_path)
    print(f"MIDI 파일이 저장되었습니다: {output_mid_path}")
    
def run_inference(args):
    """
    args.input_wav_path: 처리할 wav 파일 경로
    args.checkpoint_path: 저장된 모델 체크포인트 경로
    args.output_audio_path: 추론 결과 오디오 저장 경로
    args.output_midi_path: 추론 결과 미디(representation)를 저장할 경로
    """
    # 1. 모델 구성 (main.py 참고)
    print("악기 소스 로딩...")
    inst_srcs = load_drum_srcs(idx=N_DRUM_VSTS)  # inst_dataset 모듈의 drum 소스 로딩 함수 (DRUM_NAMES도 globals에 정의되어 있어야 함)
    inst_names = DRUM_NAMES

    print("DrummerNet 모델 생성 중...")
    drummer_net = DrummerNet(inst_srcs, inst_names, get_instset_drum(norm=args.source_norm), args)
    drummer_net = drummer_net.to(DEVICE)
    
    # 2. 체크포인트 로드
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일이 없습니다: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location=DEVICE)
    drummer_net.load_state_dict(state_dict)
    print("모델 체크포인트를 성공적으로 로드했습니다.")
    drummer_net.eval()

    # 3. 입력 wav 파일 로드 및 전처리
    if not os.path.exists(args.input_wav_path):
        raise FileNotFoundError(f"입력 wav 파일이 없습니다: {args.input_wav_path}")
    print(f"입력 wav 파일 로드 중: {args.input_wav_path}")
    wav_data, sr_loaded = librosa.load(args.input_wav_path, sr=SR)
    wav_data = wav_data / np.abs(wav_data).max()  # 정규화

    # evaluation.py의 send_pred_reduce 참고: 좌우 패딩 추가 (모델에 맞게 수정 필요)
    pad = 4080
    wav_padded = np.concatenate([np.zeros(pad, dtype=wav_data.dtype), wav_data, np.zeros(pad, dtype=wav_data.dtype)])
    input_tensor = torch.tensor(wav_padded, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 4. 모델 추론 (forward() 리턴: x_trimmed, x_hat, y_hat)
    print("모델 추론 중...")
    with torch.no_grad():
        x_trimmed, x_hat, y_hat = drummer_net(input_tensor)
    
    midi_output = y_hat.squeeze(0).cpu().numpy()
    save_midi_from_activation(midi_output, args.output_midi_path, sr=SR)
    fs = FluidSynth(args.sound_font)
    fs.midi_to_audio(args.output_midi_path, args.output_audio_path)
    print(f"오디오 출력 파일 저장됨: {args.output_audio_path}")
    print(f"미디(representation) 출력 파일 저장됨: {args.output_midi_path}")

def main():
    # argparser 모듈을 통해 필요한 인자들을 파싱합니다.
    parser = argparser.ArgParser()
    # inference 관련 인자들 (입력 wav, 체크포인트, 출력 경로 등)을 추가합니다.
    parser.parser.add_argument("--input_wav_path", type=str, default="../data_evals/BDT_DRUMS/audio/4376.wav",
                               help="입력 wav 파일 경로")
    parser.parser.add_argument("--checkpoint_path", type=str, default="results/temp_exp1/items_8369375.pth",
                               help="모델 체크포인트 파일 경로")
    parser.parser.add_argument("--output_audio_path", type=str, default="test/sample4376pth8369375.wav",
                               help="추론 결과 오디오 저장 경로")
    parser.parser.add_argument("--output_midi_path", type=str, default="test/sample4376pth8369375.mid",
                               help="추론 결과 미디(representation) 저장 경로")
    parser.parser.add_argument("--sound_font", type=str, default="~/Downloads/FluidR3_GM/FluidR3_GM.sf2",
                               help="추론 결과 미디(representation) 저장 경로")
    
    args = parser.parse()
    
    print("입력 인자:", args)
    run_inference(args)

if __name__ == '__main__':
    main()
