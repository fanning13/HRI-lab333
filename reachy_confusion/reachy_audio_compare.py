"""Compare current and raw microphone routing; restore settings on exit."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import requests
from reachy_mini import ReachyMini

NAMES = ('AUDIO_MGR_OP_L', 'AUDIO_MGR_OP_R')

def read_config(base):
    result = {}
    for name in NAMES:
        r = requests.get(f'{base}/parameter/{name}', timeout=8)
        r.raise_for_status()
        result[name] = r.json()['values']
    return result

def apply_config(base, config):
    r = requests.post(f'{base}/apply', json={
        'config': [{'name': k, 'values': v} for k, v in config.items()],
        'verify': True,
    }, timeout=15)
    r.raise_for_status()
    if not r.json().get('applied'):
        raise RuntimeError(f'Configuration not verified: {r.text}')

def measure(mini, label):
    input(f'\n{label}: press Enter when ready, then speak for 8 seconds. ')
    # Drain old data before collecting this stage.
    until = time.monotonic() + 0.5
    while time.monotonic() < until:
        mini.media.get_audio_sample()
    print('SPEAK NOW', flush=True)
    start = last = time.monotonic()
    peaks = None
    while time.monotonic() - start < 8:
        chunk = mini.media.get_audio_sample()
        if chunk is not None and np.asarray(chunk).size:
            current = np.max(np.abs(np.asarray(chunk)), axis=0)
            peaks = current if peaks is None else np.maximum(peaks, current)
        if time.monotonic() - last >= 1:
            print(f'{label}: channel peaks={peaks}', flush=True)
            peaks, last = None, time.monotonic()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-ip', default='192.168.8.199')
    p.add_argument('--restore', type=Path, help='Restore a saved configuration and exit')
    args = p.parse_args()
    base = f'http://{args.robot_ip}:8000/api/audio/config'
    if args.restore:
        apply_config(base, json.loads(args.restore.read_text()))
        print('RESTORED', flush=True)
        return
    original = read_config(base)
    backup = Path(__file__).with_name(f'audio_route_backup_{time.time_ns()}.json')
    backup.write_text(json.dumps(original, indent=2))
    print('Original settings saved:', backup, flush=True)
    with ReachyMini(host=args.robot_ip, connection_mode='network', media_backend='webrtc') as mini:
        sink = getattr(mini.media.audio, '_appsink_audio', None)
        if sink is None:
            raise RuntimeError('WebRTC audio sink not found')
        sink.set_property('max-buffers', 1)
        sink.set_property('drop', True)
        time.sleep(3)
        try:
            measure(mini, 'CURRENT')
            for a, b in ((0, 1), (2, 3)):
                apply_config(base, {NAMES[0]: [1, a], NAMES[1]: [1, b]})
                time.sleep(1)
                measure(mini, f'RAW MICS {a}/{b}')
        finally:
            restored = False
            for attempt in range(3):
                try:
                    apply_config(base, original)
                    if read_config(base) != original:
                        raise RuntimeError('Readback differs from original')
                    restored = True
                    print('RESTORED: original audio routing verified.', flush=True)
                    break
                except Exception as exc:
                    print('Restore attempt failed:', exc, flush=True)
                    time.sleep(0.5)
            if not restored:
                print(f'RESTORE REQUIRED: python reachy_audio_compare.py --robot-ip {args.robot_ip} --restore "{backup}"', flush=True)
                raise RuntimeError('Could not restore audio routing')

if __name__ == '__main__':
    main()
