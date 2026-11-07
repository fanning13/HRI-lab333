"""Inspect live robot audio without API calls, recordings, or motor commands."""
import argparse
import time
import numpy as np
from reachy_mini import ReachyMini


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-ip', default='192.168.8.199')
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error('--seconds must be 1–30')
    with ReachyMini(host=args.robot_ip, connection_mode='network', media_backend='webrtc') as mini:
        audio = mini.media.audio
        sink = getattr(audio, '_appsink_audio', None)
        if sink is None:
            raise RuntimeError('Expected a WebRTC audio sink')
        sink.set_property('max-buffers', 1)
        sink.set_property('drop', True)
        time.sleep(3)
        pad = sink.get_static_pad('sink')
        caps = pad.get_current_caps()
        print('Negotiated format:', caps.to_string() if caps else 'not available', flush=True)
        print('Speak near Reachy now; testing for', args.seconds, 'seconds.', flush=True)
        start = last = time.monotonic()
        count = 0
        peaks = None
        while time.monotonic() - start < args.seconds:
            chunk = mini.media.get_audio_sample()
            if chunk is not None and np.asarray(chunk).size:
                raw = np.asarray(chunk)
                current = np.max(np.abs(raw), axis=0)
                peaks = current if peaks is None else np.maximum(peaks, current)
                count += 1
            if time.monotonic() - last >= 1:
                print('chunks=', count, 'channel peaks=', peaks, flush=True)
                count, peaks, last = 0, None, time.monotonic()
            time.sleep(0.001)
        print('Done. No audio saved or sent to AI.', flush=True)


if __name__ == '__main__':
    main()
