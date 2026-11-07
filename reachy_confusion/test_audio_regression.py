import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
import confusion_storytelling as app

class AudioRegression(unittest.TestCase):
    def test_silence_never_sent_to_api(self):
        capture = app.AudioCapture(app._wav([np.zeros(1600)],16000),None,None,None,'webrtc_vad')
        client=Mock()
        with self.assertRaises(ValueError): app.transcribe(client,capture,'test')
        client.audio.transcriptions.create.assert_not_called()

    def test_no_detected_voice_never_sent_to_api(self):
        capture=app.AudioCapture(app._wav([np.full(1600,.001)],16000),None,None,None,'webrtc_vad')
        client=Mock()
        with self.assertRaises(ValueError): app.transcribe(client,capture,'test')
        client.audio.transcriptions.create.assert_not_called()

    def test_voice_sent_as_valid_wav(self):
        capture=app.AudioCapture(app._wav([np.full(1600,.1)],16000),10,100,1000,'webrtc_vad')
        client=Mock()
        client.audio.transcriptions.create.return_value.text=' hello '
        self.assertEqual(app.transcribe(client,capture,'test'),'hello')
        sent=client.audio.transcriptions.create.call_args.kwargs['file'][1]
        self.assertEqual(sent.read(4),b'RIFF')

    def test_backlog_drained_before_ready(self):
        mini=Mock()
        with patch.object(app.time,'monotonic',side_effect=[0,.1,.3]):
            app.prepare_recording(mini)
        mini.media.audio._appsink_audio.set_property.assert_any_call('max-buffers',1)
        mini.media.get_audio_sample.assert_called_once()

    def test_playback_waits_for_wav_duration(self):
        import pathlib
        client=Mock(); mini=Mock()
        response=Mock()
        from unittest.mock import MagicMock
        context=MagicMock(); context.__enter__.return_value=response
        client.audio.speech.with_streaming_response.create.return_value=context
        response.stream_to_file.side_effect=lambda path: pathlib.Path(path).write_bytes(app._wav([np.zeros(16000)],16000).getvalue())
        args=SimpleNamespace(demo=False,no_tts=False,tts_model='test',voice='test')
        with patch.object(app.time,'sleep') as sleep:
            self.assertTrue(app.say(client,mini,'hello',args))
            sleep.assert_called_once_with(1.3)

if __name__=='__main__': unittest.main()
