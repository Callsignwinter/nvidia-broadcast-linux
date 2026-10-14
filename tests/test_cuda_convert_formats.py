# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
import unittest
from types import SimpleNamespace
from unittest import mock

from gi.repository import Gst
from nvbroadcast.video.pipeline import VideoPipeline


class CudaConversionFormatTests(unittest.TestCase):
    def setUp(self):
        self.cache = VideoPipeline._cuda_convert_probe_results
        VideoPipeline._cuda_convert_probe_results = {}

    def tearDown(self):
        VideoPipeline._cuda_convert_probe_results = self.cache

    def test_bgra_success_does_not_enable_unsupported_yuy2(self):
        good = mock.Mock()
        good.get_bus.return_value.timed_pop_filtered.return_value = SimpleNamespace(type=Gst.MessageType.EOS)
        bad = mock.Mock()
        bad.get_bus.return_value.timed_pop_filtered.return_value = SimpleNamespace(type=Gst.MessageType.ERROR)
        with mock.patch.dict("os.environ", {"NVBROADCAST_NO_CUDA_CONVERT": "0"}), mock.patch.object(VideoPipeline, "_has_gst_element", return_value=True), mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch", side_effect=[good, bad]) as parse:
            pipeline = VideoPipeline()
            self.assertEqual(pipeline._convert_segment("BGRA"), "cudaupload ! cudaconvert ! cudadownload")
            self.assertEqual(pipeline._convert_segment("YUY2"), "videoconvert n-threads=2 qos=false")
            self.assertEqual(pipeline._convert_segment("BGRA"), "cudaupload ! cudaconvert ! cudadownload")
            self.assertEqual(parse.call_count, 2)
            self.assertIn("format=BGRA ! fakesink", parse.call_args_list[0].args[0])
            self.assertIn("format=YUY2 ! fakesink", parse.call_args_list[1].args[0])
        good.set_state.assert_any_call(Gst.State.NULL)
        bad.set_state.assert_any_call(Gst.State.NULL)

    def test_probe_failure_releases_pipeline_and_uses_cpu(self):
        broken = mock.Mock()
        broken.get_bus.side_effect = RuntimeError("CUDA unavailable")
        with mock.patch.dict("os.environ", {"NVBROADCAST_NO_CUDA_CONVERT": "0"}), mock.patch.object(VideoPipeline, "_has_gst_element", return_value=True), mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch", return_value=broken):
            self.assertFalse(VideoPipeline._cuda_convert_available("BGRA"))
        broken.set_state.assert_any_call(Gst.State.NULL)

    def test_explicit_disable_skips_gstreamer_probe(self):
        with mock.patch.dict("os.environ", {"NVBROADCAST_NO_CUDA_CONVERT": "1"}), mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch") as parse:
            self.assertFalse(VideoPipeline._cuda_convert_available("BGRA"))
            self.assertFalse(VideoPipeline._cuda_convert_available("YUY2"))
            parse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
