from __future__ import annotations

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from nerf_xray.xray_dataparser import XrayDataParserConfig
from nerf_xray.multi_camera_dataparser import MultiCameraDataParserConfig

XrayDataparser = DataParserSpecification(config=XrayDataParserConfig())
MultiCameraDataParser = DataParserSpecification(config=MultiCameraDataParserConfig())
