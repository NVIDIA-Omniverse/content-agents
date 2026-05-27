# Environment Maps

`SmartMaterials_Environment_with_Lights.exr` is retained as a historical
Kit-parity and debugging reference for comparing OVRTX behavior against the
older Kit rendering path. It is no longer the active OVRTX default after
NVBug 6206112; the default light injection now uses the `StinsonBeach.hdr`
asset packaged with the isolated OVRTX runtime.
