from .pyannote_vad import PyannoteVAD


class VADFactory:
    @staticmethod
    def create_vad_pipeline(type="pyannote", **kwargs):
        if type == "pyannote":
            return PyannoteVAD(**kwargs)
        else:
            raise ValueError(f"unknown vad pipeline")
