class VADInterface:
    async def detect_activity(self, client):
        raise NotImplementedError("this method should exist")
