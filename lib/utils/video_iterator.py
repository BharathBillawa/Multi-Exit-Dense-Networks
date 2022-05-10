import cv2

class VideoIterator:
    """Wrapper class that encapsulates video files as iterable objects.
    """
    def __init__(self, video_path, frame_size):
        self.video_path = video_path
        self.frame_size = frame_size
    
    def _init_cap(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if (self.cap.isOpened()== False): 
            assert "Error opening video stream or file"

    def __iter__(self):
        self._init_cap()
        return self

    def __len__(self):
        self._init_cap()
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __next__(self):
        ret, frame = self.cap.read()
        if ret == True:
            frame = cv2.resize(frame, self.frame_size)
            return frame

        else: 
            raise StopIteration
