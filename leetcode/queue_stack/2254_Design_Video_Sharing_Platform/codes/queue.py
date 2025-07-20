import heapq
from typing import List

class VideoSharingPlatform:
    def __init__(self):
        self.next_id = 0
        self.available_ids = [] 
        self.videos = {}  
        
    def upload(self, video: str) -> int:
        if self.available_ids:
            videoId = heapq.heappop(self.available_ids)
        else:
            videoId = self.next_id
            self.next_id += 1
        self.videos[videoId] = {
            "video": video,
            "views": 0,
            "likes": 0,
            "dislikes": 0
        }
        return videoId

    def remove(self, videoId: int) -> None:
        if videoId in self.videos:
            del self.videos[videoId]
            heapq.heappush(self.available_ids, videoId)

    def watch(self, videoId: int, startMinute: int, endMinute: int) -> str:
        if videoId not in self.videos:
            return "-1"
        video = self.videos[videoId]["video"]
        if startMinute >= len(video):
            return ""
        self.videos[videoId]["views"] += 1
        return video[startMinute:min(endMinute + 1, len(video))]

    def like(self, videoId: int) -> None:
        if videoId in self.videos:
            self.videos[videoId]["likes"] += 1

    def dislike(self, videoId: int) -> None:
        if videoId in self.videos:
            self.videos[videoId]["dislikes"] += 1

    def getLikesAndDislikes(self, videoId: int) -> List[int]:
        if videoId not in self.videos:
            return [-1]
        return [self.videos[videoId]["likes"], self.videos[videoId]["dislikes"]]

    def getViews(self, videoId: int) -> int:
        if videoId not in self.videos:
            return -1
        return self.videos[videoId]["views"]

# Your VideoSharingPlatform object will be instantiated and called as such:
# obj = VideoSharingPlatform()
# param_1 = obj.upload(video)
# obj.remove(videoId)
# param_3 = obj.watch(videoId,startMinute,endMinute)
# obj.like(videoId)
# obj.dislike(videoId)
# param_6 = obj.getLikesAndDislikes(videoId)
# param_7 = obj.getViews(videoId)