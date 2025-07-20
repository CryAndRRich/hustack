#include <bits/stdc++.h>
using namespace std;

class VideoSharingPlatform {
private:
    struct Video {
        string content;
        int views = 0;
        int likes = 0;
        int dislikes = 0;

        Video() = default;

        explicit Video(string s) : content(std::move(s)) {}
    };

    unordered_map<int, Video> videos;
    priority_queue<int, vector<int>, greater<>> availableIds;
    int nextId = 0;

public:
    VideoSharingPlatform() {}

    int upload(string video) {
        int id;
        if (!availableIds.empty()) {
            id = availableIds.top();
            availableIds.pop();
        } else {
            id = nextId++;
        }
        videos.emplace(id, Video(std::move(video)));
        return id;
    }

    void remove(int videoId) {
        if (videos.count(videoId)) {
            videos.erase(videoId);
            availableIds.push(videoId);
        }
    }

    string watch(int videoId, int startMinute, int endMinute) {
        if (!videos.count(videoId)) return "-1";
        auto& vid = videos[videoId];
        vid.views++;
        int end = min((int)vid.content.size(), endMinute + 1);
        return vid.content.substr(startMinute, end - startMinute);
    }

    void like(int videoId) {
        if (videos.count(videoId)) videos[videoId].likes++;
    }

    void dislike(int videoId) {
        if (videos.count(videoId)) videos[videoId].dislikes++;
    }

    vector<int> getLikesAndDislikes(int videoId) {
        if (!videos.count(videoId)) return {-1};
        return {videos[videoId].likes, videos[videoId].dislikes};
    }

    int getViews(int videoId) {
        return videos.count(videoId) ? videos[videoId].views : -1;
    }
};

/**
 * Your VideoSharingPlatform object will be instantiated and called as such:
 * VideoSharingPlatform* obj = new VideoSharingPlatform();
 * int param_1 = obj->upload(video);
 * obj->remove(videoId);
 * string param_3 = obj->watch(videoId,startMinute,endMinute);
 * obj->like(videoId);
 * obj->dislike(videoId);
 * vector<int> param_6 = obj->getLikesAndDislikes(videoId);
 * int param_7 = obj->getViews(videoId);
 */