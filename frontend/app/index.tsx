// Copyright (c) 2025 Mitchell Brenner
// Licensed under the GNU General Public License v3.0 (GPL-3.0-or-later)
// See LICENSE for details.

import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { useVideoPlayer, VideoView } from "expo-video";
import { useEvent } from "expo";
import * as FileSystem from "expo-file-system";
import * as MediaLibrary from "expo-media-library";

export default function UploadAndPlayVideo() {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const player = useVideoPlayer(videoUri, (player) => {
    player.loop = true;
    player.play();
  });

  const { isPlaying } = useEvent(player, "playingChange", {
    isPlaying: player.playing,
  });

  const handleSaveVideo = async () => {
    if (!videoUri) return;

    const { status } = await MediaLibrary.requestPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission denied",
        "Cannot save video without media access."
      );
      return;
    }

    try {
      await MediaLibrary.saveToLibraryAsync(videoUri);
      Alert.alert("Saved", "Video saved to your media library!");
    } catch (err) {
      console.error("Save error:", err);
      Alert.alert("Error", "Failed to save video.");
    }
  };

  const handlePickAndUpload = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "video/mp4",
        copyToCacheDirectory: true,
      });

      if (result.canceled || !result.assets?.length) return;

      const file = result.assets[0];
      setUploading(true);

      const formData = new FormData();
      formData.append("file", {
        uri: file.uri,
        name: file.name,
        type: "video/mp4",
      } as any);

      const response = await fetch(
        `${process.env.EXPO_PUBLIC_IP_ADDRESS}/process-video`,
        {
          method: "POST",
          headers: {
            "Content-Type": "multipart/form-data",
          },
          body: formData,
        }
      );

      if (!response.ok) {
        Alert.alert("Upload failed");
        return;
      }

      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        const json = await response.json();
        if (json.success === false) {
          Alert.alert(
            "No Watermark Detected",
            "This video appears to be clean already!"
          );
          return;
        }
      }

      const blob = await response.blob();
      const fileUri = FileSystem.documentDirectory + "processed.mp4";

      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64data = (reader.result as string).split(",")[1];
        if (!base64data) return;

        await FileSystem.writeAsStringAsync(fileUri, base64data, {
          encoding: FileSystem.EncodingType.Base64,
        });

        setVideoUri(fileUri);
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.error("Upload error:", err);
      Alert.alert("Error", "Something went wrong while uploading");
    } finally {
      setUploading(false);
    }
  };

  const handleClearVideo = () => {
    setVideoUri(null);
    player.pause();
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>LogoLess</Text>
      <Text style={styles.description}>
        Remove any TikTok watermark from your videos. Just upload your MP4, and
        we’ll clean it up for you—instantly.
      </Text>

      {!videoUri && !uploading && (
        <TouchableOpacity
          style={styles.uploadButton}
          onPress={handlePickAndUpload}
        >
          <Text style={styles.uploadButtonText}>Upload TikTok Video</Text>
        </TouchableOpacity>
      )}

      {uploading && (
        <>
          <ActivityIndicator style={{ marginTop: 20 }} size="large" />
        </>
      )}

      {videoUri && (
        <>
          <VideoView
            style={styles.video}
            player={player}
            allowsFullscreen
            allowsPictureInPicture
          />
          <View style={styles.controls}>
            <TouchableOpacity
              style={styles.controlButton}
              onPress={() => (isPlaying ? player.pause() : player.play())}
            >
              <Text style={styles.controlButtonText}>
                {isPlaying ? "Pause" : "Play"}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.controlButton, styles.clearButton]}
              onPress={handleClearVideo}
            >
              <Text style={styles.controlButtonText}>Clear Video</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.controlButton, { backgroundColor: "#1e90ff" }]}
              onPress={handleSaveVideo}
            >
              <Text style={styles.controlButtonText}>Save Video</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0c0c0c",
    padding: 20,
    alignItems: "center",
    justifyContent: "center",
  },
  header: {
    fontSize: 36,
    fontWeight: "bold",
    color: "#ffffff",
    marginBottom: 10,
  },
  description: {
    color: "#aaa",
    textAlign: "center",
    marginBottom: 30,
    fontSize: 16,
    lineHeight: 22,
    paddingHorizontal: 10,
  },
  uploadButton: {
    backgroundColor: "#1DB954",
    paddingVertical: 14,
    paddingHorizontal: 30,
    borderRadius: 12,
  },
  uploadButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  video: {
    width: "100%",
    height: 300,
    marginTop: 20,
    borderRadius: 12,
  },
  controls: {
    flexDirection: "row",
    marginTop: 20,
    gap: 10,
  },
  controlButton: {
    backgroundColor: "#333",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 10,
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    height: 50,
  },
  clearButton: {
    backgroundColor: "#ff4d4d",
  },
  controlButtonText: {
    color: "#fff",
    fontWeight: "600",
  },
});
