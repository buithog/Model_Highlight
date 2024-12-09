import pandas as pd

# Bước 1: Đọc danh sách nhãn từ vocab.txt, chỉ giữ lại các nhãn chính
with open("CAL500_noAudioFeatures/vocab.txt", "r") as file:
    all_labels = [line.strip() for line in file]

# Lọc chỉ giữ lại nhãn cảm xúc chính, bỏ qua các nhãn NOT
main_labels = [label for label in all_labels if not label.startswith("NOT-") and "Emotion-" in label]

# Bước 2: Đọc file hardAnnotations.txt
annotations = pd.read_csv("CAL500_noAudioFeatures/hardAnnotations.txt", header=None)

# Lọc chỉ giữ lại các cột tương ứng với nhãn chính
# Xác định index của các nhãn chính trong vocab.txt
main_label_indices = [i for i, label in enumerate(all_labels) if label in main_labels]

# Lọc các cột tương ứng với nhãn chính
filtered_annotations = annotations.iloc[:, main_label_indices]

# Bước 3: Lưu lại dữ liệu đã xử lý để sử dụng cho training
filtered_annotations.columns = main_labels  # Gắn tên cột là các nhãn chính
filtered_annotations.to_csv("filtered_annotations.csv", index=False)

print("Tiền xử lý hoàn tất. Dữ liệu đã được lưu với 18 nhãn cảm xúc.")
# Đường dẫn đến file songNames.txt
song_names_file = "CAL500_noAudioFeatures/songNames.txt"
# Đường dẫn đến thư mục lưu trữ các file âm thanh
audio_directory = "CAL500_32kps/CAL500_32kps"
# Tên file để lưu kết quả
output_file = "song_paths.txt"

# Đọc songNames.txt và tạo đường dẫn đầy đủ cho mỗi bài hát
with open(song_names_file, "r") as f:
    song_names = [line.strip() for line in f.readlines()]

# Tạo danh sách đường dẫn đầy đủ
song_paths = [f"{audio_directory}/{song_name}.mp3" for song_name in song_names]

# Ghi đường dẫn vào file output
with open(output_file, "w") as f:
    for path in song_paths:
        f.write(path + "\n")

print(f"Đã lưu đường dẫn vào file {output_file}")

# # # test file
# import os
#
# # Đường dẫn đến tệp song_paths.txt
# song_paths_file = 'song_paths.txt'
# # Đường dẫn đến thư mục chứa âm thanh
# audio_directory = 'CAL500_32kps/CAL500_32kps'
# cnt =0;
# # Đọc tệp song_paths.txt và kiểm tra từng tệp
# with open(song_paths_file, 'r') as file:
#     song_paths = file.readlines()
#
# # Kiểm tra từng đường dẫn tệp
# for song in song_paths:
#     song = song.strip()  # Loại bỏ khoảng trắng ở đầu và cuối
#     full_path = os.path.join(song)  # Tạo đường dẫn đầy đủ
#     print(f"Checking path: {full_path}")  # In đường dẫn đang kiểm tra
#     if os.path.exists(full_path):
#         print(f"{full_path} exists.")
#     else:
#         print(f"{full_path} does not exist.")
#         ++cnt
#
# print(cnt)