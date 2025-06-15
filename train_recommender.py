import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from flask_sqlalchemy import SQLAlchemy
from app import app, db, User, Song
import json, pickle, os

def train_hybrid_recommender():
    with app.app_context():
        # Bước 1: Lấy toàn bộ bài hát
        songs = Song.query.all()
        df_songs = pd.DataFrame([{
            "song_id": s.id,
            "Genre": s.genre if s.genre else "Unknown"
        } for s in songs])

        # Bước 2: Tạo tập tương tác từ lịch sử nghe (có số lượt nghe thật)
        interactions = []
        users = User.query.all()
        for user in users:
            if user.history:
                history_dict = json.loads(user.history)  # Dạng dict hoặc list
                if isinstance(history_dict, dict):
                    for sid, count in history_dict.items():
                        interactions.append((user.email, int(sid), int(count)))
                elif isinstance(history_dict, list):
                    for sid in history_dict:
                        interactions.append((user.email, int(sid), 1))

        if not interactions:
            print("⚠️ Không có dữ liệu lịch sử nghe để train.")
            return

        df_interact = pd.DataFrame(interactions, columns=['user_id', 'song_id', 'listen_count'])

        # Bước 3: Chuẩn hóa listen_count
        scaler = MinMaxScaler()
        df_interact['norm_listen'] = scaler.fit_transform(df_interact[['listen_count']])

        # Bước 4: Huấn luyện mô hình SVD
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(df_interact[['user_id', 'song_id', 'norm_listen']], reader)
        trainset = data.build_full_trainset()
        model = SVD(n_factors=50, reg_all=0.05, random_state=42)
        model.fit(trainset)

        # Bước 5: Vector bài hát
        svd_vectors = model.qi
        song_inner_ids = range(trainset.n_items)
        svd_song_ids = [int(trainset.to_raw_iid(i)) for i in song_inner_ids]

        svd_df = pd.DataFrame(svd_vectors, columns=[f"svd_{i}" for i in range(svd_vectors.shape[1])])
        svd_df['song_id'] = svd_song_ids

        # Bước 6: Metadata → OneHot → PCA
        encoder = OneHotEncoder()
        genre_encoded = encoder.fit_transform(df_songs[['Genre']]).toarray()

        pca = PCA(n_components=min(20, genre_encoded.shape[1]), random_state=42)
        meta_pca = pca.fit_transform(genre_encoded)

        meta_df = pd.DataFrame(meta_pca, columns=[f"pca_{i}" for i in range(meta_pca.shape[1])])
        meta_df['song_id'] = df_songs['song_id']

        # Bước 7: Merge SVD + Metadata
        merged = pd.merge(meta_df, svd_df, on='song_id', how='left')
        svd_cols = [col for col in merged.columns if col.startswith("svd_")]
        merged[svd_cols] = merged[svd_cols].fillna(0)

        combined_vectors = merged.drop(columns=['song_id']).values
        song_ids = merged['song_id'].values

        # Bước 8: Vector người dùng
        user_vectors = {trainset.to_raw_uid(i): model.pu[i] for i in range(trainset.n_users)}

        # Bước 9: Lưu mô hình
        os.makedirs("recommender_model", exist_ok=True)
        np.save("recommender_model/combined_vectors.npy", combined_vectors)
        np.save("recommender_model/song_ids.npy", song_ids)
        np.save("recommender_model/user_factors.npy", user_vectors)

        with open("recommender_model/encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
        with open("recommender_model/pca.pkl", "wb") as f:
            pickle.dump(pca, f)

        print(f"✅ Huấn luyện hybrid recommender hoàn tất – {len(song_ids)} bài hát.")

# Tùy chọn chạy trực tiếp file
if __name__ == '__main__':
    train_hybrid_recommender()
