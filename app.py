from datetime import datetime, timedelta
from flask import Flask, render_template, url_for, session, redirect, request, jsonify,send_from_directory
from authlib.integrations.flask_client import OAuth
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
from sqlalchemy import func
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json
from collections import Counter
from sqlalchemy import cast, Integer
from sqlalchemy.sql import func
load_dotenv()

import os
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

def get_vietnam_time():
    return datetime.utcnow() + timedelta(hours=7)

app = Flask(__name__)
app.secret_key = '696abc727'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Upload config
UPLOAD_FOLDER_IMAGES = os.path.join(os.getcwd(), 'uploads/images')
UPLOAD_FOLDER_AUDIO = os.path.join(os.getcwd(), 'uploads/audio')
os.makedirs(UPLOAD_FOLDER_IMAGES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_AUDIO, exist_ok=True)
app.config['UPLOAD_FOLDER_IMAGES'] = UPLOAD_FOLDER_IMAGES
app.config['UPLOAD_FOLDER_AUDIO'] = UPLOAD_FOLDER_AUDIO

# Init
CORS(app, origins=["http://localhost:5173"], supports_credentials=True)
db = SQLAlchemy(app)

# Models
class User(db.Model):
    email = db.Column(db.String(255), primary_key=True)
    history = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<User {self.email}>'

class Album(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text, nullable=False)
    image = db.Column(db.Text, nullable=False)
    desc = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<Album {self.name}>'

class Song(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text, nullable=False)
    image = db.Column(db.Text, nullable=False)
    file = db.Column(db.Text, nullable=False)
    desc = db.Column(db.Text, nullable=False)
    duration = db.Column(db.Text, nullable=False)
    album_id = db.Column(db.Integer, db.ForeignKey('album.id'))
    genre = db.Column(db.String(100), nullable=False, default='Unknown')
    album = db.relationship('Album', backref='songs')
    created_at = db.Column(db.DateTime, default=get_vietnam_time) 


    def __repr__(self):
        return f'<Song {self.name}>'

class PlaylistSong(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    playlist_id = db.Column(db.Integer, db.ForeignKey('playlist.id'))
    song_id = db.Column(db.Integer, db.ForeignKey('song.id'))

class Playlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text, nullable=False)
    image = db.Column(db.Text, nullable=False)
    desc = db.Column(db.Text, nullable=False)
    user_email = db.Column(db.String(255), db.ForeignKey('user.email'))  # NEW


# Utility

def name_exists(name_to_check):
    return db.session.query(User.query.filter_by(email=name_to_check).exists()).scalar()


# Create tables
with app.app_context():
    # db.drop_all()
    db.create_all()

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope' : 'profile email'},
    server_metadata_url= 'https://accounts.google.com/.well-known/openid-configuration'
)

# Routes
@app.route('/')
def home():
    email = dict(session).get('name',None)
    return f"Hello, {email}!"

@app.route('/login')
def login():
    google = oauth.create_client('google')
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    google = oauth.create_client('google')
    token = google.authorize_access_token()
    resp = google.get('userinfo')
    user_info = resp.json()
    # do something with the token and profile
    session['email'] = user_info['email']
    session['name'] = user_info['name']
    session['picture'] = user_info['picture']
    if not name_exists(session['email']):
        new_email = session['email']
        new_user = User(email=new_email,history='')
        #Do something here
        try:
            db.session.add(new_user)
            db.session.commit()
            return redirect('/')
        except Exception as e:
            print(str(e))
            return "Something went wrong",400
    else:
        try:
            get_user = User.query.filter_by(email=session['email']).first()
            history = get_user.history
            print(history)
            print(session['name'])
            #Do something Here
        except: 
            print('no history')

    return render_template('authorize.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/logout')
def logout():
    for key in list(session.keys()):
        session.pop(key)
    return redirect('/')

@app.route('/api/user')
def get_user():
    print(dict(session).get('name',None))
    return jsonify({"name": dict(session).get('name',None), "email": dict(session).get('email',None), 'picture' : dict(session).get('picture',None)})

@app.route('/api/album/<int:album_id>', methods=['GET'])
def get_album(album_id):
    album = Album.query.get(album_id)
    if album:
        return jsonify({
            "id": album.id,
            "name": album.name,
            "image": album.image,
            "desc": album.desc
        })
    return jsonify({"error": "Album not found"}), 404

@app.route('/api/album/upload', methods=['POST'])
def upload_album():
    name = request.form.get('name')
    desc = request.form.get('desc')
    image_file = request.files['image']

    image_filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
    image_file.save(image_path)
    image_url = f"{request.host_url}uploads/images/{image_filename}"

    album = Album(name=name, desc=desc, image=image_url)
    db.session.add(album)
    db.session.commit()

    return jsonify({"message": "Album uploaded successfully", "album_id": album.id})


@app.route('/api/song/<int:song_id>', methods=['GET'])
def get_song(song_id):
    song = Song.query.get(song_id)
    if not song:
        return jsonify({"error": "Song not found"}), 404
    email = session.get('email')
    if email:
        user = User.query.filter_by(email=email).first()
        if user:
            try:
                history = json.loads(user.history or "[]")
                history.append(song_id)
                history = history[-20:]  
                user.history = json.dumps(history)
                db.session.commit()
            except Exception as e:
                print(f"Lỗi cập nhật lịch sử nghe: {e}")
    return jsonify({
        "id": song.id,
        "name": song.name,
        "image": song.image,
        "file": song.file,
        "desc": song.desc,
        "duration": song.duration,
        "genre": song.genre,"album_id": song.album_id,  
        "album_name": song.album.name if song.album else None,
        "created_at": song.created_at.isoformat() if song.created_at else None
    })


@app.route('/api/songs', methods=['GET'])
def list_songs():
    songs = Song.query.all()
    return jsonify([
        {
            "id": s.id,
            "name": s.name,
            "image": s.image,
            "file": s.file,
            "desc": s.desc,
            "duration": s.duration
        } for s in songs
    ])

@app.route('/api/albums', methods=['GET'])
def list_albums():
    albums = Album.query.all()
    return jsonify([
        {
            "id": a.id,
            "name": a.name,
            "image": a.image,
            "desc": a.desc
        } for a in albums
    ])
@app.route('/api/album/<int:album_id>/songs', methods=['GET'])
def get_songs_by_album(album_id):
    songs = Song.query.filter_by(album_id=album_id).all()
    return jsonify([
        {
            "id": s.id,
            "name": s.name,
            "image": s.image,
            "file": s.file,
            "desc": s.desc,
            "duration": s.duration,
            "genre": s.genre,
            "created_at": s.created_at.isoformat() if s.created_at else None
        } for s in songs
    ])


@app.route('/api/song/upload', methods=['POST'])
def upload_song():
    name = request.form.get('name')
    desc = request.form.get('desc')
    duration = request.form.get('duration')
    album_id = request.form.get('album_id')
    genre=request.form.get('genre')
    image_file = request.files['image']
    audio_file = request.files['file']

    image_filename = secure_filename(image_file.filename)
    audio_filename = secure_filename(audio_file.filename)

    image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER_AUDIO'], audio_filename)

    image_file.save(image_path)
    audio_file.save(audio_path)

    image_url = f"{request.host_url}uploads/images/{image_filename}"
    audio_url = f"{request.host_url}uploads/audio/{audio_filename}"

    song = Song(name=name, desc=desc, duration=duration, image=image_url, file=audio_url, album_id=album_id,genre=genre)
    db.session.add(song)
    db.session.commit()

    return jsonify({"message": "Song uploaded successfully"})


@app.route('/api/song/search')
def search_song():
    query = request.args.get('q', '').strip()
    genre = request.args.get('genre', '').strip()
    duration_filter = request.args.get('duration', '').strip()

    results = Song.query

    # Tìm kiếm từ khóa đã loại bỏ khoảng trắng
    if query:
        normalized_name = func.replace(func.lower(Song.name), ' ', '')
        normalized_desc = func.replace(func.lower(Song.desc), ' ', '')
        normalized_genre = func.replace(func.lower(Song.genre), ' ', '')
        normalized_query = query.lower().replace(" ", "")

        results = results.filter(
            (normalized_name.ilike(f"%{normalized_query}%")) |
            (normalized_desc.ilike(f"%{normalized_query}%")) |
            (normalized_genre.ilike(f"%{normalized_query}%"))
        )

    # Lọc thể loại
    if genre:
        results = results.filter(func.lower(Song.genre) == genre.lower())

    # Lọc theo thời lượng mm:ss -> giây
    if duration_filter:
        try:
            minutes = cast(func.substr(Song.duration, 1, func.instr(Song.duration, ':') - 1), Integer)
            seconds = cast(func.substr(Song.duration, func.instr(Song.duration, ':') + 1), Integer)
            total_seconds = minutes * 60 + seconds

            if duration_filter == 'short':
                results = results.filter(total_seconds < 120)
            elif duration_filter == 'medium':
                results = results.filter(total_seconds.between(120, 300))
            elif duration_filter == 'long':
                results = results.filter(total_seconds > 300)
        except Exception as e:
            print("Lỗi filter duration:", e)

    songs = results.all()

    return jsonify([
        {
            "id": s.id,
            "name": s.name,
            "image": s.image,
            "file": s.file,
            "desc": s.desc,
            "duration": s.duration,
            "genre": s.genre,
            "album_name": s.album.name if s.album else None,
            "created_at": s.created_at.isoformat() if s.created_at else None
        } for s in songs
    ])


@app.route('/api/genres')
def get_genres():
    genres = db.session.query(Song.genre).distinct().all()
    cleaned = sorted({g[0].strip() for g in genres if g[0]})
    return jsonify(cleaned)


@app.route('/api/song/<int:song_id>', methods=['DELETE'])
def delete_song(song_id):
    song = Song.query.get(song_id)
    if not song:
        return jsonify({"error": "Song not found"}), 404
    db.session.delete(song)
    db.session.commit()
    return jsonify({"message": "Song deleted"})

@app.route('/api/songs/latest')
def get_latest_songs():
    songs = Song.query.order_by(Song.created_at.desc()).limit(10).all()
    return jsonify([
        {
            "id": s.id,
            "name": s.name,
            "image": s.image,
            "file": s.file,
            "desc": s.desc,
            "duration": s.duration,
            "genre": s.genre,
            "created_at": s.created_at.isoformat() if s.created_at else None
        } for s in songs
    ])



@app.route('/uploads/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_IMAGES'], filename)

@app.route('/uploads/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_AUDIO'], filename)

@app.route('/api/playlist/create', methods=['POST'])
def create_playlist():
    name = request.form.get('name')
    desc = request.form.get('desc')
    user_email = session.get('email')

    if not user_email:
        return jsonify({"error": "Unauthorized"}), 401

    image_file = request.files.get('image')
    image_url = ""
    if image_file:
        filename = secure_filename(image_file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], filename)
        image_file.save(path)
        image_url = f"{request.host_url}uploads/images/{filename}"

    playlist = Playlist(name=name, desc=desc, image=image_url, user_email=user_email)
    db.session.add(playlist)
    db.session.commit()
    return jsonify({"message": "Playlist created", "playlist_id": playlist.id})

@app.route('/api/playlist/user', methods=['GET'])
def get_user_playlists():
    user_email = session.get('email')
    if not user_email:
        return jsonify({"error": "Unauthorized"}), 401

    playlists = Playlist.query.filter_by(user_email=user_email).all()
    return jsonify([
        {
            "id": p.id,
            "name": p.name,
            "image": p.image,
            "desc": p.desc
        } for p in playlists
    ])

@app.route('/api/playlist/<int:playlist_id>/delete', methods=['DELETE'])
def delete_playlist(playlist_id):
    user_email = session.get('email')
    if not user_email:
        return jsonify({"error": "Unauthorized"}), 401

    playlist = Playlist.query.get(playlist_id)
    if not playlist or playlist.user_email != user_email:
        return jsonify({"error": "Not allowed"}), 403

    db.session.delete(playlist)
    db.session.commit()
    return jsonify({"success": True})


@app.route('/api/playlist/<int:playlist_id>', methods=['GET'])
def get_playlist_info(playlist_id):
    playlist = Playlist.query.get(playlist_id)
    if not playlist:
        return jsonify({"error": "Playlist not found"}), 404
    return jsonify({
        "id": playlist.id,
        "name": playlist.name,
        "image": playlist.image,
        "desc": playlist.desc,
        "user_email": playlist.user_email
    })

@app.route('/api/playlist/<int:playlist_id>/add_song', methods=['POST'])
def add_song_to_playlist(playlist_id):
    data = request.get_json()
    song_id = data.get('song_id')

    playlist = Playlist.query.get(playlist_id)
    song = Song.query.get(song_id)

    if not playlist or not song:
        return jsonify({"error": "Invalid playlist or song ID"}), 404

    
    existing = PlaylistSong.query.filter_by(playlist_id=playlist_id, song_id=song_id).first()
    if existing:
        return jsonify({"message": "Song already in playlist"}), 200

    link = PlaylistSong(playlist_id=playlist_id, song_id=song_id)
    db.session.add(link)
    db.session.commit()
    return jsonify({"message": "Song added to playlist"})

@app.route('/api/playlist/<int:playlist_id>/songs', methods=['GET'])
def get_songs_in_playlist(playlist_id):
    songs = db.session.query(Song).join(PlaylistSong).filter(PlaylistSong.playlist_id == playlist_id).all()
    return jsonify([
        {
            "id": s.id,
            "name": s.name,
            "image": s.image,
            "file": s.file,
            "desc": s.desc,
            "duration": s.duration
        } for s in songs
    ])

@app.route('/api/playlist/<int:playlist_id>/remove_song', methods=['POST'])
def remove_song_from_playlist(playlist_id):
    data = request.get_json()
    song_id = data.get('song_id')

    link = PlaylistSong.query.filter_by(playlist_id=playlist_id, song_id=song_id).first()
    if not link:
        return jsonify({"error": "Song not found in playlist"}), 404

    db.session.delete(link)
    db.session.commit()
    return jsonify({"message": "Song removed from playlist"})

@app.route('/api/playlist/<int:playlist_id>/upload_image', methods=['POST'])
def upload_playlist_image(playlist_id):
    playlist = Playlist.query.get(playlist_id)
    if not playlist:
        return jsonify({"error": "Playlist not found"}), 404

    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image file"}), 400

    filename = secure_filename(image_file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], filename)
    image_file.save(path)
    image_url = f"{request.host_url}uploads/images/{filename}"

    playlist.image = image_url
    db.session.commit()

    return jsonify({"message": "Image uploaded", "image": image_url})

@app.route('/api/user/history', methods=['GET'])
def get_listening_history():
    user_email = session.get('email')
    if not user_email:
        return jsonify({"error": "Unauthorized"}), 401

    user = User.query.filter_by(email=user_email).first()
    if not user or not user.history:
        return jsonify([]) 

    try:
        song_ids = json.loads(user.history)
        ordered_songs = [Song.query.get(sid) for sid in reversed(song_ids) if Song.query.get(sid)]
        return jsonify([{
            "id": s.id,
            "name": s.name,
            "image": s.image,
            "file": s.file,
            "desc": s.desc,
            "duration": s.duration,
            "genre": s.genre
        } for s in ordered_songs])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def recommend_by_genre_history(user, limit=10):
    if not user or not user.history:
        return []

    try:
        song_ids = json.loads(user.history)
        listened_songs = [Song.query.get(sid) for sid in song_ids if Song.query.get(sid)]
        genres = [s.genre for s in listened_songs if s.genre]

        if not genres:
            return []

        top_genre = Counter(genres).most_common(1)[0][0]
        recommended = Song.query.filter(Song.genre == top_genre).all()
        already_listened = set(song_ids)
        filtered = [s for s in recommended if s.id not in already_listened]

        return filtered[:limit]
    except Exception as e:
        print(f"[Genre Recommend] Error: {e}")
        return []
    
@app.route('/api/recommend_by_genre', methods=['GET'])
def api_recommend_by_genre():
    user_email = session.get('email')
    if not user_email:
        return jsonify({"error": "Unauthorized"}), 401

    user = User.query.filter_by(email=user_email).first()
    songs = recommend_by_genre_history(user, limit=10)

    return jsonify([{
        "id": s.id,
        "name": s.name,
        "image": s.image,
        "file": s.file,
        "desc": s.desc,
        "duration": s.duration,
        "genre": s.genre
    } for s in songs if s])

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


try:
    combined_vectors = np.load("recommender_model/combined_vectors.npy")
    song_ids = np.load("recommender_model/song_ids.npy")
    user_factors = np.load("recommender_model/user_factors.npy", allow_pickle=True).item()
except Exception as e:
    print("Lỗi khi load mô hình hybrid:", str(e))
    combined_vectors = None
    song_ids = None
    user_factors = {}

@app.route('/api/recommend_hybrid', methods=['GET'])
def recommend_hybrid():
    global combined_vectors, song_ids, user_factors
    try:
        user_email = session.get('email')
        if not user_email:
            return jsonify({"error": "Unauthorized"}), 401

        user = User.query.filter_by(email=user_email).first()
        if not user or not user.history or user_email not in user_factors:
            print(f"Không tìm thấy vector người dùng: {user_email}")
            return jsonify([])

        history_ids = json.loads(user.history)
        user_vector = user_factors[user_email]

        # Padding để khớp chiều
        if len(user_vector) < combined_vectors.shape[1]:
            padding = np.zeros(combined_vectors.shape[1] - len(user_vector))
            user_vector = np.hstack([user_vector, padding])
        elif len(user_vector) > combined_vectors.shape[1]:
            user_vector = user_vector[:combined_vectors.shape[1]]

      
        if np.isnan(user_vector).any():
            print("user_vector chứa NaN => thay bằng 0")
            user_vector = np.nan_to_num(user_vector)

        if np.isnan(combined_vectors).any():
            print("combined_vectors chứa NaN => thay bằng 0")
            combined_vectors = np.nan_to_num(combined_vectors)

        print(f"Độ dài vector người dùng: {user_vector.shape}")
        print(f"Số bài hát để xét gợi ý: {len(song_ids)}")
        print(f"Tất cả song_ids trong mô hình: {list(song_ids)}")
        print(f"History: {history_ids}")

        # Gợi ý
        sims = cosine_similarity([user_vector], combined_vectors).flatten()
        sorted_indices = sims.argsort()[::-1]

        recommended = []
        for idx in sorted_indices:
            song_id = int(song_ids[idx])
            if song_id not in history_ids:
                song = Song.query.get(song_id)
                if song:
                    print(f" Recommend: {song_id} - {song.name}")
                    recommended.append({
                        "id": song.id,
                        "name": song.name,
                        "image": song.image,
                        "file": song.file,
                        "desc": song.desc,
                        "duration": song.duration,
                        "genre": song.genre
                    })
                else:
                    print(f"song_id {song_id} không tìm thấy trong DB.")
            if len(recommended) >= 10:
                break

        print(f" Tổng bài hát được gợi ý: {len(recommended)}")
        return jsonify(recommended)

    except Exception as e:
        print(" [recommend_hybrid ERROR]:", str(e))
        return jsonify({"error": "Internal server error"}), 500



def ensure_model_trained():
    model_files = [
        "recommender_model/combined_vectors.npy",
        "recommender_model/song_ids.npy",
        "recommender_model/user_factors.npy"
    ]
    if not all(os.path.exists(f) for f in model_files):
        print(" Mô hình chưa có, huấn luyện lần đầu...")
        from train_recommender import train_hybrid_recommender
        train_hybrid_recommender()
    else:
        print(" Mô hình đã sẵn sàng.")

if __name__ == '__main__':
    ensure_model_trained()
    app.run(debug=True)
