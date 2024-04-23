import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import time  # For adding delays

# Set Spotify client credentials
client_id = '13af931941594c7bb35ea80e02917ace'
client_secret = 'fbf315c1f4bb4195a58ce9e42654242d'
redirect_uri = 'http://localhost/'
scope = "playlist-read-private"

# Set up the Spotify API client
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
    )
)

# Get the user's playlists
playlists = sp.current_user_playlists()

# Create a DataFrame to store playlist and track information
df = pd.DataFrame(columns=['Playlist', 'Track', 'Artist', 'Track_ID'])

# Gather track details from all playlists
track_ids = []
for playlist in playlists['items']:
    playlist_id = playlist['id']
    playlist_name = playlist['name']
    
    # Get the tracks in the playlist
    tracks = sp.playlist_tracks(playlist_id)
    
    # Iterate over each track in the playlist
    for track in tracks['items']:
        track_name = track['track']['name']
        artist_name = track['track']['artists'][0]['name']
        track_id = track['track']['id']  # Store track ID
        
        # Append track information to the DataFrame
        new_row = {
            'Playlist': playlist_name,
            'Track': track_name,
            'Artist': artist_name,
            'Track_ID': track_id,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        track_ids.append(track_id)  # Collect track IDs for audio features

# Create a DataFrame to store audio features
df_features = pd.DataFrame()

# Fetch audio features in batches of 100 tracks with a 15-second delay between requests
batch_size = 100
for i in range(0, len(track_ids), batch_size):
    track_batch = track_ids[i:i + batch_size]  # Create batches of 100
    
    # Get audio features for the current batch
    features = sp.audio_features(tracks=track_batch)
    
    # Append audio features to DataFrame
    features_df = pd.DataFrame(features)
    df_features = pd.concat([df_features, features_df], ignore_index=True)
    
    # Wait for 15 seconds to avoid exceeding rate limits
    if i + batch_size < len(track_ids):  # Avoid unnecessary delay after the last batch
        time.sleep(15)

# Print the DataFrames
print("Playlist and track information:")
print(df)
df.to_csv('Total_Canciones.csv')

print("\nAudio features:")
print(df_features)
df_features.to_csv('Audio_Features.csv')
